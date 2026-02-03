import json
import os
import argparse
import re
import time
import hashlib
from typing import Iterable

from .faiss_store import FaissStore
from .llm import embed_texts
from .settings import settings
from .text_splitter import split_text
from .entity_rules import EntityIndexParams, build_entity_index, save_entity_index


def iter_batches(items: list[str], batch_size: int) -> Iterable[list[str]]:
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def read_pdf_pages(pdf_path: str) -> list[str]:
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    extractor = (settings.ingest_extractor or "auto").strip().lower()

    if extractor in ("auto", "pdfminer"):
        try:
            from pdfminer.high_level import extract_text  # type: ignore

            full = extract_text(pdf_path) or ""
            raw_pages = full.split("\f")
            pages = [p.strip() for p in raw_pages if p is not None]
            if pages and not pages[-1]:
                pages = pages[:-1]
            return pages
        except Exception as e:
            if extractor == "pdfminer":
                raise
            print(f"[ingest] pdfminer failed, fallback to pypdf: {e}")

    if extractor in ("auto", "pypdf"):
        from pypdf import PdfReader

        reader = PdfReader(pdf_path)
        pages: list[str] = []
        for p in reader.pages:
            text = (p.extract_text() or "").strip()
            pages.append(text)
        return pages

    raise ValueError(f"Unknown ingest_extractor: {settings.ingest_extractor}")


def infer_chapter(page_text: str, last_chapter: str) -> str:
    text = (page_text or "").strip()
    if not text:
        return last_chapter

    head = "\n".join(text.splitlines()[:40])

    patterns = [
        r"(第[0-9一二三四五六七八九十百千]+[章節章节篇][^\n]{0,40})",
        r"(^[0-9]+\s*[\.、]\s*[^\n]{1,60})",
        r"(^[一二三四五六七八九十]+[、\.]\s*[^\n]{1,60})",
    ]

    import re

    for p in patterns:
        m = re.search(p, head, flags=re.MULTILINE)
        if m:
            cand = m.group(1).strip()
            # Ignore common front-matter headings that would pollute chapter tracking.
            bad = ("制作说明", "画廊", "封面", "作者简介", "ISBN", "排版", "字体", "请访问", "Pottermore", "目录")
            if any(x in cand for x in bad):
                return last_chapter
            return cand
    return last_chapter


_PAGE_NUM_RE = re.compile(r"^\s*([0-9]{1,4}|[ivxlcdm]{1,8})\s*$", flags=re.IGNORECASE)


def _strip_boilerplate_pages(pages: list[str]) -> list[str]:
    """
    Remove repeated header/footer lines (common in PDFs) and standalone page numbers.
    Heuristic: lines that appear on many pages and are short are treated as boilerplate.
    """
    if not pages:
        return pages

    def norm_line(s: str) -> str:
        s = (s or "").strip()
        s = re.sub(r"\s+", " ", s)
        return s

    from collections import Counter

    header_counts: Counter[str] = Counter()
    footer_counts: Counter[str] = Counter()
    page_lines: list[list[str]] = []

    for p in pages:
        raw = [norm_line(x) for x in (p or "").splitlines() if norm_line(x)]
        page_lines.append(raw)
        head = raw[:3]
        tail = raw[-3:] if len(raw) >= 3 else raw
        header_counts.update(head)
        footer_counts.update(tail)

    n_pages = len(pages)
    # If a short line appears on >=30% pages as header/footer, treat it as boilerplate.
    threshold = max(3, int(n_pages * 0.30))
    boiler: set[str] = set()
    for line, c in header_counts.items():
        if c >= threshold and 2 <= len(line) <= 60:
            boiler.add(line)
    for line, c in footer_counts.items():
        if c >= threshold and 2 <= len(line) <= 60:
            boiler.add(line)

    cleaned_pages: list[str] = []
    for raw in page_lines:
        kept: list[str] = []
        for line in raw:
            if line in boiler:
                continue
            if _PAGE_NUM_RE.match(line):
                continue
            kept.append(line)
        cleaned_pages.append("\n".join(kept).strip())
    return cleaned_pages


def _is_toc_page(page_text: str) -> bool:
    text = (page_text or "").strip()
    if not text:
        return False
    if "目录" in text[:200]:
        return True
    lines = [re.sub(r"\s+", " ", ln.strip()) for ln in text.splitlines() if ln.strip()]
    if len(lines) < 10:
        return False
    page_num_like = 0
    for ln in lines:
        if len(ln) > 90:
            continue
        if re.search(r"\s[0-9]{1,4}\s*$", ln):
            page_num_like += 1
    return page_num_like >= max(8, int(len(lines) * 0.6))


def _is_toc_like(page_text: str) -> bool:
    """
    Detect TOC pages that list many chapter headings without explicit page numbers.
    Common in combined/compiled books.
    """
    text = (page_text or "").strip()
    if not text:
        return False
    lines = [re.sub(r"\s+", " ", ln.strip()) for ln in text.splitlines() if ln.strip()]
    if len(lines) < 12:
        return False
    heading = 0
    short = 0
    for ln in lines:
        if len(ln) <= 36:
            short += 1
        if re.match(r"^第[0-9一二三四五六七八九十百千]+[章節章节篇]\b", ln):
            heading += 1
        elif re.match(r"^第[一二三四五六七八九十百千]+部\b", ln):
            heading += 1
        elif re.match(r"^PART\\s+[A-Z0-9]+\\b", ln, flags=re.IGNORECASE):
            heading += 1
    # Heuristic: many headings and mostly short lines => likely TOC/list page.
    return heading >= 8 and (short / max(1, len(lines))) >= 0.7


def _is_front_matter_page(page_text: str) -> bool:
    """
    Filter out non-story pages: production notes, galleries, covers, author bio, etc.
    """
    text = (page_text or "").strip()
    if not text:
        return False
    head = text[:800]
    keywords = (
        "制作说明",
        "画廊",
        "封面",
        "作者简介",
        "统一书号",
        "ISBN",
        "排版",
        "字体",
        "请访问",
        "Pottermore",
        "版权",
        "出品",
    )
    if any(k in head for k in keywords):
        return True
    # Extremely repetitive short tokens like "封面封面封面" often appear on gallery pages.
    if len(text) <= 120 and len(set(text)) <= 6:
        return True
    return False


def _ends_sentence(text: str) -> bool:
    t = (text or "").strip()
    if not t:
        return True
    return bool(re.search(r"[。！？；:：…]\s*$", t))


def _merge_pages_for_sentences(pages: list[str], *, max_merge_pages: int = 2) -> list[tuple[int, int, str]]:
    """
    Merge consecutive pages when a sentence is likely split by a PDF page break.
    Returns list of (page_start, page_end, merged_text).
    """
    out: list[tuple[int, int, str]] = []
    i = 0
    n = len(pages)
    max_merge_pages = max(1, int(max_merge_pages or 1))
    while i < n:
        start = i + 1
        end = start
        cur = (pages[i] or "").strip()
        if not cur:
            out.append((start, end, ""))
            i += 1
            continue

        merged = cur
        merged_pages = 1
        while (
            merged_pages < max_merge_pages
            and i + 1 < n
            and merged
            and not _ends_sentence(merged)
            and (pages[i + 1] or "").strip()
        ):
            nxt = (pages[i + 1] or "").strip()
            merged = (merged.rstrip() + "\n" + nxt.lstrip()).strip()
            i += 1
            merged_pages += 1
            end = i + 1

        out.append((start, end, merged))
        i += 1
    return out


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Ingest PDF into chunks + FAISS index")
    p.add_argument(
        "--chunk-only",
        "--no-embed",
        dest="chunk_only",
        action="store_true",
        help="Only extract+chunk and write chunks.jsonl; skip embeddings + FAISS build",
    )
    p.add_argument(
        "--no-resume",
        dest="no_resume",
        action="store_true",
        help="Disable resume; always rebuild FAISS index from scratch.",
    )
    return p.parse_args(argv)


def _sha256_text(s: str) -> str:
    h = hashlib.sha256()
    h.update((s or "").encode("utf-8"))
    return h.hexdigest()


def _sha256_chunk_manifest(chunks: list[dict]) -> str:
    """
    Hash chunk identity + content so resume won't reuse embeddings when chunking changes.
    """
    h = hashlib.sha256()
    for c in chunks:
        cid = str(c.get("chunk_id") or "")
        text = str(c.get("text") or "")
        h.update(cid.encode("utf-8"))
        h.update(b"\t")
        h.update(text.encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def _ingest_state_path() -> str:
    return os.path.join(settings.faiss_dir, "ingest_state.json")


def _load_state(path: str) -> dict | None:
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_state(path: str, state: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)

    pdf_path = settings.data_pdf_path
    pages = read_pdf_pages(pdf_path)
    pages = _strip_boilerplate_pages(pages)
    # Drop front-matter pages by explicit start page (1-based).
    try:
        start_page = int(getattr(settings, "ingest_content_start_page", 1) or 1)
    except Exception:
        start_page = 1
    if start_page > 1:
        for i in range(min(len(pages), start_page - 1)):
            pages[i] = ""

    cleaned_pages: list[str] = []
    for i, p in enumerate(pages, start=1):
        if not (p or "").strip():
            cleaned_pages.append("")
            continue
        # Strongly filter TOC/front-matter (especially near the beginning).
        if _is_toc_page(p) or _is_toc_like(p):
            cleaned_pages.append("")
            continue
        if i < max(80, start_page) and _is_front_matter_page(p):
            cleaned_pages.append("")
            continue
        cleaned_pages.append(p)
    pages = cleaned_pages

    nonempty_pages = sum(1 for p in pages if (p or "").strip())
    total_chars = sum(len((p or "").strip()) for p in pages)
    print(f"[ingest] pages={len(pages)} nonempty_pages={nonempty_pages} total_chars={total_chars}")
    if nonempty_pages < settings.ingest_min_nonempty_pages or total_chars < settings.ingest_min_total_chars:
        raise RuntimeError(
            "PDF 文本抽取结果过少，无法建立可用索引。\n"
            f"- pages={len(pages)} nonempty_pages={nonempty_pages} total_chars={total_chars}\n"
            "可能原因：PDF 为扫描版（图片），或文本编码/嵌入方式导致抽取失败。\n"
            "修复建议：\n"
            "1) 换用包含可复制文本的 PDF；或先对 PDF 做 OCR（例如 ocrmypdf）再 ingest。\n"
            "2) 可尝试设置 INGEST_EXTRACTOR=pdfminer 或 INGEST_EXTRACTOR=pypdf 进行切换。\n"
            "3) 如确认文档很短，可调低 INGEST_MIN_NONEMPTY_PAGES / INGEST_MIN_TOTAL_CHARS。\n"
        )

    chunks = []
    current_chapter = ""
    merged_pages = _merge_pages_for_sentences(pages, max_merge_pages=2)
    for page_start, page_end, page_text in merged_pages:
        current_chapter = infer_chapter(page_text, current_chapter)
        for j, ch in enumerate(
            split_text(
                page_text,
                max_chars=int(getattr(settings, "ingest_chunk_max_chars", 900) or 900),
                overlap_paragraphs=int(getattr(settings, "ingest_chunk_overlap_paragraphs", 1) or 1),
                hard_cut_overlap_chars=int(getattr(settings, "ingest_chunk_hard_cut_overlap_chars", 150) or 150),
                merge_min_chars=int(getattr(settings, "ingest_chunk_merge_min_chars", 200) or 200),
            )
        ):
            text = (ch.text or "").strip()
            if not text:
                continue
            # Drop tiny chunks to avoid indexing headings/TOC-like artifacts.
            if len(text) < int(getattr(settings, "ingest_min_chunk_chars", 120) or 120):
                continue
            chunk_id = f"p{page_start}_c{j}"
            chunks.append(
                {
                    "chunk_id": chunk_id,
                    "page_start": page_start,
                    "page_end": page_end,
                    "chapter": current_chapter,
                    "text": text,
                }
            )

    os.makedirs(os.path.dirname(settings.chunks_path), exist_ok=True)
    with open(settings.chunks_path, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    if bool(getattr(settings, "ingest_build_entity_index", True)) and getattr(settings, "entity_index_path", None):
        try:
            params = EntityIndexParams(
                ngram_min=int(getattr(settings, "entity_ngram_min", 2) or 2),
                ngram_max=int(getattr(settings, "entity_ngram_max", 6) or 6),
                min_df=int(getattr(settings, "entity_min_df", 2) or 2),
                max_df_ratio=float(getattr(settings, "entity_max_df_ratio", 0.20) or 0.20),
                per_chunk_keep=int(getattr(settings, "entity_per_chunk_keep", 120) or 120),
                max_postings_per_entity=int(getattr(settings, "entity_max_postings_per_entity", 2000) or 2000),
                keep_earliest_per_entity=int(getattr(settings, "entity_keep_earliest_per_entity", 200) or 200),
            )
            entity_index = build_entity_index(chunks, params=params)
            save_entity_index(str(settings.entity_index_path), entity_index)
            print(f"[ingest] entity_index ok: {settings.entity_index_path}")
        except Exception as e:
            print(f"[ingest] entity_index failed: {type(e).__name__}: {e}")

    if args.chunk_only:
        print(f"Ingest chunk-only complete: chunks={len(chunks)}")
        print(f"- chunks_path={settings.chunks_path}")
        print("- skipped: embeddings + FAISS build")
        return

    texts = [c["text"] for c in chunks]
    batch_size = int(getattr(settings, "ingest_embed_batch_size", 32) or 32)
    timeout_s = float(getattr(settings, "ingest_embed_timeout_seconds", 60.0) or 60.0)
    max_retries = int(getattr(settings, "ingest_embed_max_retries", 6) or 6)
    backoff_s = float(getattr(settings, "ingest_embed_backoff_seconds", 1.0) or 1.0)
    backoff_max_s = float(getattr(settings, "ingest_embed_backoff_max_seconds", 15.0) or 15.0)

    chunk_ids_all = [c["chunk_id"] for c in chunks]
    batches = list(range(0, len(texts), batch_size))
    print(f"[ingest] embedding: batches={len(batches)} batch_size={batch_size} timeout={timeout_s}s")

    store = FaissStore(settings.faiss_dir)
    state_path = _ingest_state_path()

    # Resume logic: use existing FAISS index + id_map length as progress, and verify chunk identity.
    processed = 0
    expected_chunks_hash = _sha256_chunk_manifest(chunks)
    embed_model = settings.openai_embed_model
    if (not args.no_resume) and store.exists() and os.path.exists(state_path):
        try:
            store.load()
            state = _load_state(state_path) or {}
            processed = int(state.get("processed") or 0)
            if processed != len(store.id_map):
                processed = len(store.id_map)

            if state.get("chunks_hash") and state["chunks_hash"] != expected_chunks_hash:
                raise RuntimeError("chunks changed since last run")
            if state.get("embed_model") and state["embed_model"] != embed_model:
                raise RuntimeError("embed model changed since last run")
            if processed > 0:
                # Verify the already indexed ids match the current chunks prefix.
                prefix = chunk_ids_all[:processed]
                if store.id_map[:processed] != prefix:
                    raise RuntimeError("existing FAISS id_map does not match current chunks")
                print(f"[ingest] resume: processed={processed}/{len(chunks)}")
        except Exception as e:
            raise RuntimeError(
                "无法从上次进度继续：现有索引/状态与当前 chunks 不一致。\n"
                "如果你确实换了知识库或改了切分参数，请删除旧索引后重跑：\n"
                f"- 删除目录：{settings.faiss_dir}\n"
                "或执行：docker compose down && rm -rf storage/faiss && docker compose up -d\n"
                f"详细错误：{type(e).__name__}: {e}"
            ) from e
    else:
        # Fresh build: start with empty index and reset state.
        store.index = None
        store.id_map = []
        processed = 0

    # Ensure checkpoint exists before we start (so users can see progress even if batch 1 fails).
    _save_state(
        state_path,
        {
            "processed": int(processed),
            "total": int(len(chunks)),
            "chunks_hash": expected_chunks_hash,
            "embed_model": embed_model,
            "updated_at": time.time(),
        },
    )

    start_batch = processed // batch_size
    for bi in range(start_batch, len(batches)):
        start = bi * batch_size
        end = min(len(texts), start + batch_size)
        if start < processed:
            continue
        batch_texts = texts[start:end]
        batch_ids = chunk_ids_all[start:end]
        attempt = 0
        while True:
            attempt += 1
            t0 = time.perf_counter()
            try:
                vecs = embed_texts(batch_texts, timeout_seconds=timeout_s)
                store.add(vecs, batch_ids)
                store.save()
                processed = end
                _save_state(
                    state_path,
                    {
                        "processed": int(processed),
                        "total": int(len(chunks)),
                        "chunks_hash": expected_chunks_hash,
                        "embed_model": embed_model,
                        "updated_at": time.time(),
                    },
                )
                dt = time.perf_counter() - t0
                print(f"[ingest] embed batch {bi+1}/{len(batches)} ok ({len(batch_texts)} texts) {dt:.2f}s")
                break
            except Exception as e:
                dt = time.perf_counter() - t0
                msg = f"[ingest] embed batch {bi+1}/{len(batches)} failed attempt {attempt}/{max_retries} ({dt:.2f}s): {type(e).__name__}: {e}"
                print(msg)
                if attempt >= max_retries:
                    raise RuntimeError(
                        "Embeddings 连接失败/超时，ingest 中止。\n"
                        "排查建议：\n"
                        "1) 检查 OPENAI_API_KEY / OPENAI_BASE_URL 是否可用（容器内能访问）。\n"
                        "2) 若使用代理/网关，确保 Docker 也配置了 HTTP(S)_PROXY。\n"
                        "3) 调小 INGEST_EMBED_BATCH_SIZE（例如 8/16），并调大 INGEST_EMBED_TIMEOUT_SECONDS（例如 60/120）。\n"
                        "4) 网络不稳时，重试或更换更稳定的网关。\n"
                        f"最后错误：{type(e).__name__}: {e}"
                    ) from e
                sleep_s = min(backoff_max_s, backoff_s * (2 ** (attempt - 1)))
                time.sleep(max(0.0, sleep_s))

    # Mark done (keep the state file for audit/debug; users can delete it if they want).
    _save_state(
        state_path,
        {
            "processed": int(processed),
            "total": int(len(chunks)),
            "done": True,
            "chunks_hash": expected_chunks_hash,
            "embed_model": embed_model,
            "updated_at": time.time(),
        },
    )

    print(f"Ingest complete: chunks={len(chunks)}")
    print(f"- chunks_path={settings.chunks_path}")
    print(f"- faiss_dir={settings.faiss_dir}")
    print(f"- state_path={state_path}")


if __name__ == "__main__":
    main()
