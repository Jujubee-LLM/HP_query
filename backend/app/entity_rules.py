import json
import math
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass


_CJK_RE = re.compile(r"[\u4e00-\u9fff]{2,}")
_EN_TITLE_RE = re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2}\b")


_DEFAULT_STOP = {
    "什么",
    "怎么",
    "为什么",
    "为何",
    "哪里",
    "哪个",
    "哪些",
    "是否",
    "能否",
    "可以",
    "一下",
    "一段",
    "原文",
    "内容",
    "章节",
    "目录",
    "作者",
    "读者",
    "问题",
    "回答",
    "如下",
    "以上",
    "如下内容",
    "原文内容",
    "要求",
    "直接",
    "明确",
    "无法确定",
    "无法回答",
}


def _iter_cjk_ngrams(text: str, n_min: int, n_max: int) -> "list[str]":
    seqs = _CJK_RE.findall(text or "")
    out: list[str] = []
    for seq in seqs:
        s = seq.strip()
        if not s:
            continue
        L = len(s)
        if L < n_min:
            continue
        # O(L * (n_max-n_min)) but bounded by chunk size; OK for offline ingest.
        for n in range(n_min, n_max + 1):
            if L < n:
                continue
            for i in range(0, L - n + 1):
                out.append(s[i : i + n])
    return out


def _extract_candidate_terms(
    text: str,
    *,
    n_min: int,
    n_max: int,
    stop: set[str],
    per_chunk_keep: int,
) -> Counter[str]:
    t = (text or "").strip()
    if not t:
        return Counter()

    counts: Counter[str] = Counter()
    for ng in _iter_cjk_ngrams(t, n_min=n_min, n_max=n_max):
        if ng in stop:
            continue
        counts[ng] += 1

    for m in _EN_TITLE_RE.findall(t):
        term = re.sub(r"\s+", " ", m.strip())
        if not term or term in stop:
            continue
        counts[term] += 1

    if per_chunk_keep > 0 and len(counts) > per_chunk_keep:
        # Prefer higher TF, then longer spans (names often longer than common n-grams).
        items = sorted(counts.items(), key=lambda x: (x[1], len(x[0])), reverse=True)[:per_chunk_keep]
        return Counter(dict(items))
    return counts


@dataclass(frozen=True)
class EntityIndexParams:
    ngram_min: int = 2
    ngram_max: int = 6
    min_df: int = 2
    max_df_ratio: float = 0.20
    per_chunk_keep: int = 120
    max_postings_per_entity: int = 2000
    keep_earliest_per_entity: int = 200


def build_entity_index(chunks: list[dict], *, params: EntityIndexParams | None = None) -> dict:
    """
    Build a simple entity index from chunks using rule-based CJK n-grams + TitleCase spans.
    Output schema (JSON-serializable):
    - meta
    - entities: [{name, df, idf}]
    - entity_to_chunks: {entity: [{chunk_id, tf}]}
    """
    params = params or EntityIndexParams()
    stop = set(_DEFAULT_STOP)

    n_chunks = len(chunks)
    if n_chunks <= 0:
        return {"meta": {"chunks": 0}, "entities": [], "entity_to_chunks": {}}

    df: Counter[str] = Counter()
    tf_sum: Counter[str] = Counter()

    # Pass 1: get DF/TF with aggressive per-chunk pruning to control vocab growth.
    for c in chunks:
        text = str(c.get("text") or "")
        local = _extract_candidate_terms(
            text,
            n_min=int(params.ngram_min),
            n_max=int(params.ngram_max),
            stop=stop,
            per_chunk_keep=int(params.per_chunk_keep),
        )
        if not local:
            continue
        df.update(local.keys())
        tf_sum.update(local)

    max_df = max(int(math.ceil(n_chunks * float(params.max_df_ratio))), int(params.min_df))
    vocab = {
        term
        for term, d in df.items()
        if int(d) >= int(params.min_df) and int(d) <= max_df and term not in stop
    }

    # Pass 2: build postings for vocab.
    postings: dict[str, list[dict]] = defaultdict(list)
    for c in chunks:
        cid = str(c.get("chunk_id") or "")
        if not cid:
            continue
        text = str(c.get("text") or "")
        try:
            page_start = int(c.get("page_start") or 0)
        except Exception:
            page_start = 0
        local = _extract_candidate_terms(
            text,
            n_min=int(params.ngram_min),
            n_max=int(params.ngram_max),
            stop=stop,
            per_chunk_keep=int(params.per_chunk_keep),
        )
        if not local:
            continue
        for term, tf in local.items():
            if term not in vocab:
                continue
            if tf <= 0:
                continue
            postings[term].append({"chunk_id": cid, "tf": int(tf), "page_start": int(page_start)})

    entities: list[dict] = []
    for term in sorted(vocab):
        d = int(df.get(term, 0))
        if d <= 0:
            continue
        idf = math.log((n_chunks + 1.0) / (d + 1.0)) + 1.0
        entities.append({"name": term, "df": d, "idf": float(idf)})

    # Sort postings by tf desc but ensure earliest pages are retained too (critical for "first appearance" queries).
    max_post = int(params.max_postings_per_entity)
    keep_early = int(params.keep_earliest_per_entity)
    entity_to_chunks: dict[str, list[dict]] = {}
    for term, items in postings.items():
        items_by_tf = sorted(items, key=lambda x: int(x.get("tf") or 0), reverse=True)
        items_by_page = sorted(items, key=lambda x: int(x.get("page_start") or 0))

        out: list[dict] = []
        seen: set[str] = set()

        # Earliest mentions first (helps "第一次出现/首次登场").
        if keep_early > 0:
            for it in items_by_page[:keep_early]:
                cid = str(it.get("chunk_id") or "")
                if cid and cid not in seen:
                    out.append(it)
                    seen.add(cid)

        # Then high-TF mentions (helps general Q&A).
        for it in items_by_tf:
            cid = str(it.get("chunk_id") or "")
            if cid and cid not in seen:
                out.append(it)
                seen.add(cid)
            if max_post > 0 and len(out) >= max_post:
                break

        if max_post > 0 and len(out) > max_post:
            out = out[:max_post]

        entity_to_chunks[term] = out

    return {
        "meta": {
            "chunks": int(n_chunks),
            "ngram_min": int(params.ngram_min),
            "ngram_max": int(params.ngram_max),
            "min_df": int(params.min_df),
            "max_df_ratio": float(params.max_df_ratio),
            "per_chunk_keep": int(params.per_chunk_keep),
            "max_postings_per_entity": int(params.max_postings_per_entity),
            "keep_earliest_per_entity": int(params.keep_earliest_per_entity),
        },
        "entities": entities,
        "entity_to_chunks": entity_to_chunks,
    }


def save_entity_index(path: str, index: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False)
    os.replace(tmp, path)
