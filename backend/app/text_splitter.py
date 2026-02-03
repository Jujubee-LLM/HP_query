from dataclasses import dataclass
import re


@dataclass
class SplitChunk:
    text: str


_HEADING_RE = re.compile(
    r"^\s*("
    r"第[0-9一二三四五六七八九十百千]+[章節章节篇]|"
    r"[0-9]{1,2}\.[0-9]{1,2}(\.[0-9]{1,2})?|"
    r"[一二三四五六七八九十]+[、\.]"
    r")"
)
_LIST_RE = re.compile(r"^\s*(?:[-•]|[0-9]{1,2}[)\.、]|[（(][0-9]{1,2}[）)])\s+")


def _normalize_text_to_paragraphs(text: str) -> list[str]:
    text = (text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    if not text:
        return []

    # Keep blank lines as paragraph separators when present.
    raw_lines = [ln.strip() for ln in text.split("\n")]
    paras: list[str] = []
    buf: list[str] = []

    def flush() -> None:
        nonlocal buf
        if buf:
            paras.append(" ".join(buf).strip())
            buf = []

    def ends_sentence(s: str) -> bool:
        return bool(re.search(r"[。！？；:：…]\s*$", s))

    def should_hard_break(line: str) -> bool:
        if not line:
            return True
        if _HEADING_RE.match(line):
            return True
        if _LIST_RE.match(line):
            return True
        # Table-like: many spaces between tokens
        if "  " in line:
            return True
        return False

    for line in raw_lines:
        if not line:
            flush()
            continue
        # Start a new paragraph for headings/list items.
        if should_hard_break(line):
            flush()
            paras.append(line.strip())
            continue

        if not buf:
            buf.append(line)
            continue

        prev = buf[-1]
        # Merge wrapped lines: if previous doesn't end a sentence, join without space for CJK.
        if not ends_sentence(prev):
            if re.search(r"[\u4e00-\u9fff]$", prev) and re.search(r"^[\u4e00-\u9fff]", line):
                buf[-1] = prev + line
            else:
                buf[-1] = (prev + " " + line).strip()
        else:
            buf.append(line)

    flush()

    # Clean up excessive spaces created by joins.
    cleaned: list[str] = []
    for p in paras:
        p = re.sub(r"\s+", " ", p).strip()
        if p:
            cleaned.append(p)
    return cleaned


def _split_to_sentences(paragraph: str) -> list[str]:
    p = (paragraph or "").strip()
    if not p:
        return []
    if _HEADING_RE.match(p) or _LIST_RE.match(p):
        return [p]
    parts = re.split(r"(?<=[。！？；])\s+", p)
    out: list[str] = []
    for s in parts:
        s = s.strip()
        if s:
            out.append(s)
    return out if out else [p]


def split_text(
    text: str,
    max_chars: int = 900,
    *,
    overlap_paragraphs: int = 1,
    hard_cut_overlap_chars: int = 150,
    merge_min_chars: int = 200,
) -> list[SplitChunk]:
    paragraphs = _normalize_text_to_paragraphs(text)
    if not paragraphs:
        return []

    # Keep paragraph id for each unit so we can apply paragraph-aligned overlap between chunks.
    units: list[tuple[str, int]] = []
    for pid, p in enumerate(paragraphs):
        for s in _split_to_sentences(p):
            units.append((s, pid))

    chunks: list[list[tuple[str, int]]] = []
    current_parts: list[tuple[str, int]] = []
    current_len = 0

    def flush() -> None:
        nonlocal current_parts, current_len
        if current_parts:
            chunks.append(current_parts)
        current_parts = []
        current_len = 0

    def add_unit(u: str, pid: int) -> None:
        nonlocal current_len
        if not current_parts:
            current_parts.append((u, pid))
            current_len = len(u)
        else:
            current_parts.append((u, pid))
            current_len += len(u) + 1

    for u, pid in units:
        u = (u or "").strip()
        if not u:
            continue

        is_heading = (
            len(u) <= 50
            and (bool(re.match(r"^\s*第[0-9一二三四五六七八九十百千]+[章節章节篇]", u)) or bool(re.match(r"^\s*[0-9]{1,2}\.[0-9]{1,2}", u)))
        )
        if is_heading and current_parts:
            flush()
            add_unit(u, pid)
            continue

        if current_len + len(u) + (1 if current_parts else 0) <= max_chars:
            add_unit(u, pid)
            continue

        if current_parts:
            flush()

        if len(u) <= max_chars:
            add_unit(u, pid)
        else:
            # Fallback: hard cut very long units, but try to cut on whitespace when possible.
            start = 0
            while start < len(u):
                end = min(len(u), start + max_chars)
                if end < len(u):
                    cut = u.rfind(" ", start, end)
                    if cut > start + max_chars * 0.6:
                        end = cut
                    else:
                        # Also try to cut on punctuation for CJK text (better than arbitrary mid-sentence cuts).
                        punct = "。！？；：…，、"
                        best = -1
                        for ch in punct:
                            idx = u.rfind(ch, start, end)
                            if idx > best:
                                best = idx
                        if best > start + int(max_chars * 0.6):
                            end = best + 1
                chunks.append([(u[start:end].strip(), pid)])
                if end >= len(u):
                    break
                start = max(0, end - int(hard_cut_overlap_chars or 0))

    flush()

    # Merge pathological tiny chunks (often caused by headings/TOC-like lines).
    merged_chunks: list[list[tuple[str, int]]] = []
    i = 0
    min_chunk_chars = int(merge_min_chars or 0)
    while i < len(chunks):
        c_units = chunks[i]
        c = "\n".join(u for u, _ in c_units).strip()
        if len(c) >= min_chunk_chars or i == len(chunks) - 1:
            merged_chunks.append(c_units)
            i += 1
            continue
        nxt_units = chunks[i + 1]
        nxt = "\n".join(u for u, _ in nxt_units).strip()
        # Don't merge into a hard chapter heading chunk.
        nxt_is_chapter_heading = bool(re.match(r"^\s*第[0-9一二三四五六七八九十百千]+[章節章节篇]", nxt))
        if nxt_is_chapter_heading:
            merged_chunks.append(c_units)
            i += 1
            continue
        merged_chunks.append(c_units + nxt_units)
        i += 2
    chunks = merged_chunks

    out: list[SplitChunk] = []
    prev_units: list[tuple[str, int]] = []
    for c_units in chunks:
        merged_units = c_units
        if prev_units and int(overlap_paragraphs or 0) > 0:
            # NOTE: Despite the name, `overlap_paragraphs` is intentionally treated as a
            # small *sentence overlap count* to avoid pathological "whole previous chunk"
            # duplication when PDF extraction collapses many lines into one huge paragraph.
            want = max(0, int(overlap_paragraphs or 0))
            if want > 0:
                overlap_units = prev_units[-want:]
                if overlap_units:
                    merged_units = overlap_units + c_units

        text = "\n".join(u for u, _ in merged_units).strip()
        if text:
            # Guardrail: never emit identical adjacent chunks.
            if not out or out[-1].text != text:
                out.append(SplitChunk(text=text))

        prev_units = c_units
    return out
