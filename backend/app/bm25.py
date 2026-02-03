import math
import os
import re
from dataclasses import dataclass


_CJK_RE = re.compile(r"[\u4e00-\u9fff]+")
_ALNUM_RE = re.compile(r"[A-Za-z0-9]+")


def _cjk_bigrams(s: str) -> list[str]:
    s = s.strip()
    if len(s) <= 1:
        return [s] if s else []
    return [s[i : i + 2] for i in range(len(s) - 1)]


def default_tokenize(text: str) -> list[str]:
    text = (text or "").strip()
    if not text:
        return []
    tokens: list[str] = []
    for m in _CJK_RE.finditer(text):
        tokens.extend(_cjk_bigrams(m.group(0)))
    for m in _ALNUM_RE.finditer(text.lower()):
        tokens.append(m.group(0))
    return tokens


@dataclass(frozen=True)
class BM25Params:
    k1: float = 1.5
    b: float = 0.75


class BM25Index:
    def __init__(self, *, params: BM25Params | None = None):
        self.params = params or BM25Params()
        self.doc_len: dict[str, int] = {}
        self.avgdl: float = 0.0
        self.df: dict[str, int] = {}
        self.doc_tf: dict[str, dict[str, int]] = {}
        # Inverted index: term -> list of (cid, tf)
        self.postings: dict[str, list[tuple[str, int]]] = {}
        self.N: int = 0

    @classmethod
    def from_chunks(cls, chunks_by_id: dict[str, dict], *, params: BM25Params | None = None) -> "BM25Index":
        idx = cls(params=params)
        total_len = 0
        for cid, c in chunks_by_id.items():
            tokens = default_tokenize(c.get("text") or "")
            tf: dict[str, int] = {}
            for t in tokens:
                tf[t] = tf.get(t, 0) + 1
            if not tf:
                continue
            idx.doc_tf[cid] = tf
            dl = sum(tf.values())
            idx.doc_len[cid] = dl
            total_len += dl
            for t, f in tf.items():
                idx.df[t] = idx.df.get(t, 0) + 1
                idx.postings.setdefault(t, []).append((cid, int(f)))

        idx.N = len(idx.doc_tf)
        idx.avgdl = (total_len / idx.N) if idx.N else 0.0
        return idx

    def _idf(self, term: str) -> float:
        n = self.df.get(term, 0)
        if n <= 0:
            return 0.0
        # BM25+ style smoothing to keep idf positive for frequent terms
        return math.log(1.0 + (self.N - n + 0.5) / (n + 0.5))

    def score(self, query: str, cid: str) -> float:
        tf = self.doc_tf.get(cid)
        if not tf or self.avgdl <= 0:
            return 0.0
        q_terms = default_tokenize(query)
        if not q_terms:
            return 0.0
        k1 = self.params.k1
        b = self.params.b
        dl = float(self.doc_len.get(cid, 0))
        denom_norm = k1 * (1.0 - b + b * (dl / self.avgdl))

        score = 0.0
        seen: set[str] = set()
        for term in q_terms:
            if term in seen:
                continue
            seen.add(term)
            f = tf.get(term, 0)
            if f <= 0:
                continue
            idf = self._idf(term)
            num = float(f) * (k1 + 1.0)
            denom = float(f) + denom_norm
            score += idf * (num / denom)
        return float(score)

    def search(self, query: str, *, top_k: int) -> list[tuple[str, float]]:
        if top_k <= 0 or self.avgdl <= 0 or self.N <= 0:
            return []
        q_terms = default_tokenize(query)
        if not q_terms:
            return []

        k1 = self.params.k1
        b = self.params.b
        scores: dict[str, float] = {}

        seen: set[str] = set()
        for term in q_terms:
            if term in seen:
                continue
            seen.add(term)
            idf = self._idf(term)
            if idf <= 0:
                continue
            plist = self.postings.get(term)
            if not plist:
                continue
            for cid, f in plist:
                dl = float(self.doc_len.get(cid, 0))
                denom_norm = k1 * (1.0 - b + b * (dl / self.avgdl))
                num = float(f) * (k1 + 1.0)
                denom = float(f) + denom_norm
                scores[cid] = scores.get(cid, 0.0) + (idf * (num / denom))

        if not scores:
            return []
        scored = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return scored[:top_k]


_CACHE: tuple[str, float, BM25Index] | None = None


def get_bm25_index(chunks_path: str, chunks_by_id: dict[str, dict]) -> BM25Index:
    global _CACHE
    mtime = os.path.getmtime(chunks_path) if os.path.exists(chunks_path) else 0.0
    if _CACHE is not None:
        path0, mtime0, idx0 = _CACHE
        if path0 == chunks_path and mtime0 == mtime:
            return idx0
    idx = BM25Index.from_chunks(chunks_by_id)
    _CACHE = (chunks_path, mtime, idx)
    return idx
