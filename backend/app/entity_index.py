import json
import os
import re
import threading
from dataclasses import dataclass


class _TrieNode:
    __slots__ = ("children", "term")

    def __init__(self) -> None:
        self.children: dict[str, "_TrieNode"] = {}
        self.term: str | None = None


class EntityTrie:
    def __init__(self) -> None:
        self.root = _TrieNode()

    def add(self, term: str) -> None:
        t = (term or "").strip()
        if not t:
            return
        node = self.root
        for ch in t:
            nxt = node.children.get(ch)
            if nxt is None:
                nxt = _TrieNode()
                node.children[ch] = nxt
            node = nxt
        node.term = t

    def extract_longest(self, text: str, *, max_hits: int = 6) -> list[str]:
        """
        Greedy longest-match scanning, non-overlapping.
        Works well for short queries; avoids O(|V|) contains checks.
        """
        s = (text or "").strip()
        if not s:
            return []

        hits: list[str] = []
        i = 0
        while i < len(s) and len(hits) < max_hits:
            node = self.root
            j = i
            best: str | None = None
            best_end: int | None = None
            while j < len(s):
                node = node.children.get(s[j])  # type: ignore[assignment]
                if node is None:
                    break
                if node.term:
                    best = node.term
                    best_end = j + 1
                j += 1
            if best and best_end is not None:
                hits.append(best)
                i = best_end
            else:
                i += 1
        return hits


@dataclass(frozen=True)
class LoadedEntityIndex:
    path: str
    mtime: float
    index: dict
    trie: EntityTrie
    idf_by_entity: dict[str, float]


_ENTITY_LOCK = threading.Lock()
_ENTITY_CACHE: LoadedEntityIndex | None = None


def load_entity_index(path: str) -> LoadedEntityIndex | None:
    if not path:
        return None
    if not os.path.exists(path):
        return None

    mtime = os.path.getmtime(path)
    global _ENTITY_CACHE
    if _ENTITY_CACHE is not None and _ENTITY_CACHE.path == path and _ENTITY_CACHE.mtime == mtime:
        return _ENTITY_CACHE

    with _ENTITY_LOCK:
        if _ENTITY_CACHE is not None and _ENTITY_CACHE.path == path and _ENTITY_CACHE.mtime == mtime:
            return _ENTITY_CACHE

        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        entities = obj.get("entities")
        e2c = obj.get("entity_to_chunks")
        trie = EntityTrie()
        idf_by_entity: dict[str, float] = {}

        def _is_reasonable_entity_term(term: str) -> bool:
            t = (term or "").strip()
            if not t:
                return False
            if len(t) < 2 or len(t) > 24:
                return False
            # Drop obvious noisy n-grams from the ingest vocabulary.
            noisy_markers = (
                "第一次",
                "首次",
                "最早",
                "初次",
                "出现",
                "登场",
                "出场",
                "第",
                "章",
                "页",
                "……",
                "\n",
                "\r",
                "\t",
            )
            if any(m in t for m in noisy_markers):
                return False
            # Avoid fragments that are clearly not entities.
            if t[0] in ("的", "了", "和", "与", "在", "是", "有", "这", "那", "他", "她", "它"):
                return False
            if any(ch.isspace() for ch in t):
                return False
            return True
        if isinstance(entities, list):
            for e in entities:
                if not isinstance(e, dict):
                    continue
                name = e.get("name")
                idf = e.get("idf")
                if isinstance(name, str) and name:
                    trie.add(name)
                    if isinstance(idf, (int, float)):
                        idf_by_entity[name] = float(idf)

        # Also add entity_to_chunks keys (often contains CJK names) so entity recall works for Chinese KBs.
        if isinstance(e2c, dict):
            for term in e2c.keys():
                if not isinstance(term, str):
                    continue
                if not _is_reasonable_entity_term(term):
                    continue
                trie.add(term)
                # If no explicit IDF is provided, keep a neutral weight.
                idf_by_entity.setdefault(term, 1.0)

        loaded = LoadedEntityIndex(path=path, mtime=mtime, index=obj, trie=trie, idf_by_entity=idf_by_entity)
        _ENTITY_CACHE = loaded
        return loaded


def entity_recall(
    query: str,
    loaded: LoadedEntityIndex,
    *,
    max_query_entities: int = 3,
    postings_per_entity: int = 20,
    intent_first_appearance: bool = False,
    intent_last_appearance: bool = False,
) -> tuple[list[str], dict[str, float], list[str]]:
    """
    Returns: (ranked_chunk_ids, score_by_chunk_id, matched_entities)
    """
    max_query_entities = max(0, int(max_query_entities or 0))
    postings_per_entity = max(0, int(postings_per_entity or 0))
    if max_query_entities <= 0 or postings_per_entity <= 0:
        return [], {}, []

    entities = loaded.trie.extract_longest(query, max_hits=max_query_entities)
    if not entities:
        return [], {}, []

    # Filter out "non-entity" matches caused by naive n-gram vocab (e.g., "第一次出现").
    # Keep this conservative: only drop when the match clearly contains temporal/action query words.
    noisy_markers = (
        "第一次",
        "首次",
        "初次",
        "出现",
        "登场",
        "说",
        "做",
        "什么",
        "怎样",
        "如何",
    )
    filtered = [e for e in entities if not any(m in e for m in noisy_markers)]
    if filtered:
        entities = filtered

    e2c = loaded.index.get("entity_to_chunks")
    if not isinstance(e2c, dict):
        return [], {}, entities

    def _parse_page_start(chunk_id: str) -> int | None:
        m = re.match(r"^p([0-9]{1,7})_c[0-9]+$", chunk_id)
        if not m:
            return None
        try:
            return int(m.group(1))
        except Exception:
            return None

    score_by_id: dict[str, float] = {}
    for e in entities:
        items = e2c.get(e)
        if not isinstance(items, list) or not items:
            continue
        idf = float(loaded.idf_by_entity.get(e, 1.0))
        # For "first/last appearance" questions, prefer earliest/latest mentions instead of high-TF passages.
        view = items
        if intent_first_appearance or intent_last_appearance:
            scored: list[tuple[int, dict]] = []
            for it in items:
                if not isinstance(it, dict):
                    continue
                cid = it.get("chunk_id")
                if not isinstance(cid, str) or not cid:
                    continue
                ps = it.get("page_start")
                if isinstance(ps, int) and ps > 0:
                    page = ps
                else:
                    page = _parse_page_start(cid)
                ps = page
                if ps is None:
                    continue
                scored.append((ps, it))
            scored.sort(key=lambda x: x[0], reverse=bool(intent_last_appearance))
            view = [it for _ps, it in scored]

        for it in view[:postings_per_entity]:
            if not isinstance(it, dict):
                continue
            cid = it.get("chunk_id")
            tf = it.get("tf")
            if not isinstance(cid, str) or not cid:
                continue
            try:
                tfv = int(tf or 0)
            except Exception:
                tfv = 0
            if tfv <= 0:
                continue
            # Light TF scaling for robustness.
            s = idf * (1.0 + (tfv ** 0.5))
            score_by_id[cid] = score_by_id.get(cid, 0.0) + float(s)

    ranked = sorted(score_by_id.items(), key=lambda x: x[1], reverse=True)
    return [cid for cid, _s in ranked], score_by_id, entities
