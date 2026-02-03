import json
import os
import threading
import time
from collections import OrderedDict
import re

from .faiss_store import FaissStore
from .llm import embed_texts
from .rerank import rerank as rerank_chunks
from .planner import build_subqueries, make_plan
from .query_expansion import expand_query
from .settings import settings
from .bm25 import get_bm25_index, default_tokenize
from .entity_index import load_entity_index, entity_recall


_CHUNKS_CACHE_LOCK = threading.Lock()
_CHUNKS_CACHE: tuple[str, float, dict[str, dict]] | None = None

_CHUNK_ORDER_CACHE_LOCK = threading.Lock()
_CHUNK_ORDER_CACHE: tuple[str, float, list[str]] | None = None


def load_chunks(chunks_path: str) -> dict[str, dict]:
    if not os.path.exists(chunks_path):
        raise RuntimeError(f"Missing chunks file: {chunks_path}. Run ingest first.")
    global _CHUNKS_CACHE
    mtime = os.path.getmtime(chunks_path)
    if _CHUNKS_CACHE is not None:
        path0, mtime0, chunks0 = _CHUNKS_CACHE
        if path0 == chunks_path and mtime0 == mtime:
            return chunks0

    with _CHUNKS_CACHE_LOCK:
        if _CHUNKS_CACHE is not None:
            path0, mtime0, chunks0 = _CHUNKS_CACHE
            if path0 == chunks_path and mtime0 == mtime:
                return chunks0

    out: dict[str, dict] = {}
    with open(chunks_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            out[obj["chunk_id"]] = obj
    _CHUNKS_CACHE = (chunks_path, mtime, out)
    return out


def _ordered_chunk_ids(chunks_path: str, chunks: dict[str, dict]) -> list[str]:
    """
    Deterministic document order for chunks (by page_start, then within-page index in chunk_id).
    Cached by chunks file mtime.
    """
    if not chunks_path or not chunks:
        return []
    if not os.path.exists(chunks_path):
        return []
    mtime = os.path.getmtime(chunks_path)

    global _CHUNK_ORDER_CACHE
    if _CHUNK_ORDER_CACHE is not None:
        path0, mtime0, ids0 = _CHUNK_ORDER_CACHE
        if path0 == chunks_path and mtime0 == mtime:
            return ids0

    with _CHUNK_ORDER_CACHE_LOCK:
        if _CHUNK_ORDER_CACHE is not None:
            path0, mtime0, ids0 = _CHUNK_ORDER_CACHE
            if path0 == chunks_path and mtime0 == mtime:
                return ids0

        items: list[tuple[tuple[int, int], str]] = []
        for cid, c in chunks.items():
            try:
                ps = int(c.get("page_start") or 0)
            except Exception:
                ps = 0
            pos = _parse_chunk_position(str(cid), page_start=ps if ps > 0 else None)
            items.append((pos, str(cid)))
        items.sort(key=lambda x: x[0])
        ids = [cid for _pos, cid in items]
        _CHUNK_ORDER_CACHE = (chunks_path, mtime, ids)
        return ids


_STORE_CACHE_LOCK = threading.Lock()
_STORE_CACHE: tuple[str, float, float, FaissStore] | None = None


def get_store() -> FaissStore:
    """
    Keep FAISS index loaded in-process for latency.
    Auto-invalidates on index/map mtime changes (e.g., after ingest).
    """
    global _STORE_CACHE
    store = FaissStore(settings.faiss_dir)
    if not store.exists():
        raise RuntimeError(f"Missing FAISS index in {settings.faiss_dir}. Run ingest first.")

    index_mtime = os.path.getmtime(store.index_path)
    map_mtime = os.path.getmtime(store.map_path)
    if _STORE_CACHE is not None:
        dir0, index_mtime0, map_mtime0, store0 = _STORE_CACHE
        if dir0 == settings.faiss_dir and index_mtime0 == index_mtime and map_mtime0 == map_mtime:
            return store0

    with _STORE_CACHE_LOCK:
        if _STORE_CACHE is not None:
            dir0, index_mtime0, map_mtime0, store0 = _STORE_CACHE
            if dir0 == settings.faiss_dir and index_mtime0 == index_mtime and map_mtime0 == map_mtime:
                return store0

        store.load()
        _STORE_CACHE = (settings.faiss_dir, index_mtime, map_mtime, store)
        return store


_TEMPORAL_INTENT_RE = re.compile(r"(第一次|首次|最早|初次|初登场|第一次出现|首次出现|第一次出场|首次出场|第一次登场)")

_TEMPORAL_NOISE_RE = re.compile(r"(怎样|怎么|如何|什么|是怎|他说|做了|说了|做什么|场景|台词|动作)")

def _extract_temporal_target_entity(question: str) -> str | None:
    """
    Extract the most likely target entity for "first/earliest appearance" questions.
    Prefer the name immediately before the temporal marker to avoid picking up noisy fragments like "是怎".
    """
    q = (question or "").strip()
    if not q:
        return None
    m = _TEMPORAL_INTENT_RE.search(q)
    if not m:
        return None
    left = q[: m.start()].strip()
    if left:
        cands = re.findall(r"[\u4e00-\u9fff·]{2,12}", left)
        # Pick the last token before the temporal marker (often the name).
        for c in reversed(cands):
            if not c:
                continue
            if _TEMPORAL_NOISE_RE.search(c):
                continue
            # Drop ultra-common function words and very generic nouns.
            if c in ("人物", "角色", "谁", "哪位", "哪人", "这人", "这个人"):
                continue
            return c
    return None

def _derive_aliases_from_text(chunks_in_order: list[str], chunks: dict[str, dict], target: str, *, max_scan: int = 80) -> list[str]:
    """
    Derive lightweight aliases for a target entity using early occurrences.
    Example: "塞德里克迪戈里" -> alias "迪戈里" (surname-only mentions are common later).
    """
    t = (target or "").strip()
    if not t:
        return []

    aliases: list[str] = [t]
    seen: set[str] = {t}

    # Only accept very "name-like" candidates; this must NOT pull in generic phrases.
    bad_chars = ("在", "的", "了", "是", "有", "这", "那", "我", "你", "他", "她", "它", "们", "与", "和")
    stop = {
        "先生",
        "女士",
        "教授",
        "队长",
        "找球手",
        "学院",
        "学生",
        "爸爸",
        "妈妈",
        "第一个",
        "第一场",
    }

    scanned = 0
    dot_surname_re = re.compile(rf"{re.escape(t)}·([\u4e00-\u9fff]{{2,4}})")
    concat_surname_re = re.compile(rf"{re.escape(t)}([\u4e00-\u9fff]{{2,4}})(?=[，。,。；;：:\n\r\t \u2014-]|$)")
    any_dot_re = re.compile(r"·([\u4e00-\u9fff]{2,4})")
    dot_vocab: set[str] = set()
    concat_cands: dict[str, int] = {}
    for cid in chunks_in_order:
        c = chunks.get(cid)
        if not c:
            continue
        text = str(c.get("text") or "")
        if t not in text:
            continue
        scanned += 1
        if scanned > max_scan:
            break

        for dm in any_dot_re.finditer(text):
            dv = (dm.group(1) or "").strip()
            if dv and 2 <= len(dv) <= 4:
                dot_vocab.add(dv)

        for m in dot_surname_re.finditer(text):
            cand = (m.group(1) or "").strip()
            if not cand or len(cand) < 2:
                continue
            if cand in stop or _TEMPORAL_NOISE_RE.search(cand):
                continue
            if any(ch in cand for ch in bad_chars):
                continue
            # Drop obvious non-name suffixes.
            if cand.endswith(("说", "道", "问", "指", "过", "大", "小")):
                continue
            if cand not in seen:
                seen.add(cand)
                aliases.append(cand)

        for m in concat_surname_re.finditer(text):
            cand = (m.group(1) or "").strip()
            if not cand or len(cand) < 2:
                continue
            if cand in stop or _TEMPORAL_NOISE_RE.search(cand):
                continue
            if any(ch in cand for ch in bad_chars):
                continue
            if cand.endswith(("说", "道", "问", "指", "过", "大", "小")):
                continue
            concat_cands[cand] = concat_cands.get(cand, 0) + 1

    # Only keep concatenated candidates that also appear as a dot-surname somewhere
    # (filters out generic verb phrases like “离开礼堂”).
    for cand, _n in sorted(concat_cands.items(), key=lambda x: (-x[1], -len(x[0]))):
        if cand in dot_vocab and cand not in seen:
            seen.add(cand)
            aliases.append(cand)

    aliases.sort(key=lambda x: len(x), reverse=True)
    return aliases

def _participation_score(text: str, aliases: list[str]) -> int:
    """
    Heuristic score for "on-stage participation" (as opposed to being merely mentioned/introduced).
    Designed for novel/event questions: match alias + functional verbs/results within a short window.
    """
    t = text or ""
    if not t or not aliases:
        return 0
    # High-confidence event/participation patterns.
    hi_verbs = (
        "抓到",
        "抓住",
        "拿到",
        "赢",
        "获胜",
        "打败",
        "击中",
        "杀死",
        "救",
        "救了",
        "追上",
        "击倒",
        "扑向",
        "冲向",
        "出手",
    )
    event_words = ("比赛", "魁地奇", "决斗", "战斗", "项目", "任务", "行动", "交锋", "对抗")

    score = 0
    if any(w in t for w in event_words):
        score += 1

    for a in aliases:
        if a not in t:
            continue
        # Explicit functional role often signals "introduced", not participation; keep low weight.
        if re.search(rf"{re.escape(a)}.{{0,8}}(?:队长|找球手|勇士|选手)", t):
            score += 1

        # Participation: alias close to a strong verb/result, or to sports-specific object.
        if re.search(rf"{re.escape(a)}.{{0,10}}(?:{'|'.join(hi_verbs)})", t):
            score += 4
        if re.search(rf"{re.escape(a)}.{{0,12}}(?:金色飞贼|飞贼|金蛋|项目)", t):
            score += 2
        # "X说/问" is weaker than participation but still indicates on-stage presence.
        if re.search(rf"{re.escape(a)}.{{0,8}}(?:说|问|答道|说道|喊道|叫道|嘟囔|告诉)", t):
            score += 2

    return score


def _parse_chunk_position(chunk_id: str, *, page_start: int | None = None) -> tuple[int, int]:
    """
    Return sortable position for a chunk: (page, within_page_index).
    Falls back to a large sentinel if parsing fails.
    """
    page = int(page_start) if isinstance(page_start, int) else 10**9
    idx = 10**9
    m = re.match(r"^p([0-9]{1,7})_c([0-9]{1,5})$", str(chunk_id or ""))
    if m:
        try:
            page = int(m.group(1))
        except Exception:
            pass
        try:
            idx = int(m.group(2))
        except Exception:
            pass
    return page, idx


def _extract_temporal_entities(question: str) -> list[str]:
    """
    Best-effort entity extraction for temporal questions.
    Prefers the ingest-built entity index trie; falls back to simple CJK span extraction.
    """
    q = (question or "").strip()
    if not q:
        return []

    # First, try a deterministic extraction anchored on the temporal marker.
    target = _extract_temporal_target_entity(q)
    if target:
        return [target]

    loaded = load_entity_index(str(getattr(settings, "entity_index_path", "") or ""))
    if loaded is not None:
        try:
            ents = loaded.trie.extract_longest(q, max_hits=3)
        except Exception:
            ents = []
        filtered = [e for e in ents if isinstance(e, str) and e and not _TEMPORAL_NOISE_RE.search(e)]
        if filtered:
            return filtered[:3]
        if ents:
            return ents[:3]

    cands = re.findall(r"[\u4e00-\u9fff·]{2,10}", q)
    out = [c for c in cands if c and not _TEMPORAL_NOISE_RE.search(c)]
    return out[:3]


_QUERY_EMB_LOCK = threading.Lock()
_QUERY_EMB_CACHE: "OrderedDict[str, list[float]]" = OrderedDict()


def _get_query_embeddings(questions: list[str], timing_detail_ms: dict[str, int]) -> dict[str, list[float]]:
    """
    Embed multiple queries with an in-process LRU cache to avoid repeated embedding calls.
    Uses a single batched embedding request for cache misses (important for query-expansion).
    """
    if not questions:
        return {}

    cache_size = int(getattr(settings, "retrieval_query_embed_cache_size", 0) or 0)

    # Preserve order while de-duplicating to avoid redundant work.
    uniq: list[str] = []
    seen: set[str] = set()
    for q in questions:
        if q not in seen:
            seen.add(q)
            uniq.append(q)

    out: dict[str, list[float]] = {}
    misses: list[str] = []

    if cache_size > 0:
        with _QUERY_EMB_LOCK:
            for q in uniq:
                hit = _QUERY_EMB_CACHE.get(q)
                if hit is not None:
                    _QUERY_EMB_CACHE.move_to_end(q, last=True)
                    out[q] = hit
                else:
                    misses.append(q)
    else:
        misses = list(uniq)

    if misses:
        t = time.perf_counter()
        embs = embed_texts(misses)
        timing_detail_ms["embed"] = timing_detail_ms.get("embed", 0) + int((time.perf_counter() - t) * 1000)

        for q, emb in zip(misses, embs):
            out[q] = emb

        if cache_size > 0:
            with _QUERY_EMB_LOCK:
                for q in misses:
                    _QUERY_EMB_CACHE[q] = out[q]
                    _QUERY_EMB_CACHE.move_to_end(q, last=True)
                while len(_QUERY_EMB_CACHE) > cache_size:
                    _QUERY_EMB_CACHE.popitem(last=False)

    return out


def _get_query_embedding(question: str, timing_detail_ms: dict[str, int]) -> list[float]:
    return _get_query_embeddings([question], timing_detail_ms)[question]


def _weighted_rrf(
    vec_rank: dict[str, int],
    bm25_rank: dict[str, int],
    *,
    alpha: float,
    k0: int,
) -> dict[str, float]:
    out: dict[str, float] = {}
    for cid, r in vec_rank.items():
        out[cid] = out.get(cid, 0.0) + (alpha / (k0 + r))
    for cid, r in bm25_rank.items():
        out[cid] = out.get(cid, 0.0) + ((1.0 - alpha) / (k0 + r))
    return out


def _unique_token_overlap_count(query: str, doc_text: str) -> int:
    q_tokens = set(default_tokenize(query))
    if not q_tokens:
        return 0
    d_tokens = set(default_tokenize(doc_text))
    if not d_tokens:
        return 0
    return sum(1 for t in q_tokens if t in d_tokens)


def _gap_cutoff(scores_desc: list[float], gap_threshold: float) -> float | None:
    if len(scores_desc) < 2:
        return None
    best_drop = 0.0
    best_i = None
    for i in range(len(scores_desc) - 1):
        drop = scores_desc[i] - scores_desc[i + 1]
        if drop > best_drop:
            best_drop = drop
            best_i = i
    if best_i is None or best_drop < gap_threshold:
        return None
    return scores_desc[best_i] - 1e-6


def _gap_cutoff_ratio(scores_desc: list[float], gap_ratio_threshold: float) -> float | None:
    if len(scores_desc) < 2:
        return None
    max_score = float(scores_desc[0])
    if max_score <= 0:
        return None
    best_drop_ratio = 0.0
    best_i = None
    for i in range(len(scores_desc) - 1):
        drop = float(scores_desc[i]) - float(scores_desc[i + 1])
        drop_ratio = drop / max_score
        if drop_ratio > best_drop_ratio:
            best_drop_ratio = drop_ratio
            best_i = i
    if best_i is None or best_drop_ratio < gap_ratio_threshold:
        return None
    return float(scores_desc[best_i]) - 1e-6


def _apply_thresholding(hits: list[dict], *, threshold_mode: str = "vector") -> tuple[list[dict], dict]:
    if not hits:
        return hits, {
            "retrieved_k": 0,
            "kept_k": 0,
            "max_score": None,
            "cutoff_score": None,
            "strategy": "empty",
        }

    scores_desc = [float(h["score"]) for h in hits]
    max_score = scores_desc[0]

    if threshold_mode in ("hybrid", "rerank"):
        # Hybrid (RRF-fused) scores are small positives; use relative cutoffs.
        cutoff = max_score * (1.0 - float(settings.retrieval_score_delta))
        gap_cut = _gap_cutoff_ratio(scores_desc, float(settings.retrieval_gap_threshold))
        if gap_cut is not None:
            cutoff = max(cutoff, gap_cut)
            strategy = "auto_ratio+gap_ratio"
        else:
            strategy = "auto_ratio"
        cutoff = max(0.0, float(cutoff))
    else:
        cutoff = max_score - float(settings.retrieval_score_delta)
        gap_cut = _gap_cutoff(scores_desc, float(settings.retrieval_gap_threshold))
        if gap_cut is not None:
            cutoff = max(cutoff, gap_cut)
            strategy = "auto_delta+gap"
        else:
            strategy = "auto_delta"

    filtered = [h for h in hits if float(h["score"]) >= cutoff]
    if len(filtered) < int(settings.retrieval_min_keep):
        filtered = hits[: int(settings.retrieval_min_keep)]
        cutoff = float(filtered[-1]["score"]) if filtered else cutoff
        strategy = strategy + "+min_keep"

    meta = {
        "retrieved_k": len(hits),
        "kept_k": len(filtered),
        "max_score": float(max_score),
        "cutoff_score": float(cutoff),
        "strategy": strategy,
    }
    return filtered, meta


def _fuse_entity_rrf(
    question: str,
    ranked_ids: list[str],
    ranked_scores: list[float],
    timing_detail_ms: dict[str, int],
    chunks: dict[str, dict] | None,
) -> tuple[list[str], list[float], dict]:
    if not bool(getattr(settings, "entity_enabled", False)):
        return ranked_ids, ranked_scores, {}

    loaded = load_entity_index(str(getattr(settings, "entity_index_path", "") or ""))
    if loaded is None:
        return ranked_ids, ranked_scores, {"entity_enabled": True, "entity_status": "no_index"}

    # Detect intent before entity recall to pick the right postings view.
    intent_first = bool(re.search(r"(第一次|首次|初次).{0,6}(出现|登场)", question))
    intent_last = bool(re.search(r"(最后一次|最终|末次).{0,6}(出现|登场)", question))

    t = time.perf_counter()
    ent_ranked, ent_score_map, ents = entity_recall(
        question,
        loaded,
        max_query_entities=int(getattr(settings, "entity_query_max", 3) or 3),
        postings_per_entity=int(getattr(settings, "entity_postings_per_entity", 20) or 20),
        intent_first_appearance=bool(intent_first),
        intent_last_appearance=bool(intent_last),
    )
    timing_detail_ms["entity"] = timing_detail_ms.get("entity", 0) + int((time.perf_counter() - t) * 1000)
    if not ent_ranked:
        return ranked_ids, ranked_scores, {"entity_enabled": True, "entity_status": "no_match", "entities": ents}

    chrono_rank: dict[str, int] = {}
    if chunks and (intent_first or intent_last):
        chrono_items: list[tuple[int, str]] = []
        for cid in ent_ranked:
            c = chunks.get(cid)
            if not c:
                continue
            try:
                ps = int(c.get("page_start") or 0)
            except Exception:
                continue
            if ps > 0:
                chrono_items.append((ps, cid))
        chrono_items.sort(key=lambda x: x[0], reverse=bool(intent_last))
        chrono_rank = {cid: i + 1 for i, (_ps, cid) in enumerate(chrono_items)}

    k0 = int(getattr(settings, "entity_rrf_k0", 60) or 60)
    wv = float(getattr(settings, "entity_rrf_weight_passage", 1.0) or 1.0)
    we = float(getattr(settings, "entity_rrf_weight_entity", 0.8) or 0.8)
    wc = 0.35 if (intent_first or intent_last) else 0.0

    vec_rank = {cid: i + 1 for i, cid in enumerate(ranked_ids)}
    ent_rank = {cid: i + 1 for i, cid in enumerate(ent_ranked)}

    # Limit fusion universe for latency: union of current recall + entity recall (both already truncated upstream).
    universe: set[str] = set(ranked_ids)
    universe.update(ent_ranked)

    fused: dict[str, float] = {}
    for cid in universe:
        s = 0.0
        vr = vec_rank.get(cid)
        if vr is not None:
            s += wv / float(k0 + vr)
        er = ent_rank.get(cid)
        if er is not None:
            s += we / float(k0 + er)
        cr = chrono_rank.get(cid)
        if wc > 0.0 and cr is not None:
            s += wc / float(k0 + cr)
        if s > 0:
            fused[cid] = s

    fused_items = sorted(fused.items(), key=lambda x: x[1], reverse=True)
    fused_ids = [cid for cid, _s in fused_items]
    fused_scores = [float(_s) for _cid, _s in fused_items]

    meta = {
        "entity_enabled": True,
        "entity_status": "ok",
        "entities": ents,
        "entity_candidates": int(len(ent_ranked)),
        "entity_score_top": float(ent_score_map.get(ent_ranked[0], 0.0)) if ent_ranked else None,
        "entity_rrf_k0": int(k0),
        "entity_rrf_weight_passage": float(wv),
        "entity_rrf_weight_entity": float(we),
        "entity_first_appearance_intent": bool(intent_first),
        "entity_last_appearance_intent": bool(intent_last),
    }
    return fused_ids, fused_scores, meta


def retrieve(question: str, top_k: int) -> tuple[list[dict], list[str], dict]:
    t0 = time.perf_counter()
    chunks = load_chunks(settings.chunks_path)
    store = get_store()
    t_loaded = time.perf_counter()

    temporal_enabled = bool(getattr(settings, "retrieval_temporal_enabled", True))
    temporal_intent = bool(_TEMPORAL_INTENT_RE.search(question or "")) if temporal_enabled else False
    temporal_candidates = int(getattr(settings, "retrieval_temporal_candidates", 80) or 80)
    temporal_keep = int(getattr(settings, "retrieval_temporal_keep", 6) or 6)

    # Deterministic path for "first/earliest appearance" questions:
    # Scan chunks in document order to find the first mention of the target entity.
    # This avoids relying on vector recall or a lossy entity index that may miss single-mention passages.
    if temporal_intent:
        target = _extract_temporal_target_entity(question)
        if target:
            ordered = _ordered_chunk_ids(settings.chunks_path, chunks)
            aliases = _derive_aliases_from_text(ordered, chunks, target, max_scan=80)
            # Provide a lightweight alias hint to help the answerer connect surname-only mentions back to the target.
            alias_primary: str | None = None
            end_noise = ("迎", "进", "出", "正", "也", "在", "说", "问", "道", "看", "望", "走")
            cand_aliases: list[str] = []
            for a in aliases:
                if a == target:
                    continue
                if not (2 <= len(a) <= 4):
                    continue
                if not re.fullmatch(r"[\u4e00-\u9fff]{2,4}", a or ""):
                    continue
                if a.endswith(end_noise):
                    continue
                cand_aliases.append(a)
            # Prefer the shortest clean alias (usually the surname).
            cand_aliases.sort(key=lambda x: (len(x), x))
            if cand_aliases:
                alias_primary = cand_aliases[0]

            # Find the earliest chunk that satisfies "on-stage participation" (heuristic score),
            # not merely "introduced" or "mentioned".
            best_idx: int | None = None
            best_score = 0
            for i, cid in enumerate(ordered):
                c = chunks.get(cid)
                if not c:
                    continue
                text = str(c.get("text") or "")
                if not any(a in text for a in aliases):
                    continue
                s = _participation_score(text, aliases)
                # Require a minimum score to treat as "appearance"; otherwise it's just a mention.
                if s >= 5:
                    best_idx = i
                    best_score = s
                    break
                # Keep the earliest "pretty good" candidate in case there's no perfect hit.
                if best_idx is None and s > best_score:
                    best_idx = i
                    best_score = s

            # Fallback: first mention of any alias.
            if best_idx is None:
                for i, cid in enumerate(ordered):
                    c = chunks.get(cid)
                    if not c:
                        continue
                    text = str(c.get("text") or "")
                    if any(a in text for a in aliases):
                        best_idx = i
                        break

            if best_idx is not None:
                # Build an evidence window around the chosen "appearance" chunk:
                # include some preceding chunks to capture the actual scene (often earlier than the "result" sentence).
                anchor = chunks.get(ordered[best_idx]) or {}
                anchor_chapter = str(anchor.get("chapter") or "")
                try:
                    anchor_ps = int(anchor.get("page_start") or 0)
                except Exception:
                    anchor_ps = 0
                try:
                    anchor_pe = int(anchor.get("page_end") or anchor_ps or 0)
                except Exception:
                    anchor_pe = anchor_ps

                start_idx = max(0, int(best_idx) - 24)
                end_idx = min(len(ordered), int(best_idx) + 24)
                # Keep a tight page neighborhood (captures the match + immediate setup).
                page_lo = max(1, anchor_ps - 3) if anchor_ps > 0 else 1
                page_hi = anchor_pe + 2 if anchor_pe > 0 else 10**9

                window: list[str] = []
                for cid in ordered[start_idx:end_idx]:
                    c = chunks.get(cid)
                    if not c:
                        continue
                    chapter = str(c.get("chapter") or "")
                    if anchor_chapter and chapter and chapter != anchor_chapter:
                        continue
                    try:
                        ps = int(c.get("page_start") or 0)
                    except Exception:
                        ps = 0
                    if ps > 0 and not (page_lo <= ps <= page_hi):
                        continue
                    window.append(cid)

                used: list[dict] = []
                for cid in window:
                    c = chunks.get(cid)
                    if not c:
                        continue
                    text = str(c.get("text") or "")
                    page_start = int(c["page_start"])
                    page_end = int(c["page_end"])
                    chapter = (c.get("chapter") or "")

                    snippet = (text[:450] + "…") if len(text) > 450 else text
                    used.append(
                        {
                            "chunk_id": cid,
                            "page_start": page_start,
                            "page_end": page_end,
                            "chapter": chapter,
                            "score": 0.0,
                            "vector_score": 0.0,
                            "bm25_score": None,
                            "rrf_score": None,
                            "text_snippet": snippet,
                            "text": text,
                        }
                    )

                if used and alias_primary:
                    # Attach an alias hint once; evidence builder may choose to expose it as a non-source hint.
                    used[0]["alias_hint"] = f"{target}={alias_primary}"

                keep_n = min(int(top_k), max(1, int(temporal_keep)))
                kept = used[:keep_n]
                meta = {
                    "retrieved_k": len(used),
                    "kept_k": len(kept),
                    "max_score": None,
                    "cutoff_score": None,
                    "strategy": "temporal_first_mention_scan",
                    "method": "doc_order_scan+temporal",
                    "temporal_enabled": True,
                    "temporal_entities": [target],
                    "temporal_aliases": aliases[:6],
                    "temporal_best_score": int(best_score),
                    "temporal_keep": int(keep_n),
                    "temporal_status": "ok" if kept else "no_match",
                    "timing_detail_ms": {"load": int((t_loaded - t0) * 1000)},
                }
                return kept, [u["chunk_id"] for u in kept], meta

    # Rerank optimizes relevance, but can conflict with "first/earliest" intent.
    rerank_enabled = bool(settings.retrieval_rerank_enabled) and not temporal_intent
    expand_enabled = bool(settings.retrieval_expand_enabled)
    plan = make_plan(question)
    multi_query_enabled = bool(expand_enabled or plan.enabled)
    candidate_k = int(top_k)
    if temporal_intent:
        candidate_k = max(candidate_k, temporal_candidates)
    if rerank_enabled:
        candidate_k = max(candidate_k, int(settings.retrieval_rerank_candidates))
    if multi_query_enabled:
        candidate_k = max(candidate_k, int(settings.retrieval_rerank_candidates), int(top_k))

    timing_detail_ms: dict[str, int] = {"load": int((t_loaded - t0) * 1000)}

    expand_status: str | None = None
    expand_queries: list[str] = [question]
    if plan.enabled:
        plan_queries = build_subqueries(plan)
        for pq in plan_queries:
            if pq and pq not in expand_queries:
                expand_queries.append(pq)
    if expand_enabled:
        t = time.perf_counter()
        extra, expand_status = expand_query(question, n=settings.retrieval_expand_n, model=settings.retrieval_expand_model)
        timing_detail_ms["expand"] = int((time.perf_counter() - t) * 1000)
        for eq in extra:
            if eq and eq not in expand_queries:
                expand_queries.append(eq)

    hybrid_alpha = settings.retrieval_hybrid_alpha
    # Multi-query recall fusion (before rerank).
    recall_by_query: list[list[tuple[str, float]]] = []
    recall_meta_extra: dict = {}
    recall_threshold_mode = "vector"
    vector_max_score = None
    orig_vec_score_map: dict[str, float] = {}

    def _bm25_fallback(q: str, *, embed_error: bool) -> tuple[list[dict], list[str], dict]:
        bm25 = get_bm25_index(settings.chunks_path, chunks)
        t = time.perf_counter()
        bm25_hits = bm25.search(q, top_k=int(top_k))
        timing_detail_ms["bm25_search"] = timing_detail_ms.get("bm25_search", 0) + int((time.perf_counter() - t) * 1000)

        used: list[dict] = []
        for cid, score in bm25_hits:
            c = chunks.get(cid)
            if not c:
                continue
            snippet = (c["text"][:450] + "…") if len(c["text"]) > 450 else c["text"]
            used.append(
                {
                    "chunk_id": cid,
                    "page_start": int(c["page_start"]),
                    "page_end": int(c["page_end"]),
                    "chapter": (c.get("chapter") or ""),
                    "score": float(score),
                    "vector_score": 0.0,
                    "bm25_score": float(score),
                    "text_snippet": snippet,
                    "text": c["text"],
                }
            )

        meta = {
            "retrieved_k": len(bm25_hits),
            "kept_k": len(used),
            "max_score": float(used[0]["score"]) if used else None,
            "cutoff_score": None,
            "strategy": "bm25_fallback",
            "method": "bm25_fallback",
            "timing_detail_ms": timing_detail_ms | ({"embed_error": 1} if embed_error else {}),
        }
        return used, [u["chunk_id"] for u in used], meta

    try:
        emb_by_q = _get_query_embeddings(expand_queries, timing_detail_ms)
    except Exception:
        if not bool(settings.retrieval_fallback_bm25_on_embed_failure):
            raise
        return _bm25_fallback(question, embed_error=True)

    for qi, q in enumerate(expand_queries):
        emb = emb_by_q[q]
        if hybrid_alpha is None:
            t = time.perf_counter()
            chunk_ids, scores = store.search(emb, top_k=candidate_k)
            timing_detail_ms["faiss_search"] = timing_detail_ms.get("faiss_search", 0) + int(
                (time.perf_counter() - t) * 1000
            )
            if q == question:
                vector_max_score = float(scores[0]) if scores else None
                orig_vec_score_map = {cid: float(s) for cid, s in zip(chunk_ids, scores)}
            recall_by_query.append([(cid, float(s)) for cid, s in zip(chunk_ids, scores)])
            recall_meta_extra = {
                "method": "vector",
                "vector_max_score": float(vector_max_score) if vector_max_score is not None else None,
            }
            recall_threshold_mode = "vector"
        else:
            alpha = float(hybrid_alpha)
            alpha = min(1.0, max(0.0, alpha))

            vec_k = max(int(candidate_k), int(settings.retrieval_hybrid_vec_k))
            bm25_k = max(int(candidate_k), int(settings.retrieval_hybrid_bm25_k))
            k0 = int(settings.retrieval_hybrid_rrf_k0)

            t = time.perf_counter()
            vec_ids, vec_scores = store.search(emb, top_k=vec_k)
            timing_detail_ms["faiss_search"] = timing_detail_ms.get("faiss_search", 0) + int(
                (time.perf_counter() - t) * 1000
            )
            if q == question:
                vector_max_score = float(vec_scores[0]) if vec_scores else None
                orig_vec_score_map = {cid: float(s) for cid, s in zip(vec_ids, vec_scores)}

            vec_rank = {cid: i + 1 for i, cid in enumerate(vec_ids)}
            bm25 = get_bm25_index(settings.chunks_path, chunks)
            t = time.perf_counter()
            bm25_hits = bm25.search(q, top_k=bm25_k)
            timing_detail_ms["bm25_search"] = timing_detail_ms.get("bm25_search", 0) + int(
                (time.perf_counter() - t) * 1000
            )
            bm25_rank = {cid: i + 1 for i, (cid, _s) in enumerate(bm25_hits)}
            fused = _weighted_rrf(vec_rank, bm25_rank, alpha=alpha, k0=k0)
            fused_items = sorted(fused.items(), key=lambda x: x[1], reverse=True)[: int(candidate_k)]
            recall_by_query.append([(cid, float(s)) for cid, s in fused_items])

            recall_meta_extra = {
                "method": "hybrid",
                "hybrid_alpha": float(alpha),
                "hybrid_vec_k": int(vec_k),
                "hybrid_bm25_k": int(bm25_k),
                "hybrid_rrf_k0": int(k0),
                "vector_max_score": float(vector_max_score) if vector_max_score is not None else None,
            }
            recall_threshold_mode = "hybrid"

    # Gate/abstain checks are based on the original query's top-1 vector score (same semantics as before).
    if vector_max_score is not None:
        # Optional overlap-based abstain: only when vector score is not confidently high.
        try:
            min_overlap = int(settings.retrieval_min_query_token_overlap)
            overlap_gate_max = float(settings.retrieval_overlap_gate_max_score)
        except Exception:
            min_overlap = 0
            overlap_gate_max = -1.0
        if (
            min_overlap > 0
            and overlap_gate_max > 0
            and vector_max_score < overlap_gate_max
            and recall_by_query
            and recall_by_query[0]
        ):
            top = chunks.get(recall_by_query[0][0][0])
            if top:
                overlap = _unique_token_overlap_count(question, top.get("text") or "")
                if overlap < min_overlap:
                    meta = {
                        "retrieved_k": len(recall_by_query[0]),
                        "kept_k": 0,
                        "max_score": float(vector_max_score),
                        "cutoff_score": None,
                        "strategy": "overlap_abstain",
                        "method": recall_meta_extra.get("method") or "retrieve",
                        "vector_max_score": float(vector_max_score),
                        "token_overlap": int(overlap),
                    }
                    if expand_enabled:
                        meta.update(
                            {
                                "expand_enabled": True,
                                "expand_n": int(settings.retrieval_expand_n),
                                "expand_queries": expand_queries,
                                "expand_status": expand_status,
                            }
                        )
                    if plan.enabled:
                        meta.update(
                            {
                                "plan_enabled": True,
                                "plan_kind": plan.kind,
                                "plan_targets": plan.targets,
                                "plan_fields": plan.fields,
                            }
                        )
                    return [], [], meta

    if settings.retrieval_min_score is not None and vector_max_score is not None:
        min_score = float(settings.retrieval_min_score)
        if vector_max_score < min_score:
            meta = {
                "retrieved_k": len(recall_by_query[0]) if recall_by_query else 0,
                "kept_k": 0,
                "max_score": float(vector_max_score),
                "cutoff_score": float(min_score),
                "strategy": "min_score_abstain",
                "method": recall_meta_extra.get("method") or "retrieve",
                "vector_max_score": float(vector_max_score),
            }
            if expand_enabled:
                meta.update(
                    {
                        "expand_enabled": True,
                        "expand_n": int(settings.retrieval_expand_n),
                        "expand_queries": expand_queries,
                        "expand_status": expand_status,
                    }
                )
            if plan.enabled:
                meta.update(
                    {
                        "plan_enabled": True,
                        "plan_kind": plan.kind,
                        "plan_targets": plan.targets,
                        "plan_fields": plan.fields,
                    }
                )
            return [], [], meta

    # If expansion is enabled, fuse per-query rankings with weighted RRF into a single recall ranking.
    if multi_query_enabled and len(recall_by_query) > 1:
        k0 = int(settings.retrieval_expand_rrf_k0)
        w0 = float(settings.retrieval_expand_weight_original)
        w1 = float(settings.retrieval_expand_weight_extra)
        fused: dict[str, float] = {}
        for i, items in enumerate(recall_by_query):
            w = w0 if i == 0 else w1
            if w <= 0:
                continue
            for r, (cid, _s) in enumerate(items, start=1):
                fused[cid] = fused.get(cid, 0.0) + (w / (k0 + r))
        fused_items = sorted(fused.items(), key=lambda x: x[1], reverse=True)[: int(candidate_k)]
        ranked_ids = [cid for cid, _s in fused_items]
        ranked_scores = [float(_s) for _cid, _s in fused_items]
        # Multi-query fused scores behave like hybrid RRF scores (small positives).
        threshold_mode = "hybrid"
        recall_meta_extra["method"] = f"{recall_meta_extra.get('method') or 'retrieve'}+expand"
    else:
        ranked_items = recall_by_query[0] if recall_by_query else []
        ranked_ids = [cid for cid, _s in ranked_items]
        ranked_scores = [float(_s) for _cid, _s in ranked_items]
        threshold_mode = recall_threshold_mode

    ranked_ids, ranked_scores, entity_meta = _fuse_entity_rrf(question, ranked_ids, ranked_scores, timing_detail_ms, chunks)
    if entity_meta:
        recall_meta_extra["method"] = f"{recall_meta_extra.get('method') or 'retrieve'}+entity"
        # RRF-like scores: treat as hybrid for cutoff semantics.
        threshold_mode = "hybrid"
        recall_meta_extra.update(entity_meta)

    # Attach vector scores (cosine) for the original query.
    # For latency, do NOT call `score_chunk_ids()` (which reconstructs vectors per id).
    vec_score_map: dict[str, float] = {}
    if ranked_ids and orig_vec_score_map:
        vec_score_map = {cid: float(orig_vec_score_map.get(cid, 0.0)) for cid in ranked_ids}

    if hybrid_alpha is None:
        # In pure vector mode without multi-query fusion, final recall score is the vector score.
        if not (multi_query_enabled and len(recall_by_query) > 1):
            ranked_scores = [float(vec_score_map.get(cid, 0.0)) for cid in ranked_ids]

    # BM25 score is computed only for the original query (debug/diagnostic).
    bm25_score_by_id: dict[str, float] = {}
    if hybrid_alpha is not None or multi_query_enabled:
        bm25 = get_bm25_index(settings.chunks_path, chunks)
        bm25_score_by_id = {cid: float(bm25.score(question, cid)) for cid in ranked_ids}

    used: list[dict] = []
    for cid, score in zip(ranked_ids, ranked_scores):
        c = chunks.get(cid)
        if not c:
            continue
        snippet = (c["text"][:450] + "…") if len(c["text"]) > 450 else c["text"]
        row = {
            "chunk_id": cid,
            "page_start": int(c["page_start"]),
            "page_end": int(c["page_end"]),
            "chapter": (c.get("chapter") or ""),
            "score": float(score),
            "vector_score": float(vec_score_map.get(cid, 0.0)),
            "text_snippet": snippet,
            "text": c["text"],
        }
        if hybrid_alpha is not None:
            row["bm25_score"] = float(bm25_score_by_id.get(cid, 0.0))
            # If not expanded, recall score is the per-query hybrid RRF score.
            if not (multi_query_enabled and len(recall_by_query) > 1):
                row["rrf_score"] = float(score)
            else:
                row["rrf_score"] = None
        elif multi_query_enabled and len(recall_by_query) > 1:
            # Multi-query fusion score behaves like RRF.
            row["rrf_score"] = float(score)
            row["bm25_score"] = float(bm25_score_by_id.get(cid, 0.0))
        used.append(row)

    # Temporal mode: reorder by document position and prefer chunks mentioning the target entity.
    if temporal_intent and used:
        ents = _extract_temporal_entities(question)
        filtered = used
        if ents:
            filtered = [u for u in used if any(e in str(u.get("text") or "") for e in ents)]
            if not filtered:
                filtered = used

        filtered.sort(key=lambda u: _parse_chunk_position(str(u.get("chunk_id") or ""), page_start=u.get("page_start")))
        keep_n = min(int(top_k), max(1, int(temporal_keep)))
        kept = filtered[:keep_n]
        meta = {
            "retrieved_k": len(filtered),
            "kept_k": len(kept),
            "max_score": float(kept[0]["score"]) if kept else None,
            "cutoff_score": None,
            "strategy": "temporal_position",
            "method": f"{(recall_meta_extra.get('method') or 'retrieve')}+temporal",
            "temporal_enabled": True,
            "temporal_entities": ents or None,
            "temporal_candidates": int(candidate_k),
            "temporal_keep": int(keep_n),
            "temporal_status": "ok" if kept else "no_match",
            "timing_detail_ms": timing_detail_ms,
        }
        # Keep meta parity with other modes.
        if expand_enabled:
            meta["expand_enabled"] = True
            meta["expand_n"] = int(settings.retrieval_expand_n)
            meta["expand_queries"] = expand_queries
            meta["expand_status"] = expand_status
        if plan.enabled:
            meta.update(
                {
                    "plan_enabled": True,
                    "plan_kind": plan.kind,
                    "plan_targets": plan.targets,
                    "plan_fields": plan.fields,
                }
            )
        meta.update(recall_meta_extra)
        return kept, [u["chunk_id"] for u in kept], meta

    rerank_status: str | None = None
    rerank_meta: dict | None = None
    if rerank_enabled and used:
        t = time.perf_counter()
        score_map, rerank_status, rerank_meta = rerank_chunks(
            question,
            used,
            model=settings.retrieval_rerank_model,
            max_chars=settings.retrieval_rerank_max_chars,
        )
        timing_detail_ms["rerank"] = int((time.perf_counter() - t) * 1000)
        if score_map:
            for u in used:
                cid = str(u.get("chunk_id") or "")
                s = float(score_map.get(cid, 0.0))
                u["rerank_score"] = s
                u["score"] = s
            used.sort(key=lambda x: float(x["score"]), reverse=True)
            threshold_mode = "rerank"
            recall_meta_extra["method"] = f"{recall_meta_extra.get('method') or 'retrieve'}+rerank"
        # Always cap to requested top_k after (attempted) rerank to keep output stable.
        used = used[: int(top_k)]

    used, meta = _apply_thresholding(used, threshold_mode=threshold_mode)
    if rerank_enabled:
        meta["rerank_enabled"] = True
        provider = (rerank_meta or {}).get("provider")
        meta["rerank_provider"] = provider
        if provider == "http":
            meta["rerank_model"] = (rerank_meta or {}).get("model_id") or "local_rerank"
            meta["rerank_provider_detail"] = (rerank_meta or {}).get("reranker_provider")
        else:
            meta["rerank_model"] = (settings.retrieval_rerank_model or settings.openai_chat_model)
        meta["rerank_candidates"] = int(candidate_k)
        meta["rerank_status"] = rerank_status
    if expand_enabled:
        meta["expand_enabled"] = True
        meta["expand_n"] = int(settings.retrieval_expand_n)
        meta["expand_queries"] = expand_queries
        meta["expand_status"] = expand_status
    if plan.enabled:
        meta.update(
            {
                "plan_enabled": True,
                "plan_kind": plan.kind,
                "plan_targets": plan.targets,
                "plan_fields": plan.fields,
            }
        )
    meta.update(recall_meta_extra)
    meta["timing_detail_ms"] = timing_detail_ms
    return used, [u["chunk_id"] for u in used], meta
