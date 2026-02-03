import re
from dataclasses import dataclass

from .alias_map import load_alias_map
from .entity_index import load_entity_index
from .settings import settings


_MAPPING_HINT_RE = re.compile(r"(分别|各自|各是谁|分别是谁|分别是|各自是)")
_CJK_TERM_RE = re.compile(r"[\u4e00-\u9fff]{1,20}")
_FIELD_SPLIT_RE = re.compile(r"[、/,，和及与]")
_FIELD_STOP = {
    "什么",
    "哪些",
    "多少",
    "谁",
    "是谁",
    "是否",
    "怎么",
    "为何",
    "为什么",
    "哪里",
    "哪位",
    "哪种",
    "分别",
    "各自",
    "情况",
    "信息",
    "内容",
    "方面",
    "问题",
    "答案",
}
_ENTITY_STOP = {
    "什么",
    "哪些",
    "多少",
    "谁",
    "是谁",
    "是否",
    "怎么",
    "为何",
    "为什么",
    "哪里",
    "哪位",
    "哪种",
    "分别",
    "各自",
}
_LIST_INTENT_RE = re.compile(r"(列出|列举|哪些|有哪些|谁是|哪几位|都有谁|都有哪些)")
_NEEDS_CRITERIA_RE = re.compile(r"(判定标准|标准|依据|何为|什么才算|如何定义|算作|算是)")
_CRITERIA_STOP = {
    "哪些",
    "有哪些",
    "谁是",
    "哪几位",
    "都有谁",
    "都有哪些",
    "的人",
    "的人物",
    "的角色",
    "的同学",
    "同学",
    "人物",
    "角色",
}


@dataclass(frozen=True)
class Plan:
    enabled: bool
    kind: str | None  # "mapping" | "multi_field"
    targets: list[str]
    fields: list[str]
    keywords: list[str]


def _is_noise_entity(e: str) -> bool:
    if not e:
        return True
    if e in _ENTITY_STOP:
        return True
    if len(e) <= 1:
        return True
    if e.isdigit():
        return True
    return False


def extract_targets(question: str, *, max_targets: int = 8) -> list[str]:
    q = (question or "").strip()
    if not q:
        return []

    amap = load_alias_map(str(getattr(settings, "aliases_path", "") or ""))
    for phrase, ents in amap.items():
        if phrase and phrase in q and ents:
            out: list[str] = []
            for e in ents:
                e = (e or "").strip()
                if e and e not in out:
                    out.append(e)
            return out[:max_targets]

    loaded = load_entity_index(str(getattr(settings, "entity_index_path", "") or ""))
    ents: list[str] = []
    if loaded is not None:
        try:
            ents = loaded.trie.extract_longest(q, max_hits=max_targets * 2)
        except Exception:
            ents = []

    cleaned: list[str] = []
    seen: set[str] = set()
    for e in ents:
        if not isinstance(e, str):
            continue
        e = e.strip()
        if not e or _is_noise_entity(e):
            continue
        if e not in seen:
            cleaned.append(e)
            seen.add(e)
        if len(cleaned) >= max_targets:
            break
    if cleaned:
        return cleaned

    spans = re.findall(r"[\u4e00-\u9fff·]{2,10}", q)
    out: list[str] = []
    for s in spans:
        s = s.replace("·", "").strip()
        if not s or _is_noise_entity(s):
            continue
        if s not in out:
            out.append(s)
        if len(out) >= max_targets:
            break
    return out


def extract_mapping_field(question: str) -> str | None:
    q = (question or "").strip()
    if not q:
        return None
    m = re.search(r"([\u4e00-\u9fff]{1,10})\s*分别(?:是|是谁|是什么)", q)
    if m:
        return (m.group(1) or "").strip()
    m = re.search(r"的\s*([\u4e00-\u9fff]{1,10})\s*分别", q)
    if m:
        return (m.group(1) or "").strip()
    return None


def extract_fields(question: str) -> list[str]:
    q = (question or "").strip()
    if not q:
        return []
    f = extract_mapping_field(q)
    if f:
        return [f]
    out: list[str] = []

    def add_terms(seg: str) -> None:
        for raw in _FIELD_SPLIT_RE.split(seg):
            term = (raw or "").strip()
            if not term:
                continue
            if term in _FIELD_STOP:
                continue
            if not _CJK_TERM_RE.fullmatch(term):
                continue
            if term not in out:
                out.append(term)

    # Pattern: "<fields>分别/各自..."
    for m in re.finditer(r"([\u4e00-\u9fff、/，和及与]{2,40})\s*(?:分别|各自)\s*(?:是|为|有哪些|有什么|是谁|是什么)?", q):
        add_terms(m.group(1))

    # Pattern: "的<fields>..."
    for m in re.finditer(r"的\s*([\u4e00-\u9fff、/，和及与]{2,40})\s*(?:是|为|有哪些|有什么|分别|分别是)?", q):
        add_terms(m.group(1))

    # Tail pattern: "<fields>是什么/有哪些/分别是"
    for m in re.finditer(r"([\u4e00-\u9fff、/，和及与]{2,40})\s*(?:是什么|有哪些|有什么|分别是|分别为)\s*$", q):
        add_terms(m.group(1))

    # Fallback: any short CJK noun phrase candidates near the end.
    if not out:
        for m in re.finditer(r"的\s*([\u4e00-\u9fff]{1,12})", q):
            term = (m.group(1) or "").strip()
            if not term or term in _FIELD_STOP:
                continue
            if term not in out:
                out.append(term)
    return out


def is_list_intent(question: str) -> bool:
    q = (question or "").strip()
    if not q:
        return False
    return bool(_LIST_INTENT_RE.search(q))


def needs_criteria_hint(question: str) -> bool:
    q = (question or "").strip()
    if not q:
        return False
    return bool(_NEEDS_CRITERIA_RE.search(q))


def extract_list_criteria(question: str) -> str | None:
    q = (question or "").strip()
    if not q:
        return None
    m = re.search(r"(?:的|中)([\u4e00-\u9fff]{2,20})\s*(?:有哪些|是谁|哪些|哪几位|都有谁|都有哪些)", q)
    if m:
        cand = (m.group(1) or "").strip()
        if cand and cand not in _CRITERIA_STOP:
            return cand
    m = re.search(r"(?:哪些|哪几位|有哪些)\s*([\u4e00-\u9fff]{2,20})", q)
    if m:
        cand = (m.group(1) or "").strip()
        if cand and cand not in _CRITERIA_STOP:
            return cand
    return None


def extract_keywords(question: str, fields: list[str], targets: list[str]) -> list[str]:
    q = (question or "").strip()
    if not q:
        return []
    keys: list[str] = []
    for f in fields:
        if f and f not in keys:
            keys.append(f)
    for t in targets:
        if t and t not in keys:
            keys.append(t)
    # Keep a few contextual nouns if present.
    tokens = re.findall(r"[A-Za-z0-9]{2,}|[\u4e00-\u9fff]{2,}", q)
    for t in tokens:
        if t in _FIELD_STOP or t in _ENTITY_STOP:
            continue
        if t in keys:
            continue
        keys.append(t)
        if len(keys) >= 6:
            break
    return keys[:6]


def make_plan(question: str) -> Plan:
    q = (question or "").strip()
    targets = extract_targets(q)
    fields = extract_fields(q)
    keywords = extract_keywords(q, fields, targets)

    mapping_like = bool(_MAPPING_HINT_RE.search(q)) and bool(fields) and len(targets) >= 2
    multi_field_like = len(fields) >= 2 and bool(targets)

    kind: str | None = "mapping" if mapping_like else ("multi_field" if multi_field_like else ("target_field" if fields and targets else None))
    enabled = bool(kind)
    return Plan(enabled=enabled, kind=kind, targets=targets, fields=fields, keywords=keywords)


def build_subqueries(plan: Plan, *, per_target: int = 2) -> list[str]:
    if not plan.enabled or not plan.targets:
        return []
    kw = " ".join(plan.keywords) if plan.keywords else ""
    out: list[str] = []
    for t in plan.targets:
        if plan.fields:
            for f in plan.fields:
                sq = f"{t} {f} {kw}".strip()
                if sq and sq not in out:
                    out.append(sq)
        else:
            sq = f"{t} {kw}".strip()
            if sq and sq not in out:
                out.append(sq)
    if plan.fields:
        gq = f"{' '.join(plan.targets)} {' '.join(plan.fields)}".strip()
        if kw:
            gq = f"{gq} {kw}".strip()
        if gq and gq not in out:
            out.append(gq)
    # Keep bounded.
    limit = max(3, len(plan.targets) * max(1, int(per_target)) + 1)
    return out[:limit]
