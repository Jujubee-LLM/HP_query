import re

from .llm import chat_complete
from .planner import extract_list_criteria, is_list_intent, make_plan, needs_criteria_hint
from .prompts import SYSTEM_PROMPT, USER_PROMPT, ABSTAIN_REWRITE_SYSTEM_PROMPT
from .rerank import rerank as rerank_chunks
from .settings import settings

_META_WORDS = ("参考信息", "资料", "原文", "片段", "页码", "章节", "知识库", "小说", "书中")

_REFUSAL_RE = re.compile(
    r"(?:"
    r"无法确认|无法确定|不确定|未知|"
    r"没有(?:提到|说明|给出|描述)|未(?:提到|说明|给出|描述)|并未(?:提到|说明|给出|描述)|"
    r"因此，?无法确认|因此，?无法确定|"
    r"无法(?:从|依据|根据).*(?:得出|确认)|"
    r"信息不足|资料不足|证据不足"
    r")"
)

def _looks_like_refusal(text: str) -> bool:
    t = (text or "").strip()
    if not t:
        return True
    # Only treat it as refusal if it's short and mostly refusal-ish.
    if len(t) <= 90 and _REFUSAL_RE.search(t):
        return True
    return False
_META_PREFIX_RE = re.compile(
    r"^\s*(?:"
    r"(?:根据|依据|参照|结合|基于)\s*(?:上述|以下|给定|提供的)?\s*(?:参考信息|资料|原文|片段)\s*[，,。:：]?\s*"
    r"|(?:在|从)\s*(?:参考信息|资料|原文|片段)\s*(?:中|里|内)\s*[，,。:：]?\s*"
    r")",
    re.IGNORECASE,
)
_META_CLAUSE_RE = re.compile(
    r"(?:"
    r"(?:参考信息|资料|原文|片段)\s*(?:中|里|内)?\s*(?:没有|未|并未)\s*(?:提到|说明|给出|描述)[^。！？\n]*"
    r"|(?:无法|不能)\s*(?:从|依据|根据)\s*(?:参考信息|资料|原文|片段)[^。！？\n]*"
    r")",
    re.IGNORECASE,
)

def _suggest_keywords(question: str, k_min: int = 3, k_max: int = 6) -> list[str]:
    q = (question or "").strip()
    if not q:
        return []
    tokens = re.findall(r"[A-Za-z0-9]{2,}|[\u4e00-\u9fff]{2,}", q)
    stop = {
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
        "台词",
        "逐字",
        "引用",
        "剧情",
        "结局",
    }
    uniq: list[str] = []
    seen: set[str] = set()
    for t in tokens:
        t = t.strip()
        if not t or t in stop:
            continue
        if t not in seen:
            uniq.append(t)
            seen.add(t)
        if len(uniq) >= k_max:
            break
    if len(uniq) < k_min:
        return uniq
    return uniq[:k_max]


def _insufficient_answer(question: str) -> str:
    return "无法回答"


def _clean_human_style(text: str) -> str:
    """
    Remove stiff meta sentences like "参考信息中没有提到..." to match the desired UX.
    If everything is removed, fall back to "无法回答".
    """
    t = (text or "").strip()
    if not t:
        return ""
    parts = re.split(r"(?<=[。！？\n])", t)
    kept: list[str] = []
    for p in parts:
        s = p.strip()
        if not s:
            continue
        if any(k in s for k in ("关键描写", "情节", "简短总结")):
            kept.append(s)
            continue
        # If it's a pure meta-clause (no content other than "can't find in ref"), drop it.
        if _META_CLAUSE_RE.search(s) and len(_META_CLAUSE_RE.sub("", s).strip()) <= 2:
            continue
        # Remove common meta prefixes while keeping the actual answer content.
        s2 = _META_PREFIX_RE.sub("", s).strip()
        # Remove embedded meta clauses like "片段中未提到..." when followed by real content.
        s2 = _META_CLAUSE_RE.sub("", s2).strip()
        # If still contains meta words, keep only if there's substantial non-meta content.
        if any(w in s2 for w in _META_WORDS):
            # Try a last cleanup: just remove the meta words themselves.
            for w in _META_WORDS:
                s2 = s2.replace(w, "")
            s2 = s2.strip()
        if not s2:
            continue
        kept.append(s2)
    return "".join(kept).strip()

def _pick_chunks_for_question(question: str, used_chunks: list[dict]) -> list[dict]:
    """
    Help the LLM by preferring chunks that contain obvious key terms from the question.
    This reduces false abstains where retrieval got "related" chunks but not the answering sentence.
    """
    if not used_chunks:
        return used_chunks
    q = (question or "").strip()
    if not q:
        return used_chunks
    plan = make_plan(q)
    key_terms: list[str] = []
    for t in plan.targets:
        if t and t not in key_terms:
            key_terms.append(t)
    for f in plan.fields:
        if f and f not in key_terms:
            key_terms.append(f)
    if not key_terms:
        return used_chunks

    def score_chunk(c: dict) -> tuple[int, float]:
        text = str(c.get("text") or c.get("text_snippet") or "")
        hit = sum(1 for kw in key_terms if kw in text)
        base = float(c.get("score") or 0.0)
        return (hit, base)

    return sorted(used_chunks, key=score_chunk, reverse=True)


def _split_sentences(text: str) -> list[str]:
    if not text:
        return []
    parts = re.split(r"(?<=[。！？；;.!?\n])", text)
    return [p.strip() for p in parts if p.strip()]


_NAME_RE = re.compile(r"[\u4e00-\u9fff]{2,4}(?:·[\u4e00-\u9fff]{2,4})?")
_LIST_STOP_NAMES = {
    "同学",
    "学生",
    "女生",
    "男生",
    "老师",
    "教授",
    "校长",
    "学院",
    "人物",
    "角色",
}
_HOUSE_WORDS = ("格兰芬多", "斯莱特林", "拉文克劳", "赫奇帕奇")


def _clean_value(val: str) -> str:
    v = (val or "").strip()
    v = re.sub(r"^[，,。；;：:\s]+", "", v)
    v = re.sub(r"[，,。；;：:\s]+$", "", v)
    v = re.sub(r"(?:等|等等)$", "", v)
    return v.strip()


def _match_value(sentence: str, target: str, field: str) -> str | None:
    t = re.escape(target)
    f = re.escape(field)
    m = re.search(fr"{t}的{f}\s*(?:是|为|：|:)?\s*([^。；;！!\n]+)", sentence)
    if m:
        return _clean_value(m.group(1))
    if target in sentence:
        m = re.search(fr"{f}\s*(?:是|为|：|:)\s*([^。；;！!\n]+)", sentence)
        if m:
            return _clean_value(m.group(1))
    if field in sentence and target in sentence:
        m = re.search(fr"{t}\s*[与和及]\s*([^，。；;、\n]+)", sentence)
        if m:
            return _clean_value(m.group(1))
    return None


def _extract_structured(plan, used_chunks: list[dict]) -> dict[str, dict[str, str]]:
    if not plan.targets or not plan.fields or not used_chunks:
        return {}
    full_text = "\n".join(str(c.get("text") or c.get("text_snippet") or "") for c in used_chunks)
    sentences = _split_sentences(full_text)
    out: dict[str, dict[str, str]] = {t: {} for t in plan.targets}
    for s in sentences:
        for t in plan.targets:
            if t not in s:
                continue
            for f in plan.fields:
                if f not in s and f"{t}的{f}" not in s:
                    continue
                if out[t].get(f):
                    continue
                val = _match_value(s, t, f)
                if val:
                    out[t][f] = val
    return out


def _extract_list_candidates(question: str, used_chunks: list[dict]) -> list[tuple[str, str, str]]:
    if not used_chunks:
        return []
    q = (question or "").strip()
    if not q:
        return []
    criteria = extract_list_criteria(q) or ""
    keywords = _suggest_keywords(q, k_min=2, k_max=8)
    terms = [t for t in ([criteria] + keywords) if t and t not in ("哪些", "列出", "列举")]
    full_text = "\n".join(str(c.get("text") or c.get("text_snippet") or "") for c in used_chunks)
    sentences = _split_sentences(full_text)
    found: dict[str, tuple[str, str]] = {}

    def _house_in_sentence(s: str) -> str | None:
        for h in _HOUSE_WORDS:
            if h in s:
                return h
        return None

    for s in sentences:
        if terms and not any(t in s for t in terms):
            continue
        for m in _NAME_RE.finditer(s):
            name = m.group(0)
            if name in _LIST_STOP_NAMES:
                continue
            if name not in found:
                house = _house_in_sentence(s)
                display = f"{name}（{house}）" if house else name
                summary = s.replace("“", "").replace("”", "").replace("‘", "").replace("’", "")
                summary = summary.replace(name, "").strip()
                # Prefer a short semantic gloss without hardcoded domain keywords.
                if criteria and criteria in summary:
                    summary = summary.replace(criteria, "").strip()
                if len(summary) > 60:
                    summary = summary[:60].rstrip() + "…"
                if not summary:
                    summary = ""
                found[name] = (display, s.strip(), summary.strip())
        if len(found) >= 8:
            break

    return list(found.values())


def _structured_complete(data: dict[str, dict[str, str]], targets: list[str], fields: list[str]) -> bool:
    if not targets or not fields:
        return False
    for t in targets:
        for f in fields:
            if not data.get(t, {}).get(f):
                return False
    return True


def _format_structured(
    data: dict[str, dict[str, str]], targets: list[str], fields: list[str], *, fill_missing: bool
) -> str:
    lines: list[str] = []
    for t in targets:
        parts: list[str] = []
        for f in fields:
            v = data.get(t, {}).get(f)
            if v:
                parts.append(f"{f}：{v}")
            elif fill_missing:
                parts.append(f"{f}：—")
        if parts:
            lines.append(f"{t}：" + "；".join(parts))
    return "\n".join(lines).strip()

def _cap_chunks_for_llm(used_chunks: list[dict]) -> list[dict]:
    k = int(getattr(settings, "rag_max_chunks", 6) or 6)
    if k <= 0:
        return used_chunks
    seen: set[tuple[int, str]] = set()
    ordered = sorted(
        used_chunks,
        key=lambda c: (
            int(c.get("page_start") or 10**9),
            str(c.get("chunk_id") or ""),
        ),
    )
    deduped: list[dict] = []
    for c in ordered:
        ps = int(c.get("page_start") or 10**9)
        cid = str(c.get("chunk_id") or "")
        key = (ps, cid)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(c)
        if len(deduped) >= k:
            break
    return deduped


def _rerank_for_evidence(question: str, used_chunks: list[dict]) -> list[dict]:
    """
    Rerank is normally applied inside `retrieval.retrieve()` and the API passes already-reranked chunks.
    This is a defensive fallback: if rerank is enabled but chunks are not reranked, rerank for evidence only.
    """
    if not used_chunks:
        return used_chunks
    if not bool(getattr(settings, "retrieval_rerank_enabled", False)):
        return used_chunks
    # If retrieval already attached rerank scores, preserve its ordering and avoid double-rerank.
    if any("rerank_score" in c for c in used_chunks):
        return used_chunks

    try:
        cand_k = int(getattr(settings, "retrieval_rerank_candidates", 24) or 24)
    except Exception:
        cand_k = 24
    candidates = used_chunks[:cand_k] if cand_k > 0 else used_chunks

    try:
        score_map, status, _meta = rerank_chunks(
            question,
            candidates,
            model=getattr(settings, "retrieval_rerank_model", None),
            max_chars=getattr(settings, "retrieval_rerank_max_chars", None),
        )
    except Exception:
        return used_chunks
    if status is not None or not score_map:
        return used_chunks

    decorated: list[dict] = []
    for c in candidates:
        cid = str(c.get("chunk_id") or "")
        if not cid:
            continue
        cc = dict(c)
        cc["rerank_score"] = float(score_map.get(cid, 0.0))
        decorated.append(cc)
    decorated.sort(key=lambda x: float(x.get("rerank_score") or 0.0), reverse=True)

    # Keep non-candidate tail in original order (they just won't influence evidence ranking).
    if len(candidates) < len(used_chunks):
        return decorated + used_chunks[len(candidates) :]
    return decorated


def _truncate_text(text: str, max_chars: int) -> str:
    t = (text or "").strip()
    if max_chars <= 0:
        return t
    if len(t) <= max_chars:
        return t
    s = t[:max_chars]
    punct = "。！？；….!?;\n"
    best = max(s.rfind(ch) for ch in punct)
    # Prefer cutting at a likely sentence boundary near the end to avoid mid-sentence truncation.
    if best >= int(max_chars * 0.6):
        s = s[: best + 1]
    return s.rstrip() + "…"


def build_evidence_block(used_chunks: list[dict]) -> str:
    lines = []
    per_chunk = int(getattr(settings, "rag_evidence_chunk_max_chars", 800) or 800)
    total_max = int(getattr(settings, "rag_evidence_total_max_chars", 6000) or 6000)
    total = 0
    for i, c in enumerate(_cap_chunks_for_llm(used_chunks), start=1):
        text = _truncate_text(str(c.get("text") or c.get("text_snippet") or ""), per_chunk)
        # Keep a global cap too (very important for latency / token count).
        if total_max > 0 and total >= total_max:
            break
        if total_max > 0 and total + len(text) > total_max:
            text = _truncate_text(text, max(0, total_max - total))
        chapter = str(c.get("chapter") or "").strip()
        chapter_part = f"，{chapter}" if chapter else ""
        page_start = c.get("page_start")
        page_end = c.get("page_end")
        page_part = ""
        if isinstance(page_start, int) and isinstance(page_end, int) and page_end != page_start:
            page_part = f"页码 {page_start}-{page_end}"
        elif page_start is not None:
            page_part = f"页码 {page_start}"
        else:
            page_part = "页码 ?"
        # Put the citation token into the evidence so the model can copy it reliably.
        lines.append(f"[来源{i}]（{page_part}{chapter_part}）：\n{text}\n")
        total += len(text)
    return "\n".join(lines).strip()


def retrieve_only_answer(question: str, used_chunks: list[dict]) -> str:
    if not used_chunks:
        return _insufficient_answer(question)
    lines: list[str] = []
    for i, c in enumerate(used_chunks, start=1):
        snippet = str(c.get("text_snippet") or "").replace("\n", " ").strip()
        if snippet:
            lines.append(f"来源{i}）{snippet}")
    if not lines:
        return _insufficient_answer(question)
    return "相关内容如下：\n\n" + "\n".join(lines)


def rag_answer(question: str, used_chunks: list[dict]) -> str:
    if not used_chunks:
        return _insufficient_answer(question)

    used_chunks = _rerank_for_evidence(question, used_chunks)
    used_chunks = _pick_chunks_for_question(question, used_chunks)
    llm_chunks = _cap_chunks_for_llm(used_chunks)

    evidence = build_evidence_block(llm_chunks)

    q = (question or "").strip()
    # Alias bridge: if retrieval attached a hint like "塞德里克=迪戈里", inject it into the query
    # to prevent surname-only evidence from triggering "没提到目标名" refusals.
    alias_hint = None
    for c in llm_chunks[:3]:
        if isinstance(c, dict) and c.get("alias_hint"):
            alias_hint = str(c.get("alias_hint") or "").strip()
            break
    if alias_hint and "=" in alias_hint:
        left, right = [s.strip() for s in alias_hint.split("=", 1)]
        if left and right and left in q and right not in q:
            q = q.replace(left, f"{left}（{right}）", 1)

    plan = make_plan(q)
    list_intent = is_list_intent(q)
    extracted = {}
    if plan.enabled and plan.targets and plan.fields:
        extracted = _extract_structured(plan, llm_chunks)
        if _structured_complete(extracted, plan.targets, plan.fields):
            formatted = _format_structured(extracted, plan.targets, plan.fields, fill_missing=False)
            if formatted:
                return formatted

        structure_hint = (
            "请先做证据内结构化抽取，优先匹配通用模板：\n"
            "1) X 的 <field> 是 Y\n"
            "2) <field>：A与B、C与D…\n"
            "然后按 targets×fields 填表输出，不要写“无法确认/未知/无法回答”等说法；"
            "未找到的字段用“—”。\n"
            f"targets：{', '.join(plan.targets)}\n"
            f"fields：{', '.join(plan.fields)}"
        )
        if extracted:
            extracted_text = _format_structured(extracted, plan.targets, plan.fields, fill_missing=False)
            if extracted_text:
                structure_hint += f"\n已抽取到的内容：\n{extracted_text}"
        q = f"{structure_hint}\n\n{q}"
    else:
        if list_intent:
            candidates = _extract_list_candidates(q, llm_chunks)
            if candidates:
                lines = []
                if needs_criteria_hint(q):
                    lines.append("判定标准：")
                    lines.append("1) 依据问题语义从原文中抽取")
                    lines.append("2) 需有明确描写或具体情节支撑")
                    lines.append("3) 不满足条件的不列出")
                    lines.append("")
                for i, (display, sentence, summary) in enumerate(candidates, start=1):
                    lines.append(f"{i}) {display}")
                    lines.append(f"   关键描写/情节：{sentence}")
                    lines.append(f"   简短总结：{summary}")
                return "\n".join(lines).strip()
            criteria = extract_list_criteria(q) or "问题描述"
            hint_lines = [
                "请按自然口吻输出，不要出现“证据/来源/片段/页码”等字眼。",
            ]
            if needs_criteria_hint(q):
                hint_lines.append("先给判定标准（3-4条，简洁表述）。")
            hint_lines.extend(
                [
                    "再给答案，只列符合者，每人两行：",
                    "1) 角色名（学院）",
                    "   关键描写/情节：一句原著里的具体描写或场景",
                    f"   简短总结：用你自己的话概括该角色为什么符合“{criteria}”",
                    "每个条目都必须同时包含“关键描写/情节”和“简短总结”。",
                    "不要讨论不符合的人；不要写“无法确认/未知/无法回答”。",
                ]
            )
            q = "\n".join(hint_lines) + f"\n\n{q}"

    user = USER_PROMPT.format(context=evidence, query=q)
    out_raw = chat_complete(SYSTEM_PROMPT, user).strip()

    if not out_raw:
        # With evidence, prefer returning snippets over refusing.
        return retrieve_only_answer(question, llm_chunks)

    # If the model abstains even though we have evidence, retry once with a more extractive prompt.
    if (
        bool(getattr(settings, "rag_retry_on_abstain", True))
        and out_raw.strip() == "无法回答"
        and evidence.strip()
    ):
        retry = chat_complete(ABSTAIN_REWRITE_SYSTEM_PROMPT, user).strip()
        if retry:
            out_raw = retry
    # Also retry if the model produced a refusal-like statement despite evidence.
    elif bool(getattr(settings, "rag_retry_on_abstain", True)) and _looks_like_refusal(out_raw) and evidence.strip():
        retry = chat_complete(ABSTAIN_REWRITE_SYSTEM_PROMPT, user).strip()
        if retry:
            out_raw = retry
    # Enforce list-style explanations when list intent is detected.
    if list_intent and evidence.strip():
        if ("关键描写" not in out_raw) or ("简短总结" not in out_raw):
            strict = (
                "请严格按固定格式输出，不得省略字段，也不得只列人名。\n"
                "输出格式：\n"
                "1) 角色名（学院）\n"
                "   关键描写/情节：一句原著里的具体描写或场景\n"
                "   简短总结：用你自己的话概括该角色为什么符合问题描述\n"
                "不要出现“证据/来源/片段/页码”等字眼；不要讨论不符合的人。\n\n"
                f"{q}"
            )
            strict_user = USER_PROMPT.format(context=evidence, query=strict)
            retry = chat_complete(SYSTEM_PROMPT, strict_user).strip()
            if retry:
                out_raw = retry

    out = _clean_human_style(out_raw).strip()
    if not out:
        return retrieve_only_answer(question, llm_chunks)
    if out.strip() == "无法回答":
        return retrieve_only_answer(question, llm_chunks)
    if _looks_like_refusal(out):
        # Never refuse when we have evidence: return a snippet-based answer as a last resort.
        return retrieve_only_answer(question, llm_chunks)
    # If the model still mentions meta words, drop them; if nothing left, abstain.
    if any(w in out for w in _META_WORDS):
        out = _clean_human_style(out).strip()
        if not out:
            return retrieve_only_answer(question, llm_chunks)
    return out
