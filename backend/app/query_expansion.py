import json

from .llm import chat_complete
from .settings import settings


_EXPAND_SYSTEM = """你是检索用“查询扩展（query expansion）”模块。
目标：在不改变原意的前提下，为原问题生成若干条更利于检索的中文查询（偏关键词/同义改写/补全核心限定词）。
约束：
1) 不要引入原问题没有的事实前提，不要改变问法方向。
2) 每条查询尽量短（建议 12~30 字），避免长句、避免加解释。
3) 只输出 JSON，不要输出任何多余文字。
输出格式：
{"queries":["...","..."]}
"""


def expand_query(question: str, *, n: int | None = None, model: str | None = None) -> tuple[list[str], str | None]:
    """
    Returns: (expanded_queries, status)
    - expanded_queries excludes the original question.
    - status is None on success, otherwise a short error string.
    """
    n = int(n if n is not None else settings.retrieval_expand_n)
    if n <= 0:
        return [], None

    model = (model or settings.retrieval_expand_model or settings.openai_chat_model).strip()
    user = f"原问题：\n{question}\n\n请生成 {n} 条扩展查询。"

    try:
        raw = chat_complete(_EXPAND_SYSTEM, user, model=model).strip()
        obj = json.loads(raw)
    except Exception as e:
        return [], f"parse_error: {type(e).__name__}"

    try:
        qs = obj.get("queries") if isinstance(obj, dict) else None
        if not isinstance(qs, list):
            return [], "bad_format"
        out: list[str] = []
        seen = {question.strip()}
        for q in qs:
            if not isinstance(q, str):
                continue
            q = q.strip()
            if not q or q in seen:
                continue
            seen.add(q)
            out.append(q)
        return out[:n], None
    except Exception as e:
        return [], f"bad_output: {type(e).__name__}"

