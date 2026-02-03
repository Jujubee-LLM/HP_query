import json
import socket
import urllib.request
from typing import Any

from .llm import chat_complete
from .settings import settings


_RERANK_SYSTEM = """你是检索重排（rerank）模型。你的任务是：给定问题和若干段落，为每个段落打“相关性分数”。
要求：
1) 分数范围为 0.0~1.0，越大越相关。
2) 只依据段落文本判断是否能直接支持回答该问题；泛泛相关但不提供答案线索的段落应给低分。
3) 必须严格输出 JSON（不要输出任何多余文字）。
输出格式：
{"scores":[{"chunk_id":"...","score":0.0}]}
"""


def _truncate_text(text: str, max_chars: int) -> str:
    t = (text or "").strip()
    if max_chars <= 0:
        return t
    if len(t) <= max_chars:
        return t
    s = t[:max_chars]
    punct = "。！？；….!?;\n"
    best = max(s.rfind(ch) for ch in punct)
    if best >= int(max_chars * 0.6):
        s = s[: best + 1]
    return s.rstrip() + "…"


def _clamp01(x: float) -> float:
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return float(x)


def rerank(
    question: str,
    chunks: list[dict],
    *,
    model: str | None = None,
    max_chars: int | None = None,
) -> tuple[dict[str, float], str | None, dict[str, Any]]:
    """
    Returns: (score_by_chunk_id, status, meta)
    - status is None on success, otherwise a short error string.
    """
    if not chunks:
        return {}, None, {"provider": None}

    max_chars = int(max_chars or settings.retrieval_rerank_max_chars)
    base_url = (settings.retrieval_rerank_base_url or "").strip() or None

    # Prefer local rerank service when configured (CPU/GPU swappable).
    if base_url:
        try:
            url = base_url.rstrip("/") + "/rerank"
            passages = []
            for c in chunks:
                cid = str(c.get("chunk_id") or "")
                text = str(c.get("text") or c.get("text_snippet") or "").strip().replace("\r\n", "\n")
                text = _truncate_text(text, int(max_chars))
                if cid and text:
                    passages.append({"chunk_id": cid, "text": text})

            payload = json.dumps({"query": question, "passages": passages}, ensure_ascii=False).encode("utf-8")
            req = urllib.request.Request(
                url,
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            timeout = float(settings.retrieval_rerank_http_timeout_seconds or 10.0)
            raw = urllib.request.urlopen(req, timeout=timeout).read().decode("utf-8", errors="replace")
            obj = json.loads(raw)
            items = obj.get("scores") if isinstance(obj, dict) else None
            if not isinstance(items, list):
                return {}, "bad_format", {"provider": "http"}
            model_id = None
            provider = "http"
            if isinstance(obj, dict):
                mid = obj.get("model_id")
                if isinstance(mid, str) and mid.strip():
                    model_id = mid.strip()
                prov = obj.get("provider")
                if isinstance(prov, str) and prov.strip():
                    provider = prov.strip()
            out: dict[str, float] = {}
            for it in items:
                if not isinstance(it, dict):
                    continue
                cid = it.get("chunk_id")
                score = it.get("score")
                if not isinstance(cid, str):
                    continue
                if isinstance(score, (int, float)):
                    out[cid] = _clamp01(float(score))
            for c in chunks:
                cid = str(c.get("chunk_id") or "")
                if cid and cid not in out:
                    out[cid] = 0.0
            meta = {"provider": "http", "reranker_provider": provider}
            if model_id is not None:
                meta["model_id"] = model_id
            return out, None, meta
        except urllib.error.HTTPError as e:
            return {}, f"http_error:{e.code}", {"provider": "http"}
        except (urllib.error.URLError, socket.timeout) as e:
            return {}, f"http_unreachable:{type(e).__name__}", {"provider": "http"}
        except Exception as e:
            return {}, f"http_failed:{type(e).__name__}", {"provider": "http"}

    model = (model or settings.retrieval_rerank_model or settings.openai_chat_model).strip()

    lines = [f"问题：\n{question}\n", "段落："]
    for i, c in enumerate(chunks, start=1):
        cid = str(c.get("chunk_id") or "")
        text = str(c.get("text") or c.get("text_snippet") or "")
        text = text.strip().replace("\r\n", "\n")
        text = _truncate_text(text, int(max_chars))
        lines.append(f"{i}) chunk_id={cid}\n{text}\n")

    user = "\n".join(lines).strip()

    try:
        raw = chat_complete(_RERANK_SYSTEM, user, model=model).strip()
        obj = json.loads(raw)
    except Exception as e:
        return {}, f"parse_error: {type(e).__name__}", {"provider": "openai_chat"}

    try:
        items = obj.get("scores") if isinstance(obj, dict) else None
        if not isinstance(items, list):
            return {}, "bad_format"
        out: dict[str, float] = {}
        for it in items:
            if not isinstance(it, dict):
                continue
            cid = it.get("chunk_id")
            score = it.get("score")
            if not isinstance(cid, str):
                continue
            if isinstance(score, (int, float)):
                out[cid] = _clamp01(float(score))
        # Ensure every requested chunk_id exists in output (missing => 0.0).
        for c in chunks:
            cid = str(c.get("chunk_id") or "")
            if cid and cid not in out:
                out[cid] = 0.0
        return out, None, {"provider": "openai_chat"}
    except Exception as e:
        return {}, f"bad_output: {type(e).__name__}", {"provider": "openai_chat"}
