import os
import json
import time
from typing import Iterator
import re

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.responses import StreamingResponse
from starlette.middleware.sessions import SessionMiddleware

from .audit import append_audit_log
from .auth import optional_user, require_admin, require_user
from .rate_limit import check_and_increment
from .quota import redeem_code
from .rag import rag_answer, retrieve_only_answer, build_evidence_block
from .retrieval import retrieve
from .llm import chat_complete, chat_complete_stream
from .prompts import SYSTEM_PROMPT, USER_PROMPT, ABSTAIN_REWRITE_SYSTEM_PROMPT
from .query_normalize import normalize_question
from .policy import should_block
from .schemas import Chunk, LoginRequest, QueryRequest, QueryResponse, RetrievalMeta
from .settings import settings

app = FastAPI(title="Web-first RAG API", version="0.1.0")

def _normalize_origin(origin: str) -> str:
    origin = (origin or "").strip()
    while origin.endswith("/"):
        origin = origin[:-1]
    return origin


def _default_allowed_origins(web_origin: str) -> list[str]:
    """
    CORS origin matching is exact; add common localhost variants to avoid subtle mismatch issues
    (e.g. localhost vs 127.0.0.1, trailing slash).
    """
    base = _normalize_origin(web_origin)
    out = []
    if base:
        out.append(base)
        if "://localhost:" in base:
            out.append(base.replace("://localhost:", "://127.0.0.1:", 1))
        if "://127.0.0.1:" in base:
            out.append(base.replace("://127.0.0.1:", "://localhost:", 1))
    # de-dupe while preserving order
    seen: set[str] = set()
    uniq: list[str] = []
    for o in out:
        if o and o not in seen:
            seen.add(o)
            uniq.append(o)
    return uniq


app.add_middleware(
    SessionMiddleware,
    secret_key=settings.session_secret,
    session_cookie="rag_session",
    same_site="lax",
    https_only=False,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_default_allowed_origins(settings.web_origin),
    # CORS origin matching is exact; keep a localhost regex to avoid dev-time mismatches.
    allow_origin_regex=r"^https?://(localhost|127\\.0\\.0\\.1)(:\\d+)?$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/auth/login")
def login(body: LoginRequest, request: Request):
    if body.username == settings.auth_admin_username and body.password == settings.auth_admin_password:
        request.session["user"] = {"username": body.username, "role": "admin"}
        return {"ok": True, "role": "admin"}
    if body.username == settings.auth_username and body.password == settings.auth_password:
        request.session["user"] = {"username": body.username, "role": "researcher"}
        return {"ok": True, "role": "researcher"}
    raise HTTPException(status_code=401, detail="Invalid credentials")


@app.post("/auth/logout")
def logout(request: Request, user=Depends(require_user)):
    request.session.clear()
    return {"ok": True}


@app.get("/auth/me")
def me(user=Depends(require_user)):
    return {"ok": True, "user": user}


@app.post("/quota/redeem")
def quota_redeem(body: dict, request: Request, user=Depends(optional_user)):
    code = str(body.get("code") or "")
    anon_id = request.session.get("anon_id")
    user_id = user.get("username") if user else ""
    balance = redeem_code(code, user_id=user_id, anon_id=anon_id)
    return {"ok": True, "balance": balance}


@app.post("/query", response_model=QueryResponse)
def query(body: QueryRequest, request: Request, user=Depends(optional_user)):
    check_and_increment(request, user)
    top_k = body.top_k or settings.retrieval_top_k_default
    blocked, block_reason = should_block(body.question)
    if blocked:
        meta = RetrievalMeta(
            retrieved_k=0,
            kept_k=0,
            max_score=None,
            cutoff_score=None,
            strategy="policy_block",
            method=None,
        )
        return QueryResponse(mode=body.mode, answer="无法回答", used_chunks=[], retrieval=meta)

    question_retrieve, question_llm, qmeta = normalize_question(body.question)
    try:
        t0 = time.perf_counter()
        used, chunk_ids, meta = retrieve(question_retrieve, top_k)
        t_retrieve = time.perf_counter()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"retrieve_failed: {type(e).__name__}: {e}")

    if body.mode == "retrieve_only":
        answer = retrieve_only_answer(question_llm, used)
    else:
        used = used[: int(settings.rag_max_chunks)]
        answer = rag_answer(question_llm, used)

    t_done = time.perf_counter()
    meta = dict(meta or {})
    meta["question_normalized"] = bool(qmeta.get("applied"))
    if qmeta.get("applied"):
        meta["question_normalize_reason"] = qmeta.get("reason")
    meta["timing_ms"] = {
        "retrieve": int((t_retrieve - t0) * 1000),
        "total": int((t_done - t0) * 1000),
    }
    try:
        print(
            json.dumps(
                {
                    "event": "timing",
                    "path": "/query",
                    "mode": body.mode,
                    "timing_ms": meta["timing_ms"],
                    "retrieved_k": meta.get("retrieved_k"),
                    "kept_k": meta.get("kept_k"),
                    "method": meta.get("method"),
                },
                ensure_ascii=False,
            )
        )
    except Exception:
        pass

    append_audit_log(
        audit_log_path=settings.audit_log_path,
        username=user["username"],
        query=body.question,
        mode=body.mode,
        top_k=top_k,
        chunk_ids=chunk_ids,
        retrieval=meta,
    )

    used_chunks = [
        Chunk(
            chunk_id=u["chunk_id"],
            page_start=u["page_start"],
            page_end=u["page_end"],
            chapter=u.get("chapter", ""),
            score=u["score"],
            vector_score=u.get("vector_score"),
            bm25_score=u.get("bm25_score"),
            rrf_score=u.get("rrf_score"),
            rerank_score=u.get("rerank_score"),
            text_snippet=u["text_snippet"],
        )
        for u in used
    ]
    return QueryResponse(
        mode=body.mode,
        answer=answer,
        used_chunks=used_chunks,
        retrieval=RetrievalMeta(**meta),
    )


@app.post("/query/stream")
def query_stream(body: QueryRequest, request: Request, user=Depends(optional_user)):
    check_and_increment(request, user)
    top_k = body.top_k or settings.retrieval_top_k_default
    blocked, block_reason = should_block(body.question)
    question_retrieve, question_llm, qmeta = normalize_question(body.question)

    def sse_event(event: str, data_obj) -> str:
        return f"event: {event}\n" + f"data: {json.dumps(data_obj, ensure_ascii=False)}\n\n"

    def gen() -> Iterator[bytes]:
        t0 = time.perf_counter()
        try:
            yield sse_event("meta", {"status": "started"}).encode("utf-8")

            if blocked:
                meta = {
                    "retrieved_k": 0,
                    "kept_k": 0,
                    "max_score": None,
                    "cutoff_score": None,
                    "strategy": "policy_block",
                    "method": None,
                    "policy_reason": block_reason,
                }
                yield sse_event(
                    "done",
                    {
                        "answer": "无法回答",
                        "used_chunks": [],
                        "retrieval": meta,
                    },
                ).encode("utf-8")
                return

            try:
                used, chunk_ids, meta = retrieve(question_retrieve, top_k)
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"retrieve_failed: {type(e).__name__}: {e}")

            used = used[: int(settings.rag_max_chunks)]
            used_chunks = [
                Chunk(
                    chunk_id=u["chunk_id"],
                    page_start=u["page_start"],
                    page_end=u["page_end"],
                    chapter=u.get("chapter", ""),
                    score=u["score"],
                    vector_score=u.get("vector_score"),
                    bm25_score=u.get("bm25_score"),
                    rrf_score=u.get("rrf_score"),
                    rerank_score=u.get("rerank_score"),
                    text_snippet=u["text_snippet"],
                )
                for u in used
            ]

            t_retrieve = time.perf_counter()
            meta = dict(meta or {})
            meta["question_normalized"] = bool(qmeta.get("applied"))
            if qmeta.get("applied"):
                meta["question_normalize_reason"] = qmeta.get("reason")
            meta["timing_ms"] = {"retrieve": int((t_retrieve - t0) * 1000)}

            append_audit_log(
                audit_log_path=settings.audit_log_path,
                username=user["username"],
                query=body.question,
                mode="rag_stream",
                top_k=top_k,
                chunk_ids=chunk_ids,
                retrieval=meta,
            )

            yield sse_event(
                "meta",
                {
                    "status": "retrieved",
                    "retrieval": meta,
                    "used_chunks": [c.model_dump() for c in used_chunks],
                },
            ).encode("utf-8")

            if not used:
                # No evidence -> do not call LLM (prevents misleading hallucinations and saves latency).
                t_done = time.perf_counter()
                meta["timing_ms"] = {
                    **(meta.get("timing_ms") or {}),
                    "first_token": int((t_done - t0) * 1000),
                    "llm": 0,
                    "total": int((t_done - t0) * 1000),
                }
                answer = retrieve_only_answer(body.question, [])
                yield sse_event(
                    "done",
                    {
                        "answer": answer.strip(),
                        "used_chunks": [],
                        "retrieval": meta,
                    },
                ).encode("utf-8")
                return

            evidence = build_evidence_block(used)
            # Alias bridge (e.g., surname-only mentions): inject alias into the query for the LLM.
            q_llm = question_llm
            try:
                alias_hint = str((used[0] or {}).get("alias_hint") or "").strip()
            except Exception:
                alias_hint = ""
            if alias_hint and "=" in alias_hint:
                left, right = [s.strip() for s in alias_hint.split("=", 1)]
                if left and right and (left in q_llm) and (right not in q_llm):
                    q_llm = q_llm.replace(left, f"{left}（{right}）", 1)
            user_prompt = USER_PROMPT.format(context=evidence, query=q_llm)

            answer_acc = ""
            first_token_at: float | None = None
            for piece in chat_complete_stream(SYSTEM_PROMPT, user_prompt):
                answer_acc += piece
                if first_token_at is None:
                    first_token_at = time.perf_counter()
                yield sse_event("delta", {"text": piece}).encode("utf-8")

            # Strip meta talk in final answer to match UX requirement.
            answer_final = (answer_acc or "").strip()
            meta_words = ("参考信息", "资料", "原文", "片段", "页码", "章节", "知识库", "小说", "书中")
            meta_prefix_re = re.compile(
                r"^\\s*(?:"
                r"(?:根据|依据|参照|结合|基于)\\s*(?:上述|以下|给定|提供的)?\\s*(?:参考信息|资料|原文|片段)\\s*[，,。:：]?\\s*"
                r"|(?:在|从)\\s*(?:参考信息|资料|原文|片段)\\s*(?:中|里|内)\\s*[，,。:：]?\\s*"
                r")",
                re.IGNORECASE,
            )
            meta_clause_re = re.compile(
                r"(?:"
                r"(?:参考信息|资料|原文|片段)\\s*(?:中|里|内)?\\s*(?:没有|未|并未)\\s*(?:提到|说明|给出|描述)[^。！？\\n]*"
                r"|(?:无法|不能)\\s*(?:从|依据|根据)\\s*(?:参考信息|资料|原文|片段)[^。！？\\n]*"
                r")",
                re.IGNORECASE,
            )
            parts = re.split(r"(?<=[。！？\\n])", answer_final)
            kept: list[str] = []
            for p in parts:
                s = p.strip()
                if not s:
                    continue
                if meta_clause_re.search(s) and len(meta_clause_re.sub("", s).strip()) <= 2:
                    continue
                s2 = meta_prefix_re.sub("", s).strip()
                s2 = meta_clause_re.sub("", s2).strip()
                if any(w in s2 for w in meta_words):
                    for w in meta_words:
                        s2 = s2.replace(w, "")
                    s2 = s2.strip()
                if not s2:
                    continue
                kept.append(s2)
            answer_final = "".join(kept).strip()
            if not answer_final:
                answer_final = "无法回答"
            # Normalize refusal variants; never refuse when evidence exists (fallback will reuse the same chunks).
            refusal_re = re.compile(
                r"(?:无法确认|无法确定|不确定|未知|没有(?:提到|说明|给出|描述)|未(?:提到|说明|给出|描述)|并未(?:提到|说明|给出|描述)|信息不足|资料不足|证据不足)"
            )
            # If streaming path still ends up abstaining, run the non-streaming answerer once as a fallback.
            # This reuses the same retrieved chunks and includes extra retry/heuristic extraction.
            if answer_final.strip() == "无法回答" or (len(answer_final) <= 120 and refusal_re.search(answer_final)):
                try:
                    answer_final = rag_answer(question_llm, used)
                except Exception:
                    answer_final = "无法回答"

            t_done = time.perf_counter()
            if first_token_at is None:
                first_token_at = t_done
            meta["timing_ms"] = {
                **(meta.get("timing_ms") or {}),
                "first_token": int((first_token_at - t0) * 1000),
                "llm": int((t_done - t_retrieve) * 1000),
                "total": int((t_done - t0) * 1000),
            }
            try:
                print(
                    json.dumps(
                        {
                            "event": "timing",
                            "path": "/query/stream",
                            "mode": "rag_stream",
                            "timing_ms": meta["timing_ms"],
                            "timing_detail_ms": meta.get("timing_detail_ms"),
                            "retrieved_k": meta.get("retrieved_k"),
                            "kept_k": meta.get("kept_k"),
                            "method": meta.get("method"),
                        },
                        ensure_ascii=False,
                    )
                )
            except Exception:
                pass

            yield sse_event(
                "done",
                {
                    "answer": answer_final.strip(),
                    "used_chunks": [c.model_dump() for c in used_chunks],
                    "retrieval": meta,
                },
            ).encode("utf-8")
        except Exception as e:
            yield sse_event("error", {"error": f"stream_failed: {type(e).__name__}: {e}"}).encode("utf-8")

    return StreamingResponse(gen(), media_type="text/event-stream")


@app.get("/admin/logs/download")
def download_logs(admin=Depends(require_admin)):
    path = settings.audit_log_path
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="No logs yet")
    return FileResponse(path, media_type="application/jsonl", filename="logs.jsonl")
