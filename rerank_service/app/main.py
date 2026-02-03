import os
import threading
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


def _sigmoid(x: float) -> float:
    # numerically stable sigmoid for moderate logits
    try:
        import math

        if x >= 0:
            z = math.exp(-x)
            return 1.0 / (1.0 + z)
        z = math.exp(x)
        return z / (1.0 + z)
    except Exception:
        return 0.0


def _clamp01(x: float) -> float:
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return float(x)


class Passage(BaseModel):
    chunk_id: str = Field(min_length=1, max_length=200)
    text: str = Field(min_length=1, max_length=20000)


class RerankRequest(BaseModel):
    query: str = Field(min_length=1, max_length=4000)
    passages: list[Passage] = Field(min_length=1, max_length=200)


class ScoreItem(BaseModel):
    chunk_id: str
    score: float


class RerankResponse(BaseModel):
    provider: str
    model_id: str
    scores: list[ScoreItem]


_MODEL_LOCK = threading.Lock()
_MODEL: Any | None = None


def _get_model():
    global _MODEL
    if _MODEL is not None:
        return _MODEL
    with _MODEL_LOCK:
        if _MODEL is not None:
            return _MODEL
        model_id = os.getenv("MODEL_ID", "BAAI/bge-reranker-base").strip()
        device = os.getenv("DEVICE", "cpu").strip() or "cpu"
        try:
            from sentence_transformers import CrossEncoder
        except Exception as e:
            raise RuntimeError(f"missing_dependency: {type(e).__name__}: {e}") from e
        _MODEL = CrossEncoder(model_id, device=device)
        return _MODEL


app = FastAPI(title="Rerank Service", version="0.1.0")


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/rerank", response_model=RerankResponse)
def rerank(body: RerankRequest):
    model = _get_model()
    model_id = os.getenv("MODEL_ID", "BAAI/bge-reranker-base").strip()
    max_chars = int(os.getenv("PASSAGE_MAX_CHARS", "800") or "800")
    batch_size = int(os.getenv("BATCH_SIZE", "16") or "16")

    pairs: list[tuple[str, str]] = []
    ids: list[str] = []
    for p in body.passages:
        t = (p.text or "").strip()
        if max_chars > 0 and len(t) > max_chars:
            t = t[:max_chars]
        pairs.append((body.query, t))
        ids.append(p.chunk_id)

    try:
        # CrossEncoder.predict returns logits by default for many models
        logits = model.predict(pairs, batch_size=batch_size, convert_to_numpy=False, show_progress_bar=False)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"inference_failed: {type(e).__name__}: {e}")

    scores: list[ScoreItem] = []
    for cid, logit in zip(ids, logits):
        try:
            v = float(logit)
        except Exception:
            v = 0.0
        scores.append(ScoreItem(chunk_id=cid, score=_clamp01(_sigmoid(v))))
    return RerankResponse(provider="sentence-transformers", model_id=model_id, scores=scores)


@app.on_event("startup")
def _startup():
    if (os.getenv("PRELOAD_MODEL") or "").strip().lower() in ("1", "true", "yes", "on"):
        _get_model()
