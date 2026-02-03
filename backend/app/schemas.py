from pydantic import BaseModel, Field
from typing import Literal


class LoginRequest(BaseModel):
    username: str
    password: str


class Chunk(BaseModel):
    chunk_id: str
    page_start: int
    page_end: int
    chapter: str = ""
    score: float
    vector_score: float | None = None
    bm25_score: float | None = None
    rrf_score: float | None = None
    rerank_score: float | None = None
    text_snippet: str


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    mode: Literal["retrieve_only", "rag"] = "rag"
    top_k: int = Field(default=16, ge=1, le=50)


class RetrievalMeta(BaseModel):
    retrieved_k: int
    kept_k: int
    max_score: float | None
    cutoff_score: float | None
    strategy: str
    method: str | None = None
    hybrid_alpha: float | None = None
    hybrid_vec_k: int | None = None
    hybrid_bm25_k: int | None = None
    hybrid_rrf_k0: int | None = None
    vector_max_score: float | None = None
    token_overlap: int | None = None
    rerank_enabled: bool | None = None
    rerank_provider: str | None = None
    rerank_model: str | None = None
    rerank_candidates: int | None = None
    rerank_status: str | None = None
    expand_enabled: bool | None = None
    expand_n: int | None = None
    expand_queries: list[str] | None = None
    expand_status: str | None = None
    temporal_enabled: bool | None = None
    temporal_entities: list[str] | None = None
    temporal_candidates: int | None = None
    temporal_keep: int | None = None
    temporal_status: str | None = None
    timing_ms: dict[str, int] | None = None
    timing_detail_ms: dict[str, int] | None = None


class QueryResponse(BaseModel):
    mode: Literal["retrieve_only", "rag"]
    answer: str
    used_chunks: list[Chunk]
    retrieval: RetrievalMeta | None = None
