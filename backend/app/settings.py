from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    openai_api_key: str
    openai_base_url: str | None = None
    openai_embed_model: str = "text-embedding-3-large"
    openai_chat_model: str = "gpt-4.1-mini"
    openai_chat_max_tokens: int = 900
    openai_timeout_seconds: float = 60.0
    openai_embed_timeout_seconds: float = 15.0
    openai_chat_timeout_seconds: float = 60.0

    # Optional split providers:
    # - If set, chat will use CHAT_*; otherwise falls back to OPENAI_*
    # - If set, embeddings will use EMBED_*; otherwise falls back to OPENAI_*
    chat_api_key: str | None = None
    chat_base_url: str | None = None
    chat_model: str | None = None
    embed_api_key: str | None = None
    embed_base_url: str | None = None
    embed_model: str | None = None

    auth_username: str = "researcher"
    auth_password: str = "change_me"
    auth_admin_username: str = "admin"
    auth_admin_password: str = "change_me_too"

    session_secret: str = "change_me_session_secret"
    web_origin: str = "http://localhost:3000"

    data_pdf_path: str = "/app/data/book.pdf"
    faiss_dir: str = "/app/storage/faiss"
    chunks_path: str = "/app/storage/chunks.jsonl"
    audit_log_path: str = "/app/storage/logs.jsonl"
    entity_index_path: str = "/app/storage/entity_index.json"
    aliases_path: str = "/app/storage/aliases.json"

    ingest_extractor: str = "auto"  # auto|pdfminer|pypdf
    ingest_min_nonempty_pages: int = 5
    ingest_min_total_chars: int = 2000
    # 1-based; pages before this are treated as front-matter and skipped (keeps page numbers stable).
    ingest_content_start_page: int = 1
    ingest_min_chunk_chars: int = 120
    # Chunking strategy (novel-friendly): keep chunks near the evidence cap to avoid truncation later.
    ingest_chunk_max_chars: int = 900
    ingest_chunk_overlap_paragraphs: int = 1
    ingest_chunk_hard_cut_overlap_chars: int = 150
    ingest_chunk_merge_min_chars: int = 200
    ingest_embed_batch_size: int = 32
    ingest_embed_timeout_seconds: float = 60.0
    ingest_embed_max_retries: int = 6
    ingest_embed_backoff_seconds: float = 1.0
    ingest_embed_backoff_max_seconds: float = 15.0
    ingest_build_entity_index: bool = True
    entity_ngram_min: int = 2
    entity_ngram_max: int = 6
    entity_min_df: int = 2
    entity_max_df_ratio: float = 0.20
    entity_per_chunk_keep: int = 120
    entity_max_postings_per_entity: int = 2000
    entity_keep_earliest_per_entity: int = 200

    retrieval_top_k_default: int = 16
    retrieval_min_keep: int = 6
    retrieval_score_delta: float = 0.20
    retrieval_gap_threshold: float = 0.18
    # Out-of-scope abstain gate: if top vector score < this, return no evidence.
    # Note: In hybrid mode, this still gates on the vector top-1 score (not the fused score).
    retrieval_min_score: float | None = None

    # Hybrid retrieval (BM25 + vector). Set RETRIEVAL_HYBRID_ALPHA to enable.
    retrieval_hybrid_alpha: float | None = 0.65  # 0..1 (vector weight), None disables hybrid
    retrieval_hybrid_vec_k: int = 50
    retrieval_hybrid_bm25_k: int = 50
    retrieval_hybrid_rrf_k0: int = 60

    # Optional rerank (final ranking) after recall (vector or hybrid).
    # When enabled, `used_chunks[].score` becomes the rerank score (final_score).
    retrieval_rerank_enabled: bool = True
    retrieval_rerank_model: str | None = None  # defaults to openai_chat_model
    retrieval_rerank_candidates: int = 32  # how many recalled chunks to rerank
    retrieval_rerank_max_chars: int = 800  # per-chunk text length sent to reranker
    retrieval_rerank_base_url: str | None = "http://rerank:9001"
    retrieval_rerank_http_timeout_seconds: float = 60.0

    # Optional query expansion (multi-query recall). Runs before recall; merges candidates with weighted RRF.
    retrieval_expand_enabled: bool = True
    retrieval_expand_model: str | None = None  # defaults to openai_chat_model
    retrieval_expand_n: int = 2  # number of extra queries
    retrieval_expand_rrf_k0: int = 60
    retrieval_expand_weight_original: float = 1.0
    retrieval_expand_weight_extra: float = 0.7

    # Optional abstain gate based on lexical overlap between query and top-1 vector hit.
    # Useful to refuse questions that are semantically "close" but not actually covered by the KB.
    retrieval_min_query_token_overlap: int = 0
    retrieval_overlap_gate_max_score: float = 0.0

    # If embeddings are slow/unavailable, optionally fall back to BM25-only retrieval (keeps service responsive).
    retrieval_fallback_bm25_on_embed_failure: bool = True

    # Cache recent query embeddings in-process (helps repeated questions; set 0 to disable).
    retrieval_query_embed_cache_size: int = 256
    entity_enabled: bool = False
    entity_query_max: int = 3
    entity_postings_per_entity: int = 20
    entity_rrf_k0: int = 60
    entity_rrf_weight_passage: float = 1.0
    entity_rrf_weight_entity: float = 0.8

    # RAG prompt shaping (latency-critical): cap how much evidence is sent to the LLM.
    rag_max_chunks: int = 10
    rag_evidence_chunk_max_chars: int = 900
    rag_evidence_total_max_chars: int = 10000
    rag_retry_on_abstain: bool = True

    # Temporal questions (first/earliest appearance): recall more candidates then reorder by document position.
    retrieval_temporal_enabled: bool = True
    retrieval_temporal_candidates: int = 80
    retrieval_temporal_keep: int = 6


settings = Settings()
