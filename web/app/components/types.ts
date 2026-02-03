export type Mode = "retrieve_only" | "rag";

export type UsedChunk = {
  chunk_id: string;
  page_start: number;
  page_end: number;
  chapter: string;
  score: number;
  vector_score?: number | null;
  bm25_score?: number | null;
  rrf_score?: number | null;
  rerank_score?: number | null;
  text_snippet: string;
};

export type RetrievalMeta = {
  retrieved_k: number;
  kept_k: number;
  max_score: number | null;
  cutoff_score: number | null;
  strategy: string;
  method?: string | null;
  hybrid_alpha?: number | null;
  hybrid_vec_k?: number | null;
  hybrid_bm25_k?: number | null;
  hybrid_rrf_k0?: number | null;
  vector_max_score?: number | null;
  token_overlap?: number | null;
  rerank_enabled?: boolean | null;
  rerank_provider?: string | null;
  rerank_model?: string | null;
  rerank_candidates?: number | null;
  rerank_status?: string | null;
  expand_enabled?: boolean | null;
  expand_n?: number | null;
  expand_queries?: string[] | null;
  expand_status?: string | null;
};

export type QueryResponse = {
  mode: Mode;
  answer: string;
  used_chunks: UsedChunk[];
  retrieval?: RetrievalMeta | null;
};
