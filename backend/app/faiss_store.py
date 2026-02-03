import json
import os
from dataclasses import dataclass

import faiss
import numpy as np


@dataclass
class StoredChunk:
    chunk_id: str
    page_start: int
    page_end: int
    text: str


class FaissStore:
    def __init__(self, faiss_dir: str):
        self.faiss_dir = faiss_dir
        self.index_path = os.path.join(faiss_dir, "index.faiss")
        self.map_path = os.path.join(faiss_dir, "id_map.json")
        self.index: faiss.Index | None = None
        self.id_map: list[str] = []

    def exists(self) -> bool:
        return os.path.exists(self.index_path) and os.path.exists(self.map_path)

    def load(self) -> None:
        self.index = faiss.read_index(self.index_path)
        with open(self.map_path, "r", encoding="utf-8") as f:
            self.id_map = json.load(f)

    def save(self) -> None:
        os.makedirs(self.faiss_dir, exist_ok=True)
        if self.index is None:
            raise RuntimeError("Index not built")
        faiss.write_index(self.index, self.index_path)
        with open(self.map_path, "w", encoding="utf-8") as f:
            json.dump(self.id_map, f, ensure_ascii=False, indent=2)

    @staticmethod
    def _normalize(vectors: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-12
        return vectors / norms

    def init_empty(self, dim: int) -> None:
        """
        Initialize an empty cosine-similarity (IP on normalized vectors) index.
        """
        if dim <= 0:
            raise ValueError("dim must be positive")
        self.index = faiss.IndexFlatIP(int(dim))
        self.id_map = []

    def build(self, embeddings: list[list[float]], chunk_ids: list[str]) -> None:
        vecs = np.array(embeddings, dtype="float32")
        vecs = self._normalize(vecs)
        dim = vecs.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(vecs)
        self.index = index
        self.id_map = list(chunk_ids)

    def add(self, embeddings: list[list[float]], chunk_ids: list[str]) -> None:
        """
        Incrementally add vectors + ids to the index.
        """
        if not embeddings:
            return
        if len(embeddings) != len(chunk_ids):
            raise ValueError("embeddings and chunk_ids length mismatch")

        vecs = np.array(embeddings, dtype="float32")
        vecs = self._normalize(vecs)
        dim = int(vecs.shape[1])

        if self.index is None:
            self.init_empty(dim)
        else:
            # Best-effort dimension check for flat indexes
            try:
                if int(self.index.d) != dim:  # type: ignore[attr-defined]
                    raise ValueError(f"Embedding dim mismatch: index={int(self.index.d)} batch={dim}")  # type: ignore[attr-defined]
            except Exception:
                pass

        self.index.add(vecs)
        self.id_map.extend(list(chunk_ids))

    def search(self, query_embedding: list[float], top_k: int) -> tuple[list[str], list[float]]:
        if self.index is None:
            raise RuntimeError("Index not loaded")
        try:
            index_dim = int(self.index.d)  # type: ignore[attr-defined]
        except Exception:
            index_dim = None
        q_dim = len(query_embedding)
        if index_dim is not None and q_dim != index_dim:
            raise ValueError(
                f"Query embedding dim mismatch: query={q_dim} index={index_dim}. "
                "Your OPENAI_EMBED_MODEL must match the model used to build the FAISS index; "
                "either switch back to the original embedding model or rebuild the index via ingest."
            )
        q = np.array([query_embedding], dtype="float32")
        q = self._normalize(q)
        scores, idxs = self.index.search(q, top_k)
        ids: list[str] = []
        sc: list[float] = []
        for i, s in zip(idxs[0].tolist(), scores[0].tolist()):
            if i < 0:
                continue
            ids.append(self.id_map[i])
            sc.append(float(s))
        return ids, sc

    def score_chunk_ids(self, query_embedding: list[float], chunk_ids: list[str]) -> list[float]:
        """
        Return cosine similarities (inner product of normalized vectors) for specific chunk_ids.
        Works with IndexFlatIP (vectors already normalized at build time).
        """
        if self.index is None:
            raise RuntimeError("Index not loaded")
        if not chunk_ids:
            return []

        q = np.array([query_embedding], dtype="float32")
        q = self._normalize(q)[0]

        id_to_pos = {cid: i for i, cid in enumerate(self.id_map)}
        out: list[float] = []
        for cid in chunk_ids:
            pos = id_to_pos.get(cid)
            if pos is None:
                out.append(0.0)
                continue
            try:
                v = self.index.reconstruct(int(pos))  # type: ignore[attr-defined]
                out.append(float(np.dot(q, np.asarray(v, dtype="float32"))))
            except Exception:
                out.append(0.0)
        return out
