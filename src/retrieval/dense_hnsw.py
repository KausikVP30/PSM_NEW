from __future__ import annotations

from typing import List, Tuple

import numpy as np
import torch

from src.data.schemas import RetrievedDoc
from src.utils.paths import resolve_device


class DenseHNSWRetriever:
    def __init__(self, documents: List[str], model_name: str, m: int = 16, ef_construction: int = 200, ef_search: int = 64, device: str = "auto") -> None:
        self.documents = documents
        self.model_name = model_name
        self.ef_search = ef_search
        self.index = None
        self.embeddings: np.ndarray | None = None
        self._use_hnsw = False
        self._source = "dense_exact"

        resolved_device = resolve_device(device)

        from sentence_transformers import SentenceTransformer
        try:
            import hnswlib  # type: ignore
        except ImportError:
            hnswlib = None  # type: ignore

        self._encoder = SentenceTransformer(model_name, device=resolved_device)
        if documents:
            self.embeddings = np.asarray(
                self._encoder.encode(documents, show_progress_bar=False, normalize_embeddings=True),
                dtype=np.float32,
            )
            if hnswlib is not None:
                dim = int(self.embeddings.shape[1])
                idx = hnswlib.Index(space="cosine", dim=dim)
                idx.init_index(max_elements=len(documents), ef_construction=ef_construction, M=m)
                idx.add_items(self.embeddings, np.arange(len(documents)))
                idx.set_ef(ef_search)
                self.index = idx
                self._use_hnsw = True
                self._source = "dense_hnsw"

    def _retrieve_hnsw(self, query: str, top_k: int) -> List[Tuple[int, float]]:
        if self.index is None or self.embeddings is None:
            return []
        q_emb = np.asarray(self._encoder.encode([query], show_progress_bar=False, normalize_embeddings=True), dtype=np.float32)
        labels, distances = self.index.knn_query(q_emb, k=min(top_k, len(self.documents)))
        # cosine distance -> similarity
        pairs = []
        for doc_idx, dist in zip(labels[0], distances[0]):
            pairs.append((int(doc_idx), float(1.0 - dist)))
        return pairs

    def _retrieve_exact(self, query: str, top_k: int) -> List[Tuple[int, float]]:
        if self.embeddings is None or not len(self.documents):
            return []
        q_emb = np.asarray(self._encoder.encode([query], show_progress_bar=False, normalize_embeddings=True), dtype=np.float32)[0]
        scores = np.dot(self.embeddings, q_emb)
        if scores.ndim != 1:
            scores = np.asarray(scores).reshape(-1)
        k = min(top_k, len(self.documents))
        if k <= 0:
            return []
        top_indices = np.argpartition(-scores, kth=k - 1)[:k]
        top_indices = top_indices[np.argsort(-scores[top_indices])]
        return [(int(doc_idx), float(scores[doc_idx])) for doc_idx in top_indices]

    def retrieve(self, query: str, top_k: int) -> List[RetrievedDoc]:
        pairs = self._retrieve_hnsw(query, top_k) if self._use_hnsw else self._retrieve_exact(query, top_k)
        return [
            RetrievedDoc(doc_id=doc_idx, text=self.documents[doc_idx], score=score, source=self._source)
            for doc_idx, score in pairs
        ]
