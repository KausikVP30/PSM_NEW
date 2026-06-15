from __future__ import annotations

from typing import Dict, List

import numpy as np

from src.data.schemas import RetrievedDoc
from .bm25_retriever import BM25Retriever
from .dense_hnsw import DenseHNSWRetriever


class HybridRetriever:
    def __init__(
        self,
        bm25: BM25Retriever,
        dense: DenseHNSWRetriever,
        sparse_weight: float,
        dense_weight: float,
    ) -> None:
        self.bm25 = bm25
        self.dense = dense
        self.sparse_weight = sparse_weight
        self.dense_weight = dense_weight

    @staticmethod
    def _normalize_scores(items: List[RetrievedDoc]) -> Dict[int, float]:
        if not items:
            return {}
        scores = np.array([i.score for i in items], dtype=float)
        min_s = float(scores.min())
        max_s = float(scores.max())
        if max_s - min_s < 1e-9:
            return {i.doc_id: 1.0 for i in items}
        return {i.doc_id: float((i.score - min_s) / (max_s - min_s)) for i in items}

    def retrieve(self, query: str, top_k_sparse: int, top_k_dense: int, top_k_fused: int) -> List[RetrievedDoc]:
        sparse = self.bm25.retrieve(query, top_k_sparse)
        dense = self.dense.retrieve(query, top_k_dense)

        sparse_norm = self._normalize_scores(sparse)
        dense_norm = self._normalize_scores(dense)

        all_ids = set(sparse_norm) | set(dense_norm)
        fused: List[RetrievedDoc] = []
        for doc_id in all_ids:
            s = sparse_norm.get(doc_id, 0.0)
            d = dense_norm.get(doc_id, 0.0)
            score = self.sparse_weight * s + self.dense_weight * d
            text = self.bm25.documents[doc_id]
            fused.append(RetrievedDoc(doc_id=doc_id, text=text, score=score, source="hybrid"))

        fused.sort(key=lambda x: x.score, reverse=True)
        return fused[:top_k_fused]
