from __future__ import annotations

from typing import List

from src.data.schemas import RetrievedDoc
from .utils import token_set


class LightweightReranker:
    def __init__(self, overlap_weight: float = 0.7, dense_weight: float = 0.3) -> None:
        self.overlap_weight = overlap_weight
        self.dense_weight = dense_weight

    def rerank(self, query: str, docs: List[RetrievedDoc], top_k: int) -> List[RetrievedDoc]:
        q = token_set(query)
        ranked: List[RetrievedDoc] = []
        for doc in docs:
            d = token_set(doc.text)
            overlap = 0.0 if not q else len(q & d) / max(1, len(q))
            blended = self.overlap_weight * overlap + self.dense_weight * doc.score
            ranked.append(RetrievedDoc(doc_id=doc.doc_id, text=doc.text, score=float(blended), source="reranked"))

        ranked.sort(key=lambda x: x.score, reverse=True)
        return ranked[:top_k]
