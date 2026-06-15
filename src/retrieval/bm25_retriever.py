from __future__ import annotations

from typing import List

import numpy as np

from src.data.schemas import RetrievedDoc
from .utils import tokenize


class BM25Retriever:
    def __init__(self, documents: List[str]) -> None:
        self.documents = documents
        from rank_bm25 import BM25Okapi  # type: ignore

        self.tokenized_docs = [tokenize(d) for d in documents]
        self.model = BM25Okapi(self.tokenized_docs)

    def retrieve(self, query: str, top_k: int) -> List[RetrievedDoc]:
        q_tokens = tokenize(query)
        scores = self.model.get_scores(q_tokens)
        if len(scores) == 0:
            return []
        top_idx = np.argsort(scores)[::-1][:top_k]
        return [
            RetrievedDoc(doc_id=int(i), text=self.documents[int(i)], score=float(scores[int(i)]), source="bm25")
            for i in top_idx
        ]
