from __future__ import annotations

from typing import List

import numpy as np

from src.data.schemas import RetrievedDoc
from .utils import tokenize


class BM25Retriever:
    def __init__(self, documents: List[str]) -> None:
        self.documents = documents
        self._use_bm25 = False
        try:
            from rank_bm25 import BM25Okapi  # type: ignore

            self.tokenized_docs = [tokenize(d) for d in documents]
            self.model = BM25Okapi(self.tokenized_docs)
            self._use_bm25 = True
        except Exception:
            # Fallback to TF-IDF cosine similarity if rank_bm25 isn't available
            from sklearn.feature_extraction.text import TfidfVectorizer

            self._vectorizer = TfidfVectorizer(max_features=10000)
            self._tfidf = self._vectorizer.fit_transform(documents)

    def retrieve(self, query: str, top_k: int) -> List[RetrievedDoc]:
        if self._use_bm25:
            q_tokens = tokenize(query)
            scores = self.model.get_scores(q_tokens)
            if len(scores) == 0:
                return []
            top_idx = np.argsort(scores)[::-1][:top_k]
            return [
                RetrievedDoc(doc_id=int(i), text=self.documents[int(i)], score=float(scores[int(i)]), source="bm25")
                for i in top_idx
            ]

        # TF-IDF fallback
        q_vec = self._vectorizer.transform([query])
        scores = (self._tfidf @ q_vec.T).toarray().reshape(-1)
        order = np.argsort(scores)[::-1][:top_k]
        return [RetrievedDoc(doc_id=int(i), text=self.documents[int(i)], score=float(scores[int(i)]), source="bm25_fallback") for i in order]
