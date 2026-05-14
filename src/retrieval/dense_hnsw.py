from __future__ import annotations

from typing import List, Tuple

import numpy as np

from src.data.schemas import RetrievedDoc


class DenseHNSWRetriever:
    def __init__(self, documents: List[str], model_name: str, m: int = 16, ef_construction: int = 200, ef_search: int = 64) -> None:
        self.documents = documents
        self.model_name = model_name
        self.ef_search = ef_search
        self.index = None
        self.embeddings: np.ndarray | None = None
        self.fallback = None

        try:
            from sentence_transformers import SentenceTransformer
            import hnswlib

            self._encoder = SentenceTransformer(model_name)
            self.embeddings = np.asarray(self._encoder.encode(documents, show_progress_bar=False, normalize_embeddings=True), dtype=np.float32)
            dim = int(self.embeddings.shape[1])

            idx = hnswlib.Index(space="cosine", dim=dim)
            idx.init_index(max_elements=len(documents), ef_construction=ef_construction, M=m)
            idx.add_items(self.embeddings, np.arange(len(documents)))
            idx.set_ef(ef_search)
            self.index = idx
        except Exception:
            # Fallback keeps the project runnable if HNSW or encoder setup is unavailable.
            from sklearn.feature_extraction.text import TfidfVectorizer

            self._encoder = None
            vec = TfidfVectorizer(max_features=10000)
            self.fallback = vec.fit_transform(documents)
            self._vectorizer = vec

    def _retrieve_hnsw(self, query: str, top_k: int) -> List[Tuple[int, float]]:
        q_emb = np.asarray(self._encoder.encode([query], show_progress_bar=False, normalize_embeddings=True), dtype=np.float32)
        labels, distances = self.index.knn_query(q_emb, k=min(top_k, len(self.documents)))
        # cosine distance -> similarity
        pairs = []
        for doc_idx, dist in zip(labels[0], distances[0]):
            pairs.append((int(doc_idx), float(1.0 - dist)))
        return pairs

    def _retrieve_fallback(self, query: str, top_k: int) -> List[Tuple[int, float]]:
        q = self._vectorizer.transform([query])
        scores = (self.fallback @ q.T).toarray().reshape(-1)
        order = np.argsort(scores)[::-1][:top_k]
        return [(int(i), float(scores[int(i)])) for i in order]

    def retrieve(self, query: str, top_k: int) -> List[RetrievedDoc]:
        if self.index is not None:
            pairs = self._retrieve_hnsw(query, top_k)
            source = "dense_hnsw"
        else:
            pairs = self._retrieve_fallback(query, top_k)
            source = "dense_fallback"

        return [
            RetrievedDoc(doc_id=doc_idx, text=self.documents[doc_idx], score=score, source=source)
            for doc_idx, score in pairs
        ]
