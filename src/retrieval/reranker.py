from __future__ import annotations

from typing import List
import logging
import torch

from src.data.schemas import RetrievedDoc
from .utils import token_set
from src.utils.paths import resolve_device


class LightweightReranker:
    def __init__(
        self,
        overlap_weight: float = 0.8,
        dense_weight: float = 0.2,
        doc_score_min: float = 0.0,
        doc_text_max_chars: int = 300,
        cross_encoder_model: str = "",
    ) -> None:
        self.overlap_weight = overlap_weight
        self.dense_weight = dense_weight
        self.doc_score_min = float(doc_score_min)
        self.doc_text_max_chars = int(doc_text_max_chars)
        self.cross_encoder_model = cross_encoder_model.strip()
        self._cross_encoder = None
        self._logger = logging.getLogger(__name__)
        
        if not self.cross_encoder_model:
            raise RuntimeError("cross_encoder_model is required for GPU-only reranking. Set a valid CrossEncoder model name.")
        
        resolved_device = resolve_device("auto")
        try:
            from sentence_transformers import CrossEncoder

            self._cross_encoder = CrossEncoder(self.cross_encoder_model, device=resolved_device)
        except Exception as exc:
            raise RuntimeError(f"Failed to load CrossEncoder '{self.cross_encoder_model}' on GPU: {exc}") from exc

    def rerank(self, query: str, docs: List[RetrievedDoc], top_k: int) -> List[RetrievedDoc]:
        if not docs:
            return []
        
        candidates = [doc for doc in docs if doc.score >= self.doc_score_min]
        if not candidates:
            return []
            
        pairs = [(query, doc.text[: self.doc_text_max_chars]) for doc in candidates]
        scores = self._cross_encoder.predict(pairs)
        ranked = [
            RetrievedDoc(
                doc_id=doc.doc_id,
                text=doc.text[: self.doc_text_max_chars],
                score=float(score),
                source="cross_encoder_reranked",
            )
            for doc, score in zip(candidates, scores)
        ]
        ranked.sort(key=lambda x: x.score, reverse=True)
        return ranked[:top_k]
