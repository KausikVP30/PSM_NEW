"""Semantic memory store using embeddings for similarity-based lookup."""
from __future__ import annotations

import os
import pickle
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from src.utils.paths import resolve_device


@dataclass
class MemoryItemWithEmbedding:
    """Memory item with precomputed embedding."""
    query: str
    query_embedding: np.ndarray  # Shape: (embedding_dim,)
    answer: str
    context: str
    quality_score: float  # Combined quality metric (0.0-1.0)
    evidence_supported: bool = False
    timestamp: int = 0


class EmbeddingMemoryStore:
    """Memory store using semantic embeddings for similarity-based lookup.
    
    Features:
    - Semantic similarity matching using embeddings
    - Quality-based confidence scoring (reranker + ROUGE)
    - GPU support for embedding computation
    - Disk persistence of embeddings
    - Backward compatible with token-based lookups
    """

    def __init__(
        self,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        similarity_threshold: float = 0.75,
        write_rouge_threshold: float = 0.7,
        write_reranker_threshold: float = 0.7,
        min_answer_f1: float = 0.5,
        min_quality: float = 0.75,
        require_evidence_support: bool = True,
        quality_metric_weights: Dict[str, float] | None = None,
        max_items: int = 500,
        persist_embeddings: bool = True,
        persist_dir: str = "outputs/memory_embeddings",
        device: str = "auto",
        flush_every: int = 25,
    ) -> None:
        """Initialize semantic memory store.
        
        Args:
            embedding_model: Model name for embeddings
            similarity_threshold: Minimum similarity for memory hit (0.0-1.0)
            max_items: Maximum number of items to store
            persist_embeddings: Whether to save embeddings to disk
            persist_dir: Directory for persisting embeddings
            device: "auto", "gpu", "cpu" - device for embeddings
        """
        self.embedding_model_name = embedding_model
        self.similarity_threshold = similarity_threshold
        self.write_rouge_threshold = write_rouge_threshold
        self.write_reranker_threshold = write_reranker_threshold
        self.min_answer_f1 = float(min_answer_f1)
        self.min_quality = float(min_quality)
        self.require_evidence_support = bool(require_evidence_support)
        weights = quality_metric_weights or {}
        self.reranker_quality_weight = float(weights.get("reranker_score", 0.2))
        self.rouge_quality_weight = float(weights.get("rouge_l_score", 0.8))
        total_weight = self.reranker_quality_weight + self.rouge_quality_weight
        if total_weight <= 0:
            self.reranker_quality_weight = 0.2
            self.rouge_quality_weight = 0.8
        else:
            self.reranker_quality_weight /= total_weight
            self.rouge_quality_weight /= total_weight
        self.max_items = max_items
        self.persist_embeddings = persist_embeddings
        self.persist_dir = persist_dir
        self.device = self._resolve_device(device)
        self._logger = logging.getLogger(__name__)

        self._embedding_model = None
        self._items: List[MemoryItemWithEmbedding] = []
        self._timestamp_counter = 0
        self._dirty_count = 0
        self.flush_every = int(flush_every)

        # Create persist directory if needed
        if self.persist_embeddings:
            os.makedirs(self.persist_dir, exist_ok=True)
            self._load_from_disk()

    def _resolve_device(self, device: str) -> str:
        """Resolve device to 'cpu' or 'cuda:X'."""
        from src.memory.embedding_utils import resolve_embedding_device
        return resolve_embedding_device(device)

    def _get_embedding_model(self):
        """Lazy load embedding model."""
        if self._embedding_model is None:
            from sentence_transformers import SentenceTransformer
            self._embedding_model = SentenceTransformer(
                self.embedding_model_name,
                device=self.device
            )
        return self._embedding_model

    def _embed_query(self, query: str) -> np.ndarray:
        """Compute embedding for query.
        
        Args:
            query: Query text
            
        Returns:
            Embedding vector as numpy array
        """
        model = self._get_embedding_model()
        embedding = model.encode(query, convert_to_numpy=True, normalize_embeddings=True)
        return embedding

    def _compute_quality_score(
        self,
        reranker_score: float,
        rouge_l_score: float,
    ) -> float:
        """Compute combined quality score.
        
        Args:
            reranker_score: Score from reranking (0.0-1.0)
            rouge_l_score: ROUGE-L F-measure (0.0-1.0)
            
        Returns:
            Combined quality score (0.0-1.0)
        """
        #quality = self.reranker_quality_weight * reranker_score + self.rouge_quality_weight * rouge_l_score
        quality = reranker_score
        return float(np.clip(quality, 0.0, 1.0))

    def add(
        self,
        query: str,
        answer: str,
        reranker_score: float,
        answer_f1: float = 0.0,
        rouge_l_score: float = 0.0,
        context: str = "",
        evidence_supported: bool = False,
    ) -> None:
        """Add query-answer pair to memory with quality score.
        
        Args:
            query: Input query text
            answer: Generated answer
            reranker_score: Top document reranking score (0.0-1.0)
            answer_f1: Token F1 of the generated answer vs gold answer
            rouge_l_score: ROUGE-L match score if available (0.0-1.0)
        """
        # Only store high-quality, evidence-backed entries to avoid reinforcing noisy answers.
        if self.require_evidence_support and not evidence_supported:
            return

        # Compute quality score for stored metadata only.
        quality = self._compute_quality_score(reranker_score, rouge_l_score)
        
        if quality < self.min_quality:
            return

        # Compute embedding
        try:
            embedding = self._embed_query(query)
        except Exception as e:
            # Fallback: return without storing if embedding fails
            self._logger.warning("Embedding failed during add(): %s", e)
            return

        # Create memory item
        item = MemoryItemWithEmbedding(
            query=query,
            query_embedding=np.asarray(embedding, dtype=np.float32),
            answer=answer,
            context=context,
            quality_score=quality,
            evidence_supported=bool(evidence_supported),
            timestamp=self._timestamp_counter,
        )
        self._timestamp_counter += 1

        # Add to store
        self._items.append(item)

        # Enforce max size (FIFO eviction)
        if len(self._items) > self.max_items:
            self._items.pop(0)

        # Persist if enabled
        # Persist periodically instead of on every insert (avoids O(N) pickle cost per add)
        if self.persist_embeddings:
            self._dirty_count += 1
            if self._dirty_count >= self.flush_every:
                try:
                    self._save_to_disk()
                    self._dirty_count = 0
                except Exception as e:
                    self._logger.warning("Periodic save_to_disk failed: %s", e)

    def lookup_item(self, query: str) -> Tuple[Optional[MemoryItemWithEmbedding], float]:
        if not self._items:
            return None, 0.0

        try:
            query_embedding = self._embed_query(query)
        except Exception as e:
            self._logger.warning("Embedding failed during lookup(): %s", e)
            return None, 0.0

        # best_item = None
        # best_candidate_score = 0.0
        # best_similarity = 0.0

        # for item in self._items:
        #     similarity = self._cosine_similarity(query_embedding, item.query_embedding)
        #     quality = float(getattr(item, "quality_score", 0.0))
        #     effective_score = similarity * quality

        #     if (
        #         similarity >= self.similarity_threshold
        #         and quality >= self.min_quality
        #         and effective_score > best_candidate_score
        #     ):
        #         best_candidate_score = effective_score
        #         best_similarity = similarity
        #         best_item = item

        # return best_item, best_similarity


        best_item = None
        best_similarity = 0.0
        best_quality = 0.0

        SIMILARITY_MARGIN = 0.02

        for item in self._items:

            similarity = self._cosine_similarity(
                query_embedding,
                item.query_embedding,
            )

            quality = float(getattr(item, "quality_score", 0.0))

            if (
                similarity < self.similarity_threshold
                or quality < self.min_quality
            ):
                continue

            if similarity > best_similarity:
                best_similarity = similarity
                best_quality = quality
                best_item = item

            elif abs(similarity - best_similarity) <= SIMILARITY_MARGIN:
                if quality > best_quality:
                    best_similarity = similarity
                    best_quality = quality
                    best_item = item

        return best_item, best_similarity

    def lookup(self, query: str) -> Tuple[Optional[str], float]:
        """Backward-compatible answer lookup."""
        item, score = self.lookup_item(query)
        return (item.answer if item else None), score

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors.
        
        Args:
            a, b: Embedding vectors
            
        Returns:
            Cosine similarity (0.0-1.0)
        """
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)

        if norm_a < 1e-8 or norm_b < 1e-8:
            return 0.0

        return float(np.dot(a, b) / (norm_a * norm_b))

    def _save_to_disk(self) -> None:
        """Save embeddings to disk."""
        if not self.persist_embeddings:
            return

        filepath = os.path.join(self.persist_dir, "memory_store.pkl")
        try:
            with open(filepath, "wb") as f:
                pickle.dump(self._items, f)
        except Exception:
            pass

    def _load_from_disk(self) -> None:
        """Load embeddings from disk."""
        if not self.persist_embeddings:
            return

        filepath = os.path.join(self.persist_dir, "memory_store.pkl")
        if not os.path.exists(filepath):
            return

        try:
            with open(filepath, "rb") as f:
                self._items = pickle.load(f)
            for item in self._items:
                try:
                    item.query_embedding = np.asarray(item.query_embedding, dtype=np.float32)
                except Exception:
                    pass
            if self._items:
                self._timestamp_counter = max(
                    (item.timestamp for item in self._items),
                    default=0,
                ) + 1
        except Exception:
            pass

    def size(self) -> int:
        """Return current number of stored items."""
        return len(self._items)

    def size(self) -> int:
        """Return current number of stored items."""
        return len(self._items)

    def flush(self) -> None:
        """Force a save regardless of the dirty counter — call at end of a run."""
        if self.persist_embeddings and self._dirty_count > 0:
            try:
                self._save_to_disk()
                self._dirty_count = 0
            except Exception as e:
                self._logger.warning("flush() save_to_disk failed: %s", e)

    def clear(self) -> None:
        """Clear all stored items."""
        self._items = []
        self._timestamp_counter = 0
        if self.persist_embeddings:
            try:
                filepath = os.path.join(self.persist_dir, "memory_store.pkl")
                if os.path.exists(filepath):
                    os.remove(filepath)
            except Exception:
                pass
