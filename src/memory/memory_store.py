from __future__ import annotations

from dataclasses import dataclass
from typing import List

from src.retrieval.utils import token_set


@dataclass
class MemoryItem:
    query: str
    answer: str
    confidence: float


class MemoryStore:
    def __init__(self, max_items: int = 500) -> None:
        self.max_items = max_items
        self._items: List[MemoryItem] = []

    def add(self, query: str, answer: str, confidence: float) -> None:
        self._items.append(MemoryItem(query=query, answer=answer, confidence=confidence))
        if len(self._items) > self.max_items:
            self._items.pop(0)

    def lookup(self, query: str) -> tuple[str | None, float]:
        if not self._items:
            return None, 0.0
        q = token_set(query)
        best_text = None
        best_score = 0.0
        for item in self._items:
            q2 = token_set(item.query)
            overlap = 0.0 if not q else len(q & q2) / max(1, len(q | q2))
            score = 0.7 * overlap + 0.3 * item.confidence
            if score > best_score:
                best_score = score
                best_text = item.answer
        return best_text, float(best_score)
