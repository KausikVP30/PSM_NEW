from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RoutingDecision:
    route: str
    confidence: float
    reason: str


class ConfidenceGate:
    def __init__(self, threshold: float) -> None:
        self.threshold = threshold

    def decide(self, memory_score: float, top_doc_score: float) -> RoutingDecision:
        confidence = max(memory_score, top_doc_score)
        if memory_score >= self.threshold:
            return RoutingDecision(route="memory_hit", confidence=confidence, reason="memory score above threshold")
        if top_doc_score >= self.threshold:
            return RoutingDecision(route="retrieval_hit", confidence=confidence, reason="retrieval score above threshold")
        return RoutingDecision(route="fallback", confidence=confidence, reason="confidence below threshold")
