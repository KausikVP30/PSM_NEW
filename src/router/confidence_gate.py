from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RoutingDecision:
    route: str
    confidence: float
    reason: str


class ConfidenceGate:
    """Gate for memory-based decisions only.
    
    Architecture:
    - If memory lookup score >= threshold: MEMORY_HIT (use memory answer)
    - If memory lookup score < threshold or no match: RETRIEVAL_HIT (always retrieve)
    
    There is NO fallback route - we always have a valid path.
    """
    
    def __init__(self, threshold: float) -> None:
        self.threshold = threshold

    def decide(self, memory_score: float, memory_confidence: float | None = None) -> RoutingDecision:
        """Decide if memory hit is confident enough to use.
        
        Args:
            memory_score: Similarity score from memory lookup (0.0 to 1.0)
            
        Returns:
            RoutingDecision with route as 'memory_hit' or 'retrieval_hit'
        """
        confidence = memory_score if memory_confidence is None else min(memory_score, memory_confidence)
        if memory_score >= self.threshold and confidence >= self.threshold:
            return RoutingDecision(
                route="memory_hit",
                confidence=confidence,
                reason="memory score and confidence above threshold"
            )
        return RoutingDecision(
            route="retrieval_hit",
            confidence=confidence,
            reason="memory score or confidence below threshold - retrieve from corpus"
        )
