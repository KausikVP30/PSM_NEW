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
    - If similarity >= similarity_threshold AND quality confidence >= quality_threshold: MEMORY_HIT
    - Otherwise: RETRIEVAL_HIT (always retrieve)
    
    There is NO fallback route - we always have a valid path.
    """
    
    def __init__(self, similarity_threshold: float, quality_threshold: float) -> None:
        self.similarity_threshold = similarity_threshold
        self.quality_threshold = quality_threshold

    def decide(self, memory_score: float, memory_confidence: float | None = None) -> RoutingDecision:
        """Decide if memory hit is confident enough to use.
        
        Args:
            memory_score: Similarity score from memory lookup (0.0 to 1.0), i.e. tau_s check
            memory_confidence: Entry quality score (0.0 to 1.0), i.e. tau_c check
            
        Returns:
            RoutingDecision with route as 'memory_hit' or 'retrieval_hit'
        """
        confidence = memory_confidence if memory_confidence is not None else memory_score
        if memory_score >= self.similarity_threshold and confidence >= self.quality_threshold:
            return RoutingDecision(
                route="memory_hit",
                confidence=confidence,
                reason="similarity and quality both above threshold"
            )
        return RoutingDecision(
            route="retrieval_hit",
            confidence=confidence,
            reason="similarity or quality below threshold - retrieve from corpus"
        )
