from src.router.confidence_gate import ConfidenceGate


def test_confidence_gate_routes() -> None:
    gate = ConfidenceGate(threshold=0.5)
    assert gate.decide(memory_score=0.7, top_doc_score=0.3).route == "memory_hit"
    assert gate.decide(memory_score=0.1, top_doc_score=0.6).route == "retrieval_hit"
    assert gate.decide(memory_score=0.1, top_doc_score=0.2).route == "fallback"
