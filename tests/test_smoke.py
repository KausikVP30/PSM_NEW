from src.router.confidence_gate import ConfidenceGate


def test_confidence_gate_routes() -> None:
    gate = ConfidenceGate(threshold=0.5)
    assert gate.decide(memory_score=0.7).route == "memory_hit"
    assert gate.decide(memory_score=0.1).route == "retrieval_hit"
