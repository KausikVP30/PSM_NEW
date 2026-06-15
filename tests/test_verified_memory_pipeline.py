from __future__ import annotations

from dataclasses import dataclass

from src.data.schemas import QASample, RetrievedDoc
from src.pipeline.rag_pipeline import RAGPipeline
from src.router.confidence_gate import ConfidenceGate


@dataclass
class FakeMemoryItem:
    answer: str
    context: str
    quality_score: float


class FakeMemory:
    def __init__(self, item: FakeMemoryItem | None, score: float) -> None:
        self.item = item
        self.score = score
        self.add_calls = 0

    def lookup_item(self, query: str):
        return self.item, self.score

    def add(self, *args, **kwargs) -> None:
        self.add_calls += 1

    def size(self) -> int:
        return 1 if self.item else 0


class FailingRetriever:
    def retrieve(self, *args, **kwargs):
        raise AssertionError("retrieval should be skipped on memory hit")


class FakeRetriever:
    def __init__(self) -> None:
        self.calls = 0

    def retrieve(self, *args, **kwargs):
        self.calls += 1
        return [RetrievedDoc(0, "Paris is the capital of France.", 1.0, "test")]


class FakeReranker:
    def rerank(self, query, docs, top_k):
        return docs[:top_k]


class FakeGenerator:
    def __init__(self) -> None:
        self.calls = []

    def generate(self, question, docs, memory_context=None):
        self.calls.append((question, docs, memory_context))
        return "Paris"


def _pipeline(memory, retriever):
    generator = FakeGenerator()
    pipe = RAGPipeline(
        retriever=retriever,
        reranker=FakeReranker(),
        gate=ConfidenceGate(threshold=0.85),
        memory=memory,
        assembler=None,
        generator=generator,
        retrieval_cfg={"top_k_final": 3},
        gating_cfg={"memory_write_f1_threshold": 0.8},
    )
    return pipe, generator


def test_memory_hit_skips_retrieval_and_uses_memory_context() -> None:
    memory = FakeMemory(FakeMemoryItem("Paris", "Paris is the capital of France.", 0.9), 0.9)
    pipe, generator = _pipeline(memory, FailingRetriever())

    result = pipe.run([QASample("1", "What is the capital of France?", ["Paris"], [], {})])

    assert result.records[0].route == "memory_hit"
    assert result.metrics["memory.skipped_retrieval"] == 1.0
    assert generator.calls[0][1] == []
    assert generator.calls[0][2] == "Paris is the capital of France."


def test_low_confidence_memory_uses_retrieval() -> None:
    retriever = FakeRetriever()
    pipe, generator = _pipeline(FakeMemory(None, 0.0), retriever)

    result = pipe.run([QASample("1", "What is the capital of France?", ["Paris"], [], {})])

    assert result.records[0].route == "retrieval_hit"
    assert retriever.calls == 1
    assert generator.calls[0][1]
