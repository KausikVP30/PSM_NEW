from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from src.data.schemas import PredictionRecord, QASample, RetrievedDoc
from src.evaluation.metrics import evaluate_predictions
from src.generation.generator import Generator
from src.memory.memory_store import MemoryStore
from src.prompt.assembler import PromptAssembler
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.reranker import LightweightReranker
from src.router.confidence_gate import ConfidenceGate


@dataclass
class PipelineResult:
    records: List[PredictionRecord]
    metrics: Dict[str, float]


class RAGPipeline:
    def __init__(
        self,
        retriever: HybridRetriever,
        reranker: LightweightReranker,
        gate: ConfidenceGate,
        memory: MemoryStore,
        assembler: PromptAssembler,
        generator: Generator,
        retrieval_cfg: Dict[str, float],
        gating_cfg: Dict[str, float],
    ) -> None:
        self.retriever = retriever
        self.reranker = reranker
        self.gate = gate
        self.memory = memory
        self.assembler = assembler
        self.generator = generator
        self.retrieval_cfg = retrieval_cfg
        self.gating_cfg = gating_cfg

    def _expand_retrieval(self, question: str) -> List[RetrievedDoc]:
        expand_k = int(self.gating_cfg.get("fallback_expand_k", 40))
        fused = self.retriever.retrieve(
            question,
            top_k_sparse=expand_k,
            top_k_dense=expand_k,
            top_k_fused=expand_k,
        )
        reranked = self.reranker.rerank(question, fused, top_k=int(self.retrieval_cfg.get("top_k_final", 5)))
        return reranked

    def run(self, samples: List[QASample]) -> PipelineResult:
        records: List[PredictionRecord] = []

        for sample in samples:
            fused = self.retriever.retrieve(
                sample.question,
                top_k_sparse=int(self.retrieval_cfg.get("top_k_sparse", 30)),
                top_k_dense=int(self.retrieval_cfg.get("top_k_dense", 30)),
                top_k_fused=int(self.retrieval_cfg.get("top_k_fused", 20)),
            )
            reranked = self.reranker.rerank(
                sample.question,
                fused,
                top_k=int(self.retrieval_cfg.get("top_k_final", 5)),
            )

            memory_answer, memory_score = self.memory.lookup(sample.question)
            top_doc_score = reranked[0].score if reranked else 0.0
            decision = self.gate.decide(memory_score=memory_score, top_doc_score=top_doc_score)

            final_docs = reranked
            if decision.route == "fallback":
                final_docs = self._expand_retrieval(sample.question)

            prompt = self.assembler.assemble(sample.question, final_docs, memory_answer if decision.route == "memory_hit" else None)
            prediction = self.generator.generate(prompt, final_docs)

            self.memory.add(sample.question, prediction, decision.confidence)

            records.append(
                PredictionRecord(
                    sample_id=sample.sample_id,
                    question=sample.question,
                    prediction=prediction,
                    confidence=decision.confidence,
                    route=decision.route,
                    references=[d.text for d in final_docs],
                    gold_answers=sample.answers,
                )
            )

        metrics = evaluate_predictions(records)
        route_counts = {
            "memory_hit": sum(1 for r in records if r.route == "memory_hit"),
            "retrieval_hit": sum(1 for r in records if r.route == "retrieval_hit"),
            "fallback": sum(1 for r in records if r.route == "fallback"),
        }
        metrics.update({f"routes.{k}": float(v) for k, v in route_counts.items()})
        return PipelineResult(records=records, metrics=metrics)
