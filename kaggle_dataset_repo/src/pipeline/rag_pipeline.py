from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List

from src.data.schemas import PredictionRecord, QASample, RetrievedDoc
from src.evaluation.metrics import evaluate_predictions, rouge_l
from src.generation.generator import Generator
from src.memory.embedding_memory_store import EmbeddingMemoryStore
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
        memory: EmbeddingMemoryStore,
        assembler: PromptAssembler,
        generator: Generator,
        retrieval_cfg: Dict[str, float],
        gating_cfg: Dict[str, float],
    ) -> None:
        self._logger = logging.getLogger(__name__)
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
        memory_scores: List[float] = []

        for sample in samples:
            # Step 1: Check memory first
            memory_answer, memory_score = self.memory.lookup(sample.question)
            decision = self.gate.decide(memory_score=memory_score)
            memory_scores.append(float(memory_score))

            # Step 2: Determine final docs and prediction based on memory decision
            memory_context = None
            used_memory = False
            used_retrieval = False
            if decision.route == "memory_hit" and memory_answer and memory_answer.strip():
                # Use memory as context only; generation still runs.
                memory_context = memory_answer.strip()
                final_docs = []
                used_memory = True
            else:
                used_retrieval = True
                # RETRIEVAL_HIT: Always retrieve if memory missed or confidence too low
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
                final_docs = reranked

            prediction = self.generator.generate(
                sample.question,
                final_docs,
                memory_context=memory_context,
            )

            if prediction is None:
                prediction = ""

            # Step 3: Store in memory for future hits
            # Compute reranker score (top doc if available)
            reranker_score = final_docs[0].score if final_docs else 0.0
            # Compute ROUGE-L if references available
            rouge_l_score = 0.0
            if sample.answers and prediction:
                rouge_l_score = rouge_l(prediction, sample.answers)
            self.memory.add(
                sample.question,
                prediction,
                reranker_score=reranker_score,
                rouge_l_score=rouge_l_score,
            )

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

            self._logger.info(
                "memory_debug=%s",
                {
                    "sample_id": sample.sample_id,
                    "memory_score": float(memory_score),
                    "used_memory": used_memory,
                    "used_retrieval": used_retrieval,
                    "route": decision.route,
                },
            )

        metrics = evaluate_predictions(records)
        route_counts = {
            "memory_hit": sum(1 for r in records if r.route == "memory_hit"),
            "retrieval_hit": sum(1 for r in records if r.route == "retrieval_hit"),
        }
        metrics.update({f"routes.{k}": float(v) for k, v in route_counts.items()})
        metrics.update({
            "memory.size": float(self.memory.size()),
            "memory.hit_rate": float(route_counts["memory_hit"]) / len(records) if records else 0.0,
            "memory.avg_score": float(sum(memory_scores) / len(memory_scores)) if memory_scores else 0.0,
            "memory.max_score": float(max(memory_scores)) if memory_scores else 0.0,
        })
        return PipelineResult(records=records, metrics=metrics)
