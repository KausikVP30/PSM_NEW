from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List
import logging
import time
import json
import os
import re
import string

from src.data.schemas import PredictionRecord, QASample, RetrievedDoc
from src.evaluation.metrics import evaluate_predictions, exact_match, rouge_l, token_f1
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
        generator: Any,
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

    @staticmethod
    def _contains_gold(text: str, gold_answers: List[str]) -> bool:
        def normalize(value: str) -> str:
            value = (value or "").lower()
            value = "".join(ch if ch not in string.punctuation else " " for ch in value)
            value = re.sub(r"\b(a|an|the)\b", " ", value)
            return " ".join(value.split())

        haystack = normalize(text)
        if not haystack:
            return False
        return any(normalize(g) in haystack for g in gold_answers if normalize(g))

    def _retrieval_recall_flags(self, docs: List[RetrievedDoc], answers: List[str]) -> Dict[str, float]:
        flags: Dict[str, float] = {}
        for k in (1, 3, 5, 10):
            joined = " ".join(d.text for d in docs[:k])
            flags[f"retrieval.recall@{k}"] = 1.0 if self._contains_gold(joined, answers) else 0.0
        return flags

    def _log_retrieved_docs(self, sample: QASample, docs: List[RetrievedDoc], answers: List[str], memory_score: float, memory_confidence: float) -> None:
        if not self._logger.isEnabledFor(logging.INFO):
            return

        top_docs = []
        for doc in docs[:3]:
            snippet = (doc.text or "")[:240].replace("\n", " ")
            top_docs.append(
                {
                    "doc_id": doc.doc_id,
                    "score": round(float(doc.score), 4),
                    "source": doc.source,
                    "snippet": snippet,
                }
            )

        self._logger.info(
            "retrieval_debug=%s",
            {
                "sample_id": sample.sample_id,
                "question": sample.question,
                "gold_answer": answers[0] if answers else "",
                "memory_score": round(float(memory_score), 4),
                "memory_confidence": round(float(memory_confidence), 4),
                "gold_in_retrieved_top10": self._contains_gold(" ".join(d.text for d in docs[:10]), answers),
                "top_docs": top_docs,
            },
        )

    def run(self, samples: List[QASample], predictions_path: str | None = None, progress_path: str | None = None) -> PipelineResult:
        records: List[PredictionRecord] = []
        memory_scores: List[float] = []
        memory_hit_correct = 0
        memory_hit_count = 0
        retrieval_recall_sums = {f"retrieval.recall@{k}": 0.0 for k in (1, 3, 5, 10)}
        gold_in_context_count = 0

        total = len(samples)

        # prepare progress file if requested
        if progress_path:
            try:
                os.makedirs(os.path.dirname(progress_path), exist_ok=True)
            except Exception:
                pass
            progress = {
                "total_samples": total,
                "processed": 0,
                "last_sample": None,
                "avg_latency_seconds": None,
            }
            try:
                with open(progress_path, "w", encoding="utf-8") as pf:
                    json.dump(progress, pf)
            except Exception:
                pass

        for idx, sample in enumerate(samples, start=1):
            sample_start = time.time()

            # Step 1: Check verified semantic memory first. A memory hit skips retrieval.
            memory_item, memory_score = self.memory.lookup_item(sample.question)
            memory_confidence = float(getattr(memory_item, "quality_score", 0.0)) if memory_item else 0.0
            decision = self.gate.decide(memory_score=memory_score, memory_confidence=memory_confidence)
            memory_scores.append(float(memory_score))

            top_doc_score = 0.0
            memory_context = None
            used_memory = False
            used_retrieval = False
            # If memory confident enough, skip retrieval and use memory as context
            if memory_item and decision.route == "memory_hit":
                stored_context = getattr(memory_item, "context", "") or getattr(memory_item, "answer", "")
                memory_context = stored_context.strip()
                final_docs = []
                used_memory = True
                memory_hit_count += 1
            else:
                decision = self.gate.decide(memory_score=0.0, memory_confidence=0.0)
                # Always perform retrieval for the non-skip paths
                used_retrieval = True
                fused = self.retriever.retrieve(
                    sample.question,
                    top_k_sparse=int(self.retrieval_cfg.get("top_k_sparse", 30)),
                    top_k_dense=int(self.retrieval_cfg.get("top_k_dense", 30)),
                    top_k_fused=int(self.retrieval_cfg.get("top_k_fused", 20)),
                )
                reranked = self.reranker.rerank(
                    sample.question,
                    fused,
                    top_k=max(5, int(self.retrieval_cfg.get("top_k_final", 5))),
                )
                final_docs = reranked
                top_doc_score = (
                    reranked[0].score
                    if reranked
                    else (
                        fused[0].score
                        if fused
                        else 0
                    )
                )
                recall_flags = self._retrieval_recall_flags(final_docs, sample.answers)
                for key, value in recall_flags.items():
                    retrieval_recall_sums[key] += value
                if idx <= 3:
                    self._log_retrieved_docs(sample, reranked or fused, sample.answers, memory_score, memory_confidence)

            prediction = self.generator.generate(sample.question, final_docs[:5], memory_context=memory_context)
            if prediction is None:
                prediction = ""

            # Step 3: Compute quality metrics for memory storage
            # Compute ROUGE-L score for this prediction
            rouge_score = rouge_l(prediction, sample.answers)
            f1_score = token_f1(prediction, sample.answers)
            em_score = exact_match(prediction, sample.answers)
            context_for_storage = memory_context if used_memory else "\n\n".join(
                d.text
                for d in final_docs[:3]
            )
            evidence_supported = self._contains_gold(context_for_storage, sample.answers)
            if evidence_supported:
                gold_in_context_count += 1
            if used_memory and em_score:
                memory_hit_correct += 1
            
            # Store in memory with quality score
            # Use reranker/top_doc_score only when retrieval was used to produce the answer
            reranker_score_for_storage = top_doc_score if used_retrieval else 0.0
            self.memory.add(
                sample.question,
                prediction,
                reranker_score=reranker_score_for_storage,
                answer_f1=f1_score,
                rouge_l_score=rouge_score,
                context=context_for_storage,
                evidence_supported=evidence_supported and (em_score > 0.0 or f1_score >= float(self.gating_cfg.get("memory_write_f1_threshold", 0.8))),
            )

            rec = PredictionRecord(
                sample_id=sample.sample_id,
                question=sample.question,
                prediction=prediction,
                confidence=decision.confidence,
                route=decision.route,
                references=[memory_context] if used_memory and memory_context else [d.text for d in final_docs],
                gold_answers=sample.answers,
            )
            records.append(rec)

            self._logger.info(
                "memory_debug=%s",
                {
                    "sample_id": sample.sample_id,
                    "memory_score": float(memory_score),
                    "memory_confidence": float(memory_confidence),
                    "used_memory": used_memory,
                    "used_retrieval": used_retrieval,
                    "route": decision.route,
                    "memory_hit": bool(memory_item and decision.route == "memory_hit"),
                },
            )

            # per-sample latency and retrieval summary
            sample_latency = time.time() - sample_start
            retrieved_summary = " || ".join(f"{d.doc_id}:{d.score:.4f}:{d.source}" for d in final_docs) if final_docs else "memory_hit"

            # append to incremental predictions CSV if requested
            if predictions_path:
                try:
                    from src.utils.io import append_prediction_row

                    extra = {
                        "latency_seconds": float(sample_latency),
                        "retrieved_summary": retrieved_summary,
                        "top_doc_score": float(top_doc_score),
                        "memory_size": self.memory.size(),
                        "memory_score": float(memory_score),
                        "memory_quality": float(getattr(memory_item, "quality_score", 0.0)) if memory_item else 0.0,
                        "used_memory": used_memory,
                        "used_retrieval": used_retrieval,
                        "evidence_supported": evidence_supported,
                    }
                    append_prediction_row(rec, predictions_path, extra=extra)
                except Exception:
                    pass

            # update simple progress JSON
            if progress_path:
                try:
                    progress["processed"] = idx
                    progress["last_sample"] = rec.sample_id
                    prev_avg = progress.get("avg_latency_seconds") or 0.0
                    if prev_avg:
                        progress["avg_latency_seconds"] = (prev_avg * (idx - 1) + sample_latency) / idx
                    else:
                        progress["avg_latency_seconds"] = sample_latency
                    with open(progress_path, "w", encoding="utf-8") as pf:
                        json.dump(progress, pf)
                except Exception:
                    pass

        metrics = evaluate_predictions(records)
        route_counts = {
            "memory_hit": sum(1 for r in records if r.route == "memory_hit"),
            "retrieval_hit": sum(1 for r in records if r.route == "retrieval_hit"),
        }
        metrics.update({f"routes.{k}": float(v) for k, v in route_counts.items()})
        retrieval_count = max(1, route_counts["retrieval_hit"])
        for key, value in retrieval_recall_sums.items():
            metrics[key] = float(value / retrieval_count)
        metrics.update({
            "memory.size": float(self.memory.size()),
            "memory.hit_rate": float(route_counts["memory_hit"]) / len(records) if records else 0.0,
            "memory.precision": float(memory_hit_correct / memory_hit_count) if memory_hit_count else 0.0,
            "memory.skipped_retrieval": float(memory_hit_count),
            "memory.avg_score": float(sum(memory_scores) / len(memory_scores)) if memory_scores else 0.0,
            "memory.max_score": float(max(memory_scores)) if memory_scores else 0.0,
            "context.gold_in_context_rate": float(gold_in_context_count / len(records)) if records else 0.0,
        })
        
        return PipelineResult(records=records, metrics=metrics)
