#!/usr/bin/env python3
"""Unified runner for multi-dataset, multi-scale RAG evaluation."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
import random
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

cwd = os.getcwd()
if cwd not in sys.path:
    sys.path.insert(0, cwd)
src_path = os.path.join(cwd, "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)
os.environ["PYTHONPATH"] = cwd + (os.pathsep + os.environ.get("PYTHONPATH", "") if os.environ.get("PYTHONPATH") else "")

from src.config import load_settings
from src.data import load_dataset
from src.data.paraphrase import paraphrase_samples
from src.data.schemas import QASample
from src.generation import OllamaGenerator
from src.logging_utils import get_logger
from src.memory.embedding_memory_store import EmbeddingMemoryStore
from src.pipeline import RAGPipeline
from src.pipeline.rag_pipeline import PipelineResult
from src.prompt import PromptAssembler
from src.retrieval import BM25Retriever, DenseHNSWRetriever, HybridRetriever, LightweightReranker
from src.router import ConfidenceGate
from src.utils.paths import resolve_device
from src.utils.runtime import is_kaggle, resolve_output_path
from src.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Multi-dataset, multi-scale RAW vs paraphrased memory experiment."
    )
    parser.add_argument(
        "--config",
        default="config_multi_dataset.yaml",
        help="Path to base YAML config",
    )
    parser.add_argument(
        "--dataset",
        default="all",
        choices=["triviaqa", "nq", "hotpotqa", "all"],
        help="Dataset to run (default: all)",
    )
    parser.add_argument(
        "--mode",
        default="all",
        choices=["smoke", "subset", "full", "all"],
        help="Run mode (default: all)",
    )
    return parser.parse_args()


def _load_samples_for_dataset(data_cfg: Dict[str, Any], dataset: str, mode: str, max_subset: int) -> List[QASample]:
    """Load samples for a specific dataset and mode."""
    original_dataset_name = data_cfg.get("dataset_name")
    data_cfg["dataset_name"] = dataset
    try:
        if mode == "subset":
            samples = load_dataset(data_cfg, mode="full")
            if max_subset > 0:
                samples = samples[:max_subset]
            return samples
        return load_dataset(data_cfg, mode=mode)
    finally:
        data_cfg["dataset_name"] = original_dataset_name


def _finalize_metrics(
    result: PipelineResult,
    elapsed: float,
    n_samples: int,
    predictions_csv: str,
) -> Dict[str, float]:
    metrics = dict(result.metrics)
    metrics["runtime_seconds"] = float(elapsed)
    metrics["num_samples"] = float(n_samples)

    try:
        if os.path.exists(predictions_csv):
            df = pd.read_csv(predictions_csv)
            if "latency_seconds" in df.columns and not df.empty:
                metrics["avg_latency_seconds"] = float(df["latency_seconds"].mean())
                metrics["total_latency_seconds"] = float(df["latency_seconds"].sum())
    except Exception:
        pass

    return metrics


def _clear_path(path: str) -> None:
    try:
        if path and os.path.exists(path):
            os.remove(path)
    except Exception:
        pass


def _clear_memory_dir(persist_dir: str) -> None:
    if persist_dir and os.path.isdir(persist_dir):
        shutil.rmtree(persist_dir, ignore_errors=True)


def _create_pipeline(settings, corpus: List[str], dataset: str, persist_dir: str) -> RAGPipeline:
    """Create RAG pipeline with given corpus and dataset-specific memory persist_dir."""
    retrieval_cfg = settings.retrieval
    rerank_cfg = settings.reranking
    gating_cfg = settings.gating
    memory_cfg = settings.memory

    bm25 = BM25Retriever(corpus)
    dense = DenseHNSWRetriever(
        corpus,
        model_name=str(retrieval_cfg.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2")),
        m=int(retrieval_cfg.get("hnsw_m", 16)),
        ef_construction=int(retrieval_cfg.get("hnsw_ef_construction", 200)),
        ef_search=int(retrieval_cfg.get("hnsw_ef_search", 64)),
    )
    hybrid = HybridRetriever(
        bm25=bm25,
        dense=dense,
        sparse_weight=float(retrieval_cfg.get("sparse_weight", 0.5)),
        dense_weight=float(retrieval_cfg.get("dense_weight", 0.5)),
    )
    reranker = LightweightReranker(
        overlap_weight=float(rerank_cfg.get("overlap_weight", 0.7)),
        dense_weight=float(rerank_cfg.get("dense_weight", 0.3)),
        doc_score_min=float(retrieval_cfg.get("doc_score_min", 0.0)),
        doc_text_max_chars=int(retrieval_cfg.get("doc_text_max_chars", 300)),
        cross_encoder_model=str(rerank_cfg.get("cross_encoder_model", "")),
    )
    gate = ConfidenceGate(threshold=float(gating_cfg.get("confidence_threshold", 0.60)))

    memory = EmbeddingMemoryStore(
        embedding_model=str(memory_cfg.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2")),
        similarity_threshold=float(memory_cfg.get("similarity_threshold", 0.60)),
        write_rouge_threshold=float(memory_cfg.get("write_rouge_threshold", 0.50)),
        write_reranker_threshold=float(memory_cfg.get("write_reranker_threshold", 0.20)),
        min_quality=float(memory_cfg.get("min_quality", 0.50)),
        require_evidence_support=bool(memory_cfg.get("require_evidence_support", False)),
        quality_metric_weights=memory_cfg.get("quality_metric_weights", {}),
        max_items=int(memory_cfg.get("max_items", 500)),
        persist_embeddings=bool(memory_cfg.get("persist_embeddings", True)),
        persist_dir=persist_dir,
        device=str(memory_cfg.get("device", "auto")),
    )

    assembler = PromptAssembler(max_chars=4000)
    generation_cfg = settings.generation
    config_model_name = str(generation_cfg.get("model_name", "llama3")).strip()
    ollama_model_name = config_model_name.removeprefix("ollama:").strip() or "llama3"
    if not ollama_model_name.lower().startswith("llama3"):
        raise RuntimeError(f"This experiment path only supports Ollama Llama 3, got: {ollama_model_name}")
    ollama_endpoint = os.environ.get(
        "OLLAMA_ENDPOINT",
        str(generation_cfg.get("ollama_endpoint", "http://127.0.0.1:11435")),
    ).strip()
    generator = OllamaGenerator(
        ollama_endpoint=ollama_endpoint,
        model_name=ollama_model_name,
        max_tokens=int(generation_cfg.get("max_new_tokens", 64)),
        temperature=float(generation_cfg.get("temperature", 0.0)),
    )

    return RAGPipeline(
        retriever=hybrid,
        reranker=reranker,
        gate=gate,
        memory=memory,
        assembler=assembler,
        generator=generator,
        retrieval_cfg=retrieval_cfg,
        gating_cfg=gating_cfg,
    )


def main() -> None:
    args = parse_args()
    settings = load_settings(args.config)

    seed = int(settings.run.get("seed", 42))
    set_seed(seed)

    log_file = settings.output.get("log_file", "outputs/logs/multi_dataset.log")
    logger = get_logger("multi_dataset_experiment", log_file)
    logger.info("Starting multi-dataset RAW vs paraphrased experiment")
    logger.info("Kaggle detected: %s", is_kaggle())

    device = resolve_device(str(settings.run.get("device", "auto")))
    logger.info("Resolved device: %s", device)

    max_subset = int(settings.run.get("max_samples_subset", 200))
    max_smoke = int(settings.run.get("max_samples_smoke", 20))

    datasets = ["triviaqa", "nq", "hotpotqa"] if args.dataset == "all" else [args.dataset]
    modes = ["smoke", "subset", "full"] if args.mode == "all" else [args.mode]

    for dataset in datasets:
        for mode in modes:
            logger.info("=" * 60)
            logger.info("Running dataset=%s mode=%s", dataset, mode)
            logger.info("=" * 60)

            current_max = max_smoke if mode == "smoke" else max_subset
            raw_samples = _load_samples_for_dataset(settings.data, dataset, mode, current_max)
            logger.info("Loaded %d RAW samples for %s (%s)", len(raw_samples), dataset, mode)
            if not raw_samples:
                logger.warning("No samples loaded for %s (%s), skipping", dataset, mode)
                continue

            preview_count = min(5, len(raw_samples))
            for sample in random.sample(raw_samples, preview_count):
                context_snippet = (sample.context or " ".join(sample.documents[:2]))[:300].replace("\n", " ")
                logger.info(
                    "sample_debug=%s",
                    {
                        "dataset": dataset,
                        "mode": mode,
                        "sample_id": sample.sample_id,
                        "question": sample.question,
                        "answer": sample.answers[0] if sample.answers else "",
                        "context_snippet": context_snippet,
                    },
                )

            corpus: List[str] = []
            for sample in raw_samples:
                corpus.extend(sample.documents)
            deduped_corpus: List[str] = []
            seen_corpus: set[str] = set()
            for doc in corpus:
                text = " ".join(str(doc or "").split()).strip()
                if not text or text in seen_corpus:
                    continue
                seen_corpus.add(text)
                deduped_corpus.append(text)
            corpus = deduped_corpus
            if not corpus:
                logger.warning("No documents found for %s (%s), skipping", dataset, mode)
                continue

            persist_dir = f"outputs/memory_embeddings/{dataset}_{mode}"
            _clear_memory_dir(persist_dir)

            pipeline = _create_pipeline(settings, corpus, dataset, persist_dir)

            predictions_raw = resolve_output_path("", f"predictions/{dataset}_{mode}_raw.csv")
            predictions_paraphrased = resolve_output_path("", f"predictions/{dataset}_{mode}_paraphrased.csv")
            progress_raw = resolve_output_path("", f"metrics/progress_{dataset}_{mode}_raw.json")
            progress_paraphrased = resolve_output_path("", f"metrics/progress_{dataset}_{mode}_paraphrased.json")
            comparison_json = resolve_output_path("", f"metrics/{dataset}_{mode}_comparison.json")

            for path in (predictions_raw, predictions_paraphrased, progress_raw, progress_paraphrased):
                _clear_path(path)

            os.makedirs(os.path.dirname(predictions_raw), exist_ok=True)
            os.makedirs(os.path.dirname(comparison_json), exist_ok=True)

            # Phase 1: RAW — populate memory
            pipeline.memory.clear()
            logger.info("Phase 1: RAW run (%d samples)", len(raw_samples))
            raw_start = time.time()
            raw_result = pipeline.run(
                raw_samples,
                predictions_path=predictions_raw,
                progress_path=progress_raw,
            )
            raw_elapsed = time.time() - raw_start
            raw_metrics = _finalize_metrics(raw_result, raw_elapsed, len(raw_samples), predictions_raw)
            logger.info(
                "Phase 1 complete | memory.size=%.0f memory.hit_rate=%.3f runtime=%.2fs",
                raw_metrics.get("memory.size", 0),
                raw_metrics.get("memory.hit_rate", 0),
                raw_elapsed,
            )

            # Phase 2: PARAPHRASED — test semantic memory (same pipeline, same memory, same corpus)
            paraphrased_samples = paraphrase_samples(raw_samples)
            logger.info("Phase 2: PARAPHRASED run (%d samples)", len(paraphrased_samples))
            paraphrased_start = time.time()
            paraphrased_result = pipeline.run(
                paraphrased_samples,
                predictions_path=predictions_paraphrased,
                progress_path=progress_paraphrased,
            )
            paraphrased_elapsed = time.time() - paraphrased_start
            paraphrased_metrics = _finalize_metrics(
                paraphrased_result,
                paraphrased_elapsed,
                len(paraphrased_samples),
                predictions_paraphrased,
            )
            logger.info(
                "Phase 2 complete | memory.hit_rate=%.3f routes.memory_hit=%.0f runtime=%.2fs",
                paraphrased_metrics.get("memory.hit_rate", 0),
                paraphrased_metrics.get("routes.memory_hit", 0),
                paraphrased_elapsed,
            )

            comparison = {
                "raw_run": raw_metrics,
                "paraphrased_run": paraphrased_metrics,
            }
            with open(comparison_json, "w", encoding="utf-8") as f:
                json.dump(comparison, f, indent=2)

            logger.info("Comparison written to %s", comparison_json)
            print(f"Done: {dataset} {mode} -> {comparison_json}")

    print("All experiments complete")


if __name__ == "__main__":
    main()
