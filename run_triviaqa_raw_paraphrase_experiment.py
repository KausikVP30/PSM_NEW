#!/usr/bin/env python3
"""Two-phase TriviaQA experiment: RAW (memory populate) then PARAPHRASED (memory test)."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from typing import Any, Dict

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
        description="TriviaQA RAW vs paraphrased two-phase memory experiment."
    )
    parser.add_argument(
        "--config",
        default="config_triviaqa_paraphrase_experiment.yaml",
        help="Path to YAML config",
    )
    parser.add_argument(
        "--mode",
        default=None,
        choices=["smoke", "subset", "full"],
        help="Override run mode (default: subset from config)",
    )
    return parser.parse_args()


def _load_samples(data_cfg: Dict[str, Any], mode: str, max_subset: int) -> list[QASample]:
    if mode == "subset":
        samples = load_dataset(data_cfg, mode="full")
        if max_subset > 0:
            samples = samples[:max_subset]
        return samples
    return load_dataset(data_cfg, mode=mode)


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


def main() -> None:
    args = parse_args()
    settings = load_settings(args.config)

    mode = args.mode or settings.run.get("mode", "subset")
    seed = int(settings.run.get("seed", 42))
    set_seed(seed)

    log_file = settings.output.get("log_file", "outputs/logs/triviaqa_raw_paraphrase.log")
    logger = get_logger("triviaqa_raw_paraphrase", log_file)
    logger.info("Starting RAW vs paraphrased experiment | mode=%s", mode)
    logger.info("Kaggle detected: %s", is_kaggle())

    device = resolve_device(str(settings.run.get("device", "auto")))
    logger.info("Resolved device: %s", device)

    max_subset = int(settings.run.get("max_samples_subset", 200))
    raw_samples = _load_samples(settings.data, mode, max_subset)
    logger.info("Loaded %d RAW TriviaQA samples", len(raw_samples))
    if not raw_samples:
        raise RuntimeError("No TriviaQA samples loaded. Check dataset path and config.")

    corpus: list[str] = []
    for sample in raw_samples:
        corpus.extend(sample.documents)
    corpus = [doc for doc in corpus if doc and doc.strip()]
    if not corpus:
        raise RuntimeError("No documents found in dataset. Check dataset paths/config.")

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

    persist_dir = str(memory_cfg.get("persist_dir", "outputs/memory_embeddings_triviaqa_paraphrase"))
    _clear_memory_dir(persist_dir)

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
    logger.info(
        "Initialized memory | similarity_threshold=%.2f confidence_threshold=%.2f persist_dir=%s",
        memory.similarity_threshold,
        gate.threshold,
        persist_dir,
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

    pipeline = RAGPipeline(
        retriever=hybrid,
        reranker=reranker,
        gate=gate,
        memory=memory,
        assembler=assembler,
        generator=generator,
        retrieval_cfg=retrieval_cfg,
        gating_cfg=gating_cfg,
    )

    predictions_raw = resolve_output_path(
        "",
        "predictions/predictions_raw.csv",
    )
    predictions_paraphrased = resolve_output_path(
        "",
        "predictions/predictions_paraphrased.csv",
    )
    progress_raw = resolve_output_path("", "metrics/progress_raw.json")
    progress_paraphrased = resolve_output_path("", "metrics/progress_paraphrased.json")
    comparison_json = resolve_output_path(
        str(settings.output.get("metrics_json", "")),
        "metrics/raw_paraphrase_comparison.json",
    )

    for path in (predictions_raw, predictions_paraphrased, progress_raw, progress_paraphrased):
        _clear_path(path)

    os.makedirs(os.path.dirname(predictions_raw), exist_ok=True)
    os.makedirs(os.path.dirname(comparison_json), exist_ok=True)

    # Phase 1: RAW — populate memory
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
    print("RAW vs paraphrased experiment complete")
    print(f"RAW predictions: {predictions_raw}")
    print(f"Paraphrased predictions: {predictions_paraphrased}")
    print(f"Comparison metrics: {comparison_json}")


if __name__ == "__main__":
    main()
