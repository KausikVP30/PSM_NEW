from __future__ import annotations

import argparse
import time
import sys
import os
import random

import pandas as pd

# Ensure the current working directory and the local `src/` folder are on Python path.
# This helps when the notebook/kernel extracts files into /kaggle/working so imports like
# `from src.*` work without requiring a package install or notebook path tweaks.
cwd = os.getcwd()
if cwd not in sys.path:
    sys.path.insert(0, cwd)
src_path = os.path.join(cwd, "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)
os.environ["PYTHONPATH"] = cwd + (os.pathsep + os.environ.get("PYTHONPATH", "") if os.environ.get("PYTHONPATH") else "")

from src.config import load_settings
from src.data import load_dataset
from src.generation import OllamaGenerator
from src.logging_utils import get_logger
from src.memory.embedding_memory_store import EmbeddingMemoryStore
from src.pipeline import RAGPipeline
from src.prompt import PromptAssembler
from src.retrieval import BM25Retriever, DenseHNSWRetriever, HybridRetriever, LightweightReranker
from src.router import ConfidenceGate
from src.utils.io import write_metrics_json
from src.utils.paths import resolve_device
from src.utils.runtime import is_kaggle, resolve_output_path
from src.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run lightweight confidence-gated RAG experiment.")
    parser.add_argument("--config", default="config.yaml", help="Path to YAML config")
    parser.add_argument("--mode", default=None, choices=["smoke", "subset", "full"], help="Override run mode")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = load_settings(args.config)

    mode = args.mode or settings.run.get("mode", "smoke")
    seed = int(settings.run.get("seed", 42))
    set_seed(seed)

    logger = get_logger("rag_v1", settings.output.get("log_file", "outputs/logs/run.log"))
    logger.info("Starting experiment | mode=%s", mode)
    logger.info("Kaggle detected: %s", is_kaggle())

    device = resolve_device(str(settings.run.get("device", "auto")))
    logger.info("Resolved device: %s", device)

    # For subset mode, load the full merged dataset then slice according to config.
    if mode == "subset":
        samples = load_dataset(settings.data, mode="full")
        try:
            max_subset = int(settings.run.get("max_samples_subset", 200))
            if max_subset > 0:
                samples = samples[:max_subset]
        except Exception:
            pass
    else:
        samples = load_dataset(settings.data, mode=mode)
    logger.info("Loaded %d samples", len(samples))

    if samples:
        preview_count = min(5, len(samples))
        preview_samples = random.sample(samples, preview_count)
        for sample in preview_samples:
            context_snippet = (sample.context or " ".join(sample.documents[:2]))[:300].replace("\n", " ")
            logger.info(
                "sample_debug=%s",
                {
                    "sample_id": sample.sample_id,
                    "question": sample.question,
                    "answer": sample.answers[0] if sample.answers else "",
                    "context_snippet": context_snippet,
                },
            )

    # Build retrieval corpus from all available documents in the loaded samples.
    corpus = []
    for s in samples:
        corpus.extend(s.documents)
    deduped_corpus = []
    seen_corpus = set()
    for chunk in corpus:
        text = " ".join(str(chunk or "").split()).strip()
        if not text or text in seen_corpus:
            continue
        seen_corpus.add(text)
        deduped_corpus.append(text)
    corpus = deduped_corpus
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
        device=str(settings.run.get("device", "gpu")),
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
    gate = ConfidenceGate(threshold=float(gating_cfg.get("confidence_threshold", 0.85)))
    
    # Initialize semantic embedding-based memory store
    memory = EmbeddingMemoryStore(
        embedding_model=str(memory_cfg.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2")),
        similarity_threshold=float(memory_cfg.get("similarity_threshold", 0.75)),
        write_rouge_threshold=float(memory_cfg.get("write_rouge_threshold", 0.7)),
        write_reranker_threshold=float(memory_cfg.get("write_reranker_threshold", 0.7)),
        min_quality=float(memory_cfg.get("min_quality", 0.75)),
        require_evidence_support=bool(memory_cfg.get("require_evidence_support", True)),
        quality_metric_weights=memory_cfg.get("quality_metric_weights", {}),
        max_items=int(memory_cfg.get("max_items", 500)),
        persist_embeddings=bool(memory_cfg.get("persist_embeddings", True)),
        persist_dir=str(memory_cfg.get("persist_dir", "outputs/memory_embeddings")),
        device=str(memory_cfg.get("device", "auto")),
    )
    logger.info(
        "Initialized semantic memory with similarity_threshold=%.2f, device=%s",
        memory.similarity_threshold,
        memory.device,
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

    # Resolve output paths early so the pipeline can write incremental progress.
    predictions_csv = resolve_output_path(str(settings.output.get("predictions_csv", "")), "predictions/predictions.csv")
    metrics_json = resolve_output_path(str(settings.output.get("metrics_json", "")), "metrics/metrics.json")
    # simple progress path (overwritten during the run)
    progress_json = resolve_output_path("", "metrics/progress.json")

    for path in (predictions_csv, progress_json):
        try:
            if path and os.path.exists(path):
                os.remove(path)
        except Exception:
            pass

    start = time.time()
    result = pipeline.run(samples, predictions_path=predictions_csv, progress_path=progress_json)
    elapsed = time.time() - start

    metrics = dict(result.metrics)
    metrics["runtime_seconds"] = float(elapsed)
    metrics["num_samples"] = float(len(samples))

    # predictions_csv and metrics_json already resolved above

    # Ensure output directories exist so writes don't fail in fresh environments
    try:
        os.makedirs(os.path.dirname(predictions_csv), exist_ok=True)
    except Exception:
        pass
    try:
        os.makedirs(os.path.dirname(metrics_json), exist_ok=True)
    except Exception:
        pass

    latency_seconds = None
    total_latency_seconds = None
    try:
        if os.path.exists(predictions_csv):
            df = pd.read_csv(predictions_csv)
            if "latency_seconds" in df.columns and not df.empty:
                latency_seconds = float(df["latency_seconds"].mean())
                total_latency_seconds = float(df["latency_seconds"].sum())
    except Exception:
        latency_seconds = None
        total_latency_seconds = None

    if latency_seconds is not None:
        metrics["avg_latency_seconds"] = float(latency_seconds)
    if total_latency_seconds is not None:
        metrics["total_latency_seconds"] = float(total_latency_seconds)

    write_metrics_json(metrics, metrics_json)

    logger.info("Finished | runtime=%.2fs | metrics=%s", elapsed, metrics)
    print("Run complete")
    print(f"Predictions: {predictions_csv}")
    print(f"Metrics: {metrics_json}")


if __name__ == "__main__":
    main()
