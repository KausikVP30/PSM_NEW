from __future__ import annotations

import argparse
import time
import sys
import os

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
from src.generation import Generator
from src.logging_utils import get_logger
from src.memory import MemoryStore
from src.pipeline import RAGPipeline
from src.prompt import PromptAssembler
from src.retrieval import BM25Retriever, DenseHNSWRetriever, HybridRetriever, LightweightReranker
from src.router import ConfidenceGate
from src.utils.io import write_metrics_json, write_predictions_csv
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

    samples = load_dataset(settings.data, mode=mode)
    logger.info("Loaded %d samples", len(samples))

    # Build retrieval corpus from all available documents in the loaded samples.
    corpus = []
    for s in samples:
        corpus.extend(s.documents)
    corpus = [c for c in corpus if c and c.strip()]
    if not corpus:
        raise RuntimeError("No documents found in dataset. Check dataset paths/config.")

    retrieval_cfg = settings.retrieval
    rerank_cfg = settings.reranking
    gating_cfg = settings.gating

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
    )
    gate = ConfidenceGate(threshold=float(gating_cfg.get("confidence_threshold", 0.45)))
    memory = MemoryStore(max_items=500)
    assembler = PromptAssembler(max_chars=4000)
    generator = Generator(
        enabled=bool(settings.generation.get("enabled", False)),
        model_name=str(settings.generation.get("model_name", "google/flan-t5-small")),
        max_new_tokens=int(settings.generation.get("max_new_tokens", 64)),
        temperature=float(settings.generation.get("temperature", 0.0)),
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

    start = time.time()
    result = pipeline.run(samples)
    elapsed = time.time() - start

    metrics = dict(result.metrics)
    metrics["runtime_seconds"] = float(elapsed)
    metrics["num_samples"] = float(len(samples))

    predictions_csv = resolve_output_path(str(settings.output.get("predictions_csv", "")), "predictions/predictions.csv")
    metrics_json = resolve_output_path(str(settings.output.get("metrics_json", "")), "metrics/metrics.json")

    # Ensure output directories exist so writes don't fail in fresh environments
    try:
        os.makedirs(os.path.dirname(predictions_csv), exist_ok=True)
    except Exception:
        pass
    try:
        os.makedirs(os.path.dirname(metrics_json), exist_ok=True)
    except Exception:
        pass

    write_predictions_csv(result.records, predictions_csv)
    write_metrics_json(metrics, metrics_json)

    logger.info("Finished | runtime=%.2fs | metrics=%s", elapsed, metrics)
    print("Run complete")
    print(f"Predictions: {predictions_csv}")
    print(f"Metrics: {metrics_json}")


if __name__ == "__main__":
    main()
