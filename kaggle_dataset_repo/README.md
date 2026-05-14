# Lightweight V1 Confidence-Gated RAG (Kaggle-Ready)

This project implements a reliability-first RAG pipeline with:
- Hybrid retrieval: BM25 + HNSW dense search
- Lightweight reranking stage
- Confidence-gated fallback routing
- Deterministic pipeline execution
- CSV predictions and JSON metrics outputs

## Quick Start

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Configure settings in `config.yaml`.
3. Run smoke test mode:
   ```bash
   python run_experiment.py --mode smoke
   ```
4. Run validation subset:
   ```bash
   python run_experiment.py --mode subset
   ```

## Output Artifacts
- `outputs/predictions/predictions.csv`
- `outputs/metrics/metrics.json`
- `outputs/logs/run.log`

## Kaggle Notes
- Place datasets/models under `/kaggle/input/...` and configure paths in `config.yaml`.
- The pipeline is CPU-safe by default and enables GPU if available.
- If you leave `data.dataset_paths` empty on Kaggle, the runner will try to auto-discover `nq.jsonl`, `triviaqa.jsonl`, and `hotpotqa.jsonl` under `/kaggle/input`.
- Use `notebooks/00_kaggle_run.ipynb` as the ready-to-run Kaggle notebook entrypoint.
