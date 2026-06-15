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
- If you leave `data.dataset_paths` empty on Kaggle, the runner will try to auto-discover `nq.jsonl`, `triviaqa.jsonl`, `hotpotqa.jsonl`, and `qqp.jsonl` under `/kaggle/input`.
- Use `notebooks/00_kaggle_run.ipynb` as the ready-to-run Kaggle notebook entrypoint.

## Dataset Prep
- Prepare Quora Question Pairs locally with `python scripts/prepare_quora_qqp.py --output data/quora_qqp.jsonl`.
- The prep script downloads the GLUE `qqp` split via Hugging Face Datasets and stores a normalized JSONL artifact for later reuse.

## QQP Runs
- Smoke: `python run_experiment.py --config config_qqp_smoke.yaml`
- Subset: `python run_experiment.py --config config_qqp_subset.yaml`
- Full: `python run_experiment.py --config config_qqp_full.yaml`
- QQP is loaded as a duplicate-question classification variant: `question1` is the query, `question2` is the evidence doc, and the gold answer is `duplicate` or `not duplicate`.

## Multi-Dataset RAW vs Paraphrased
- Prepare the dataset artifacts first:
  - `python scripts/prepare_triviaqa_full.py --output data/triviaqa_full.jsonl`
  - `python scripts/prepare_nq.py --output data/nq_full.jsonl`
  - `python scripts/prepare_hotpotqa.py --output data/hotpotqa_full.jsonl`
- Then run the shared experiment config:
  - `python run_multi_dataset_experiment.py --config config_multi_dataset.yaml --dataset all --mode all`
- The runner keeps the same retrieval, generation, memory, and evaluation settings across datasets and modes.
