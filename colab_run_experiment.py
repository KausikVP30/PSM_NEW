"""Colab helper to run the project's RAG experiment using TriviaQA.

Usage in Colab (example):
1. Upload or `git clone` this repo into `/content/PSM_NEW`.
2. In Colab runtime ensure you are in the repo root, then run:

    !python colab_run_experiment.py --model google/flan-t5-small --mode smoke

Options:
  --model     : Hugging Face model name or local path (defaults to google/flan-t5-small)
  --dataset   : Which dataset to prepare: triviaqa (default), nq, hotpotqa
  --mode      : smoke|subset|full (default: smoke)
  --no-install: skip pip installs (if you already installed deps)
  --max-samples: max examples to download from HF dataset (default: 200)

The script will:
 - (optionally) install needed Python packages in Colab
 - download TriviaQA via `datasets` and write a JSONL file compatible with the project's loader
 - create a lightweight YAML config pointing to the generated file and outputs under `/content/outputs`
 - set `MODEL_PATH_OVERRIDE` if `--model` points to a local path
 - invoke `run_experiment.py --config <config>`

"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import Any, Dict, List


def detect_colab() -> bool:
    try:
        import google.colab  # type: ignore

        return True
    except Exception:
        return False


def pip_install(packages: List[str]) -> None:
    subprocess.check_call(["python", "-m", "pip", "install", "--upgrade", *packages])


def prepare_triviaqa(out_path: Path, max_samples: int = 200) -> int:
    from datasets import load_dataset

    ds = load_dataset("trivia_qa", "rc", split=f"validation[:{max_samples}]")
    rows: List[Dict[str, Any]] = []
    
    for ex in ds:
        q = ex.get("question") or ""
        
        # Extract answers
        ans = ex.get("answer", {})
        if isinstance(ans, dict):
            answer_field = {"aliases": ans.get("aliases") or [ans.get("text", "")]}
        elif isinstance(ans, str):
            answer_field = {"aliases": [ans]}
        else:
            answer_field = {"aliases": []}

        # Extract documents from context.paragraphs (HF TriviaQA structure)
        documents: List[str] = []
        context = ex.get("context", {})
        if isinstance(context, dict):
            paragraphs = context.get("paragraphs", [])
            if isinstance(paragraphs, list):
                for para in paragraphs:
                    if isinstance(para, dict):
                        text = para.get("text", "").strip()
                        if text:
                            documents.append(text)
                    elif isinstance(para, str):
                        if para.strip():
                            documents.append(para.strip())
        
        # Fallback: if no docs found, use a placeholder
        if not documents and answer_field["aliases"]:
            documents = [f"Information related to {answer_field['aliases'][0]}"]

        out_obj = {
            "question": q,
            "answer": answer_field,
            "documents": documents,  # Use 'documents' field for compatibility with loader
            "question_id": ex.get("question_id", "") or str(hash(q)),
        }
        rows.append(out_obj)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return len(rows)


def write_config(path: Path, dataset_file: str, model_name: str, mode: str) -> None:
    import yaml

    cfg: Dict[str, Any] = {
        "run": {"seed": 42, "mode": mode, "device": "auto"},
        "data": {
            "dataset_name": "triviaqa",
            "dataset_paths": {"nq": "", "triviaqa": dataset_file, "hotpotqa": ""},
            "auto_discover_kaggle_inputs": False,
        },
        "retrieval": {
            "top_k_sparse": 30,
            "top_k_dense": 30,
            "top_k_fused": 20,
            "sparse_weight": 0.5,
            "dense_weight": 0.5,
            "embedding_model": "sentence-transformers/all-mpnet-base-v2",
        },
        "reranking": {"enabled": True, "overlap_weight": 0.7, "dense_weight": 0.3},
        "gating": {"confidence_threshold": 0.55, "fallback_expand_k": 40},
        "generation": {"enabled": True, "model_name": model_name, "max_new_tokens": 64, "temperature": 0.0},
        "evaluation": {"compute_rouge_l": True, "compute_bleu": True},
        "output": {"predictions_csv": "outputs/predictions/predictions.csv", "metrics_json": "outputs/metrics/metrics.json", "log_file": "outputs/logs/run.log"},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.1-8B-Instruct", help="HF model name or local path")
    parser.add_argument("--dataset", type=str, default="triviaqa", choices=["triviaqa", "nq", "hotpotqa"], help="Which dataset to prepare")
    parser.add_argument("--mode", type=str, default="smoke", choices=["smoke", "subset", "full"], help="Run mode")
    parser.add_argument("--no-install", action="store_true", help="Skip pip installs")
    parser.add_argument("--max-samples", type=int, default=200, help="Max examples to download from HF datasets")
    args = parser.parse_args()

    colab = detect_colab()
    if colab and not args.no_install:
        print("Installing runtime dependencies (this may take a few minutes)...")
        pip_install([
            "transformers[torch]",
            "datasets",
            "sentence-transformers",
            "hnswlib",
            "rank_bm25",
            "scikit-learn",
            "pyyaml",
        ])

    # Prepare dataset file
    data_dir = Path("/content/psm_data") if detect_colab() else Path("./data_colab")
    data_dir.mkdir(parents=True, exist_ok=True)
    dataset_file = data_dir / f"{args.dataset}.jsonl"
    if args.dataset == "triviaqa":
        print(f"Downloading TriviaQA (up to {args.max_samples}) and writing to {dataset_file}...")
        count = prepare_triviaqa(dataset_file, max_samples=args.max_samples)
        print(f"Wrote {count} examples")
    else:
        print("Only TriviaQA preparation implemented by helper; to use other datasets, provide JSON/JSONL manually.")

    # Write config
    cfg_path = Path("/content/colab_config.yaml") if detect_colab() else Path("colab_config.yaml")
    write_config(cfg_path, str(dataset_file), args.model, args.mode)
    print(f"Wrote config to {cfg_path}")

    # If model is a local path on Colab (for example on Drive), set MODEL_PATH_OVERRIDE
    model_path = args.model
    if Path(model_path).exists():
        os.environ["MODEL_PATH_OVERRIDE"] = str(Path(model_path).as_posix())
        print(f"Set MODEL_PATH_OVERRIDE={os.environ['MODEL_PATH_OVERRIDE']}")

    # Run the repo's run_experiment.py
    print("Running experiment...")
    cmd = ["python", "run_experiment.py", "--config", str(cfg_path), "--mode", args.mode]
    subprocess.check_call(cmd)


if __name__ == "__main__":
    main()
