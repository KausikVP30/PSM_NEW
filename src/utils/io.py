from __future__ import annotations

import json
import os
from dataclasses import asdict
from typing import Any, Dict, List

import pandas as pd

from src.data.schemas import PredictionRecord
from .paths import ensure_parent


def write_predictions_csv(records: List[PredictionRecord], path: str) -> None:
    ensure_parent(path)
    rows = []
    for r in records:
        item = asdict(r)
        item["references"] = " || ".join(item["references"])
        item["gold_answers"] = " || ".join(item["gold_answers"])
        item["expected_answer"] = r.gold_answers[0] if r.gold_answers else ""
        item["output_answer"] = r.prediction
        item["retrieved_context"] = item["references"]
        rows.append(item)
    pd.DataFrame(rows).to_csv(path, index=False)


def write_metrics_json(metrics: Dict[str, Any], path: str) -> None:
    ensure_parent(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)


def append_prediction_row(record: PredictionRecord, path: str, extra: Dict[str, Any] | None = None) -> None:
    """Append a single prediction row to CSV at `path`. Adds `extra` fields if provided.

    This is safe to call repeatedly during a run; creates parent dir and header if missing.
    """
    ensure_parent(path)
    item = asdict(record)
    item["references"] = " || ".join(item["references"])
    item["gold_answers"] = " || ".join(item["gold_answers"])
    item["expected_answer"] = record.gold_answers[0] if record.gold_answers else ""
    item["output_answer"] = record.prediction
    item["retrieved_context"] = item["references"]
    if extra:
        for k, v in extra.items():
            item[k] = v

    df = pd.DataFrame([item])
    # append mode; write header only if file doesn't exist
    write_header = not os.path.exists(path)
    df.to_csv(path, mode="a", header=write_header, index=False)
