from __future__ import annotations

import json
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
        rows.append(item)
    pd.DataFrame(rows).to_csv(path, index=False)


def write_metrics_json(metrics: Dict[str, Any], path: str) -> None:
    ensure_parent(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
