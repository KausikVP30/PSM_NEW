from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from .schemas import QQPSample


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _coalesce(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _normalize_label(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        label = int(value)
    except (TypeError, ValueError):
        return None
    if label < 0:
        return None
    return label


def _load_json(path: str) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _load_path(path: str) -> List[Dict[str, Any]]:
    if not path:
        return []
    p = Path(path)
    if not p.exists():
        return []
    if p.suffix.lower() == ".jsonl":
        return _load_jsonl(path)
    data = _load_json(path)
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("data", "examples", "rows"):
            if key in data and isinstance(data[key], list):
                return data[key]
    return []


def _normalize_row(row: Dict[str, Any], idx: int) -> QQPSample:
    question1 = _as_text(row.get("question1") or row.get("question_1") or row.get("sentence1") or row.get("text1"))
    question2 = _as_text(row.get("question2") or row.get("question_2") or row.get("sentence2") or row.get("text2"))
    label = _normalize_label(row.get("label"))
    if label is None:
        label = _normalize_label(row.get("is_duplicate"))

    sample_id = str(_coalesce(row.get("pair_id"), row.get("id"), row.get("idx"), f"qqp-{idx}"))
    metadata: Dict[str, str] = {}
    existing_metadata = row.get("metadata")
    if isinstance(existing_metadata, dict):
        metadata.update({str(key): str(value) for key, value in existing_metadata.items() if value is not None})
    split = _as_text(row.get("split") or row.get("subset") or metadata.get("split"))
    metadata["dataset"] = "quora_qqp"
    if split:
        metadata["split"] = split
    source = _as_text(row.get("source"))
    if source:
        metadata["source"] = source

    return QQPSample(
        pair_id=sample_id,
        question1=question1,
        question2=question2,
        label=label,
        metadata=metadata,
    )


def load_qqp_dataset(path: str) -> List[QQPSample]:
    rows = _load_path(path)
    return [_normalize_row(row, idx) for idx, row in enumerate(rows)]
