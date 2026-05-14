from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from .schemas import QASample
from src.utils.runtime import resolve_dataset_paths


def _as_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v) for v in value if v is not None]
    return [str(value)]


def _load_json(path: str) -> Any:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _normalize_nq(rows: List[Dict[str, Any]]) -> List[QASample]:
    samples: List[QASample] = []
    for idx, row in enumerate(rows):
        q = row.get("question") or row.get("query") or ""
        answers = _as_list(row.get("answers") or row.get("answer"))
        docs = _as_list(row.get("documents") or row.get("contexts") or row.get("context"))
        sample_id = str(row.get("id") or f"nq-{idx}")
        samples.append(QASample(sample_id=sample_id, question=q, answers=answers, documents=docs, metadata={"dataset": "nq"}))
    return samples


def _normalize_triviaqa(rows: List[Dict[str, Any]]) -> List[QASample]:
    samples: List[QASample] = []
    for idx, row in enumerate(rows):
        q = row.get("question") or ""
        answer_obj = row.get("answer", {})
        answers = _as_list(answer_obj.get("aliases") if isinstance(answer_obj, dict) else answer_obj)
        docs = _as_list(row.get("documents") or row.get("entity_pages") or row.get("context"))
        sample_id = str(row.get("question_id") or row.get("id") or f"triviaqa-{idx}")
        samples.append(QASample(sample_id=sample_id, question=q, answers=answers, documents=docs, metadata={"dataset": "triviaqa"}))
    return samples


def _normalize_hotpot(rows: List[Dict[str, Any]]) -> List[QASample]:
    samples: List[QASample] = []
    for idx, row in enumerate(rows):
        q = row.get("question") or ""
        answers = _as_list(row.get("answer"))
        context = row.get("context", [])
        docs: List[str] = []
        if isinstance(context, list):
            for item in context:
                if isinstance(item, list) and len(item) == 2 and isinstance(item[1], list):
                    docs.append(" ".join(str(s) for s in item[1]))
        sample_id = str(row.get("_id") or row.get("id") or f"hotpotqa-{idx}")
        samples.append(QASample(sample_id=sample_id, question=q, answers=answers, documents=docs, metadata={"dataset": "hotpotqa"}))
    return samples


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


def _synthetic_dataset() -> List[QASample]:
    docs = [
        "Paris is the capital city of France and is known for the Eiffel Tower.",
        "Berlin is the capital city of Germany.",
        "Tokyo is the capital of Japan and one of the largest cities in the world.",
        "The Pacific Ocean is the largest and deepest ocean on Earth.",
    ]
    return [
        QASample("demo-1", "What is the capital of France?", ["Paris"], docs, {"dataset": "synthetic"}),
        QASample("demo-2", "Which ocean is the largest on Earth?", ["Pacific Ocean", "The Pacific Ocean"], docs, {"dataset": "synthetic"}),
    ]


def load_dataset(data_cfg: Dict[str, Any], mode: str) -> List[QASample]:
    if data_cfg.get("dataset_name") == "synthetic_demo":
        return _synthetic_dataset()

    paths = resolve_dataset_paths(data_cfg.get("dataset_paths", {}), dataset_name=str(data_cfg.get("dataset_name", "")))
    nq = _normalize_nq(_load_path(paths.get("nq", "")))
    triviaqa = _normalize_triviaqa(_load_path(paths.get("triviaqa", "")))
    hotpotqa = _normalize_hotpot(_load_path(paths.get("hotpotqa", "")))

    merged = [*nq, *triviaqa, *hotpotqa]
    if not merged:
        return _synthetic_dataset()

    if mode == "smoke":
        return merged[:20]
    if mode == "subset":
        return merged[:200]
    return merged
