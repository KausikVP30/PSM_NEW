from __future__ import annotations

import json
import html
import re
from pathlib import Path
from typing import Any, Dict, List

from .schemas import QASample
from src.retrieval.chunking import chunk_documents, clean_text
from src.utils.runtime import resolve_dataset_paths


def _as_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v) for v in value if v is not None]
    if isinstance(value, dict):
        aliases = value.get("aliases")
        if isinstance(aliases, list):
            return [str(v) for v in aliases if v is not None and str(v).strip()]
        text = value.get("text")
        if text is not None and str(text).strip():
            return [str(text).strip()]
        return []
    return [str(value)]


def _is_fake_support_doc(text: str) -> bool:
    return str(text or "").strip().lower().startswith("supporting evidence for ")


def _clean_documents(docs: List[str]) -> List[str]:
    cleaned: List[str] = []
    for doc in docs:
        text = clean_text(str(doc or ""))
        if not text or _is_fake_support_doc(text):
            continue
        if text not in cleaned:
            cleaned.append(text)
    return cleaned


def _build_context(docs: List[str], max_chunks: int = 3, max_chars: int = 1200) -> str:
    pieces: List[str] = []
    for doc in docs[:max_chunks]:
        text = clean_text(doc)
        if not text:
            continue
        pieces.append(text[:max_chars])
    return "\n\n".join(pieces)


def _load_json(path: str) -> Any:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_jsonl(path: str, limit: int | None = None) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if limit is not None and len(rows) >= limit:
                break
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
        docs = _clean_documents(_as_list(row.get("documents") or row.get("contexts") or row.get("context")))
        docs = chunk_documents(docs, min_tokens=200, max_tokens=350, overlap=60)
        sample_id = str(row.get("id") or f"nq-{idx}")
        samples.append(
            QASample(
                sample_id=sample_id,
                question=q,
                answers=answers,
                documents=docs,
                metadata={"dataset": "nq"},
                context=_build_context(docs),
            )
        )
    return samples


def _normalize_triviaqa(rows: List[Dict[str, Any]]) -> List[QASample]:
    samples: List[QASample] = []
    for idx, row in enumerate(rows):
        q = row.get("question") or ""
        answer_obj = row.get("answer", {})
        answers = _as_list(answer_obj.get("aliases") if isinstance(answer_obj, dict) else answer_obj)
        docs = _clean_documents(_as_list(row.get("documents") or row.get("entity_pages") or row.get("context")))
        docs = chunk_documents(docs, min_tokens=200, max_tokens=350, overlap=60)
        sample_id = str(row.get("question_id") or row.get("id") or f"triviaqa-{idx}")
        samples.append(
            QASample(
                sample_id=sample_id,
                question=q,
                answers=answers,
                documents=docs,
                metadata={"dataset": "triviaqa"},
                context=_build_context(docs),
            )
        )
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
        docs = _clean_documents(docs)
        docs = chunk_documents(docs, min_tokens=200, max_tokens=350, overlap=60)
        sample_id = str(row.get("_id") or row.get("id") or f"hotpotqa-{idx}")
        samples.append(
            QASample(
                sample_id=sample_id,
                question=q,
                answers=answers,
                documents=docs,
                metadata={"dataset": "hotpotqa"},
                context=_build_context(docs),
            )
        )
    return samples


def _normalize_qqp(rows: List[Dict[str, Any]]) -> List[QASample]:
    samples: List[QASample] = []
    for idx, row in enumerate(rows):
        pair_id = str(row.get("pair_id") or row.get("id") or row.get("idx") or f"qqp-{idx}")
        question1 = row.get("question1") or row.get("question_1") or row.get("sentence1") or ""
        question2 = row.get("question2") or row.get("question_2") or row.get("sentence2") or ""
        label_value = row.get("label")
        if label_value is None:
            label_value = row.get("is_duplicate")

        try:
            label = int(label_value) if label_value is not None else None
        except (TypeError, ValueError):
            label = None

        if label == 1:
            answers = ["duplicate"]
        elif label == 0:
            answers = ["not duplicate"]
        else:
            answers = []
        metadata = {"dataset": "qqp", "pair_id": pair_id}
        if label is not None:
            metadata["label"] = str(label)
        split = row.get("metadata", {}).get("split") if isinstance(row.get("metadata"), dict) else None
        if split:
            metadata["split"] = str(split)

        samples.append(
            QASample(
                sample_id=pair_id,
                question=str(question1).strip(),
                answers=answers,
                documents=[str(question2).strip()] if str(question2).strip() else [],
                metadata=metadata,
                context=str(question2).strip(),
            )
        )
    return samples


def _normalize_preprocessed(rows: List[Dict[str, Any]], dataset_name: str) -> List[QASample]:
    """Normalize preprocessed format with question, answer.aliases, documents fields directly."""
    samples: List[QASample] = []
    for idx, row in enumerate(rows):
        q = row.get("question") or row.get("query") or ""
        answers = _as_list(row.get("answer"))
        docs = _clean_documents(_as_list(row.get("documents") or row.get("contexts") or row.get("context")))
        docs = chunk_documents(docs, min_tokens=200, max_tokens=350, overlap=60)
        sample_id = str(row.get("question_id") or row.get("id") or f"{dataset_name}-{idx}")
        metadata = row.get("metadata", {})
        if "dataset" not in metadata:
            metadata["dataset"] = dataset_name
        samples.append(
            QASample(
                sample_id=sample_id,
                question=q,
                answers=answers,
                documents=docs,
                metadata=metadata,
                context=_build_context(docs),
            )
        )
    return samples


def _load_path(path: str, limit: int | None = None) -> List[Dict[str, Any]]:
    if not path:
        return []
    p = Path(path)
    if not p.exists():
        return []
    if p.suffix.lower() == ".jsonl":
        return _load_jsonl(path, limit=limit)
    data = _load_json(path)
    if isinstance(data, list):
        return data[:limit] if limit is not None else data
    if isinstance(data, dict):
        for key in ("data", "examples", "rows"):
            if key in data and isinstance(data[key], list):
                rows = data[key]
                return rows[:limit] if limit is not None else rows
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
    dataset_name = str(data_cfg.get("dataset_name", "") or "").strip().lower()

    loaders = {
        "nq": _normalize_nq,
        "triviaqa": _normalize_triviaqa,
        "hotpotqa": lambda rows: _normalize_preprocessed(rows, "hotpotqa"),
        "qqp": _normalize_qqp,
    }
    dataset_order = ["nq", "triviaqa", "hotpotqa", "qqp"]
    if dataset_name and dataset_name != "all":
        dataset_order = [dataset_name] if dataset_name in loaders else []

    mode_limit: int | None = None
    if mode == "smoke":
        mode_limit = 20
    elif mode == "subset":
        mode_limit = 200

    merged: List[QASample] = []
    for key in dataset_order:
        if mode_limit is not None and len(merged) >= mode_limit:
            break
        remaining = None if mode_limit is None else max(0, mode_limit - len(merged))
        loaded_rows = _load_path(paths.get(key, ""), limit=remaining)
        if not loaded_rows:
            continue
        merged.extend(loaders[key](loaded_rows))

    had_loaded_rows = bool(merged)
    if data_cfg.get("skip_no_evidence", True):
        merged = [sample for sample in merged if sample.documents]
    if not merged:
        requested_dataset = str(data_cfg.get("dataset_name", "") or "").strip().lower()
        has_explicit_paths = any(str(path).strip() for path in (data_cfg.get("dataset_paths", {}) or {}).values())
        if had_loaded_rows or requested_dataset in {"nq", "triviaqa", "hotpotqa", "qqp", "all"} or has_explicit_paths:
            return []
        return _synthetic_dataset()

    return merged
