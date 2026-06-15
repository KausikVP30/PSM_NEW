#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


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


def _coalesce(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _iter_dataset(split: str):
    from datasets import load_dataset

    return load_dataset("glue", "qqp", split=split)


def build_rows(split: str, max_samples: int | None = None) -> Iterable[Dict[str, Any]]:
    ds = _iter_dataset(split)
    for index, example in enumerate(ds):
        if max_samples is not None and index >= max_samples:
            break

        question1 = _as_text(example.get("question1") or example.get("question_1") or example.get("sentence1"))
        question2 = _as_text(example.get("question2") or example.get("question_2") or example.get("sentence2"))
        label = _normalize_label(example.get("label"))
        sample_id = str(_coalesce(example.get("idx"), example.get("id"), f"qqp-{split}-{index}"))

        row: Dict[str, Any] = {
            "pair_id": sample_id,
            "question1": question1,
            "question2": question2,
            "metadata": {"dataset": "quora_qqp", "split": split},
        }
        if label is not None:
            row["label"] = label
        yield row


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare Quora Question Pairs in JSONL format")
    parser.add_argument("--output", default="data/quora_qqp.jsonl", help="Output JSONL path")
    parser.add_argument("--split", default="train", help="HF GLUE QQP split to export")
    parser.add_argument("--max-samples", type=int, default=None, help="Optional cap for quick tests")
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with output_path.open("w", encoding="utf-8") as handle:
        for row in build_rows(args.split, args.max_samples):
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1

    print(f"Wrote {count} Quora QQP rows to {output_path}")


if __name__ == "__main__":
    main()
