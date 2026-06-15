#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from glob import glob
from pathlib import Path
from typing import Any, Dict, Iterable, List

from datasets import Dataset, concatenate_datasets


def _as_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if item is not None and str(item).strip()]
    text = str(value).strip()
    return [text] if text else []


def _extract_documents(example: Dict[str, Any]) -> List[str]:
    documents: List[str] = []

    # Support both the usual list-of-pairs format and the cached dict format.
    context = example.get("context", [])
    if isinstance(context, list):
        for item in context:
            if isinstance(item, list) and len(item) == 2:
                title, sentences = item
                if isinstance(sentences, list):
                    doc_text = " ".join(str(s) for s in sentences)
                    if doc_text.strip():
                        documents.append(doc_text.strip())
    elif isinstance(context, dict):
        titles = context.get("title", [])
        sentences = context.get("sentences", [])
        if isinstance(sentences, list):
            for idx, sentence_list in enumerate(sentences):
                if isinstance(sentence_list, list):
                    title = str(titles[idx]).strip() if isinstance(titles, list) and idx < len(titles) else ""
                    doc_text = " ".join(str(s) for s in sentence_list).strip()
                    if title and doc_text:
                        documents.append(f"{title}: {doc_text}")
                    elif doc_text:
                        documents.append(doc_text)

    return documents


def _iter_dataset(split: str):
    from datasets import load_dataset

    try:
        return load_dataset("hotpot_qa", "fullwiki", split=split)
    except Exception:
        cache_root = Path(os.environ.get("HF_DATASETS_CACHE", Path.home() / ".cache" / "huggingface" / "datasets"))
        dataset_root = cache_root / "hotpot_qa" / "fullwiki" / "0.0.0"
        patterns = []
        if split == "validation":
            patterns = [dataset_root / "*" / "hotpot_qa-validation.arrow"]
        elif split == "test":
            patterns = [dataset_root / "*" / "hotpot_qa-test.arrow"]
        else:
            patterns = [dataset_root / "*" / "hotpot_qa-train-*.arrow"]

        shard_paths: List[str] = []
        for pattern in patterns:
            shard_paths.extend(glob(str(pattern)))
        shard_paths = sorted(shard_paths)
        if not shard_paths:
            raise
        datasets = [Dataset.from_file(path) for path in shard_paths]
        return concatenate_datasets(datasets)


def build_rows(split: str, max_samples: int | None = None) -> Iterable[Dict[str, Any]]:
    ds = _iter_dataset(split)
    for index, example in enumerate(ds):
        if max_samples is not None and index >= max_samples:
            break

        question = str(example.get("question", "")).strip()
        if not question:
            continue

        answer = example.get("answer", "")
        answers = _as_list(answer)
        if not answers:
            continue

        documents = _extract_documents(example)
        if not documents:
            continue

        # Deduplicate documents
        seen = set()
        unique_docs = []
        for doc in documents:
            if doc not in seen:
                seen.add(doc)
                unique_docs.append(doc)

        yield {
            "question_id": str(example.get("_id") or example.get("id") or f"hotpotqa-{index}"),
            "question": question,
            "answer": {"aliases": answers},
            "documents": unique_docs,
            "metadata": {"dataset": "hotpotqa", "split": split},
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare HotpotQA in project JSONL format")
    parser.add_argument("--output", default="data/hotpotqa_full.jsonl", help="Output JSONL path")
    parser.add_argument("--split", default="validation", help="HF HotpotQA split to export (validation/test)")
    parser.add_argument("--max-samples", type=int, default=None, help="Optional cap for quick tests")
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with output_path.open("w", encoding="utf-8") as handle:
        for row in build_rows(args.split, args.max_samples):
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1

    print(f"Wrote {count} HotpotQA rows to {output_path}")


if __name__ == "__main__":
    main()
