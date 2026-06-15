#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List


def _as_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if item is not None and str(item).strip()]
    if isinstance(value, dict):
        aliases = value.get("aliases")
        if isinstance(aliases, list):
            return [str(item) for item in aliases if item is not None and str(item).strip()]
        text = value.get("text")
        return [str(text)] if text else []
    text = str(value).strip()
    return [text] if text else []


def _append_text(documents: List[str], *parts: Any) -> None:
    text = " ".join(str(part).strip() for part in parts if part is not None and str(part).strip())
    if text and text not in documents:
        documents.append(text)


def _extract_from_nested(value: Any, documents: List[str]) -> None:
    if isinstance(value, str):
        _append_text(documents, value)
        return
    if isinstance(value, list):
        for item in value:
            _extract_from_nested(item, documents)
        return
    if not isinstance(value, dict):
        return

    title = value.get("title") or value.get("filename") or value.get("doc_id") or ""
    for key in ("text", "content", "context", "search_context", "description"):
        if key in value:
            item = value[key]
            if isinstance(item, str):
                _append_text(documents, title, item)
            else:
                _extract_from_nested(item, documents)
    for key in ("paragraphs", "wiki_context", "results", "documents", "entity_pages", "search_results"):
        if key in value:
            _extract_from_nested(value[key], documents)


def _extract_documents(example: Dict[str, Any]) -> List[str]:
    documents: List[str] = []
    for key in ("entity_pages", "search_results", "context", "documents"):
        if key in example:
            _extract_from_nested(example[key], documents)

    return documents


def _iter_dataset(split: str):
    from datasets import load_dataset

    return load_dataset("trivia_qa", "rc", split=split)


def build_rows(split: str, max_samples: int | None = None) -> Iterable[Dict[str, Any]]:
    ds = _iter_dataset(split)
    for index, example in enumerate(ds):
        if max_samples is not None and index >= max_samples:
            break

        question = str(example.get("question", "")).strip()
        answer = example.get("answer", {})
        answers = _as_list(answer)
        if isinstance(answer, dict):
            answers = _as_list(answer.get("aliases") or answer.get("text") or answers)

        documents = _extract_documents(example)
        if not documents:
            continue

        yield {
            "question_id": str(example.get("question_id") or example.get("id") or f"triviaqa-{index}"),
            "question": question,
            "answer": {"aliases": answers},
            "documents": documents,
            "metadata": {"dataset": "triviaqa", "split": split},
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare TriviaQA in project JSONL format")
    parser.add_argument("--output", default="data/triviaqa_full.jsonl", help="Output JSONL path")
    parser.add_argument("--split", default="validation", help="HF TriviaQA split to export")
    parser.add_argument("--max-samples", type=int, default=None, help="Optional cap for quick tests")
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with output_path.open("w", encoding="utf-8") as handle:
        for row in build_rows(args.split, args.max_samples):
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1

    print(f"Wrote {count} TriviaQA rows to {output_path}")


if __name__ == "__main__":
    main()
