#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from glob import glob
from pathlib import Path
from typing import Any, Dict, Iterable, List

from datasets import Dataset


def _as_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if item is not None and str(item).strip()]
    if isinstance(value, dict):
        text = value.get("text")
        return [str(text)] if text else []
    text = str(value).strip()
    return [text] if text else []


def _extract_span(tokens: List[str], start: int, end: int) -> str:
    if start < 0 or end < 0 or start >= end or end > len(tokens):
        return ""
    return " ".join(str(t) for t in tokens[start:end]).strip()


def _extract_documents(example: Dict[str, Any]) -> List[str]:
    documents: List[str] = []

    tokens_obj = example.get("document", {}).get("tokens") if isinstance(example.get("document"), dict) else example.get("tokens")
    tokens = tokens_obj.get("token", []) if isinstance(tokens_obj, dict) else tokens_obj
    if not isinstance(tokens, list):
        tokens = []

    # Long answer candidates
    long_answers = example.get("long_answer_candidates", [])
    if isinstance(long_answers, dict):
        starts = long_answers.get("start_token", [])
        ends = long_answers.get("end_token", [])
        for start_token, end_token in zip(starts, ends):
            doc_text = _extract_span(tokens, int(start_token), int(end_token))
            if doc_text:
                documents.append(doc_text)
    else:
        for la in long_answers:
            if isinstance(la, dict):
                start_token = int(la.get("start_token", 0))
                end_token = int(la.get("end_token", 0))
                doc_text = _extract_span(tokens, start_token, end_token)
                if doc_text:
                    documents.append(doc_text)

    # Annotation spans
    annotations = example.get("annotations", [])
    if isinstance(annotations, dict):
        long_answers = annotations.get("long_answer", [])
        short_answers = annotations.get("short_answers", [])
        for long_answer in long_answers:
            if isinstance(long_answer, dict):
                start_token = int(long_answer.get("start_token", -1))
                end_token = int(long_answer.get("end_token", -1))
                doc_text = _extract_span(tokens, start_token, end_token)
                if doc_text:
                    documents.append(doc_text)
        for short_answer in short_answers:
            if isinstance(short_answer, dict):
                starts = short_answer.get("start_token", [])
                ends = short_answer.get("end_token", [])
                for start_token, end_token in zip(starts, ends):
                    doc_text = _extract_span(tokens, int(start_token), int(end_token))
                    if doc_text:
                        documents.append(doc_text)
    else:
        for ann in annotations:
            if isinstance(ann, dict):
                long_answer = ann.get("long_answer", {})
                if isinstance(long_answer, dict):
                    start_token = int(long_answer.get("start_token", -1))
                    end_token = int(long_answer.get("end_token", -1))
                    doc_text = _extract_span(tokens, start_token, end_token)
                    if doc_text:
                        documents.append(doc_text)

                short_answers = ann.get("short_answers", [])
                for sa in short_answers:
                    if isinstance(sa, dict):
                        start_token = int(sa.get("start_token", -1))
                        end_token = int(sa.get("end_token", -1))
                        doc_text = _extract_span(tokens, start_token, end_token)
                        if doc_text:
                            documents.append(doc_text)
    
    return documents


def _iter_dataset(split: str):
    from datasets import load_dataset

    try:
        return load_dataset("natural_questions", split=split)
    except Exception:
        cache_root = Path(os.environ.get("HF_DATASETS_CACHE", Path.home() / ".cache" / "huggingface" / "datasets"))
        pattern = cache_root / "natural_questions" / "default" / "0.0.0" / "*" / "natural_questions-*.arrow"
        shard_paths = sorted(glob(str(pattern)))
        if not shard_paths:
            raise

        def _cached_rows():
            for path in shard_paths:
                for row in Dataset.from_file(path):
                    yield row

        return _cached_rows()


def build_rows(split: str, max_samples: int | None = None) -> Iterable[Dict[str, Any]]:
    ds = _iter_dataset(split)
    for index, example in enumerate(ds):
        if max_samples is not None and index >= max_samples:
            break

        question_obj = example.get("question", "")
        if isinstance(question_obj, dict):
            question = str(question_obj.get("text", "")).strip()
        else:
            question = str(question_obj).strip()
        if not question:
            continue

        # Get answers from annotations
        answers: List[str] = []
        annotations = example.get("annotations", [])
        tokens_obj = example.get("document", {}).get("tokens") if isinstance(example.get("document"), dict) else example.get("tokens")
        tokens = tokens_obj.get("token", []) if isinstance(tokens_obj, dict) else tokens_obj
        if not isinstance(tokens, list):
            tokens = []

        if isinstance(annotations, dict):
            short_answers = annotations.get("short_answers", [])
            yes_no_answers = annotations.get("yes_no_answer", [])
            for short_answer, yes_no in zip(short_answers, yes_no_answers):
                if isinstance(short_answer, dict):
                    starts = short_answer.get("start_token", [])
                    ends = short_answer.get("end_token", [])
                    texts = short_answer.get("text", [])
                    for text in texts:
                        if str(text).strip():
                            answers.append(str(text).strip())
                    for start_token, end_token in zip(starts, ends):
                        ans_text = _extract_span(tokens, int(start_token), int(end_token))
                        if ans_text:
                            answers.append(ans_text)
                try:
                    yes_no_int = int(yes_no)
                except (TypeError, ValueError):
                    yes_no_int = -1
                if yes_no_int != -1:
                    answers.append("yes" if yes_no_int == 1 else "no")
        else:
            for ann in annotations:
                if isinstance(ann, dict):
                    short_answers = ann.get("short_answers", [])
                    for sa in short_answers:
                        if isinstance(sa, dict):
                            start_token = sa.get("start_token", -1)
                            end_token = sa.get("end_token", -1)
                            ans_text = _extract_span(tokens, int(start_token), int(end_token))
                            if ans_text:
                                answers.append(ans_text)
                    # Also check yes/no answers
                    yes_no = ann.get("yes_no_answer")
                    if yes_no is not None:
                        answers.append("yes" if yes_no else "no")

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
            "question_id": str(example.get("example_id") or example.get("id") or f"nq-{index}"),
            "question": question,
            "answer": {"aliases": answers},
            "documents": unique_docs,
            "metadata": {"dataset": "nq", "split": split},
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare Natural Questions in project JSONL format")
    parser.add_argument("--output", default="data/nq_full.jsonl", help="Output JSONL path")
    parser.add_argument("--split", default="validation", help="HF Natural Questions split to export (validation/test)")
    parser.add_argument("--max-samples", type=int, default=None, help="Optional cap for quick tests")
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with output_path.open("w", encoding="utf-8") as handle:
        for row in build_rows(args.split, args.max_samples):
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1

    print(f"Wrote {count} Natural Questions rows to {output_path}")


if __name__ == "__main__":
    main()
