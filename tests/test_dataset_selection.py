from __future__ import annotations

import json

from src.data.dataset import load_dataset


def test_dataset_loader_honors_dataset_name(tmp_path) -> None:
    triviaqa_path = tmp_path / "triviaqa.jsonl"
    nq_path = tmp_path / "nq.jsonl"
    hotpotqa_path = tmp_path / "hotpotqa.jsonl"

    triviaqa_row = {
        "question_id": "triviaqa-1",
        "question": "Who directed Titanic?",
        "answer": {"aliases": ["James Cameron"]},
        "documents": ["Titanic was directed by James Cameron."],
        "metadata": {"dataset": "triviaqa"},
    }
    nq_row = {
        "id": "nq-1",
        "question": "What is the capital of France?",
        "answers": ["Paris"],
        "documents": ["Paris is the capital of France."],
        "metadata": {"dataset": "nq"},
    }
    hotpot_row = {
        "_id": "hotpotqa-1",
        "question": "Who wrote Hamlet?",
        "answer": "William Shakespeare",
        "context": [["Hamlet", ["Hamlet was written by William Shakespeare."]]],
        "metadata": {"dataset": "hotpotqa"},
    }

    for path, row in (
        (triviaqa_path, triviaqa_row),
        (nq_path, nq_row),
        (hotpotqa_path, hotpot_row),
    ):
        with path.open("w", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")

    triviaqa_only = load_dataset(
        {
            "dataset_name": "triviaqa",
            "skip_no_evidence": True,
            "dataset_paths": {
                "triviaqa": str(triviaqa_path),
                "nq": str(nq_path),
                "hotpotqa": str(hotpotqa_path),
            },
        },
        mode="full",
    )

    all_datasets = load_dataset(
        {
            "dataset_name": "all",
            "skip_no_evidence": True,
            "dataset_paths": {
                "triviaqa": str(triviaqa_path),
                "nq": str(nq_path),
                "hotpotqa": str(hotpotqa_path),
            },
        },
        mode="full",
    )

    assert len(triviaqa_only) == 1
    assert triviaqa_only[0].metadata["dataset"] == "triviaqa"
    assert triviaqa_only[0].sample_id == "triviaqa-1"

    assert len(all_datasets) == 3
    assert {sample.metadata["dataset"] for sample in all_datasets} == {"triviaqa", "nq", "hotpotqa"}


def test_dataset_loader_returns_empty_for_missing_requested_dataset(tmp_path) -> None:
    empty_path = tmp_path / "empty.jsonl"
    empty_path.write_text("", encoding="utf-8")

    samples = load_dataset(
        {
            "dataset_name": "nq",
            "skip_no_evidence": True,
            "dataset_paths": {"nq": str(empty_path)},
        },
        mode="full",
    )

    assert samples == []
