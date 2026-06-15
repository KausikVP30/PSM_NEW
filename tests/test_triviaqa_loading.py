from __future__ import annotations

import json

from src.data.dataset import load_dataset


def test_triviaqa_loader_filters_fake_support_docs(tmp_path) -> None:
    data_path = tmp_path / "triviaqa.jsonl"
    rows = [
        {
            "question_id": "fake",
            "question": "Who?",
            "answer": {"aliases": ["Someone"]},
            "documents": ["Supporting evidence for Someone"],
        },
        {
            "question_id": "real",
            "question": "Capital?",
            "answer": {"aliases": ["Paris"]},
            "documents": ["Paris is the capital of France."],
        },
    ]
    with data_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")

    samples = load_dataset(
        {
            "dataset_name": "triviaqa",
            "dataset_paths": {"triviaqa": str(data_path)},
        },
        mode="full",
    )

    assert len(samples) == 1
    assert samples[0].sample_id == "real"
    assert samples[0].documents == ["Paris is the capital of France."]
