from __future__ import annotations

import json

from src.data.dataset import load_dataset
from src.data.qqp import load_qqp_dataset


def test_qqp_loader_round_trips_jsonl(tmp_path) -> None:
    data_path = tmp_path / "quora_qqp.jsonl"
    rows = [
        {
            "pair_id": "pair-1",
            "question1": "How are you?",
            "question2": "How do you do?",
            "label": 1,
            "metadata": {"dataset": "quora_qqp", "split": "train"},
        },
        {
            "pair_id": "pair-2",
            "question1": "What is AI?",
            "question2": "What is artificial intelligence?",
            "label": 0,
            "metadata": {"dataset": "quora_qqp", "split": "train"},
        },
    ]
    with data_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")

    samples = load_qqp_dataset(str(data_path))

    assert len(samples) == 2
    assert samples[0].pair_id == "pair-1"
    assert samples[0].question1 == "How are you?"
    assert samples[0].question2 == "How do you do?"
    assert samples[0].label == 1
    assert samples[0].metadata["dataset"] == "quora_qqp"
    assert samples[0].metadata["split"] == "train"
    assert samples[1].label == 0


def test_qqp_loader_works_through_experiment_dataset_loader(tmp_path) -> None:
    data_path = tmp_path / "quora_qqp.jsonl"
    rows = [
        {
            "pair_id": "pair-1",
            "question1": "How are you?",
            "question2": "How do you do?",
            "label": 1,
            "metadata": {"dataset": "quora_qqp", "split": "train"},
        },
        {
            "pair_id": "pair-2",
            "question1": "What is AI?",
            "question2": "What is artificial intelligence?",
            "label": 0,
            "metadata": {"dataset": "quora_qqp", "split": "train"},
        },
    ]
    with data_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")

    samples = load_dataset(
        {
            "dataset_name": "qqp",
            "skip_no_evidence": True,
            "dataset_paths": {"qqp": str(data_path)},
        },
        mode="full",
    )

    assert len(samples) == 2
    assert samples[0].question == "How are you?"
    assert samples[0].documents == ["How do you do?"]
    assert samples[0].answers == ["duplicate"]
    assert samples[1].answers == ["not duplicate"]
