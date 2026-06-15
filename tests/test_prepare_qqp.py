from __future__ import annotations

import json
import sys

import scripts.prepare_quora_qqp as prepare_quora_qqp


def test_prepare_quora_qqp_build_rows_with_sample_cap(monkeypatch, tmp_path) -> None:
    class FakeDataset:
        def __iter__(self):
            yield {"idx": 0, "question1": "Q1", "question2": "Q2", "label": 1}
            yield {"idx": 2, "question1": "Q3", "question2": "Q4", "label": 0}

    monkeypatch.setattr(prepare_quora_qqp, "_iter_dataset", lambda split: FakeDataset())

    output_path = tmp_path / "quora_qqp.jsonl"
    rows = list(prepare_quora_qqp.build_rows("train", max_samples=1))
    assert len(rows) == 1
    assert rows[0]["pair_id"] == "0"
    assert rows[0]["label"] == 1

    monkeypatch.setattr(
        sys,
        "argv",
        ["prepare_quora_qqp.py", "--output", str(output_path), "--split", "train", "--max-samples", "1"],
    )
    prepare_quora_qqp.main()

    with output_path.open("r", encoding="utf-8") as handle:
        written = [json.loads(line) for line in handle if line.strip()]

    assert len(written) == 1
    assert written[0]["question1"] == "Q1"
    assert written[0]["question2"] == "Q2"
    assert written[0]["label"] == 1
