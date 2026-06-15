from __future__ import annotations

from src.data.paraphrase import paraphrase_question, paraphrase_samples
from src.data.schemas import QASample


def test_who_directed_paraphrase() -> None:
    result = paraphrase_question("Who directed Titanic?")
    assert "Titanic" in result
    assert "whom" in result.lower()


def test_capital_of_france_paraphrase() -> None:
    result = paraphrase_question("What is the capital of France?")
    assert result == "Which city is the capital of France?"


def test_paraphrase_samples_preserves_answers_and_documents() -> None:
    sample = QASample(
        sample_id="qa1",
        question="What is the capital of France?",
        answers=["Paris"],
        documents=["Paris is the capital of France."],
        metadata={"dataset": "triviaqa"},
    )
    out = paraphrase_samples([sample])
    assert len(out) == 1
    assert out[0].sample_id == "qa1-paraphrased"
    assert out[0].question != sample.question
    assert out[0].answers == sample.answers
    assert out[0].documents == sample.documents
    assert out[0].metadata == sample.metadata
