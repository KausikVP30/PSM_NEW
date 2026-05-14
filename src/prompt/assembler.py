from __future__ import annotations

from typing import List

from src.data.schemas import RetrievedDoc


class PromptAssembler:
    def __init__(self, max_chars: int = 4000) -> None:
        self.max_chars = max_chars

    def assemble(self, question: str, docs: List[RetrievedDoc], memory_answer: str | None = None) -> str:
        blocks = ["You are a QA assistant.", f"Question: {question}"]
        if memory_answer:
            blocks.append(f"Prior memory hint: {memory_answer}")

        for i, doc in enumerate(docs, start=1):
            blocks.append(f"Doc {i}: {doc.text}")

        blocks.append("Answer concisely and factually based on evidence.")
        prompt = "\n".join(blocks)
        return prompt[: self.max_chars]
