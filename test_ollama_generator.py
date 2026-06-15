#!/usr/bin/env python3
from __future__ import annotations

import os
import sys


sys.path.insert(0, "/fab3/btech/2024/kausik.patra24b/PSM_New_V1/PSM_NEW")

from src.data.schemas import RetrievedDoc
from src.generation.generator_ollama import OllamaGenerator


def main() -> None:
    os.environ.setdefault("RAG_GENERATION_DEBUG", "1")

    question = "Who was the man behind The Chipmunks?"
    docs = [
        RetrievedDoc(
            doc_id=0,
            text=(
                "David Seville was the man behind The Chipmunks. He created and produced the group and was the "
                "original voice behind the project."
            ),
            score=0.99,
            source="test",
        )
    ]

    generator = OllamaGenerator(
        ollama_endpoint=os.environ.get("OLLAMA_ENDPOINT", "http://127.0.0.1:11435"),
        model_name=os.environ.get("OLLAMA_MODEL_NAME", "llama3"),
        max_tokens=48,
        temperature=0.1,
    )

    answer = generator.generate(question, docs)
    print(answer)
    print("Expected: David Seville")

    if answer.strip().lower() != "david seville" and "david seville" not in answer.lower():
        raise SystemExit(1)


if __name__ == "__main__":
    main()