from __future__ import annotations

from typing import List

from src.data.schemas import RetrievedDoc


class Generator:
    def __init__(self, enabled: bool = False, model_name: str = "", max_new_tokens: int = 64, temperature: float = 0.0) -> None:
        self.enabled = enabled
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self._pipeline = None

        if self.enabled:
            try:
                from transformers import pipeline

                self._pipeline = pipeline("text2text-generation", model=model_name)
            except Exception:
                self.enabled = False
                self._pipeline = None

    def _extractive_fallback(self, docs: List[RetrievedDoc]) -> str:
        if not docs:
            return "I do not have enough context to answer confidently."
        text = docs[0].text.strip()
        if not text:
            return "I do not have enough context to answer confidently."
        return text.split(".")[0].strip()

    def generate(self, prompt: str, docs: List[RetrievedDoc]) -> str:
        if self.enabled and self._pipeline is not None:
            try:
                out = self._pipeline(prompt, max_new_tokens=self.max_new_tokens, do_sample=self.temperature > 0, temperature=self.temperature)
                if out and isinstance(out, list):
                    return str(out[0].get("generated_text", "")).strip() or self._extractive_fallback(docs)
            except Exception:
                return self._extractive_fallback(docs)
        return self._extractive_fallback(docs)
