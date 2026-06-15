from __future__ import annotations

import os
import traceback
from typing import List

import torch

from src.data.schemas import RetrievedDoc


class Generator:
    """T5-based (flan-t5) seq2seq generator using transformers pipeline."""

    def __init__(
        self, enabled: bool = False, model_name: str = "", max_new_tokens: int = 64, temperature: float = 0.0
    ) -> None:
        self.enabled = enabled
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self._pipeline = None
        self._debug = os.environ.get("RAG_GENERATION_DEBUG", "").lower() in {"1", "true", "yes"}

        if self.enabled:
            self._init_model()

    def _init_model(self) -> None:
        """Initialize T5 model using transformers pipeline."""
        try:
            print(f"[Generator] Initializing {self.model_name} for text-to-text generation...")

            from transformers import pipeline
            import torch

            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is required for generation, but no GPU is available.")

            # Use text2text-generation pipeline for T5 models
            self._pipeline = pipeline(
                "text2text-generation",
                model=self.model_name,
                device=0,
                torch_dtype=torch.float16,
            )

            print(f"[Generator] Model loaded successfully")

        except Exception as e:
            print(f"[Generator] Failed to initialize model: {type(e).__name__}: {e}")
            traceback.print_exc()
            self.enabled = False
            self._pipeline = None

    def _format_prompt(self, question: str, docs: List[RetrievedDoc], memory_context: str | None = None) -> str:
        """Format prompt for T5 (task-oriented format)."""
        context_parts = []
        if memory_context:
            context_parts.append(f"Memory hint: {memory_context[:1000]}")
        for idx, doc in enumerate(docs[:3], start=1):
            text = (doc.text or "").strip()
            if text:
                context_parts.append(f"{idx}. {text[:1000]}")
        context = "\n".join(context_parts) if context_parts else "No external context provided."
        return f"question: {question}\ncontext:\n{context}\nanswer:"

    def generate(self, question: str, docs: List[RetrievedDoc], memory_context: str | None = None) -> str:
        """Generate answer using T5 model."""
        if not self.enabled or self._pipeline is None:
            return self._extractive_fallback(docs, memory_context=memory_context)

        try:
            # Format prompt
            prompt = self._format_prompt(question, docs, memory_context=memory_context)

            if self._debug:
                print(f"[Generator DEBUG] Input prompt: {prompt}")

            # Generate using pipeline
            outputs = self._pipeline(
                prompt,
                max_length=self.max_new_tokens,
                num_beams=1,
                do_sample=False,
            )

            # Extract answer from output
            answer = ""
            if outputs and isinstance(outputs, list) and len(outputs) > 0:
                answer = outputs[0].get("generated_text", "").strip()

            if self._debug:
                print(f"[Generator DEBUG] Raw output: {answer}")

            # Use fallback if no answer generated
            if not answer:
                answer = self._extractive_fallback(docs, memory_context=memory_context)

            if self._debug:
                print(f"[Generator DEBUG] Final answer: {answer}\n")

            return answer

        except Exception as e:
            print(f"[Generator] Generation failed: {type(e).__name__}: {e}")
            if self._debug:
                traceback.print_exc()
            return self._extractive_fallback(docs, memory_context=memory_context)

    def _extractive_fallback(self, docs: List[RetrievedDoc], memory_context: str | None = None) -> str:
        """Fallback: return first sentence of top doc."""
        if memory_context:
            hint = memory_context.strip()
            if hint:
                return hint
        if not docs:
            return "I do not have enough context to answer confidently."
        text = docs[0].text.strip()
        if not text:
            return "I do not have enough context to answer confidently."
        # Return first sentence or first 100 chars
        sentence = text.split(".")[0].strip()
        if not sentence:
            sentence = text[:100].strip()
        return sentence
