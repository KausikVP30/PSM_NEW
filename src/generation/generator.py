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
                import os
                import traceback

                hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")

                # Try a simple text2text pipeline first (works for many models)
                try:
                    print(f"[Generator] Trying text2text-generation pipeline for {model_name}")
                    self._pipeline = pipeline("text2text-generation", model=model_name)
                    print("[Generator] text2text pipeline initialized")
                except Exception:
                    print("[Generator] text2text pipeline failed, trying text-generation with trust_remote_code/device_map")
                    try:
                        # Prefer device_map='auto' to place model on GPU if available
                        kwargs = {"trust_remote_code": True}
                        # Pass token if available for gated models
                        if hf_token:
                            kwargs["use_auth_token"] = hf_token
                        # device_map may be set by the pipeline automatically when available
                        self._pipeline = pipeline("text-generation", model=model_name, **kwargs)
                        print("[Generator] text-generation pipeline initialized")
                    except Exception:
                        print("[Generator] Failed to initialize any pipeline for generation")
                        traceback.print_exc()
                        self.enabled = False
                        self._pipeline = None
            except Exception:
                # Unexpected error during import or init
                import traceback

                print("[Generator] Unexpected error initializing generation pipeline")
                traceback.print_exc()
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
