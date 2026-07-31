from __future__ import annotations

import json
import logging
import os
import re
import urllib.error
import urllib.request
from typing import List

from src.data.schemas import RetrievedDoc


logger = logging.getLogger(__name__)


class OllamaGenerator:
    def __init__(
        self,
        ollama_endpoint: str = "http://127.0.0.1:11435",
        model_name: str = "llama3",
        max_tokens: int = 64,
        temperature: float = 0.1,
        timeout_seconds: int = 120,
    ) -> None:
        self.ollama_endpoint = ollama_endpoint.rstrip("/")
        self.model_name = model_name.strip() or "llama3"
        self.max_tokens = int(max_tokens)
        self.temperature = float(temperature)
        self.timeout_seconds = int(timeout_seconds)
        self._debug = os.environ.get("RAG_GENERATION_DEBUG", "").lower() in {"1", "true", "yes"}

        self._validate_server()

    def _validate_server(self) -> None:
        request = urllib.request.Request(f"{self.ollama_endpoint}/api/tags", method="GET")
        try:
            with urllib.request.urlopen(request, timeout=10) as response:
                if response.status != 200:
                    raise RuntimeError(f"Ollama endpoint returned HTTP {response.status}")
        except Exception as exc:
            raise RuntimeError(
                f"Unable to reach Ollama at {self.ollama_endpoint}. Start the server before running generation."
            ) from exc

    # 🔥 FIXED: HARD CAP TOTAL CONTEXT SIZE
    def _format_docs(self, docs: List[RetrievedDoc]) -> str:
        MAX_TOTAL_CONTEXT = 1500  # total characters allowed
        MAX_DOCS = 2              # limit number of docs

        lines: List[str] = []
        current_len = 0

        for idx, doc in enumerate(docs[:MAX_DOCS], start=1):
            text = (doc.text or "").strip()
            if not text:
                continue

            remaining = MAX_TOTAL_CONTEXT - current_len
            if remaining <= 0:
                break

            snippet = text[:remaining]
            lines.append(f"{idx}. {snippet}")
            current_len += len(snippet)

        return "\n\n".join(lines)

    def _build_payload(self, question: str, docs: List[RetrievedDoc], memory_context: str | None = None) -> dict:
        system_prompt = (
            "Answer using only the provided context. "
            "Return only the final short answer. "
            "No explanation, no preamble, no supporting evidence. "
            "If the answer is not present in the context, return unknown."
        )

        sections = []

        # 🔥 FIXED: LIMIT MEMORY SIZE
        if memory_context:
            memory_context = memory_context.strip()[:1500]
            sections.append(f"Memory hint:\n{memory_context}")

        docs_text = self._format_docs(docs)
        if docs_text:
            sections.append(f"Context:\n{docs_text}")

        sections.append(f"Question:\n{question}")

        user_prompt = "\n\n".join(sections)

        # 🔍 DEBUG (optional)
        if self._debug:
            print("PROMPT LENGTH:", len(user_prompt))

        options = {
            "num_predict": self.max_tokens,
            "temperature": self.temperature,
        }

        num_gpu = os.environ.get("OLLAMA_NUM_GPU")
        if num_gpu is not None and num_gpu.strip():
            try:
                options["num_gpu"] = int(num_gpu)
            except ValueError:
                logger.warning("Ignoring invalid OLLAMA_NUM_GPU value: %s", num_gpu)

        return {
            "model": self.model_name,
            "system": system_prompt,
            "prompt": user_prompt,
            "stream": False,
            "options": options,
        }

    def _normalize_answer(self, text: str) -> str:
        normalized = " ".join(text.strip().split())

        normalized = re.sub(
            r"^(?:the\s+)?(?:correct\s+)?answer\s+is\s+",
            "",
            normalized,
            flags=re.IGNORECASE,
        )
        normalized = re.sub(
            r"^(?:based on the provided context,\s*)",
            "",
            normalized,
            flags=re.IGNORECASE,
        )
        normalized = re.sub(
            r"^(?:the\s+)?final\s+answer\s+is\s+",
            "",
            normalized,
            flags=re.IGNORECASE,
        )

        normalized = normalized.strip().strip('"').strip("'")

        if normalized.endswith((".", "!", "?")):
            normalized = normalized[:-1].rstrip()

        return normalized

    def generate(self, question: str, docs: List[RetrievedDoc], memory_context: str | None = None) -> str:
        payload = self._build_payload(question, docs, memory_context=memory_context)
        payload_json = json.dumps(payload).encode("utf-8")

        request = urllib.request.Request(
            f"{self.ollama_endpoint}/api/generate",
            data=payload_json,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        logger.info("Ollama prompt payload: %s", payload)

        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                raw_text = response.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as exc:
            error_text = exc.read().decode("utf-8", errors="replace") if exc.fp else ""
            raise RuntimeError(
                f"Ollama generation failed with HTTP {exc.code}: {error_text or exc.reason}"
            ) from exc
        except Exception as exc:
            raise RuntimeError(f"Ollama generation request failed: {exc}") from exc

        logger.info("Ollama raw response: %s", raw_text)

        try:
            data = json.loads(raw_text)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Ollama returned invalid JSON: {raw_text}") from exc

        if "error" in data and data["error"]:
            raise RuntimeError(f"Ollama error: {data['error']}")

        answer = self._normalize_answer(str(data.get("response", "")))

        if not answer:
            raise RuntimeError(f"Ollama returned an empty answer: {data}")

        logger.info("Ollama final answer: %s", answer)

        return answer

    def generate_raw(self, system_prompt: str, user_prompt: str) -> str:
        """Generic single-turn completion, bypassing the QA-specific prompt builder."""
        payload = {
            "model": self.model_name,
            "system": system_prompt,
            "prompt": user_prompt,
            "stream": False,
            "options": {"num_predict": 64, "temperature": 0.3},
        }
        payload_json = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            f"{self.ollama_endpoint}/api/generate",
            data=payload_json,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                raw_text = response.read().decode("utf-8", errors="replace")
        except Exception as exc:
            raise RuntimeError(f"Ollama generate_raw failed: {exc}") from exc

        data = json.loads(raw_text)
        if "error" in data and data["error"]:
            raise RuntimeError(f"Ollama error: {data['error']}")
        return str(data.get("response", "")).strip()