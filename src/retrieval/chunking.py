from __future__ import annotations

import html
import re
from typing import Iterable, List

TAG_RE = re.compile(r"<[^>]+>")
WHITESPACE_RE = re.compile(r"\s+")
WORD_RE = re.compile(r"\S+")


def clean_text(text: str) -> str:
    """Normalize noisy wiki-style text into a compact retrieval snippet."""
    value = html.unescape(str(text or ""))
    value = TAG_RE.sub(" ", value)
    value = value.replace("``", '"').replace("''", '"')
    value = WHITESPACE_RE.sub(" ", value)
    return value.strip()


def chunk_text(
    text: str,
    min_tokens: int = 200,
    max_tokens: int = 350,
    overlap: int = 60,
) -> List[str]:
    """Split text into overlapping token windows.

    The windows are sized for retrieval passages, not final prompts.
    """
    cleaned = clean_text(text)
    if not cleaned:
        return []

    tokens = WORD_RE.findall(cleaned)
    if len(tokens) <= max_tokens:
        return [cleaned]

    overlap = max(0, min(overlap, max_tokens - 1))
    step = max(1, max_tokens - overlap)
    chunks: List[str] = []

    start = 0
    while start < len(tokens):
        end = min(len(tokens), start + max_tokens)
        chunk = " ".join(tokens[start:end]).strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(tokens):
            break
        start += step

    # Merge tiny trailing fragments into the previous chunk.
    merged: List[str] = []
    for chunk in chunks:
        if merged and len(WORD_RE.findall(chunk)) < min_tokens // 2:
            merged[-1] = f"{merged[-1]} {chunk}".strip()
        else:
            merged.append(chunk)

    return merged


def chunk_documents(
    documents: Iterable[str],
    min_tokens: int = 200,
    max_tokens: int = 350,
    overlap: int = 60,
) -> List[str]:
    """Chunk a list of documents into retrieval passages."""
    passages: List[str] = []
    seen: set[str] = set()

    for document in documents:
        for chunk in chunk_text(document, min_tokens=min_tokens, max_tokens=max_tokens, overlap=overlap):
            normalized = chunk.strip()
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            passages.append(normalized)

    return passages
