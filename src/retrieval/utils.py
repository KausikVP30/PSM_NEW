from __future__ import annotations

import re
from typing import List, Set

TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+")

STOPWORDS = {
    "the", "is", "a", "an", "of", "in", "on", "at",
    "who", "what", "when", "where", "why", "how",
    "to", "for", "by", "with", "and", "or",
    "was", "were", "be", "been", "are", "as",
}


def tokenize(text: str) -> List[str]:
    return [t.lower() for t in TOKEN_PATTERN.findall(text or "")]


def token_set(text: str) -> Set[str]:
    tokens = tokenize(text)
    return {t for t in tokens if t not in STOPWORDS}