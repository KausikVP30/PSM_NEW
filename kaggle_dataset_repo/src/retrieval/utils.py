from __future__ import annotations

import re
from typing import List, Set

TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+")


def tokenize(text: str) -> List[str]:
    return [t.lower() for t in TOKEN_PATTERN.findall(text or "")]


def token_set(text: str) -> Set[str]:
    return set(tokenize(text))
