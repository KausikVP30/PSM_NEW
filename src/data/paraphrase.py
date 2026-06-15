from __future__ import annotations

import re
from dataclasses import replace
from typing import List

from .schemas import QASample


def paraphrase_question(question: str) -> str:
    """Apply a lightweight rule-based paraphrase (no API, no models)."""
    q = (question or "").strip()
    if not q:
        return q

    has_qmark = q.endswith("?")
    core = q[:-1].strip() if has_qmark else q

    # Who <verb> <rest>?  e.g. "Who directed Titanic?" -> "Titanic was directed by whom?"
    m = re.match(r"^Who\s+(\w+)\s+(.+)$", core, re.IGNORECASE)
    if m:
        verb, rest = m.group(1), m.group(2).strip()
        return f"{rest} was {verb} by whom?"

    # What is the <noun> of <X>?  e.g. capital of France
    m = re.match(r"^What\s+is\s+the\s+(\w+)\s+of\s+(.+)$", core, re.IGNORECASE)
    if m:
        noun, subject = m.group(1), m.group(2).strip()
        if noun.lower() in {"capital", "city", "country", "state"}:
            return f"Which city is the {noun} of {subject}?"
        return f"Which is the {noun} of {subject}?"

    # What is <X>?
    m = re.match(r"^What\s+is\s+(.+)$", core, re.IGNORECASE)
    if m:
        return f"Which is {m.group(1).strip()}?"

    # When did/was <X>?
    m = re.match(r"^When\s+(did|was|were)\s+(.+)$", core, re.IGNORECASE)
    if m:
        return f"At what time {m.group(1).lower()} {m.group(2).strip()}?"

    # Where is/was <X>?
    m = re.match(r"^Where\s+(is|was|were)\s+(.+)$", core, re.IGNORECASE)
    if m:
        return f"In which place {m.group(1).lower()} {m.group(2).strip()}?"

    # How did/does <X>?
    m = re.match(r"^How\s+(did|does|do|was|were)\s+(.+)$", core, re.IGNORECASE)
    if m:
        return f"In what way {m.group(1).lower()} {m.group(2).strip()}?"

    # Which <rest>? -> What <rest>?
    m = re.match(r"^Which\s+(.+)$", core, re.IGNORECASE)
    if m:
        return f"What {m.group(1).strip()}?"

    # Fallback: rephrase as an indirect question
    topic = core.rstrip(".")
    return f"What is known about {topic}?"


def paraphrase_samples(samples: List[QASample]) -> List[QASample]:
    """Return copies with paraphrased questions; answers and documents unchanged."""
    paraphrased: List[QASample] = []
    for sample in samples:
        paraphrased.append(
            replace(
                sample,
                sample_id=f"{sample.sample_id}-paraphrased",
                question=paraphrase_question(sample.question),
            )
        )
    return paraphrased
