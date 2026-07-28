from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class QASample:
    sample_id: str
    question: str
    answers: List[str]
    documents: List[str]
    metadata: Dict[str, str] = field(default_factory=dict)
    context: str = ""


@dataclass
class QQPSample:
    pair_id: str
    question1: str
    question2: str
    label: Optional[int] = None
    metadata: Dict[str, str] = field(default_factory=dict)


@dataclass
class RetrievedDoc:
    doc_id: int
    text: str
    score: float
    source: str


@dataclass
class PredictionRecord:
    sample_id: str
    question: str
    prediction: str
    confidence: float
    route: str
    references: List[str]
    gold_answers: List[str]
