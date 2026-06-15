from __future__ import annotations

from collections import Counter
import re
import string
from typing import Dict, List

try:
    from rouge_score import rouge_scorer  # type: ignore
    HAVE_ROUGE = True
except Exception:
    HAVE_ROUGE = False

from src.data.schemas import PredictionRecord


def _normalize(text: str) -> str:
    """TriviaQA/SQuAD-style answer normalization."""
    text = (text or "").lower()
    text = "".join(ch if ch not in string.punctuation else " " for ch in text)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def _normalized_tokens(text: str) -> List[str]:
    normalized = _normalize(text)
    return normalized.split() if normalized else []


def exact_match(pred: str, golds: List[str]) -> float:
    p = _normalize(pred)
    return 1.0 if any(p == _normalize(g) for g in golds) else 0.0


def token_f1(pred: str, golds: List[str]) -> float:
    pred_toks = _normalized_tokens(pred)
    if not pred_toks:
        return 0.0

    best = 0.0
    for g in golds:
        gold_toks = _normalized_tokens(g)
        common = Counter(pred_toks) & Counter(gold_toks)
        num_same = sum(common.values())
        if num_same == 0:
            continue
        precision = num_same / len(pred_toks)
        recall = num_same / max(1, len(gold_toks))
        f1 = 2 * precision * recall / (precision + recall)
        best = max(best, f1)
    return best


def bleu_unigram(pred: str, golds: List[str]) -> float:
    pred_toks = _normalized_tokens(pred)
    if not pred_toks:
        return 0.0
    best = 0.0
    for g in golds:
        gold_toks = _normalized_tokens(g)
        overlap = Counter(pred_toks) & Counter(gold_toks)
        score = sum(overlap.values()) / len(pred_toks)
        best = max(best, score)
    return best


def rouge_l(pred: str, golds: List[str]) -> float:
    # Prefer library scorer when available for better accuracy
    if HAVE_ROUGE:
        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        best = 0.0
        for g in golds:
            best = max(best, scorer.score(g, pred)["rougeL"].fmeasure)
        return best

    # Fallback: token-based longest common subsequence ratio approximation
    def lcs_len(a: List[str], b: List[str]) -> int:
        # dynamic programming LCS
        m, n = len(a), len(b)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(m - 1, -1, -1):
            for j in range(n - 1, -1, -1):
                if a[i] == b[j]:
                    dp[i][j] = dp[i + 1][j + 1] + 1
                else:
                    dp[i][j] = max(dp[i + 1][j], dp[i][j + 1])
        return dp[0][0]

    p_toks = _normalized_tokens(pred)
    if not p_toks:
        return 0.0
    best = 0.0
    for g in golds:
        g_toks = _normalized_tokens(g)
        if not g_toks:
            continue
        lcs = lcs_len(p_toks, g_toks)
        # use F-measure like combination of precision & recall on LCS
        precision = lcs / len(p_toks)
        recall = lcs / len(g_toks)
        if precision + recall == 0:
            f = 0.0
        else:
            f = 2 * precision * recall / (precision + recall)
        best = max(best, f)
    return best


def evaluate_predictions(records: List[PredictionRecord]) -> Dict[str, float]:
    if not records:
        return {"exact_match": 0.0, "token_f1": 0.0, "bleu": 0.0, "rougeL": 0.0}

    em = sum(exact_match(r.prediction, r.gold_answers) for r in records) / len(records)
    f1 = sum(token_f1(r.prediction, r.gold_answers) for r in records) / len(records)
    bleu = sum(bleu_unigram(r.prediction, r.gold_answers) for r in records) / len(records)
    rouge = sum(rouge_l(r.prediction, r.gold_answers) for r in records) / len(records)

    return {
        "exact_match": float(em),
        "token_f1": float(f1),
        "bleu": float(bleu),
        "rougeL": float(rouge),
    }
