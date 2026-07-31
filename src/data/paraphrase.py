from __future__ import annotations

from dataclasses import replace
from typing import List, Tuple

from .schemas import QASample


def paraphrase_question_llm(question: str, generator) -> str:
    """Generate a paraphrase using the LLM. `generator` must be an OllamaGenerator."""
    system_prompt = (
        "You rephrase questions using different wording while preserving their "
        "exact meaning and the same correct answer. Return only the rephrased "
        "question, nothing else."
    )
    user_prompt = f"Original question: {question}\nRephrased question:"
    result = generator.generate_raw(system_prompt, user_prompt)
    result = result.strip().strip('"').strip("'")
    if not result.endswith("?"):
        result = result.rstrip(".") + "?"
    return result


def verify_paraphrase(original: str, paraphrase: str, nli_model, threshold: float = 0.8) -> Tuple[bool, float]:
    """Bidirectional entailment check using a CrossEncoder NLI model.
    Returns (passed, min_score). Verify label index against your checkpoint's
    id2label before running at scale (see note in the chat reply)."""
    if not paraphrase or paraphrase.strip().lower() == original.strip().lower():
        return False, 0.0
    scores = nli_model.predict([(original, paraphrase), (paraphrase, original)])
    entail_fwd = float(scores[0][-1])
    entail_bwd = float(scores[1][-1])
    min_score = min(entail_fwd, entail_bwd)
    return min_score >= threshold, min_score


def paraphrase_samples_verified(
    samples: List[QASample],
    generator,
    nli_model,
    threshold: float = 0.8,
    max_retries: int = 1,
) -> Tuple[List[QASample], List[dict]]:
    """Paraphrase + verify each sample. Returns (kept_samples, audit_log).
    kept_samples only includes samples that passed verification;
    audit_log has one entry per INPUT sample (pass or fail) for reporting."""
    kept: List[QASample] = []
    audit_log: List[dict] = []

    for sample in samples:
        passed = False
        best_paraphrase = ""
        best_score = 0.0
        for attempt in range(max_retries + 1):
            candidate = paraphrase_question_llm(sample.question, generator)
            ok, score = verify_paraphrase(sample.question, candidate, nli_model, threshold)
            if score > best_score:
                best_paraphrase, best_score = candidate, score
            if ok:
                passed = True
                best_paraphrase = candidate
                break

        audit_log.append({
            "sample_id": sample.sample_id,
            "original": sample.question,
            "paraphrase": best_paraphrase,
            "entailment_score": best_score,
            "passed": passed,
        })

        if passed:
            kept.append(
                replace(
                    sample,
                    sample_id=f"{sample.sample_id}-paraphrased",
                    question=best_paraphrase,
                )
            )

    return kept, audit_log