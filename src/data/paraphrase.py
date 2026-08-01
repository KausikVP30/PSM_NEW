from __future__ import annotations

from dataclasses import replace
from typing import List, Tuple

from .schemas import QASample


def paraphrase_question_llm(question: str, generator) -> str:
    """Generate a paraphrase using the LLM. `generator` must be an OllamaGenerator."""
    system_prompt = (
        "You are an expert paraphrasing assistant.\n\n"
        "Rewrite the given question using substantially different wording while "
        "preserving exactly the same meaning and exactly the same answer.\n\n"
        "Rules:\n"
        "- Do NOT answer the question.\n"
        "- Do NOT add or remove information.\n"
        "- Return only a single rewritten question.\n"
        "- Do not include explanations, prefixes, or quotation marks."
    )
    user_prompt = f"Original question: {question}\nRephrased question:"
    result = generator.generate_raw(system_prompt, user_prompt)
    result = result.strip()

    prefixes = [
        "Rephrased question:",
        "Rephrased:",
        "Question:",
        "Paraphrase:",
    ]

    for p in prefixes:
        if result.lower().startswith(p.lower()):
            result = result[len(p):].strip()

    result = result.strip('"').strip("'")
    if not result.endswith("?"):
        result = result.rstrip(".") + "?"
    return result


def verify_paraphrase(original: str, paraphrase: str, nli_model, threshold: float = 0.65) -> Tuple[bool, float]:
    """
    Bidirectional entailment verification using a CrossEncoder NLI model.

    Returns
    -------
    (bool, float)
        (passed, average_entailment_score)
    """
    if not paraphrase or paraphrase.strip().lower() == original.strip().lower():
        return False, 0.0
    scores = nli_model.predict([(original, paraphrase), (paraphrase, original)])
    ENTAILMENT_INDEX = 1

    entail_fwd = float(scores[0][ENTAILMENT_INDEX])
    entail_bwd = float(scores[1][ENTAILMENT_INDEX])    
    
    score = (entail_fwd + entail_bwd) / 2.0
    
    print(
        f"\nOriginal: {original}\n"
        f"Paraphrase: {paraphrase}\n"
        f"Forward: {entail_fwd:.3f}\n"
        f"Backward: {entail_bwd:.3f}\n"
        f"Average: {score:.3f}\n"
    )

    return score >= threshold, score

def paraphrase_samples_verified(
    samples: List[QASample],
    generator,
    nli_model,
    threshold: float = 0.65,
    max_retries: int = 3,
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
        else:
            kept.append(sample)

    return kept, audit_log