"""
Confidence assessment and ambiguity detection.

Given a ranked list of fused candidates, decides whether the top result is
"confident" (commit to top-1) or "ambiguous" (surface alternatives to user).

Confidence is LOW when ANY of:
  1. top-1 combined score < min_score_threshold
  2. gap between top-1 and top-2 < score_gap_threshold
  3. variance of top-5 combined scores < variance_threshold

Deduplication: normalize text (lowercase, strip punctuation, collapse whitespace)
before counting alternatives so near-duplicates ("he dresses himself" /
"He dresses himself.") don't count as separate candidates.

Run standalone for local verification:
    python -m src.confidence
"""

from __future__ import annotations

import math
import re
import unicodedata
from pathlib import Path


# ---------------------------------------------------------------------------
# Text normalisation for deduplication
# ---------------------------------------------------------------------------

def _normalize_text(text: str) -> str:
    text = text.lower()
    text = unicodedata.normalize("NFKD", text)
    text = re.sub(r"[^\w\s]", "", text)   # strip punctuation
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _deduplicate(candidates: list[dict]) -> list[dict]:
    """Keep only the highest-scoring instance of each normalised text."""
    seen: dict[str, dict] = {}
    for c in candidates:
        key = _normalize_text(c["text"])
        if key not in seen or c["combined"] > seen[key]["combined"]:
            seen[key] = c
    # Preserve original sort order (combined descending)
    result = sorted(seen.values(), key=lambda x: x["combined"], reverse=True)
    # Restore normalised key for downstream use
    for c in result:
        c["_norm_text"] = _normalize_text(c["text"])
    return result


# ---------------------------------------------------------------------------
# Softmax confidence
# ---------------------------------------------------------------------------

def _softmax_top1_prob(scores: list[float]) -> float:
    """Softmax probability of the top-1 element over the provided scores."""
    if not scores:
        return 0.0
    max_s = max(scores)
    exps = [math.exp(s - max_s) for s in scores]
    return exps[0] / sum(exps)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def assess_confidence(
    ranked_candidates: list[dict],
    config: dict,
) -> dict:
    """
    Parameters
    ----------
    ranked_candidates : list[dict]
        Already sorted by combined score descending (output of fusion.fuse).
        Each dict must have at least "text" and "combined" keys.
    config : dict
        The "confidence" section of config.yaml.

    Returns
    -------
    {
        "status":          "confident" | "ambiguous",
        "confidence":      float,   # softmax probability of top-1 over top-5
        "alternatives":    list[dict],  # top-N with text/score/rank
        "trigger_reason":  str | None,  # which signal tripped (debug)
    }
    """
    thresh_score: float = config.get("min_score_threshold", -3.0)
    thresh_gap: float = config.get("score_gap_threshold", 0.15)
    thresh_var: float = config.get("variance_threshold", 0.05)
    alt_min: int = config.get("top_n_alternatives_min", 3)
    alt_max: int = config.get("top_n_alternatives_max", 5)
    cutoff_gap: float = config.get("alternative_cutoff_gap", 0.4)

    deduped = _deduplicate(ranked_candidates)

    if not deduped:
        return {
            "status": "ambiguous",
            "confidence": 0.0,
            "alternatives": [],
            "trigger_reason": "no_candidates",
        }

    top5_scores = [c["combined"] for c in deduped[:5]]
    top1_score = top5_scores[0]
    top2_score = top5_scores[1] if len(top5_scores) > 1 else top1_score - 999

    # Variance (population variance)
    if len(top5_scores) > 1:
        mean = sum(top5_scores) / len(top5_scores)
        variance = sum((s - mean) ** 2 for s in top5_scores) / len(top5_scores)
    else:
        variance = 999.0  # single candidate → infinite certainty on variance signal

    # Confidence signals
    score_ok = top1_score > thresh_score
    gap_ok = (top1_score - top2_score) > thresh_gap
    variance_ok = variance > thresh_var

    # Determine trigger reason
    triggers = []
    if not score_ok:
        triggers.append("low_absolute_score")
    if not gap_ok:
        triggers.append("small_gap_to_top2")
    if not variance_ok:
        triggers.append("low_variance_top5")
    trigger_reason = ", ".join(triggers) if triggers else None

    # Confidence probability
    confidence = _softmax_top1_prob(top5_scores)

    if not triggers:
        status = "confident"
        alternatives = [
            {"text": deduped[0]["text"], "score": round(top1_score, 4), "rank": 1}
        ]
    else:
        status = "ambiguous"
        # Collect alternatives within the cutoff gap
        alts = []
        for i, c in enumerate(deduped):
            if i >= alt_max:
                break
            if i > 0 and (top1_score - c["combined"]) > cutoff_gap:
                break
            alts.append(
                {"text": c["text"], "score": round(c["combined"], 4), "rank": i + 1}
            )
        # Always return at least alt_min (pad from deduped if available)
        while len(alts) < alt_min and len(alts) < len(deduped):
            i = len(alts)
            c = deduped[i]
            alts.append(
                {"text": c["text"], "score": round(c["combined"], 4), "rank": i + 1}
            )
        alternatives = alts

    return {
        "status": status,
        "confidence": round(confidence, 4),
        "alternatives": alternatives,
        "trigger_reason": trigger_reason,
    }


# ---------------------------------------------------------------------------
# Local verification
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import yaml

    here = Path(__file__).parent.parent
    with open(here / "config.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    conf_cfg = cfg["confidence"]

    print("=" * 60)
    print("confidence — local verification")
    print("=" * 60)

    # Case 1: confident (clear top-1)
    candidates_confident = [
        {"text": "he dresses himself", "combined": -1.20},
        {"text": "he addresses himself", "combined": -2.50},
        {"text": "e dresse imself", "combined": -3.80},
        {"text": "he processes himself", "combined": -4.10},
        {"text": "they dress themselves", "combined": -4.50},
    ]

    # Case 2: ambiguous (top-2 very close)
    candidates_ambiguous = [
        {"text": "he dresses himself", "combined": -1.80},
        {"text": "he dresses themself", "combined": -1.85},
        {"text": "he addresses himself", "combined": -2.00},
        {"text": "he dresses itself", "combined": -2.10},
        {"text": "he stresses himself", "combined": -2.15},
    ]

    # Case 3: near-duplicates (with punctuation)
    candidates_dupes = [
        {"text": "He dresses himself.", "combined": -1.80},
        {"text": "he dresses himself", "combined": -1.82},
        {"text": "he dresses themself", "combined": -2.10},
    ]

    for label, cands in [
        ("CONFIDENT", candidates_confident),
        ("AMBIGUOUS (close gap)", candidates_ambiguous),
        ("DEDUPLICATED", candidates_dupes),
    ]:
        result = assess_confidence(cands, conf_cfg)
        print(f"\n  [{label}]")
        print(f"    status           : {result['status']}")
        print(f"    confidence       : {result['confidence']}")
        print(f"    trigger_reason   : {result['trigger_reason']}")
        print(f"    alternatives     :")
        for alt in result["alternatives"]:
            print(f"      rank={alt['rank']}  score={alt['score']}  '{alt['text']}'")

    print("\nDONE")
