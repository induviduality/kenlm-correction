"""
Score fusion: combines whisper, gpt2, and fingerprint scores into a single
combined score using severity-adapted weights from config.

combined = α·norm(whisper) + β·norm(gpt2) + γ·norm(fingerprint)

Each dimension is min-max normalised across the candidate set so that no single
scorer dominates due to magnitude differences (GPT2 log-probs are ~5× larger
in magnitude than fingerprint scores). Normalisation is applied only when std > 0.

Run standalone for local verification:
    python -m src.fusion
"""

from __future__ import annotations

import math
from pathlib import Path


def _minmax_normalise(values: list[float]) -> list[float]:
    """Normalise a list of floats to [0, 1] using min-max scaling.
    If all values are equal, return 0.5 for each."""
    lo, hi = min(values), max(values)
    rng = hi - lo
    if rng < 1e-12:
        return [0.5] * len(values)
    return [(v - lo) / rng for v in values]


def fuse(
    candidates: list[dict],
    severity: str,
    weights_config: dict,
) -> list[dict]:
    """
    Adds a "combined" field to each candidate dict and returns the list
    sorted by combined score descending.

    Each candidate dict must already contain:
      - "whisper_score"      (float, from Whisper n-best or 0.0 for generated)
      - "gpt2_score"         (float, length-norm log-prob from DistilGPT2)
      - "fingerprint_score"  (float, length-norm log-prob from fingerprint alignment)

    weights_config is the fusion_weights section of config.yaml, e.g.:
      { "severe": {"alpha": 0.3, "beta": 0.3, "gamma": 0.4}, ... }
    """
    if not candidates:
        return candidates

    # Normalise severity key; fall back to "moderate" if unknown
    sev = severity.lower()
    if sev not in weights_config:
        sev = "moderate"
    w = weights_config[sev]
    alpha: float = w["alpha"]   # whisper weight
    beta: float = w["beta"]     # gpt2 weight
    gamma: float = w["gamma"]   # fingerprint weight

    delta: float = w.get("delta", 0.0)   # word-validity weight (optional)

    # Extract raw score vectors
    ws = [c.get("whisper_score", 0.0) for c in candidates]
    gs = [c.get("gpt2_score", 0.0) for c in candidates]
    fs = [c.get("fingerprint_score", 0.0) for c in candidates]
    # word_validity_score is already in [0,1] so no normalisation needed
    vs = [c.get("word_validity_score", 0.5) for c in candidates]

    # Normalise log-prob dimensions independently to [0, 1]
    ws_n = _minmax_normalise(ws)
    gs_n = _minmax_normalise(gs)
    fs_n = _minmax_normalise(fs)

    for c, wn, gn, fn, vn in zip(candidates, ws_n, gs_n, fs_n, vs):
        c["combined"] = alpha * wn + beta * gn + gamma * fn + delta * vn

    candidates.sort(key=lambda x: x["combined"], reverse=True)
    return candidates


# ---------------------------------------------------------------------------
# Local verification
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import yaml

    here = Path(__file__).parent.parent
    with open(here / "config.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    weights_config = cfg["fusion_weights"]

    candidates = [
        {
            "text": "he dresses himself",
            "whisper_score": -2.3,
            "gpt2_score": -1.8,
            "fingerprint_score": -0.9,
        },
        {
            "text": "he addresses himself",
            "whisper_score": -3.1,
            "gpt2_score": -2.1,
            "fingerprint_score": -1.5,
        },
        {
            "text": "e dresse imself",
            "whisper_score": -1.9,
            "gpt2_score": -4.5,
            "fingerprint_score": -2.0,
        },
        {
            "text": "he dresses themself",
            "whisper_score": -3.5,
            "gpt2_score": -2.3,
            "fingerprint_score": -1.2,
        },
    ]

    print("=" * 60)
    print("fusion — local verification")
    print("=" * 60)

    for severity in ["severe", "moderate", "mild"]:
        import copy
        cands = copy.deepcopy(candidates)
        fused = fuse(cands, severity, weights_config)
        w = weights_config[severity]
        print(f"\n  severity={severity}  alpha={w['alpha']} beta={w['beta']} gamma={w['gamma']}")
        for i, c in enumerate(fused, 1):
            print(
                f"    {i}. combined={c['combined']:.4f}  "
                f"(w={c['whisper_score']:.2f} g={c['gpt2_score']:.2f} "
                f"f={c['fingerprint_score']:.2f})  '{c['text']}'"
            )

    print("\nDONE")
