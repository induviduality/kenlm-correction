"""
Score fusion: combines whisper, gpt2, and fingerprint scores into a single
combined score using severity-adapted weights from config.

combined = α·whisper + β·gpt2 + γ·fingerprint

Run standalone for local verification:
    python -m src.fusion
"""

from __future__ import annotations

from pathlib import Path


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
    # Normalise severity key; fall back to "moderate" if unknown
    sev = severity.lower()
    if sev not in weights_config:
        sev = "moderate"
    w = weights_config[sev]
    alpha: float = w["alpha"]   # whisper weight
    beta: float = w["beta"]     # gpt2 weight
    gamma: float = w["gamma"]   # fingerprint weight

    for c in candidates:
        c["combined"] = (
            alpha * c.get("whisper_score", 0.0)
            + beta * c.get("gpt2_score", 0.0)
            + gamma * c.get("fingerprint_score", 0.0)
        )

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
