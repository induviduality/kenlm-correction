"""
End-to-end correction pipeline.

Flow:
  1. Generate phoneme-derived candidates (candidate_gen)
  2. Merge with whisper_nbest; deduplicate by normalised text
  3. Batch-score all candidates with GPT2 (single GPU call)
  4. Score each with fingerprint plausibility
  5. Fuse scores using severity-adapted weights
  6. Assess confidence
  7. Return output contract

Output contract:
  {
    "status":        "confident" | "ambiguous",
    "corrected":     str,
    "alternatives":  [{"text": str, "score": float, "rank": int}, ...],
    "candidates":    [...],   # full ranked list for debug
    "selected_score": float,
    "confidence":    float,
  }

Run standalone for local verification:
    python -m src.pipeline
"""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path

from . import candidate_gen as cgen
from . import scoring
from . import fusion as fus
from . import confidence as conf


# ---------------------------------------------------------------------------
# Text normalisation (same as confidence.py — kept local to avoid circular import)
# ---------------------------------------------------------------------------

def _normalize(text: str) -> str:
    text = text.lower()
    text = unicodedata.normalize("NFKD", text)
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def correct(
    whisper_nbest: list[dict],
    observed_phonemes: list[str],
    fingerprint: dict,
    config: dict,
) -> dict:
    """
    Parameters
    ----------
    whisper_nbest : list[dict]
        Each entry: {"text": str, "score": float}
        score = Whisper log-probability (higher = better).
    observed_phonemes : list[str]
        Arpabet phoneme sequence from the phoneme recognizer.
    fingerprint : dict
        Full speaker fingerprint JSON.
    config : dict
        Full config.yaml loaded as dict.

    Returns
    -------
    Output contract dict (see module docstring).
    """
    cgen_cfg = config.get("candidate_generation", {})
    scoring_cfg = config.get("scoring", {})
    model_cfg = config.get("model", {})
    severity = fingerprint.get("severity", "moderate")

    # ------------------------------------------------------------------
    # 1. Phoneme-derived candidates
    # ------------------------------------------------------------------
    phoneme_candidates = cgen.generate_candidates(
        observed_phonemes=observed_phonemes,
        fingerprint=fingerprint,
        top_k_per_phoneme=cgen_cfg.get("top_k_per_phoneme", 3),
        beam_width=cgen_cfg.get("beam_width", 10),
        max_candidates=cgen_cfg.get("max_candidates", 20),
    )

    # ------------------------------------------------------------------
    # 2. Merge whisper_nbest + phoneme candidates; deduplicate
    # ------------------------------------------------------------------
    merged: dict[str, dict] = {}  # norm_text → candidate dict

    for entry in whisper_nbest:
        text = entry.get("text", "").strip()
        if not text:
            continue
        key = _normalize(text)
        if key not in merged or entry.get("score", -999) > merged[key].get("whisper_score", -999):
            merged[key] = {
                "text": text,
                "whisper_score": scoring.whisper_score(entry.get("score", 0.0)),
                "source": "whisper",
            }

    for c in phoneme_candidates:
        key = _normalize(c["text"])
        if key not in merged:
            merged[key] = {
                "text": c["text"],
                "whisper_score": 0.0,   # not in whisper n-best
                "source": "phoneme",
                "generation_score": c.get("generation_score", 0.0),
            }

    all_candidates = list(merged.values())

    # ------------------------------------------------------------------
    # 3. Batch GPT2 scoring
    # ------------------------------------------------------------------
    texts = [c["text"] for c in all_candidates]
    gpt2_scores = scoring.gpt2_score(
        texts,
        batch_size=scoring_cfg.get("gpt2_batch_size", 40),
        model_name=model_cfg.get("gpt2_name", "distilgpt2"),
        device_cfg=model_cfg.get("device", "auto"),
        max_length=scoring_cfg.get("gpt2_max_length", 128),
    )
    for c, s in zip(all_candidates, gpt2_scores):
        c["gpt2_score"] = s

    # ------------------------------------------------------------------
    # 4. Fingerprint scoring
    # ------------------------------------------------------------------
    smoothing = scoring_cfg.get("fingerprint_smoothing", 0.01)
    for c in all_candidates:
        c["fingerprint_score"] = scoring.fingerprint_score(
            hypothesis=c["text"],
            observed_phonemes=observed_phonemes,
            fingerprint=fingerprint,
            smoothing=smoothing,
        )

    # ------------------------------------------------------------------
    # 5. Fuse scores
    # ------------------------------------------------------------------
    ranked = fus.fuse(all_candidates, severity, config["fusion_weights"])

    # Attach rank
    for i, c in enumerate(ranked):
        c["rank"] = i + 1

    # ------------------------------------------------------------------
    # 6. Confidence assessment
    # ------------------------------------------------------------------
    conf_result = conf.assess_confidence(ranked, config.get("confidence", {}))

    # ------------------------------------------------------------------
    # 7. Build output contract
    # ------------------------------------------------------------------
    top1 = ranked[0] if ranked else {}
    return {
        "status": conf_result["status"],
        "corrected": top1.get("text", ""),
        "alternatives": conf_result["alternatives"],
        "candidates": [
            {
                "text": c["text"],
                "rank": c["rank"],
                "combined": round(c.get("combined", 0.0), 4),
                "whisper_score": round(c.get("whisper_score", 0.0), 4),
                "gpt2_score": round(c.get("gpt2_score", 0.0), 4),
                "fingerprint_score": round(c.get("fingerprint_score", 0.0), 4),
                "source": c.get("source", ""),
            }
            for c in ranked
        ],
        "selected_score": round(top1.get("combined", 0.0), 4),
        "confidence": conf_result["confidence"],
        "trigger_reason": conf_result.get("trigger_reason"),
    }


# ---------------------------------------------------------------------------
# Local verification
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import yaml
    import pprint

    here = Path(__file__).parent.parent
    with open(here / "config.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    with open(here / "fingerprints" / "TORGO_F01.json", encoding="utf-8") as f:
        fingerprint = json.load(f)

    # Load sample input from demo_data if available, else use hardcoded
    sample_path = here / "demo_data" / "sample_inputs.json"
    if sample_path.exists():
        with open(sample_path, encoding="utf-8") as f:
            samples = json.load(f)
        sample = samples[0]
        whisper_nbest = sample["whisper_nbest"]
        observed_phonemes = sample["observed_phonemes"]
        print(f"Loaded sample: {sample.get('reference', 'unknown')}")
    else:
        whisper_nbest = [
            {"text": "e dresse imself", "score": -2.3},
            {"text": "he dresses himself", "score": -2.8},
            {"text": "he addresses himself", "score": -3.1},
            {"text": "he dresses themself", "score": -3.4},
            {"text": "he processes himself", "score": -3.8},
        ]
        observed_phonemes = ["IY", "D", "R", "EH", "S", "IY", "M", "S", "EH", "L", "F"]

    print("=" * 60)
    print("pipeline — local verification")
    print(f"Speaker: {fingerprint['speaker_id']}  severity: {fingerprint['severity']}")
    print(f"Observed phonemes: {observed_phonemes}")
    print(f"Whisper top-1: '{whisper_nbest[0]['text']}'")
    print("=" * 60)

    result = correct(whisper_nbest, observed_phonemes, fingerprint, cfg)

    print(f"\nStatus     : {result['status']}")
    print(f"Corrected  : '{result['corrected']}'")
    print(f"Confidence : {result['confidence']}")
    print(f"Trigger    : {result.get('trigger_reason')}")
    print(f"\nAlternatives:")
    for alt in result["alternatives"]:
        print(f"  rank={alt['rank']}  score={alt['score']}  '{alt['text']}'")
    print(f"\nTop candidates (all {len(result['candidates'])}):")
    for c in result["candidates"][:8]:
        print(
            f"  #{c['rank']}  combined={c['combined']:7.4f}  "
            f"w={c['whisper_score']:6.3f} g={c['gpt2_score']:6.3f} "
            f"f={c['fingerprint_score']:6.3f}  [{c['source']}] '{c['text']}'"
        )

    print("\nDONE")
