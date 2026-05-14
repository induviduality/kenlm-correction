"""
Phoneme-driven candidate generation.

Algorithm:
1. Build inverse error map from fingerprint["error_map"]:
   given observed phoneme O, retrieve the top-K intended phonemes most likely
   to have produced O for this speaker.
2. Beam-search over observed_phonemes to build top-K intended phoneme sequences.
3. Segment each intended sequence into English words via CMU dict reverse lookup.
4. Return up to max_candidates unique text candidates with generation scores.

Run standalone for local verification:
    python -m src.candidate_gen
"""

from __future__ import annotations

import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

import nltk

# ---------------------------------------------------------------------------
# One-time globals (built at first use, shared across calls)
# ---------------------------------------------------------------------------
_phoneme_to_words: dict[tuple, list[str]] | None = None


def _strip_stress(phoneme: str) -> str:
    return re.sub(r"\d", "", phoneme)


def _ensure_cmu_dict() -> dict[tuple, list[str]]:
    global _phoneme_to_words
    if _phoneme_to_words is not None:
        return _phoneme_to_words

    try:
        nltk.data.find("corpora/cmudict")
    except LookupError:
        print("[candidate_gen] Downloading NLTK cmudict …", file=sys.stderr)
        nltk.download("cmudict", quiet=True)

    cmu = nltk.corpus.cmudict.dict()
    rev: dict[tuple, list[str]] = {}
    for word, pron_list in cmu.items():
        for pron in pron_list:
            key = tuple(_strip_stress(p) for p in pron)
            rev.setdefault(key, []).append(word)

    _phoneme_to_words = rev
    return rev


# ---------------------------------------------------------------------------
# Inverse error map
# ---------------------------------------------------------------------------

def _build_inverse_map(fingerprint: dict) -> dict[str, list[tuple[str, float]]]:
    """
    Returns: { observed_phoneme: [(intended_phoneme, probability), ...] }
    sorted by probability descending.

    Sources:
    - error_map "intended>observed": P(observed | intended)
    - identity: P(correct | intended) = 1 - sum_of_errors_for_that_intended
    """
    error_map: dict[str, float] = fingerprint.get("error_map", {})
    intended_counts: dict[str, int] = fingerprint.get("intended_phoneme_counts", {})

    # Accumulate total error rate per intended phoneme
    total_err_for: dict[str, float] = defaultdict(float)
    inverse: dict[str, dict[str, float]] = defaultdict(dict)  # obs -> {int: best_prob}

    for key, prob in error_map.items():
        parts = key.split(">")
        if len(parts) != 2:
            continue
        intended, observed = parts
        total_err_for[intended] += prob
        # Keep highest prob if the same (intended, observed) pair appears twice
        prev = inverse[observed].get(intended, 0.0)
        if prob > prev:
            inverse[observed][intended] = prob

    # Add identity mapping for every known intended phoneme
    for phoneme in intended_counts:
        correct_prob = max(1.0 - total_err_for.get(phoneme, 0.0), 0.05)
        prev = inverse[phoneme].get(phoneme, 0.0)
        if correct_prob > prev:
            inverse[phoneme][phoneme] = correct_prob

    # Sort each list by probability descending
    result: dict[str, list[tuple[str, float]]] = {}
    for obs, cands in inverse.items():
        result[obs] = sorted(cands.items(), key=lambda x: x[1], reverse=True)

    return result


# ---------------------------------------------------------------------------
# Beam search over observed phoneme sequence
# ---------------------------------------------------------------------------

def _beam_search(
    observed: list[str],
    inverse_map: dict[str, list[tuple[str, float]]],
    top_k: int,
    beam_width: int,
) -> list[tuple[list[str], float]]:
    """
    State: (intended_seq_so_far, cumulative_log_score)
    At each position, expand with top-k intended phonemes for the observed phoneme.
    SIL as intended is skipped (silence is not a real phoneme in words).
    """
    beam: list[tuple[list[str], float]] = [([], 0.0)]

    for obs_p in observed:
        candidates = inverse_map.get(obs_p, [(obs_p, 0.5)])[:top_k]
        new_beam: list[tuple[list[str], float]] = []

        for seq, score in beam:
            for intended_p, prob in candidates:
                if intended_p == "SIL":
                    # Observed SIL means a phoneme was dropped; include the
                    # most-likely intended phoneme as an insertion in the
                    # intended sequence so the word can still be segmented.
                    continue
                new_score = score + math.log(max(prob, 1e-10))
                new_beam.append((seq + [intended_p], new_score))

        new_beam.sort(key=lambda x: x[1], reverse=True)
        beam = new_beam[:beam_width]

    return beam


# ---------------------------------------------------------------------------
# DP word segmentation over phoneme sequence
# ---------------------------------------------------------------------------

def _segment_to_words(
    phoneme_seq: list[str],
    phoneme_to_words: dict[tuple, list[str]],
    max_word_phonemes: int = 10,
    max_paths_per_pos: int = 4,
) -> list[list[str]]:
    """
    Dynamic-programming segmentation of a phoneme sequence into English words.
    Returns list of word-lists (each list is one complete segmentation).
    """
    n = len(phoneme_seq)
    # dp[i] = list of word-sequences that consume phoneme_seq[0:i]
    dp: list[list[list[str]] | None] = [None] * (n + 1)
    dp[0] = [[]]

    for i in range(n):
        if dp[i] is None:
            continue
        max_len = min(max_word_phonemes, n - i)
        for length in range(1, max_len + 1):
            key = tuple(phoneme_seq[i : i + length])
            if key not in phoneme_to_words:
                continue
            words = phoneme_to_words[key]
            best_word = words[0]
            if dp[i + length] is None:
                dp[i + length] = []
            for path in dp[i][:max_paths_per_pos]:
                if len(dp[i + length]) < max_paths_per_pos * 2:
                    dp[i + length].append(path + [best_word])

    return dp[n] if dp[n] is not None else []


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_candidates(
    observed_phonemes: list[str],
    fingerprint: dict,
    top_k_per_phoneme: int = 3,
    beam_width: int = 10,
    max_candidates: int = 20,
) -> list[dict]:
    """
    Returns: [{"text": str, "generation_score": float, "source": "phoneme"}, ...]

    generation_score is length-normalized log-probability of the beam path.
    """
    phoneme_to_words = _ensure_cmu_dict()
    inverse_map = _build_inverse_map(fingerprint)

    beam = _beam_search(observed_phonemes, inverse_map, top_k_per_phoneme, beam_width)

    candidates: list[dict] = []
    seen: set[str] = set()

    for intended_seq, log_score in beam:
        if not intended_seq:
            continue
        norm_score = log_score / len(intended_seq)
        segmentations = _segment_to_words(intended_seq, phoneme_to_words)
        for words in segmentations:
            text = " ".join(words)
            if not text or text in seen:
                continue
            seen.add(text)
            candidates.append(
                {"text": text, "generation_score": norm_score, "source": "phoneme"}
            )
            if len(candidates) >= max_candidates:
                return candidates

    return candidates


# ---------------------------------------------------------------------------
# Local verification
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import os

    # Locate fingerprint relative to this file
    here = Path(__file__).parent.parent
    fp_path = here / "fingerprints" / "TORGO_F01.json"
    with open(fp_path, encoding="utf-8") as f:
        fingerprint = json.load(f)

    # From the implementation plan: "he dresses himself"
    observed_phonemes = ["IY", "D", "R", "EH", "S", "IY", "M", "S", "EH", "L", "F"]

    print("=" * 60)
    print("candidate_gen — local verification")
    print(f"Speaker: {fingerprint['speaker_id']}  severity: {fingerprint['severity']}")
    print(f"Observed phonemes: {observed_phonemes}")
    print("=" * 60)

    print("\n[1] Inverse error map (top 5 entries):")
    inv = _build_inverse_map(fingerprint)
    for obs, cands in list(inv.items())[:5]:
        print(f"  observed={obs:6s}  →  {cands[:3]}")

    print("\n[2] Beam search (top 5 sequences):")
    inv_map = _build_inverse_map(fingerprint)
    beam = _beam_search(observed_phonemes, inv_map, top_k=3, beam_width=10)
    for seq, score in beam[:5]:
        print(f"  score={score:.3f}  seq={seq}")

    print("\n[3] Generated candidates:")
    candidates = generate_candidates(observed_phonemes, fingerprint, max_candidates=20)
    if candidates:
        for i, c in enumerate(candidates, 1):
            print(f"  {i:2d}. score={c['generation_score']:.3f}  text='{c['text']}'")
    else:
        print("  (no candidates — CMU dict segmentation found no paths)")
        print("  Beam sequences for debug:")
        for seq, score in beam[:3]:
            print(f"    {seq}")

    print("\nDONE")
