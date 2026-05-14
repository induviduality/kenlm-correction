"""
Three scoring functions:

  gpt2_score(sentences)        – batched length-normalised log-prob from DistilGPT2
  fingerprint_score(...)       – phoneme-alignment log-prob against the speaker fingerprint
  whisper_score(log_prob)      – pass-through from Whisper n-best

DistilGPT2 is loaded once at module import (lazy, on first call to gpt2_score).

Run standalone for local verification:
    python -m src.scoring
"""

from __future__ import annotations

import math
import re
import sys
from pathlib import Path
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# DistilGPT2 — lazy singleton
# ---------------------------------------------------------------------------
_model = None
_tokenizer = None
_device = None
_max_length = 128


def _load_gpt2(model_name: str = "distilgpt2", device_cfg: str = "auto") -> None:
    global _model, _tokenizer, _device, _max_length
    if _model is not None:
        return

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if device_cfg == "auto":
        _device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        _device = device_cfg

    print(f"[scoring] Loading {model_name} on {_device} …", file=sys.stderr)
    _tokenizer = AutoTokenizer.from_pretrained(model_name)
    _tokenizer.pad_token = _tokenizer.eos_token
    _model = AutoModelForCausalLM.from_pretrained(model_name).to(_device)
    _model.eval()
    print("[scoring] Model loaded.", file=sys.stderr)


def gpt2_score(
    sentences: list[str],
    batch_size: int = 40,
    model_name: str = "distilgpt2",
    device_cfg: str = "auto",
    max_length: int = 128,
) -> list[float]:
    """
    Returns length-normalised log-probability for each sentence.
    Higher (less negative) is better.
    Processed in batches for GPU efficiency.
    """
    import torch
    import torch.nn.functional as F

    _load_gpt2(model_name, device_cfg)

    all_scores: list[float] = []

    for i in range(0, len(sentences), batch_size):
        batch = sentences[i : i + batch_size]

        encodings = _tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
            return_attention_mask=True,
        )
        input_ids = encodings["input_ids"].to(_device)
        attention_mask = encodings["attention_mask"].to(_device)

        with torch.no_grad():
            outputs = _model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits  # [B, T, V]

        log_probs = F.log_softmax(logits, dim=-1)  # [B, T, V]

        for j in range(len(batch)):
            ids = input_ids[j]
            mask = attention_mask[j]
            seq_len = int(mask.sum().item())

            if seq_len <= 1:
                all_scores.append(-10.0)
                continue

            # Predict token[1:seq_len] from positions [0:seq_len-1]
            token_log_probs = (
                log_probs[j, : seq_len - 1, :]
                .gather(1, ids[1:seq_len].unsqueeze(-1))
                .squeeze(-1)
            )  # [seq_len-1]

            score = token_log_probs.mean().item()
            all_scores.append(score)

    return all_scores


# ---------------------------------------------------------------------------
# g2p helper — lazy singleton
# ---------------------------------------------------------------------------
_g2p = None


def _get_g2p():
    global _g2p
    if _g2p is None:
        import nltk
        nltk.download("averaged_perceptron_tagger_eng", quiet=True)
        nltk.download("cmudict", quiet=True)
        from g2p_en import G2p

        _g2p = G2p()
    return _g2p


def _strip_stress(phoneme: str) -> str:
    return re.sub(r"\d", "", phoneme)


def _phonemize(text: str) -> list[str]:
    """Convert text to a list of Arpabet phonemes (stress stripped, no spaces)."""
    g2p = _get_g2p()
    raw = g2p(text.lower())
    # Strip stress FIRST, then filter — avoids dropping "IH1", "EH1" etc.
    result = []
    for p in raw:
        stripped = _strip_stress(p)
        if stripped and stripped.isalpha():
            result.append(stripped)
    return result


# ---------------------------------------------------------------------------
# Levenshtein alignment
# ---------------------------------------------------------------------------

def _levenshtein_align(
    seq1: list[str], seq2: list[str]
) -> list[tuple[Optional[str], Optional[str]]]:
    """
    Returns list of (elem_from_seq1_or_None, elem_from_seq2_or_None).
    None in first slot = insertion from seq2 (no intended match).
    None in second slot = deletion from seq1 (intended phoneme was dropped).
    """
    m, n = len(seq1), len(seq2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if seq1[i - 1] == seq2[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,       # deletion (intended not observed)
                dp[i][j - 1] + 1,       # insertion (observed not intended)
                dp[i - 1][j - 1] + cost,
            )

    alignment: list[tuple] = []
    i, j = m, n
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            cost = 0 if seq1[i - 1] == seq2[j - 1] else 1
            if dp[i][j] == dp[i - 1][j - 1] + cost:
                alignment.append((seq1[i - 1], seq2[j - 1]))
                i -= 1
                j -= 1
                continue
        if i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            alignment.append((seq1[i - 1], None))  # intended dropped
            i -= 1
        else:
            alignment.append((None, seq2[j - 1]))   # unexpected observed
            j -= 1

    return list(reversed(alignment))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fingerprint_score(
    hypothesis: str,
    observed_phonemes: list[str],
    fingerprint: dict,
    smoothing: float = 0.01,
) -> float:
    """
    1. Phonemise hypothesis → intended phoneme sequence.
    2. Levenshtein-align intended ↔ observed.
    3. For each aligned pair (intended, observed):
         - if match: score = log(1 - total_error_rate_for_this_phoneme)
         - if intended → observed substitution: score = log(error_map[key])
         - if intended → None (dropped):   score = log(error_map["X>SIL"])
         - if None → observed (insertion): skip
    4. Return length-normalised mean log-prob.
    """
    error_map: dict[str, float] = fingerprint.get("error_map", {})

    # Pre-compute total error rate per intended phoneme
    total_err: dict[str, float] = {}
    for key, prob in error_map.items():
        parts = key.split(">")
        if len(parts) == 2:
            intended_p = parts[0]
            total_err[intended_p] = total_err.get(intended_p, 0.0) + prob

    intended_phonemes = _phonemize(hypothesis)
    if not intended_phonemes:
        return math.log(smoothing)

    alignment = _levenshtein_align(intended_phonemes, observed_phonemes)

    log_probs: list[float] = []
    for intended_p, observed_p in alignment:
        if intended_p is None:
            # Unexpected observed phoneme — skip (insertion noise)
            continue

        if observed_p is None:
            # Intended phoneme was dropped → "X>SIL"
            key = f"{intended_p}>SIL"
            prob = error_map.get(key, smoothing)
        elif intended_p == observed_p:
            # Correct production
            prob = max(1.0 - total_err.get(intended_p, 0.0), smoothing)
        else:
            # Substitution
            key = f"{intended_p}>{observed_p}"
            prob = error_map.get(key, smoothing)

        log_probs.append(math.log(max(prob, 1e-10)))

    if not log_probs:
        return math.log(smoothing)
    return float(np.mean(log_probs))


def whisper_score(log_prob: float) -> float:
    """Pass-through: Whisper n-best scores are already log-probabilities."""
    return float(log_prob)


# ---------------------------------------------------------------------------
# Word-validity score — Brown corpus frequency list
#
# Uses words that appear ≥ 2 times in the Brown corpus (a representative
# sample of American English). This captures inflected forms like "dresses",
# "himself", "addresses" but excludes obscure proper nouns and non-words like
# "eade", "reh", "dresse", "imself".
# ---------------------------------------------------------------------------
_brown_word_set: set[str] | None = None


def _get_brown_word_set() -> set[str]:
    global _brown_word_set
    if _brown_word_set is not None:
        return _brown_word_set
    import nltk
    for corpus in ("brown", "averaged_perceptron_tagger_eng"):
        try:
            nltk.data.find(f"corpora/{corpus}" if corpus == "brown" else f"taggers/{corpus}")
        except LookupError:
            nltk.download(corpus, quiet=True)
    from nltk.corpus import brown as brown_corpus
    freq: dict[str, int] = {}
    for word in brown_corpus.words():
        w = word.lower()
        freq[w] = freq.get(w, 0) + 1
    # Keep words that appear at least twice (filters out hapax legomena and names)
    _brown_word_set = set(w for w, c in freq.items() if c >= 2)
    return _brown_word_set


_SINGLE_LETTER_WORDS = {"a", "i"}  # only valid standalone single-letter words


def word_validity_score(text: str) -> float:
    """
    Fraction of words in `text` that are common English.

    Rules:
    - Single-letter tokens: only "a" and "i" count as valid.
    - Multi-letter tokens: must appear ≥ 2 times in the Brown corpus.

    Correctly scores:
      "he dresses himself"    → 1.0  (all in Brown)
      "e dresse imself"       → 0.0  ("e" excluded; "dresse"/"imself" not in Brown)
      "eade reh seam shh elf" → 0.4  ("seam" + "elf" in Brown)
    """
    word_set = _get_brown_word_set()
    tokens = re.sub(r"[^\w\s]", "", text.lower()).split()
    if not tokens:
        return 0.0

    def _valid(t: str) -> bool:
        if len(t) == 1:
            return t in _SINGLE_LETTER_WORDS
        return t in word_set

    valid = sum(1 for t in tokens if _valid(t))
    return valid / len(tokens)


# ---------------------------------------------------------------------------
# Local verification
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json
    import time

    here = Path(__file__).parent.parent
    fp_path = here / "fingerprints" / "TORGO_F01.json"
    with open(fp_path, encoding="utf-8") as f:
        fingerprint = json.load(f)

    sentences = [
        "he dresses himself",
        "he addresses himself",
        "e dresse imself",
        "they dress themselves",
        "he processes himself",
    ]
    observed = ["IY", "D", "R", "EH", "S", "IY", "M", "S", "EH", "L", "F"]

    print("=" * 60)
    print("scoring — local verification")
    print("=" * 60)

    # --- GPT2 ---
    print("\n[1] GPT2 scores (DistilGPT2, length-normalised log-prob):")
    t0 = time.time()
    gpt2_scores = gpt2_score(sentences)
    elapsed = time.time() - t0
    for sent, s in zip(sentences, gpt2_scores):
        print(f"  {s:7.4f}  '{sent}'")
    print(f"  elapsed: {elapsed*1000:.1f} ms for {len(sentences)} sentences")

    # --- Fingerprint ---
    print("\n[2] Fingerprint scores (phoneme alignment vs. observed):")
    for sent in sentences:
        intended = _phonemize(sent)
        fp_s = fingerprint_score(sent, observed, fingerprint)
        print(f"  {fp_s:7.4f}  '{sent}'  (intended: {intended})")

    # --- Whisper pass-through ---
    print("\n[3] Whisper score pass-through:")
    print(f"  whisper_score(-2.3) = {whisper_score(-2.3)}")

    print("\nDONE")
