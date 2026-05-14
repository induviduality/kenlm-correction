"""
Decode observed phonemes from an audio file using wav2vec2 CTC.

Converts audio → Arpabet phoneme sequence for building eval test cases.
Based on the wav2vec2-lv-60-espeak-cv-ft model, with IPA→Arpabet mapping.

Usage:
    python eval/audio_to_phonemes.py path/to/audio.wav
    python eval/audio_to_phonemes.py path/to/audio.wav --format json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torchaudio

# ---------------------------------------------------------------------------
# IPA → Arpabet mapping (wav2vec2 outputs IPA; our pipeline uses Arpabet)
# ---------------------------------------------------------------------------

IPA_TO_ARPABET = {
    # Vowels
    "i": "IY", "ɪ": "IH", "e": "EY", "ɛ": "EH", "æ": "AE",
    "ɑ": "AA", "ɔ": "AO", "o": "OW", "ʊ": "UH", "u": "UW",
    "ʌ": "AH", "ə": "AH", "ɝ": "ER", "ɚ": "ER",
    # Diphthongs
    "aɪ": "AY", "aʊ": "AW", "ɔɪ": "OY", "eɪ": "EY", "oʊ": "OW",
    # Consonants
    "p": "P", "b": "B", "t": "T", "d": "D", "k": "K", "ɡ": "G",
    "g": "G", "f": "F", "v": "V", "θ": "TH", "ð": "DH",
    "s": "S", "z": "Z", "ʃ": "SH", "ʒ": "ZH",
    "h": "HH", "m": "M", "n": "N", "ŋ": "NG",
    "l": "L", "ɹ": "R", "r": "R", "w": "W", "j": "Y",
    "tʃ": "CH", "dʒ": "JH",
    # Flap
    "ɾ": "D",
}


def _ipa_to_arpabet(ipa_tokens: list[str]) -> list[str]:
    """Convert a list of IPA phoneme tokens to Arpabet."""
    result = []
    i = 0
    while i < len(ipa_tokens):
        tok = ipa_tokens[i]
        # Try two-character combos for affricates/diphthongs
        if i + 1 < len(ipa_tokens):
            combo = tok + ipa_tokens[i + 1]
            if combo in IPA_TO_ARPABET:
                result.append(IPA_TO_ARPABET[combo])
                i += 2
                continue
        if tok in IPA_TO_ARPABET:
            result.append(IPA_TO_ARPABET[tok])
        # Skip unmapped tokens (stress marks, spaces, etc.)
        i += 1
    return result


# ---------------------------------------------------------------------------
# CTC decoding
# ---------------------------------------------------------------------------

def decode_phonemes(wav_path: str, model_id: str = "facebook/wav2vec2-lv-60-espeak-cv-ft") -> list[str]:
    """Decode audio to Arpabet phonemes via wav2vec2 CTC."""
    from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
    from transformers.models.wav2vec2_phoneme.tokenization_wav2vec2_phoneme import (
        Wav2Vec2PhonemeCTCTokenizer,
    )

    # Bypass phonemizer backend check (only need decoding)
    Wav2Vec2PhonemeCTCTokenizer.init_backend = lambda self, *a, **kw: None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    processor = Wav2Vec2Processor.from_pretrained(model_id)
    model = Wav2Vec2ForCTC.from_pretrained(model_id).to(device)
    model.eval()

    # Load and resample audio
    import soundfile as sf
    audio_data, sr = sf.read(wav_path)
    waveform = torch.from_numpy(audio_data).float()
    if waveform.ndim > 1:
        waveform = waveform.mean(dim=-1)
    if sr != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
        waveform = resampler(waveform.unsqueeze(0)).squeeze(0)

    # CTC forward pass
    input_values = processor(
        waveform.numpy(), sampling_rate=16000, return_tensors="pt"
    ).input_values.to(device)

    with torch.inference_mode():
        logits = model(input_values).logits[0]

    # Greedy CTC decode: collapse consecutive duplicates and blanks
    predicted_ids = torch.argmax(logits, dim=-1)
    collapsed = []
    prev_id = -1
    for pid in predicted_ids.tolist():
        if pid != prev_id and pid != 0:
            collapsed.append(pid)
        prev_id = pid

    # Decode to IPA tokens
    ipa_tokens = []
    for token_id in collapsed:
        phoneme = processor.decode([token_id]).strip()
        if phoneme and phoneme not in ("<pad>", "<s>", "</s>", "|", " "):
            ipa_tokens.append(phoneme)

    # Convert IPA → Arpabet
    return _ipa_to_arpabet(ipa_tokens)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Decode Arpabet phonemes from an audio file"
    )
    parser.add_argument("audio", type=str, help="Path to audio file (wav, flac, etc.)")
    parser.add_argument(
        "--model",
        type=str,
        default="facebook/wav2vec2-lv-60-espeak-cv-ft",
        help="HuggingFace model ID for phoneme CTC",
    )
    parser.add_argument(
        "--format",
        choices=["text", "json"],
        default="text",
        help="Output format: 'text' for space-separated, 'json' for list",
    )
    args = parser.parse_args()

    phonemes = decode_phonemes(args.audio, model_id=args.model)

    if args.format == "json":
        print(json.dumps(phonemes))
    else:
        print(" ".join(phonemes))


if __name__ == "__main__":
    main()
