"""
End-to-end CLI: audio file + speaker fingerprint -> corrected transcript.

Usage:
    uv run python transcribe.py <audio_file> <fingerprint_key> [--model base] [--verbose]

Examples:
    uv run python transcribe.py my_audio.wav TORGO_F01
    uv run python transcribe.py my_audio.wav TORGO_F01 --model small --verbose

Fingerprint keys match filenames in fingerprints/ (without .json):
    TORGO_F01  TORGO_F03  TORGO_M01  TORGO_M02  TORGO_M03  TORGO_M04  TORGO_M05
    UASpeech_F02  UASpeech_F03  UASpeech_F04  UASpeech_M01  UASpeech_M04 ...

Flow:
    1. Whisper (multi-temperature) -> n-best list
    2. g2p_en on Whisper top-1 -> observed phonemes (Arpabet)
    3. Correction pipeline -> ranked candidates + confidence
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))


# ---------------------------------------------------------------------------
# Step 1: Whisper n-best via multiple temperatures
# ---------------------------------------------------------------------------

def _whisper_nbest(audio_path: str, model_name: str) -> list[dict]:
    import whisper

    print(f"[1/3] Running Whisper ({model_name}) on {audio_path} ...", file=sys.stderr)
    model = whisper.load_model(model_name)

    nbest: list[dict] = []
    seen: set[str] = set()
    for temp in [0.0, 0.2, 0.4, 0.6, 0.8]:
        result = model.transcribe(audio_path, temperature=temp, language="en")
        text = result["text"].strip()
        score = float(result.get("avg_logprob", -2.0))
        key = text.lower()
        if text and key not in seen:
            seen.add(key)
            nbest.append({"text": text, "score": score})

    if not nbest:
        print("ERROR: Whisper produced no output.", file=sys.stderr)
        sys.exit(1)

    print(f"  Whisper top-1: '{nbest[0]['text']}' (score={nbest[0]['score']:.3f})", file=sys.stderr)
    print(f"  {len(nbest)} unique hypotheses", file=sys.stderr)
    return nbest


# ---------------------------------------------------------------------------
# Step 2: Observed phonemes from Whisper top-1 via g2p_en
# ---------------------------------------------------------------------------

def _phonemize(text: str) -> list[str]:
    import nltk
    nltk.download("averaged_perceptron_tagger_eng", quiet=True)
    nltk.download("cmudict", quiet=True)
    from g2p_en import G2p

    g2p = G2p()
    raw = g2p(text.lower())
    phonemes = []
    for p in raw:
        stripped = re.sub(r"\d", "", p)
        if stripped and stripped.isalpha():
            phonemes.append(stripped)
    return phonemes


def _get_observed_phonemes(nbest: list[dict]) -> list[str]:
    print("[2/3] Deriving observed phonemes from Whisper top-1 via g2p ...", file=sys.stderr)
    phonemes = _phonemize(nbest[0]["text"])
    print(f"  Phonemes: {' '.join(phonemes)}", file=sys.stderr)
    return phonemes


# ---------------------------------------------------------------------------
# Step 3: Correction pipeline
# ---------------------------------------------------------------------------

def _run_pipeline(
    nbest: list[dict],
    phonemes: list[str],
    fingerprint: dict,
    config: dict,
) -> dict:
    from src import pipeline
    print("[3/3] Running correction pipeline ...", file=sys.stderr)
    return pipeline.correct(nbest, phonemes, fingerprint, config)


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

def _print_result(result: dict, verbose: bool) -> None:
    status = result["status"]
    corrected = result["corrected"]
    confidence = result["confidence"]

    print()
    print("=" * 60)
    print(f"  Corrected : {corrected}")
    print(f"  Status    : {status}  (confidence={confidence:.3f})")
    if result.get("trigger_reason"):
        print(f"  Trigger   : {result['trigger_reason']}")
    print("=" * 60)

    alts = result["alternatives"]
    if status == "ambiguous" and len(alts) > 1:
        print("\nAlternatives (review and pick the best):")
        for a in alts:
            marker = ">>>" if a["rank"] == 1 else "   "
            print(f"  {marker} #{a['rank']}  score={a['score']:.4f}  '{a['text']}'")

    if verbose:
        print("\nAll candidates:")
        for c in result["candidates"]:
            print(
                f"  #{c['rank']:2d}  combined={c['combined']:.4f}"
                f"  w={c['whisper_score']:6.3f}"
                f"  g={c['gpt2_score']:6.3f}"
                f"  f={c['fingerprint_score']:6.3f}"
                f"  v={c.get('word_validity_score', 0):.2f}"
                f"  [{c['source']}]  '{c['text']}'"
            )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Correct dysarthric speech: audio -> corrected transcript",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("audio", help="Path to audio file (.wav, .mp3, .m4a, ...)")
    parser.add_argument(
        "speaker",
        help="Fingerprint key, e.g. TORGO_F01. Run with --list-speakers to see all.",
    )
    parser.add_argument(
        "--model",
        default="base",
        choices=["tiny", "base", "small", "medium", "large"],
        help="Whisper model size (default: base)",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Show full candidate table")
    parser.add_argument(
        "--list-speakers",
        action="store_true",
        help="Print available fingerprint keys and exit",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output result as JSON instead of human-readable text",
    )
    args = parser.parse_args()

    # Load config + fingerprints
    with open(HERE / "config.yaml", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    fingerprints: dict[str, dict] = {}
    for fp_file in sorted((HERE / "fingerprints").glob("*.json")):
        with open(fp_file, encoding="utf-8") as f:
            fingerprints[fp_file.stem] = json.load(f)

    if args.list_speakers:
        print("Available speaker fingerprints:")
        for k in sorted(fingerprints):
            sev = fingerprints[k].get("severity", "?")
            print(f"  {k:<30}  severity={sev}")
        return

    if args.speaker not in fingerprints:
        print(f"ERROR: fingerprint '{args.speaker}' not found.", file=sys.stderr)
        print(f"Run with --list-speakers to see available keys.", file=sys.stderr)
        sys.exit(1)

    if not Path(args.audio).exists():
        print(f"ERROR: audio file not found: {args.audio}", file=sys.stderr)
        sys.exit(1)

    fingerprint = fingerprints[args.speaker]

    nbest = _whisper_nbest(args.audio, args.model)
    phonemes = _get_observed_phonemes(nbest)
    result = _run_pipeline(nbest, phonemes, fingerprint, config)

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        _print_result(result, args.verbose)


if __name__ == "__main__":
    main()
