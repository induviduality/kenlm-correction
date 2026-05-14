"""
Script 03 — Observed Phonemes (wav2vec2 CTC)

For each calibration utterance, decode the full audio into an observed
phoneme sequence using wav2vec2's CTC phoneme model.

This replaces the old segment-based decoding approach which relied on
MMS_FA's time windows and caused false silence detections for dysarthric
speech with irregular pacing.

Usage:
    python scripts/03_observed_phonemes.py --speaker F03
"""

import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torchaudio
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from transformers.models.wav2vec2_phoneme.tokenization_wav2vec2_phoneme import Wav2Vec2PhonemeCTCTokenizer

# Monkey patch init_backend to bypass phonemizer check since we only need decoding
Wav2Vec2PhonemeCTCTokenizer.init_backend = lambda self, *args, **kwargs: None


def parse_args():
    parser = argparse.ArgumentParser(description="Decode observed phonemes from full audio via CTC")
    parser.add_argument("--speaker", type=str, required=True, help="Speaker ID, e.g. F01, M04")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset, e.g. TORGO, UASpeech")
    parser.add_argument(
        "--output_root",
        type=str,
        default=None,
        help="Output root directory (default: outputs/ in project root)",
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default="facebook/wav2vec2-lv-60-espeak-cv-ft",
        help="HuggingFace model ID for phoneme CTC model",
    )
    return parser.parse_args()


def decode_full_audio(
    waveform: torch.Tensor,
    sr: int,
    processor: Wav2Vec2Processor,
    model: Wav2Vec2ForCTC,
    device: torch.device,
) -> list[str]:
    """
    Decode the full audio waveform into a sequence of phonemes via CTC greedy decode.
    Returns a list of eSpeak IPA phoneme tokens.
    """
    input_values = processor(
        waveform.numpy(), sampling_rate=sr, return_tensors="pt"
    ).input_values.to(device)

    with torch.inference_mode():
        logits = model(input_values).logits[0]  # (T, vocab_size)

    # Greedy CTC decode
    predicted_ids = torch.argmax(logits, dim=-1)

    # CTC collapse: remove consecutive duplicates and blanks (id=0)
    collapsed = []
    prev_id = -1
    for pid in predicted_ids.tolist():
        if pid != prev_id and pid != 0:  # 0 is blank/pad
            collapsed.append(pid)
        prev_id = pid

    # Decode token IDs to phoneme strings
    phonemes = []
    for token_id in collapsed:
        phoneme = processor.decode([token_id]).strip()
        if phoneme and phoneme not in ("<pad>", "<s>", "</s>", "|", " "):
            phonemes.append(phoneme)

    return phonemes


def main():
    args = parse_args()
    project_root = Path(__file__).resolve().parent.parent
    output_root = Path(args.output_root) if args.output_root else project_root / "outputs"

    speaker_id = args.speaker
    identifier = f"{args.dataset}_{speaker_id}"
    cal_dir = output_root / "calibration" / identifier
    observed_dir = output_root / "observed_phonemes" / identifier
    observed_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== Decoding observed phonemes for speaker: {args.dataset}_{speaker_id} ===")

    # Load calibration manifest
    manifest_path = cal_dir / "calibration_manifest.csv"
    manifest = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            manifest.append(row)

    print(f"Loaded {len(manifest)} calibration utterances")

    if len(manifest) == 0:
        print("\033[91m" + f"[ERROR] 0 calibration utterances found for speaker {speaker_id}. Cannot decode phonemes." + "\033[0m")
        sys.exit(1)

    # Validate manifest has required columns
    required_manifest_fields = {"utt_id", "wav_path"}
    first_row_keys = set(manifest[0].keys())
    missing_fields = required_manifest_fields - first_row_keys
    if missing_fields:
        print("\033[91m" + f"[ERROR] Calibration manifest missing required columns: {missing_fields}" + "\033[0m")
        sys.exit(1)

    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load CTC phoneme model
    print(f"Loading CTC model: {args.model_id}...")
    processor = Wav2Vec2Processor.from_pretrained(args.model_id)
    model = Wav2Vec2ForCTC.from_pretrained(args.model_id).to(device)
    model.eval()
    print("Model loaded.")

    success_count = 0
    fail_count = 0
    total_phonemes = 0

    for i, row in enumerate(manifest):
        utt_id = row["utt_id"]
        wav_path = row["wav_path"]

        # Validate audio file exists
        if not os.path.exists(wav_path):
            print(f"  [{i+1}/{len(manifest)}] {utt_id}: SKIP (audio file missing: {wav_path})")
            fail_count += 1
            continue

        # Load audio
        try:
            import soundfile as sf
            audio_data, sr = sf.read(wav_path)
        except Exception as e:
            print(f"  [{i+1}/{len(manifest)}] {utt_id}: SKIP (audio unreadable: {e})")
            fail_count += 1
            continue

        waveform = torch.from_numpy(audio_data).float()
        if waveform.ndim > 1:
            waveform = waveform.mean(dim=-1)  # stereo to mono

        # Resample to 16kHz if needed
        if sr != 16000:
            waveform_2d = waveform.unsqueeze(0)
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
            waveform = resampler(waveform_2d).squeeze(0)
            sr = 16000

        # Decode full audio
        try:
            phonemes = decode_full_audio(waveform, sr, processor, model, device)
        except Exception as e:
            print(f"  [{i+1}/{len(manifest)}] {utt_id}: ERROR ({e})")
            fail_count += 1
            continue

        if not phonemes:
            print(f"  [{i+1}/{len(manifest)}] {utt_id}: SKIP (no phonemes decoded)")
            fail_count += 1
            continue

        # Write observed phonemes CSV
        out_path = observed_dir / f"{utt_id}.csv"
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["phoneme"])
            writer.writeheader()
            for ph in phonemes:
                writer.writerow({"phoneme": ph})

        print(f"  [{i+1}/{len(manifest)}] {utt_id}: {len(phonemes)} phonemes observed")
        success_count += 1
        total_phonemes += len(phonemes)

    print(f"\n=== Observed phoneme decoding complete ===")
    print(f"Success: {success_count}/{len(manifest)}")
    print(f"Failed:  {fail_count}/{len(manifest)}")
    print(f"Total observed phonemes: {total_phonemes}")
    print(f"Output:  {observed_dir}")

    if success_count == 0:
        print("\033[91m" + "[ERROR] All utterances failed. 0 observed phoneme files produced." + "\033[0m")
        sys.exit(1)


if __name__ == "__main__":
    main()