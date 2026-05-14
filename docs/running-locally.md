# Running the Pipeline Locally

## Setup

```powershell
cd d:\Code\att-hackathon\kenlm-correction
uv sync
```

---

## Option 1 — Audio file + speaker (end-to-end)

The `transcribe.py` CLI takes an audio file and a speaker fingerprint key, runs Whisper internally, derives phonemes, and returns the corrected transcript.

```powershell
# List available speaker fingerprints
uv run python transcribe.py --list-speakers

# Run on your audio file
uv run python transcribe.py your_audio.wav TORGO_F01

# With full candidate/score table
uv run python transcribe.py your_audio.wav TORGO_F01 --verbose

# Better Whisper accuracy (downloads ~500 MB, slower)
uv run python transcribe.py your_audio.wav TORGO_F01 --model small

# Raw JSON output
uv run python transcribe.py your_audio.wav TORGO_F01 --json
```

Supported audio formats: `.wav`, `.mp3`, `.m4a`, `.flac`, and anything ffmpeg can read.

**Note on phonemes:** phonemes are derived from Whisper's top-1 output via g2p, not directly from the audio signal. This means they reflect what Whisper heard — which is the garbled dysarthric output the fingerprint is designed to work with.

---

## Option 2 — Gradio UI (interactive)

Best for testing with manual inputs or when you want to inspect the debug table.

```powershell
uv run python app.py
# Open http://localhost:7860
```

Paste Whisper n-best JSON and observed phonemes directly in the UI. The debug panel shows all score columns (Whisper, GPT2, Fingerprint, WordValidity, Source).

---

## Option 3 — REST API

```powershell
uv run uvicorn api:app --reload --port 8000
# Swagger UI at http://localhost:8000/docs
```

List available speakers:
```powershell
curl http://localhost:8000/speakers
```

Call `/correct`:
```powershell
curl -X POST http://localhost:8000/correct `
  -H "Content-Type: application/json" `
  -d '{
    "speaker_id": "F01",
    "dataset": "TORGO",
    "whisper_nbest": [
      {"text": "e dresse imself", "score": -2.3},
      {"text": "he dresses himself", "score": -2.8}
    ],
    "observed_phonemes": ["IY","D","R","EH","S","IY","M","S","EH","L","F"]
  }'
```

---

## Option 4 — Python directly (no audio needed)

Uses the built-in demo samples to verify the pipeline works end-to-end:

```powershell
uv run python -m src.pipeline
```

Or with your own inputs:

```python
import json, yaml
from src import pipeline

with open("config.yaml") as f:
    config = yaml.safe_load(f)
with open("fingerprints/TORGO_F01.json", encoding="utf-8") as f:
    fingerprint = json.load(f)

result = pipeline.correct(
    whisper_nbest=[
        {"text": "e dresse imself", "score": -2.3},
        {"text": "he dresses himself", "score": -2.8},
    ],
    observed_phonemes=["IY", "D", "R", "EH", "S", "IY", "M", "S", "EH", "L", "F"],
    fingerprint=fingerprint,
    config=config,
)
print(result["corrected"], result["status"], result["confidence"])
```

---

## Understanding the output

| Field | What it means |
|---|---|
| `corrected` | Top-ranked correction |
| `status` | `confident` or `ambiguous` |
| `confidence` | Softmax probability over top-5 candidates (0–1) |
| `alternatives` | Shown when ambiguous — review and pick the best |
| `trigger_reason` | Why confidence was flagged (score gap, variance, absolute score) |

When `status` is `ambiguous`, the correct answer is almost always in `alternatives`. The pipeline is intentionally conservative — it prefers surfacing options over silently outputting the wrong correction.

---

## Running the evaluation

```powershell
uv run python eval/run_eval.py
```

Runs 4 conditions (Whisper baseline, GPT2-only, full pipeline, oracle) across 30 test cases and prints a WER table per severity tier.
