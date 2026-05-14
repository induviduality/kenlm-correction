# Dysarthric Speech Correction Pipeline

A post-ASR correction layer for dysarthric speech. Takes Whisper n-best hypotheses and observed phonemes as input, then uses speaker-specific error fingerprints + language model reranking to produce corrected transcriptions.

The system knows when it doesn't know — ambiguous cases surface ranked alternatives instead of committing to a wrong answer.

## How it works

```
Whisper n-best + Observed phonemes + Speaker fingerprint
        │                │                    │
        ▼                ▼                    ▼
   ┌─────────────────────────────────────────────┐
   │  1. Candidate generation (phoneme trie)     │
   │  2. Merge with Whisper n-best, deduplicate  │
   │  3. GPT2 language model scoring             │
   │  4. Fingerprint plausibility scoring        │
   │  5. Severity-weighted score fusion          │
   │  6. Confidence assessment                   │
   └─────────────────────────────────────────────┘
        │                              │
        ▼                              ▼
   "confident"                    "ambiguous"
   → corrected text               → ranked alternatives
```

## Project structure

```
├── app.py                  # Gradio UI (HF Spaces entry point)
├── api.py                  # FastAPI service endpoint
├── config.yaml             # All tunable parameters
├── requirements.txt
│
├── src/
│   ├── candidate_gen.py    # Phoneme-derived candidate generation
│   ├── scoring.py          # GPT2, fingerprint, and whisper scoring
│   ├── fusion.py           # Severity-weighted score fusion
│   ├── confidence.py       # Confidence assessment & ambiguity detection
│   └── pipeline.py         # End-to-end correction pipeline
│
├── fingerprints/           # Speaker error fingerprints (JSON)
│   ├── TORGO_F01.json
│   ├── UASpeech_M07.json
│   └── ...
│
├── eval/
│   ├── run_eval.py         # Evaluation harness (WER tables)
│   ├── test_set.json       # Test cases with n-best + phonemes
│   └── audio_to_phonemes.py # Utility: audio → Arpabet phonemes
│
└── demo_data/
    └── sample_inputs.json  # Sample inputs for the Gradio demo
```

## Setup

### With uv (recommended)

```bash
uv sync
```

For GPU (T4 / CUDA 12.x), install torch separately after sync:
```bash
uv pip install torch --index-url https://download.pytorch.org/whl/cu121
```

Activate:
```bash
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # Linux / macOS
```

Or skip activation and prefix commands with `uv run python ...`.

### With pip

```bash
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cu121  # GPU
```

## Usage

### Gradio demo

```bash
python app.py
```

### FastAPI service

```bash
uvicorn api:app --host 0.0.0.0 --port 8000
```

### Pipeline (Python)

```python
from src.pipeline import correct
import yaml, json

with open("config.yaml") as f:
    config = yaml.safe_load(f)
with open("fingerprints/TORGO_F01.json") as f:
    fingerprint = json.load(f)

result = correct(
    whisper_nbest=[
        {"text": "e dresse imself", "score": 0.42},
        {"text": "he dresses himself", "score": 0.21},
    ],
    observed_phonemes=["IY", "D", "R", "EH", "S", "IY", "M", "S", "EH", "L", "F"],
    fingerprint=fingerprint,
    config=config,
)

print(result["status"])     # "confident" or "ambiguous"
print(result["corrected"])  # "he dresses himself"
```

### Evaluation

```bash
python eval/run_eval.py
```

Outputs a WER table per severity tier comparing:
1. Whisper top-1 (baseline)
2. Whisper + GPT2 rerank
3. Full pipeline top-1
4. Oracle from alternatives (best-case if user picks correctly)

Plus ambiguity rate per tier.

## Input / Output contracts

**Input:**
```json
{
  "speaker_id": "F01",
  "fingerprint": { "speaker_id": "F01", "severity": "severe", "error_map": { ... } },
  "whisper_nbest": [
    {"text": "e dresse imself", "score": 0.42}
  ],
  "observed_phonemes": ["IY", "D", "R", "EH", "S", "IY", "M", "S", "EH", "L", "F"]
}
```

**Output:**
```json
{
  "status": "confident",
  "corrected": "he dresses himself",
  "alternatives": [
    {"text": "he dresses himself", "score": 0.42, "rank": 1}
  ],
  "confidence": 0.78
}
```

When `status == "ambiguous"`, `alternatives` contains 3–5 ranked options for the user to choose from.

## Configuration

All thresholds are in `config.yaml`:

- **Fusion weights** — severity-adapted α/β/γ for whisper, GPT2, and fingerprint scores
- **Candidate generation** — beam width, max candidates
- **Confidence** — score thresholds, gap thresholds, variance thresholds
- **Model** — GPT2 variant, device selection
