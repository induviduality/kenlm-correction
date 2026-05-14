# Evaluation

Scripts for evaluating the dysarthric speech correction pipeline.

## Files

| File | Description |
|---|---|
| `run_eval.py` | Main evaluation harness. Runs all test cases through 4 conditions and prints a WER table per severity tier. |
| `test_set.json` | Test cases with pre-computed Whisper n-best lists and observed phonemes. |
| `audio_to_phonemes.py` | Utility to decode Arpabet phonemes from an audio file (for building new test cases). |

## Running the evaluation

```bash
# From project root
python eval/run_eval.py

# With custom paths
python eval/run_eval.py --test-set eval/test_set.json --config config.yaml --fingerprints-dir fingerprints/
```

### What it measures

For each test case, four conditions are compared:

1. **Whisper top-1** — baseline WER (no correction)
2. **Whisper n-best + GPT2 rerank** — intermediate WER (language model only)
3. **Full pipeline top-1** — system WER (phoneme candidates + fingerprint + GPT2 + fusion)
4. **Oracle from alternatives** — best-case WER if the user always picks the correct alternative

Output includes:
- Per-severity WER table (severe / moderate / mild)
- Ambiguity rate per tier (% of inputs flagged as ambiguous)
- A copy-pasteable Markdown table for reports

## Building new test cases

Each entry in `test_set.json` needs:

```json
{
  "id": "F01_test_01",
  "speaker_id": "F01",
  "dataset": "TORGO",
  "severity": "severe",
  "reference": "he dresses himself",
  "whisper_nbest": [
    {"text": "e dresse imself", "score": -2.3},
    ...
  ],
  "observed_phonemes": ["IY", "D", "R", "EH", "S", ...]
}
```

- **`speaker_id` + `dataset`** must match a fingerprint file in `fingerprints/` (e.g. `TORGO_F01.json`).
- **`severity`** should be one of `severe`, `moderate`, or `mild` (maps to fusion weights in `config.yaml`).
- **`observed_phonemes`** are Arpabet symbols. Generate them from audio using:

```bash
python eval/audio_to_phonemes.py path/to/audio.wav
python eval/audio_to_phonemes.py path/to/audio.wav --format json
```
