# Testing Guide

Step-by-step verification for each component of the pipeline, in build order.
Run from the **project root** unless stated otherwise.

---

## Prerequisites

### With uv (recommended)

```bash
uv sync
```

This reads `pyproject.toml` + `uv.lock` and creates a `.venv` with all dependencies pinned.

For CUDA (T4 / GPU):
```bash
uv sync
uv pip install torch --index-url https://download.pytorch.org/whl/cu121
```

Activate the environment:
```bash
# Windows
.venv\Scripts\activate

# Linux / macOS
source .venv/bin/activate
```

Then prefix every command in this guide with `uv run` if you prefer not to activate:
```bash
uv run python -m src.scoring
uv run python eval/run_eval.py
uv run python app.py
```

### With pip (alternative)

```bash
pip install -r requirements.txt
```

---

Confirm GPU access (optional but recommended):
```bash
python -c "import torch; print(torch.cuda.is_available())"
# True
```

---

## Step 1 — `src/scoring.py`

Tests GPT2 language model scoring, fingerprint plausibility scoring, and the Whisper pass-through.

```bash
python -m src.scoring
```

**Example output:**
```
============================================================
scoring — local verification
============================================================
[scoring] Loading distilgpt2 on cuda … 
[scoring] Model loaded.

[1] GPT2 scores (DistilGPT2, length-normalised log-prob):
  -2.1843  'he dresses himself'
  -2.4217  'he addresses himself'
  -4.8901  'e dresse imself'
  -2.6534  'they dress themselves'
  -2.9102  'he processes himself'
  elapsed: 87.4 ms for 5 sentences

[2] Fingerprint scores (phoneme alignment vs. observed):
  -0.9134  'he dresses himself'    (intended: ['HH', 'IY', 'D', 'R', 'EH', 'S', 'IH', 'Z', 'HH', 'IH', 'M', 'S', 'EH', 'L', 'F'])
  -1.5442  'he addresses himself'  (intended: ['HH', 'IY', 'AH', 'D', 'R', 'EH', 'S', 'IH', 'Z', ...])
  -2.0011  'e dresse imself'       (intended: ['IY', 'D', 'R', 'EH', 'S', 'IY', 'M', 'S', 'EH', 'L', 'F'])
  -1.8203  'they dress themselves' (intended: [...])
  -2.1347  'he processes himself'  (intended: [...])

[3] Whisper score pass-through:
  whisper_score(-2.3) = -2.3

DONE
```

**What to check:**
- `"he dresses himself"` has the highest (least negative) GPT2 and fingerprint scores
- `"e dresse imself"` scores very poorly on GPT2 (ungrammatical)
- Elapsed time is under 200ms for 5 sentences

---

## Step 2 — `src/candidate_gen.py`

Tests phoneme-to-word candidate generation using the CMU dict trie and speaker fingerprint.

```bash
python -m src.candidate_gen
```

**Example output:**
```
============================================================
candidate_gen — local verification
============================================================
Speaker: F01  severity: severe
Observed phonemes: ['IY', 'D', 'R', 'EH', 'S', 'IY', 'M', 'S', 'EH', 'L', 'F']

Building inverse error map for F01...
Beam search: top_k=3, beam_width=10, max_candidates=20

Generated 18 candidates:
  #01  score=-1.2341  'he dresses himself'
  #02  score=-1.4782  'he dresses themself'
  #03  score=-1.6103  'he addresses himself'
  #04  score=-1.8234  'he dresses its self'
  #05  score=-2.0011  'he presses himself'
  #06  score=-2.1445  'he stresses himself'
  #07  score=-2.2891  'he dressed himself'
  #08  score=-2.4102  'he processes himself'
  ...

DONE
```

**What to check:**
- At least 10 candidates are returned
- Candidates are real English words/phrases
- The reference sentence (`"he dresses himself"`) appears near the top

---

## Step 3 — `src/fusion.py`

Tests severity-weighted score combination.

```bash
python -m src.fusion
```

**Example output:**
```
============================================================
fusion — local verification
============================================================

  severity=severe  alpha=0.3 beta=0.3 gamma=0.4
    1. combined=-1.4300  (w=-2.30 g=-1.80 f=-0.90)  'he dresses himself'
    2. combined=-1.9910  (w=-3.10 g=-2.10 f=-1.50)  'he addresses himself'
    3. combined=-2.7800  (w=-3.50 g=-2.30 f=-1.20)  'he dresses themself'
    4. combined=-3.0500  (w=-1.90 g=-4.50 f=-2.00)  'e dresse imself'

  severity=moderate  alpha=0.4 beta=0.3 gamma=0.3
    1. combined=-1.5100  (w=-2.30 g=-1.80 f=-0.90)  'he dresses himself'
    2. combined=-2.0800  (w=-3.10 g=-2.10 f=-1.50)  'he addresses himself'
    3. combined=-2.8300  (w=-3.50 g=-2.30 f=-1.20)  'he dresses themself'
    4. combined=-3.0800  (w=-1.90 g=-4.50 f=-2.00)  'e dresse imself'

  severity=mild  alpha=0.5 beta=0.3 gamma=0.2
    1. combined=-1.5900  (w=-2.30 g=-1.80 f=-0.90)  'he dresses himself'
    2. combined=-2.1600  (w=-3.10 g=-2.10 f=-1.50)  'he addresses himself'
    3. combined=-2.9900  (w=-3.50 g=-2.30 f=-1.20)  'he dresses themself'
    4. combined=-3.0100  (w=-1.90 g=-4.50 f=-2.00)  'e dresse imself'

DONE
```

**What to check:**
- Combined scores differ across severity tiers (weights are being applied)
- Severe tier penalises the garbled candidate more heavily (higher fingerprint weight)
- Ranking is consistent: grammatical sentence stays on top

---

## Step 4 — `src/confidence.py`

Tests the ambiguity detection logic and alternative generation.

```bash
python -m src.confidence
```

**Example output:**
```
============================================================
confidence — local verification
============================================================

  [CONFIDENT]
    status           : confident
    confidence       : 0.9312
    trigger_reason   : None
    alternatives     :
      rank=1  score=-1.2000  'he dresses himself'

  [AMBIGUOUS (close gap)]
    status           : ambiguous
    confidence       : 0.5124
    trigger_reason   : small_gap_to_top2
    alternatives     :
      rank=1  score=-1.8000  'he dresses himself'
      rank=2  score=-1.8500  'he dresses themself'
      rank=3  score=-2.0000  'he addresses himself'

  [DEDUPLICATED]
    status           : ambiguous
    confidence       : 0.5891
    trigger_reason   : small_gap_to_top2
    alternatives     :
      rank=1  score=-1.8000  'He dresses himself.'
      rank=2  score=-2.1000  'he dresses themself'

DONE
```

**What to check:**
- Confident case: only 1 alternative returned, `trigger_reason` is `None`
- Ambiguous case: 3+ alternatives, `trigger_reason` names the tripped signal
- Deduplication: `"He dresses himself."` and `"he dresses himself"` collapse to 1 entry

---

## Step 5 — `src/pipeline.py`

End-to-end test on a single sample using TORGO F01 fingerprint.

```bash
python -m src.pipeline
```

**Example output:**
```
============================================================
pipeline — local verification
============================================================
Loaded sample: he dresses himself
Speaker: F01  severity: severe
Observed phonemes: ['IY', 'D', 'R', 'EH', 'S', 'IY', 'M', 'S', 'EH', 'L', 'F']
Whisper top-1: 'e dresse imself'

Status     : confident
Corrected  : 'he dresses himself'
Confidence : 0.8743
Trigger    : None

Alternatives:
  rank=1  score=-1.4102  'he dresses himself'

Top candidates (all 22):
  #1  combined=-1.4102  w=-2.800 g=-2.184 f=-0.913  [whisper] 'he dresses himself'
  #2  combined=-1.7834  w=-3.100 g=-2.421 f=-1.544  [whisper] 'he addresses himself'
  #3  combined=-1.9201  w=-3.400 g=-2.653 f=-1.203  [whisper] 'he dresses themself'
  #4  combined=-2.1445  w=0.000  g=-2.341 f=-1.820  [phoneme] 'he dresses its self'
  #5  combined=-2.3891  w=-3.800 g=-2.910 f=-2.101  [whisper] 'he processes himself'
  ...

DONE
```

**What to check:**
- `corrected` is not the garbled Whisper top-1 (`"e dresse imself"`)
- Both `[whisper]` and `[phoneme]` sources appear in the candidate list
- All 4 scores are populated on every row

---

## Step 6 — Full evaluation (`eval/run_eval.py`)

Runs all 30 test cases through 4 conditions and prints a WER table per severity tier.

```bash
python eval/run_eval.py
```

**Example output:**
```
Test set        : eval/test_set.json
Config          : config.yaml
Fingerprints dir: fingerprints/

  F01_test_01                     ref='he dresses himself'      W=0.667  G=0.333  P=0.000  O=0.000
  F01_test_02                     ref='share'            [AMB]  W=1.000  G=1.000  P=0.000  O=0.000
  ...

================================================================================
EVALUATION RESULTS
================================================================================

### Severity: SEVERE  (n=14)
ID                              Whisper    GPT2  Pipeline  Oracle  Ambig  Trigger
------------------------------------------------------------------------------------------
F01_test_01                       0.667   0.333     0.000   0.000      N
F01_test_02                       1.000   1.000     0.000   0.000      Y  small_gap_to_top2
...
------------------------------------------------------------------------------------------
AVERAGE                           0.612   0.418     0.201   0.094   43%

### Severity: MODERATE  (n=6)
...
AVERAGE                           0.389   0.278     0.167   0.056   33%

### Severity: MILD  (n=10)
...
AVERAGE                           0.180   0.120     0.060   0.020   10%

### OVERALL  (n=30)
  Whisper WER   : 0.452
  GPT2 WER      : 0.312
  Pipeline WER  : 0.168
  Oracle WER    : 0.063
  Ambiguity rate: 33%


## Markdown table

| Severity | n | WER Whisper | WER GPT2 | WER Pipeline | WER Oracle | Ambiguity Rate |
|----------|---|------------|----------|--------------|------------|----------------|
| severe   | 14 | 0.612 | 0.418 | 0.201 | 0.094 | 43% |
| moderate |  6 | 0.389 | 0.278 | 0.167 | 0.056 | 33% |
| mild     | 10 | 0.180 | 0.120 | 0.060 | 0.020 | 10% |
```

**What to check:**
- `Pipeline WER ≤ GPT2 WER ≤ Whisper WER` — pipeline should improve on every tier
- `Oracle WER ≤ Pipeline WER` — oracle is always the best-case ceiling
- Ambiguity rate is highest for severe speakers, near zero for mild
- No `[SKIP]` lines in the per-case output

---

## Step 7 — Gradio app

```bash
python app.py
```

Visit the printed local URL (e.g. `http://127.0.0.1:7860`). Then:

1. Select speaker **TORGO — F01 (severe)**
2. Select input **"he dresses himself"** from the dropdown
3. Click **Correct**

**Expected UI state:**
```
Corrected:   he dresses himself          ← large output text
Confidence:  [● Confident — 0.87]        ← green badge
Alternatives panel: hidden              ← only shown when ambiguous
```

Try a more ambiguous case (e.g. F01 "yet he still thinks as swiftly as ever"):
```
Corrected:   he still thinks as swiftly as ever
Confidence:  [◐ Ambiguous — please confirm]   ← yellow badge
Alternatives:
  1. he still thinks as swiftly as ever    (0.4231)
  2. yet he still thinks as swiftly as ever (0.3987)
  3. he still stinks as swiftly as ever    (0.3102)
```

---

## Step 8 — FastAPI service (optional)

```bash
uvicorn api:app --reload
# INFO: Uvicorn running on http://127.0.0.1:8000
```

Then in a second terminal:

```bash
curl -s -X POST http://localhost:8000/correct \
  -H "Content-Type: application/json" \
  -d '{
    "speaker_id": "F01",
    "dataset": "TORGO",
    "whisper_nbest": [
      {"text": "e dresse imself", "score": 0.42},
      {"text": "he dresses himself", "score": 0.21},
      {"text": "he addresses himself", "score": 0.14}
    ],
    "observed_phonemes": ["IY", "D", "R", "EH", "S", "IY", "M", "S", "EH", "L", "F"]
  }' | python -m json.tool
```

**Example response:**
```json
{
  "status": "confident",
  "corrected": "he dresses himself",
  "alternatives": [
    {"text": "he dresses himself", "score": -1.4102, "rank": 1}
  ],
  "selected_score": -1.4102,
  "confidence": 0.8743,
  "trigger_reason": null
}
```

---

## Quick smoke test (all modules, one command)

```bash
python -m src.scoring && \
python -m src.candidate_gen && \
python -m src.fusion && \
python -m src.confidence && \
python -m src.pipeline && \
python eval/run_eval.py
```

All steps should complete with `DONE` and no Python tracebacks.
