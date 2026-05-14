## Updated implementation plan — for Claude Code handoff

### Environment
- **Compute**: T4 VM (16GB VRAM, single GPU)
- **OS**: Ubuntu with CUDA 12.x
- **Python**: 3.10+
- **External services** (assumed available, not built here): fine-tuned Whisper, phoneme recognizer, fingerprint generator
- **Language model**: DistilGPT2 from HF Hub (replaces KenLM)

### Input contract
```json
{
  "speaker_id": "F01",
  "fingerprint": { /* full JSON in TORGO_F01.json format */ },
  "whisper_nbest": [
    {"text": "e dresse imself", "score": -2.3},
    /* ... 10 entries ... */
  ],
  "observed_phonemes": ["IY", "D", "R", "EH", "S", "IY", "M", "S", "EH", "L", "F"]
}
```

### Output contract
```json
{
  "status": "confident" | "ambiguous",
  "corrected": "he dresses himself",
  "alternatives": [
    {"text": "he dresses himself", "score": 0.42, "rank": 1},
    {"text": "he dresses themself", "score": 0.39, "rank": 2},
    {"text": "he addresses himself", "score": 0.35, "rank": 3}
  ],
  "candidates": [ /* full ranked list with all sub-scores, for debug */ ],
  "selected_score": 0.42,
  "confidence": 0.78
}
```

**Behavior**:
- If `status == "confident"`: client uses `corrected` directly
- If `status == "ambiguous"`: client should surface `alternatives` to the user for selection (UX layer)

---

## Confidence + ambiguity logic

A new module `confidence.py` decides whether to commit to top-1 or surface alternatives.

### Confidence is "low" when **any** of:

1. **Top-1 absolute score below threshold** (`min_score_threshold`, default = -3.0 in log-prob space after fusion)
2. **Top-1 and top-2 are too close** (gap below `score_gap_threshold`, default = 0.15)
3. **Top-K candidates cluster tightly** (variance of top-5 scores below `variance_threshold`, default = 0.05)

All thresholds configurable in `config.yaml`.

### How many alternatives to return

- Always return `top_n_alternatives_min` (default 3) when ambiguous
- Cap at `top_n_alternatives_max` (default 5)
- Cut off alternatives whose score gap from top-1 exceeds `alternative_cutoff_gap` (default 0.4) — these are too unlikely to surface

### Deduplication

Whisper n-best and phoneme-derived candidates often produce near-duplicates ("he dresses himself" / "he dresses himself."). Normalize before counting alternatives:
- Lowercase, strip punctuation, collapse whitespace
- Hash → dedupe → keep highest-scoring instance

---

## `config.yaml` (single source of truth)

```yaml
fusion_weights:
  severe:   { alpha: 0.3, beta: 0.3, gamma: 0.4 }
  moderate: { alpha: 0.4, beta: 0.3, gamma: 0.3 }
  mild:     { alpha: 0.5, beta: 0.3, gamma: 0.2 }

candidate_generation:
  top_k_per_phoneme: 3
  beam_width: 10
  max_candidates: 20

confidence:
  min_score_threshold: -3.0
  score_gap_threshold: 0.15
  variance_threshold: 0.05
  top_n_alternatives_min: 3
  top_n_alternatives_max: 5
  alternative_cutoff_gap: 0.4

scoring:
  fingerprint_smoothing: 0.01
  gpt2_batch_size: 40
  gpt2_max_length: 128

model:
  gpt2_name: "distilgpt2"
  device: "cuda"
```

---

## Project structure

```
dysarthric-correction/
├── app.py                          # Gradio entry, HF Spaces runs this
├── api.py                          # Optional FastAPI service
├── requirements.txt
├── config.yaml
├── README.md
│
├── fingerprints/
│   ├── TORGO_F01.json
│   ├── TORGO_M01.json
│   ├── UASpeech_F03.json
│   └── UASpeech_M07.json
│
├── src/
│   ├── __init__.py
│   ├── candidate_gen.py
│   ├── scoring.py
│   ├── fusion.py
│   ├── confidence.py               # NEW
│   └── pipeline.py
│
├── eval/
│   ├── run_eval.py
│   └── test_set.json
│
└── demo_data/
    └── sample_inputs.json
```

---

## Component specs

### `requirements.txt`
```
transformers>=4.40
torch>=2.2
g2p-en
nltk
numpy
gradio>=4.0
jiwer
pyyaml
huggingface_hub
fastapi
uvicorn
```

T4 install:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

---

### `src/candidate_gen.py`

```python
def generate_candidates(
    observed_phonemes: list[str],
    fingerprint: dict,
    top_k_per_phoneme: int = 3,
    beam_width: int = 10,
    max_candidates: int = 20,
) -> list[dict]:
    """Returns: [{"text": str, "generation_score": float}, ...]"""
```

**Build at module load**:
- CMU dict phoneme trie (single global)
- Per-speaker inverse error map cache (dict keyed by `speaker_id`)

**Algorithm**:
1. Build inverse error map from `fingerprint["error_map"]`
2. For each observed phoneme, retrieve top-3 likely intended phonemes
3. Beam-search expand to top-K intended phoneme sequences (K=10)
4. Segment each via phoneme trie lookup into words
5. Return ~20 candidates max

**Performance**: ~50ms per call on CPU.

---

### `src/scoring.py`

Three pure functions:

```python
def gpt2_score(sentences: list[str], batch_size: int = 40) -> list[float]:
    """Batched length-normalized log-prob from DistilGPT2. GPU."""

def fingerprint_score(
    hypothesis: str,
    observed_phonemes: list[str],
    fingerprint: dict,
    smoothing: float = 0.01,
) -> float:
    """
    1. Phonemize hypothesis with g2p_en
    2. Levenshtein-align intended ↔ observed phonemes
    3. For each aligned pair: log(error_map.get(f"{intended}>{observed}", smoothing))
    4. Return length-normalized mean log-prob
    """

def whisper_score(log_prob: float) -> float:
    """Pass-through from Whisper n-best."""
```

**DistilGPT2 setup**: Load once at module import to GPU. Process all candidates in one batched forward pass (~50ms for 40 candidates on T4).

---

### `src/fusion.py`

```python
def fuse(
    candidates: list[dict],
    severity: str,
    weights_config: dict,
) -> list[dict]:
    """
    combined = α·whisper + β·gpt2 + γ·fingerprint
    Weights pulled from config by severity.
    Returns same list, sorted by combined score descending.
    Each candidate dict gains a "combined" field.
    """
```

---

### `src/confidence.py` (NEW)

```python
def assess_confidence(
    ranked_candidates: list[dict],
    config: dict,
) -> dict:
    """
    Returns:
      {
        "status": "confident" | "ambiguous",
        "confidence": float,  # 0-1
        "alternatives": list[dict],  # top-N normalized
        "trigger_reason": str | None,  # for debug: which threshold tripped
      }
    """
```

**Logic**:
1. Deduplicate candidates (normalize text, hash, keep best score per group)
2. Compute three confidence signals:
   - `score_ok = top_1.combined > min_score_threshold`
   - `gap_ok = (top_1.combined - top_2.combined) > score_gap_threshold`
   - `variance_ok = variance(top_5.combined) > variance_threshold`
3. Status:
   - All three pass → `"confident"`, return only top-1 as alternative
   - Any fails → `"ambiguous"`, return top-N alternatives subject to `alternative_cutoff_gap`
4. Confidence score: softmax over top-5 fused scores, return the top-1 probability
5. Log which signal tripped (`trigger_reason`) — useful for tuning thresholds during eval

---

### `src/pipeline.py`

```python
def correct(
    whisper_nbest: list[dict],
    observed_phonemes: list[str],
    fingerprint: dict,
    config: dict,
) -> dict:
    """End-to-end correction. Returns the output contract."""
```

**Flow**:
1. Generate phoneme-derived candidates
2. Merge with `whisper_nbest`; deduplicate
3. Batch-score all candidates with GPT2 (single GPU call)
4. Score each with fingerprint plausibility
5. Fuse scores using severity-adapted weights
6. Run confidence assessment
7. Return output contract

---

### `app.py` (Gradio)

UI elements:
- **Speaker dropdown**: loads fingerprint from `fingerprints/`
- **Input**: pre-computed `(n-best, phonemes)` from `demo_data/sample_inputs.json` dropdown
- **Primary output**: corrected sentence in large font
- **Confidence badge**: green ("Confident") or yellow ("Ambiguous — please confirm")
- **Alternatives panel**: shows when ambiguous, lets user click an alternative
- **Debug expander**: full ranked candidate table with all 4 scores + which confidence signal tripped

The confidence-driven alternatives panel is your **judging killer feature** — it shows the system knows when it doesn't know.

---

### `api.py` (optional, for clean service demo)

```python
@app.post("/correct")
def correct_endpoint(req: CorrectionRequest) -> CorrectionResponse:
    return pipeline.correct(...)
```

Useful for demoing this as a service that the mobile app would call.

---

### `eval/run_eval.py`

For each test speaker × 20 sentences in `test_set.json`:
1. Whisper top-1 only (baseline WER)
2. Whisper n-best + GPT2 rerank (intermediate WER)
3. Full pipeline, top-1 (system WER)
4. Full pipeline, oracle from alternatives (best-case WER if user always picks correctly)

Output: markdown WER table per speaker + **ambiguity rate** (% of inputs flagged ambiguous) per severity tier.

The ambiguity rate metric is important for the judging narrative — proves your system fails gracefully rather than silently.

---

## Build order (T4 VM, 4 hours)

| Hour | Task | Verification |
|---|---|---|
| **0:00-0:30** | Setup repo, `pip install`, verify GPU access, download DistilGPT2 (cached automatically) | `nvidia-smi`, `transformers` loads on cuda |
| **0:30-1:30** | `candidate_gen.py` (hardest piece) | 20 candidates from F01 sample input |
| **1:30-2:15** | `scoring.py` — 3 functions | GPT2 scores 10 sentences in <100ms |
| **2:15-2:45** | `fusion.py` + `confidence.py` | Ambiguous case returns 3 alternatives |
| **2:45-3:15** | `pipeline.py` + `app.py` | End-to-end on 1 sample |
| **3:15-3:45** | `run_eval.py`, generate WER table | Per-speaker table with ambiguity rate |
| **3:45-4:00** | HF Spaces deploy | Live URL |

---

## Hosting

**HF Spaces, T4 small** (recommended over CPU Basic for this workload):
- T4 inference: ~200ms total per request (GPT2 batched + everything else)
- CPU Basic: ~3-5 sec per request (GPT2 on CPU is slow)
- Cost: ~$0.60/hr during demo only, ~₹50 for the judging session

Free tier (CPU Basic) is acceptable as backup but the demo experience is noticeably slower.

**Models**:
- DistilGPT2: auto-cached by `transformers` library, no manual download
- Fingerprints: commit to repo (small JSONs)
- Sample inputs: commit to repo

**Cold start**: ~20 sec (DistilGPT2 loads from HF cache, fingerprints load from disk).

---

## Claude Code handoff checklist

When you start the Claude Code session, give it:

1. ✅ This plan (paste it)
2. ✅ The `TORGO_F01.json` sample for reference
3. ✅ Sample n-best format (3-5 example entries)
4. ✅ Sample observed phonemes format
5. ✅ A sentence telling it to **build in the order above** and verify each component before moving on
6. ✅ The `config.yaml` exactly as written above
7. ✅ Permission to install packages and use GPU
8. ✅ Instruction to write `eval/run_eval.py` last and use it to validate the full pipeline before deployment

**One critical instruction to give Claude Code**: "Do not implement Whisper n-best extraction or phoneme recognition — assume those are provided as input. Build only the correction layer."
