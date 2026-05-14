"""
FastAPI service wrapper for the correction pipeline.

Endpoints:
  POST /correct          — full correction
  GET  /health           — liveness check
  GET  /speakers         — list available fingerprint keys

Launch:
    uvicorn api:app --reload --port 8000

Example curl:
    curl -X POST http://localhost:8000/correct \
         -H "Content-Type: application/json" \
         -d @demo_data/sample_inputs.json   # single object, not array
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

import yaml
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Load config + fingerprints at startup
# ---------------------------------------------------------------------------
HERE = Path(__file__).parent
CONFIG_PATH = HERE / "config.yaml"
FINGERPRINTS_DIR = HERE / "fingerprints"

with open(CONFIG_PATH, encoding="utf-8") as f:
    CONFIG = yaml.safe_load(f)


def _load_fingerprints() -> dict[str, dict]:
    fps = {}
    for fp_file in sorted(FINGERPRINTS_DIR.glob("*.json")):
        with open(fp_file, encoding="utf-8") as f:
            data = json.load(f)
        fps[fp_file.stem] = data
    return fps


FINGERPRINTS = _load_fingerprints()

# Lazy pipeline import
sys.path.insert(0, str(HERE))
from src import pipeline as _pipeline

# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class WhisperEntry(BaseModel):
    text: str
    score: float = 0.0


class CorrectionRequest(BaseModel):
    speaker_id: str = Field(..., description="e.g. 'F01'; used to look up fingerprint_key if fingerprint omitted")
    dataset: str = Field("TORGO", description="Dataset name, e.g. 'TORGO' or 'UASpeech'")
    fingerprint: Optional[dict] = Field(None, description="Full fingerprint JSON; if omitted, loaded from fingerprints/")
    whisper_nbest: list[WhisperEntry] = Field(..., description="Whisper n-best list")
    observed_phonemes: list[str] = Field(..., description="Arpabet phoneme sequence")


class Alternative(BaseModel):
    text: str
    score: float
    rank: int


class CandidateDetail(BaseModel):
    text: str
    rank: int
    combined: float
    whisper_score: float
    gpt2_score: float
    fingerprint_score: float
    source: str


class CorrectionResponse(BaseModel):
    status: str
    corrected: str
    alternatives: list[Alternative]
    candidates: list[CandidateDetail]
    selected_score: float
    confidence: float
    trigger_reason: Optional[str] = None


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Dysarthric Speech Correction API",
    description="Corrects dysarthric ASR output using speaker fingerprints + DistilGPT2.",
    version="1.0.0",
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/speakers")
def list_speakers():
    return {"speakers": sorted(FINGERPRINTS.keys())}


@app.post("/correct", response_model=CorrectionResponse)
def correct_endpoint(req: CorrectionRequest) -> CorrectionResponse:
    # Resolve fingerprint
    fp = req.fingerprint
    if fp is None:
        key = f"{req.dataset}_{req.speaker_id}"
        if key not in FINGERPRINTS:
            raise HTTPException(
                status_code=404,
                detail=f"Fingerprint not found: '{key}'. Available: {sorted(FINGERPRINTS.keys())}",
            )
        fp = FINGERPRINTS[key]

    nbest = [{"text": e.text, "score": e.score} for e in req.whisper_nbest]

    result = _pipeline.correct(
        whisper_nbest=nbest,
        observed_phonemes=req.observed_phonemes,
        fingerprint=fp,
        config=CONFIG,
    )
    return CorrectionResponse(**result)
