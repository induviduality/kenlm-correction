"""
Gradio UI for the dysarthric speech correction pipeline.

UI elements:
  - Speaker dropdown   → loads fingerprint from fingerprints/
  - Sample input dropdown → pre-computed (n-best, phonemes) from demo_data/
  - Primary output: corrected sentence in large font
  - Confidence badge: green (Confident) or yellow (Ambiguous)
  - Alternatives panel: shown when ambiguous, click to select
  - Debug expander: full ranked candidate table

Launch:
    python app.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import gradio as gr
import yaml

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
HERE = Path(__file__).parent
FINGERPRINTS_DIR = HERE / "fingerprints"
DEMO_DATA_PATH = HERE / "demo_data" / "sample_inputs.json"
CONFIG_PATH = HERE / "config.yaml"

# ---------------------------------------------------------------------------
# Load config + demo data at startup
# ---------------------------------------------------------------------------
with open(CONFIG_PATH, encoding="utf-8") as f:
    CONFIG = yaml.safe_load(f)

with open(DEMO_DATA_PATH, encoding="utf-8") as f:
    DEMO_SAMPLES = json.load(f)

DEMO_BY_ID: dict[str, dict] = {s["id"]: s for s in DEMO_SAMPLES}

# ---------------------------------------------------------------------------
# Load fingerprints
# ---------------------------------------------------------------------------
def _load_fingerprints() -> dict[str, dict]:
    fps = {}
    for fp_file in sorted(FINGERPRINTS_DIR.glob("*.json")):
        with open(fp_file, encoding="utf-8") as f:
            data = json.load(f)
        # Key by "DATASET_SPEAKERID" from filename stem
        fps[fp_file.stem] = data
    return fps

FINGERPRINTS = _load_fingerprints()

# ---------------------------------------------------------------------------
# Lazy pipeline import (avoids loading DistilGPT2 on import)
# ---------------------------------------------------------------------------
_pipeline_module = None

def _get_pipeline():
    global _pipeline_module
    if _pipeline_module is None:
        sys.path.insert(0, str(HERE))
        from src import pipeline as _p
        _pipeline_module = _p
    return _pipeline_module


# ---------------------------------------------------------------------------
# Core correction function called by Gradio
# ---------------------------------------------------------------------------

def run_correction(
    fingerprint_key: str,
    sample_id: str,
    custom_nbest_json: str,
    custom_phonemes: str,
) -> tuple[str, str, str, str]:
    """
    Returns: (corrected_text, confidence_badge_html, alternatives_md, debug_md)
    """
    # ── Resolve fingerprint ──────────────────────────────────────────────
    if fingerprint_key not in FINGERPRINTS:
        return "Error: fingerprint not found", "", "", ""
    fingerprint = FINGERPRINTS[fingerprint_key]

    # ── Resolve input ────────────────────────────────────────────────────
    if sample_id and sample_id in DEMO_BY_ID:
        sample = DEMO_BY_ID[sample_id]
        whisper_nbest = sample["whisper_nbest"]
        observed_phonemes = sample["observed_phonemes"]
    else:
        # Parse custom inputs
        try:
            whisper_nbest = json.loads(custom_nbest_json)
            if not isinstance(whisper_nbest, list):
                raise ValueError("Expected a JSON array")
        except Exception as e:
            return f"Error parsing n-best JSON: {e}", "", "", ""
        observed_phonemes = [p.strip().upper() for p in custom_phonemes.split() if p.strip()]
        if not observed_phonemes:
            return "Error: no observed phonemes provided", "", "", ""

    # ── Run pipeline ─────────────────────────────────────────────────────
    pipeline = _get_pipeline()
    try:
        result = pipeline.correct(whisper_nbest, observed_phonemes, fingerprint, CONFIG)
    except Exception as e:
        import traceback
        return f"Pipeline error: {e}\n{traceback.format_exc()}", "", "", ""

    # ── Format outputs ───────────────────────────────────────────────────
    corrected = result["corrected"]

    if result["status"] == "confident":
        badge = (
            '<div style="display:inline-block;background:#22c55e;color:white;'
            'padding:6px 18px;border-radius:20px;font-weight:bold;font-size:1.1em;">'
            "✓ Confident</div>"
        )
    else:
        badge = (
            '<div style="display:inline-block;background:#eab308;color:white;'
            'padding:6px 18px;border-radius:20px;font-weight:bold;font-size:1.1em;">'
            "⚠ Ambiguous — please confirm</div>"
        )
    badge += f'<span style="margin-left:12px;color:#888;">confidence={result["confidence"]:.2f}</span>'

    alts = result["alternatives"]
    if len(alts) <= 1:
        alternatives_md = "_No alternatives (top result is confident)_"
    else:
        lines = ["| Rank | Text | Score |", "|------|------|-------|"]
        for a in alts:
            lines.append(f"| {a['rank']} | **{a['text']}** | {a['score']} |")
        alternatives_md = "\n".join(lines)

    cands = result["candidates"]
    debug_lines = [
        "| # | Text | Combined | Whisper | GPT2 | Fingerprint | Source |",
        "|---|------|----------|---------|------|-------------|--------|",
    ]
    for c in cands:
        debug_lines.append(
            f"| {c['rank']} | {c['text']} | {c['combined']} | "
            f"{c['whisper_score']} | {c['gpt2_score']} | "
            f"{c['fingerprint_score']} | {c['source']} |"
        )
    trigger = result.get("trigger_reason") or "—"
    debug_lines.append(f"\n**Confidence trigger:** {trigger}")
    debug_md = "\n".join(debug_lines)

    return corrected, badge, alternatives_md, debug_md


# ---------------------------------------------------------------------------
# Gradio interface
# ---------------------------------------------------------------------------

def build_ui() -> gr.Blocks:
    fingerprint_choices = sorted(FINGERPRINTS.keys())
    sample_choices = ["(custom input)"] + [
        f"{s['id']} — {s.get('reference', '')} [{s['severity']}]"
        for s in DEMO_SAMPLES
    ]
    sample_id_map = {
        f"{s['id']} — {s.get('reference', '')} [{s['severity']}]": s["id"]
        for s in DEMO_SAMPLES
    }

    default_nbest = json.dumps(
        [
            {"text": "e dresse imself", "score": -2.3},
            {"text": "he dresses himself", "score": -2.8},
            {"text": "he addresses himself", "score": -3.1},
        ],
        indent=2,
    )
    default_phonemes = "IY D R EH S IY M S EH L F"

    with gr.Blocks(title="Dysarthric Speech Correction", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            "# Dysarthric Speech Correction\n"
            "Corrects dysarthric ASR output using speaker-specific phoneme fingerprints and DistilGPT2 reranking."
        )

        with gr.Row():
            with gr.Column(scale=1):
                fp_dropdown = gr.Dropdown(
                    choices=fingerprint_choices,
                    value=fingerprint_choices[0] if fingerprint_choices else None,
                    label="Speaker fingerprint",
                )
                sample_dropdown = gr.Dropdown(
                    choices=sample_choices,
                    value=sample_choices[1] if len(sample_choices) > 1 else sample_choices[0],
                    label="Sample input (or choose 'custom input')",
                )
                gr.Markdown("**Custom input** (used when 'custom input' is selected above)")
                nbest_input = gr.Textbox(
                    label="Whisper n-best JSON array",
                    value=default_nbest,
                    lines=6,
                )
                phonemes_input = gr.Textbox(
                    label="Observed phonemes (space-separated Arpabet)",
                    value=default_phonemes,
                    lines=2,
                )
                run_btn = gr.Button("Correct", variant="primary")

            with gr.Column(scale=2):
                gr.Markdown("### Corrected output")
                corrected_box = gr.Textbox(
                    label="Best correction",
                    lines=2,
                    interactive=False,
                )
                badge_html = gr.HTML(label="Confidence")

                with gr.Accordion("Alternatives (shown when ambiguous)", open=True):
                    alternatives_md = gr.Markdown()

                with gr.Accordion("Debug: full candidate table", open=False):
                    debug_md = gr.Markdown()

        def on_run(fp_key, sample_label, nbest_json, phonemes_str):
            sid = sample_id_map.get(sample_label, "")
            return run_correction(fp_key, sid, nbest_json, phonemes_str)

        run_btn.click(
            fn=on_run,
            inputs=[fp_dropdown, sample_dropdown, nbest_input, phonemes_input],
            outputs=[corrected_box, badge_html, alternatives_md, debug_md],
        )

    return demo


if __name__ == "__main__":
    ui = build_ui()
    ui.launch(server_name="0.0.0.0", server_port=7860, share=False)
