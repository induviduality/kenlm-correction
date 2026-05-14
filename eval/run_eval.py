"""
Evaluation harness: WER table per speaker + ambiguity rate.

For each test case in eval/test_set.json, runs 4 conditions:
  1. Whisper top-1 only                  (baseline WER)
  2. Whisper n-best + GPT2 rerank        (intermediate WER)
  3. Full pipeline, top-1               (system WER)
  4. Full pipeline, oracle from alts    (best-case WER if user picks correctly)

Outputs a Markdown WER table per severity tier + ambiguity rate column.

Run:
    python eval/run_eval.py
    python eval/run_eval.py --test-set eval/test_set.json --config config.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

# Ensure project root is on path
HERE = Path(__file__).parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

import jiwer
from src import pipeline, scoring, fusion, confidence as conf_mod
from src.candidate_gen import generate_candidates


# ---------------------------------------------------------------------------
# WER helper
# ---------------------------------------------------------------------------

def _wer(reference: str, hypothesis: str) -> float:
    if not reference.strip():
        return 0.0
    return jiwer.wer(reference.lower(), hypothesis.lower())


# ---------------------------------------------------------------------------
# Condition 2: Whisper n-best + GPT2 only (no fingerprint)
# ---------------------------------------------------------------------------

def _gpt2_rerank(whisper_nbest: list[dict], config: dict) -> str:
    """Re-rank whisper n-best using GPT2 only, return top-1 text."""
    model_cfg = config.get("model", {})
    scoring_cfg = config.get("scoring", {})
    texts = [e["text"] for e in whisper_nbest]
    gpt2_scores = scoring.gpt2_score(
        texts,
        batch_size=scoring_cfg.get("gpt2_batch_size", 40),
        model_name=model_cfg.get("gpt2_name", "distilgpt2"),
        device_cfg=model_cfg.get("device", "auto"),
        max_length=scoring_cfg.get("gpt2_max_length", 128),
    )
    best_idx = max(range(len(gpt2_scores)), key=lambda i: gpt2_scores[i])
    return texts[best_idx]


# ---------------------------------------------------------------------------
# Oracle: best WER achievable from pipeline alternatives
# ---------------------------------------------------------------------------

def _oracle_from_alts(alts: list[dict], reference: str) -> str:
    """Return the alternative with lowest WER vs. reference."""
    if not alts:
        return ""
    return min(alts, key=lambda a: _wer(reference, a["text"]))["text"]


# ---------------------------------------------------------------------------
# Main eval loop
# ---------------------------------------------------------------------------

def run_eval(test_set_path: Path, config_path: Path, fingerprints_dir: Path):
    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    with open(test_set_path, encoding="utf-8") as f:
        test_cases = json.load(f)

    # Load fingerprints
    fingerprints: dict[str, dict] = {}
    for fp_file in sorted(fingerprints_dir.glob("*.json")):
        with open(fp_file, encoding="utf-8") as f:
            data = json.load(f)
        fingerprints[fp_file.stem] = data

    rows = []

    for tc in test_cases:
        sid = tc["speaker_id"]
        dataset = tc.get("dataset", "TORGO")
        key = f"{dataset}_{sid}"
        if key not in fingerprints:
            print(f"  [SKIP] fingerprint not found: {key}", flush=True)
            continue

        fp = fingerprints[key]
        reference = tc["reference"]
        whisper_nbest = tc["whisper_nbest"]
        observed = tc["observed_phonemes"]
        severity = tc.get("severity", fp.get("severity", "moderate"))

        print(f"  {tc['id']:30s}  ref='{reference}'", end=" ", flush=True)

        # Condition 1: Whisper top-1
        w_top1 = whisper_nbest[0]["text"]
        wer_w = _wer(reference, w_top1)

        # Condition 2: GPT2 rerank
        gpt2_best = _gpt2_rerank(whisper_nbest, config)
        wer_g = _wer(reference, gpt2_best)

        # Conditions 3 + 4: full pipeline
        result = pipeline.correct(whisper_nbest, observed, fp, config)
        top1 = result["corrected"]
        wer_p = _wer(reference, top1)

        oracle_text = _oracle_from_alts(result["alternatives"], reference)
        wer_o = _wer(reference, oracle_text) if oracle_text else wer_p

        is_ambiguous = result["status"] == "ambiguous"

        rows.append({
            "id": tc["id"],
            "speaker_id": sid,
            "severity": severity,
            "reference": reference,
            "whisper_top1": w_top1,
            "gpt2_best": gpt2_best,
            "pipeline_top1": top1,
            "oracle": oracle_text or top1,
            "wer_whisper": round(wer_w, 3),
            "wer_gpt2": round(wer_g, 3),
            "wer_pipeline": round(wer_p, 3),
            "wer_oracle": round(wer_o, 3),
            "ambiguous": is_ambiguous,
            "trigger": result.get("trigger_reason", ""),
        })
        tag = "[AMB]" if is_ambiguous else "     "
        print(
            f"{tag}  W={wer_w:.3f}  G={wer_g:.3f}  P={wer_p:.3f}  O={wer_o:.3f}",
            flush=True,
        )

    # ──────────────────────────────────────────────────────────────────────
    # Print per-severity summary table
    # ──────────────────────────────────────────────────────────────────────
    severity_groups: dict[str, list[dict]] = {}
    for r in rows:
        severity_groups.setdefault(r["severity"], []).append(r)

    print("\n" + "=" * 80)
    print("EVALUATION RESULTS")
    print("=" * 80)

    all_rows_flat: list[dict] = []

    for sev in ["severe", "moderate", "mild"]:
        group = severity_groups.get(sev, [])
        if not group:
            continue
        n = len(group)
        avg_w = sum(r["wer_whisper"] for r in group) / n
        avg_g = sum(r["wer_gpt2"] for r in group) / n
        avg_p = sum(r["wer_pipeline"] for r in group) / n
        avg_o = sum(r["wer_oracle"] for r in group) / n
        amb_rate = sum(1 for r in group if r["ambiguous"]) / n

        print(f"\n### Severity: {sev.upper()}  (n={n})")
        print(
            f"{'ID':<30}  {'Whisper':>7}  {'GPT2':>7}  {'Pipeline':>8}  {'Oracle':>6}  {'Ambig':>5}  Trigger"
        )
        print("-" * 90)
        for r in group:
            flag = "Y" if r["ambiguous"] else "N"
            print(
                f"{r['id']:<30}  {r['wer_whisper']:>7.3f}  {r['wer_gpt2']:>7.3f}  "
                f"{r['wer_pipeline']:>8.3f}  {r['wer_oracle']:>6.3f}  {flag:>5}  {r['trigger']}"
            )
        print("-" * 90)
        print(
            f"{'AVERAGE':<30}  {avg_w:>7.3f}  {avg_g:>7.3f}  {avg_p:>8.3f}  {avg_o:>6.3f}  {amb_rate:>4.0%}"
        )
        all_rows_flat.extend(group)

    # Overall
    if all_rows_flat:
        n = len(all_rows_flat)
        print(f"\n### OVERALL  (n={n})")
        print(
            f"  Whisper WER   : {sum(r['wer_whisper'] for r in all_rows_flat)/n:.3f}"
        )
        print(
            f"  GPT2 WER      : {sum(r['wer_gpt2'] for r in all_rows_flat)/n:.3f}"
        )
        print(
            f"  Pipeline WER  : {sum(r['wer_pipeline'] for r in all_rows_flat)/n:.3f}"
        )
        print(
            f"  Oracle WER    : {sum(r['wer_oracle'] for r in all_rows_flat)/n:.3f}"
        )
        print(
            f"  Ambiguity rate: {sum(1 for r in all_rows_flat if r['ambiguous'])/n:.0%}"
        )

    # ──────────────────────────────────────────────────────────────────────
    # Markdown table (for copy-paste into report)
    # ──────────────────────────────────────────────────────────────────────
    print("\n\n## Markdown table\n")
    print("| Severity | n | WER Whisper | WER GPT2 | WER Pipeline | WER Oracle | Ambiguity Rate |")
    print("|----------|---|------------|----------|--------------|------------|----------------|")
    for sev in ["severe", "moderate", "mild"]:
        group = severity_groups.get(sev, [])
        if not group:
            continue
        n = len(group)
        avg_w = sum(r["wer_whisper"] for r in group) / n
        avg_g = sum(r["wer_gpt2"] for r in group) / n
        avg_p = sum(r["wer_pipeline"] for r in group) / n
        avg_o = sum(r["wer_oracle"] for r in group) / n
        amb = sum(1 for r in group if r["ambiguous"]) / n
        print(
            f"| {sev:<8} | {n} | {avg_w:.3f} | {avg_g:.3f} | {avg_p:.3f} | {avg_o:.3f} | {amb:.0%} |"
        )

    return rows


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluate the correction pipeline")
    parser.add_argument(
        "--test-set",
        type=Path,
        default=ROOT / "eval" / "test_set.json",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "config.yaml",
    )
    parser.add_argument(
        "--fingerprints-dir",
        type=Path,
        default=ROOT / "fingerprints",
    )
    args = parser.parse_args()

    print(f"Test set        : {args.test_set}")
    print(f"Config          : {args.config}")
    print(f"Fingerprints dir: {args.fingerprints_dir}")
    print()

    run_eval(args.test_set, args.config, args.fingerprints_dir)


if __name__ == "__main__":
    main()
