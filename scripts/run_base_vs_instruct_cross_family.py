#!/usr/bin/env python
"""Cross-family base-vs-instruct replication (Experiment #3).

For each model family (mistral, gemma, qwen), loads the BASE model but uses
the INSTRUCT tokenizer for prompt formatting (base models lack chat
templates). Runs clean pass, trains probes, then runs C4a/C4d/C4c_matched
conditions. Saves results to results/base_model/{family}/.

Usage::

    python scripts/run_base_vs_instruct_cross_family.py --family mistral
    python scripts/run_base_vs_instruct_cross_family.py --family gemma
    python scripts/run_base_vs_instruct_cross_family.py --family qwen
"""
from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "4")

import argparse
import csv
import pickle
import sys
from pathlib import Path

import joblib
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import HF_TOKEN, RESULTS_DIR
from src.cross_model import (
    _ChatTemplateWrapper,
    collect_clean,
    filter_jury_exclude_subject,
    run_cross_experiment_full,
    train_subject_probes,
)
from src.data import load_artifacts
from src.prompts import (
    build_prompt_no_attribution_matched,
    build_prompt_self_framing,
    build_prompt_user_role_jury,
)
from transformers import AutoModelForCausalLM, AutoTokenizer

FAMILY_CONFIG = {
    "mistral": {
        "base_id": "mistralai/Mistral-7B-v0.3",
        "instruct_id": "mistralai/Mistral-7B-Instruct-v0.3",
        "jury_key": "mistral",
    },
    "gemma": {
        "base_id": "google/gemma-2-9b",
        "instruct_id": "google/gemma-2-9b-it",
        "jury_key": "gemma",
    },
    "qwen": {
        "base_id": "Qwen/Qwen2.5-7B",
        "instruct_id": "Qwen/Qwen2.5-7B-Instruct",
        "jury_key": "qwen",
    },
}

CONDITIONS = {
    "c4a": build_prompt_user_role_jury,
    "c4d": build_prompt_self_framing,
    "c4c_matched": build_prompt_no_attribution_matched,
}

DEVICE = "cuda:0"


def bootstrap_ci(values, n_boot=1000, seed=42, ci=0.95):
    rng = np.random.RandomState(seed)
    arr = np.asarray(values, dtype=float)
    means = np.array([rng.choice(arr, size=len(arr), replace=True).mean()
                       for _ in range(n_boot)])
    lo = np.percentile(means, (1 - ci) / 2 * 100)
    hi = np.percentile(means, (1 + ci) / 2 * 100)
    return float(lo), float(hi)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--family", required=True, choices=list(FAMILY_CONFIG.keys()))
    p.add_argument("--threshold", type=float, default=0.8)
    args = p.parse_args()

    family = args.family
    cfg = FAMILY_CONFIG[family]
    base_id = cfg["base_id"]
    instruct_id = cfg["instruct_id"]
    jury_key = cfg["jury_key"]

    out_dir = RESULTS_DIR / "base_model" / family
    out_dir.mkdir(parents=True, exist_ok=True)
    probes_dir = RESULTS_DIR / "probes" / f"base_model_{family}"
    probes_dir.mkdir(parents=True, exist_ok=True)

    print(f"Family: {family}")
    print(f"Base model: {base_id}")
    print(f"Instruct tokenizer: {instruct_id}")
    print(f"Output: {out_dir}")

    # Load artifacts
    art = load_artifacts()
    questions = art["known_questions"]
    jury_strong_raw = art["jury_strong"]
    jury_strong = filter_jury_exclude_subject(jury_strong_raw, jury_key)
    print(f"Jury lineup (after removing {jury_key}): {list(jury_strong.keys())}")

    # Load tokenizer from the Instruct variant (base models lack chat templates)
    print(f"\nLoading Instruct tokenizer from {instruct_id}...")
    raw_tok = AutoTokenizer.from_pretrained(instruct_id, token=HF_TOKEN)
    if raw_tok.pad_token is None:
        raw_tok.pad_token = raw_tok.eos_token
    tok = _ChatTemplateWrapper(raw_tok)
    if tok._merge_system:
        print("  (system→user merge active)")

    # Load the BASE model
    print(f"Loading base model {base_id} on {DEVICE}...")
    model = AutoModelForCausalLM.from_pretrained(
        base_id,
        torch_dtype=torch.bfloat16,
        device_map={"": DEVICE},
        token=HF_TOKEN,
    )
    model.eval()
    n_layers = len(model.model.layers)
    d_model = model.config.hidden_size
    print(f"  layers={n_layers}, d_model={d_model}")

    # Step 1: Clean pass
    print(f"\n[1] Clean pass on {family} base model...")
    threshold = args.threshold
    base_art = collect_clean(model, tok, questions, threshold=threshold)
    n_pass = int(base_art.passing_mask.sum())
    print(f"  {n_pass}/{len(questions)} pass P(correct) > {threshold}")
    print(f"  mean clean P(correct): {base_art.clean_truth_probs.mean():.3f}")

    if n_pass < 50:
        print(f"  WARNING: only {n_pass} pass at {threshold} — lowering to 0.5")
        threshold = 0.5
        base_art = collect_clean(model, tok, questions, threshold=threshold)
        n_pass = int(base_art.passing_mask.sum())
        print(f"  {n_pass}/{len(questions)} pass at threshold={threshold}")

    if n_pass < 12:
        print(f"  FATAL: only {n_pass} pass even at 0.5 — cannot proceed")
        with open(out_dir / "summary.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["condition", "yield_pct", "onset", "final_probe", "n_questions"])
            w.writeheader()
        print("  wrote empty summary.csv")
        return 1

    passing_indices = np.where(base_art.passing_mask)[0].astype(np.int64)
    print(f"  LDA layer: {base_art.lda_layer}")

    # Step 2: Train probes
    print(f"\n[2] Training probes on {family} base clean activations...")
    pass_labels = np.array(
        [questions[i]["answer"] for i in passing_indices], dtype=np.int64
    )
    pass_acts = base_art.clean_activations[passing_indices].astype(np.float32)
    n_layers_train = pass_acts.shape[1]
    base_probes, avg_probe_accs = train_subject_probes(
        pass_acts, pass_labels, n_layers=n_layers_train
    )
    joblib.dump(base_probes, probes_dir / "final_probes.joblib")
    joblib.dump(avg_probe_accs, probes_dir / "avg_probe_accs.joblib")
    print(f"  saved {n_layers_train} probes -> {probes_dir}")

    # Save clean artifacts
    with open(out_dir / "clean.pkl", "wb") as f:
        pickle.dump({
            "model_id": base_id,
            "instruct_tokenizer_id": instruct_id,
            "num_layers": n_layers,
            "hidden_dim": d_model,
            "clean_truth_probs": base_art.clean_truth_probs,
            "clean_answer_probs": base_art.clean_answer_probs,
            "passing_mask": base_art.passing_mask,
            "n_pass": n_pass,
            "threshold": threshold,
            "lda_layer": base_art.lda_layer,
            "lda_centroids": base_art.lda_centroids,
            "avg_probe_accs": avg_probe_accs,
            "mean_clean_p_correct": float(base_art.clean_truth_probs.mean()),
        }, f)
    with open(out_dir / "lda.pkl", "wb") as f:
        pickle.dump(base_art.lda, f)

    # Step 3: Run C4a, C4d, C4c_matched
    summary_rows = []
    for cond_name, builder in CONDITIONS.items():
        print(f"\n{'=' * 60}")
        print(f"CONDITION: {cond_name} (family={family}, base, n={n_pass})")
        print(f"{'=' * 60}")

        result = run_cross_experiment_full(
            builder, jury_strong, model, tok,
            passing_indices=passing_indices,
            questions=questions,
            lda=base_art.lda,
            centroids=base_art.lda_centroids,
            lda_layer=base_art.lda_layer,
            subject_probes=base_probes,
            description=f"base_{family}/{cond_name}",
        )

        cond_path = out_dir / f"{cond_name}.pkl"
        with open(cond_path, "wb") as f:
            pickle.dump(result, f)

        yield_pct = result["yield_rate"] * 100
        ci = bootstrap_ci(result["yielded"].astype(float))
        ci = (ci[0] * 100, ci[1] * 100)
        onset = result.get("onset")
        final_probe = result["probe_accs"][-1] if result["probe_accs"] else None

        row = {
            "condition": cond_name,
            "yield_pct": round(yield_pct, 2),
            "ci_lo": round(ci[0], 2),
            "ci_hi": round(ci[1], 2),
            "onset": onset,
            "final_probe": round(final_probe, 4) if final_probe is not None else None,
            "n_questions": n_pass,
        }
        summary_rows.append(row)
        print(f"  -> yield={yield_pct:.1f}% [{ci[0]:.1f}, {ci[1]:.1f}], "
              f"onset={onset}, final_probe={row['final_probe']}")

    # Release model
    del model
    torch.cuda.empty_cache()

    # Write summary CSV
    csv_path = out_dir / "summary.csv"
    fieldnames = list(summary_rows[0].keys())
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(summary_rows)
    print(f"\nSaved summary -> {csv_path}")

    # Print table
    print(f"\n=== {family.upper()} BASE MODEL RESULTS ===")
    print(f"Base: {base_id} | Tokenizer: {instruct_id}")
    print(f"Clean filter: {n_pass}/{len(questions)} pass at threshold={threshold}")
    print(f"\n{'Condition':<15} {'Yield%':>8} {'95% CI':>18} {'Onset':>7} {'FinalProbe':>11}")
    print("-" * 65)
    for r in summary_rows:
        onset_str = str(r["onset"]) if r["onset"] is not None else "-"
        fp_str = f"{r['final_probe']:.4f}" if r["final_probe"] is not None else "-"
        print(f"{r['condition']:<15} {r['yield_pct']:>8.1f} "
              f"[{r['ci_lo']:>6.1f}, {r['ci_hi']:>6.1f}] {onset_str:>7} {fp_str:>11}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
