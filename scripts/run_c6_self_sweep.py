#!/usr/bin/env python
"""Run the C6 self-framing sweep: all 5 gradient points, suffixed or unsuffixed.

Output files:
    results/c6_s_{kw}v{kc}.pkl             (suffixed, default)
    results/c6_s_{kw}v{kc}_nosuffix.pkl    (--nosuffix)

Prints progress after each gradient point and a summary table when done.
Run run_c6_bootstrap_figure.py after this to update the headline figure.

Usage:
    python scripts/run_c6_self_sweep.py
    python scripts/run_c6_self_sweep.py --nosuffix
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import RESULTS_DIR  # noqa: E402
from src.data import load_artifacts  # noqa: E402
from src.experiment import run_experiment  # noqa: E402
from src.lda import CleanLDA  # noqa: E402
from src.model import get_model_and_tokenizer  # noqa: E402
from src.priming_ablation import run_experiment_unsuffixed, strip_priming_suffix  # noqa: E402
from src.prompts import (  # noqa: E402
    assign_agents_per_question,
    build_prompt_c6_self_framing,
)

GRADIENT_POINTS = [(0, 4), (1, 3), (2, 2), (3, 1), (4, 0)]


def _load_c6_corpora(art: dict) -> tuple[dict, dict]:
    phi_wrong_path = RESULTS_DIR / "jury_responses_phi_wrong.json"
    correct_path = RESULTS_DIR / "jury_responses_correct.json"
    for p in (phi_wrong_path, correct_path):
        if not p.exists():
            raise SystemExit(f"Missing {p}")
    with open(phi_wrong_path) as f:
        jury_phi_wrong = json.load(f)
    with open(correct_path) as f:
        jury_correct = json.load(f)
    jury_wrong = {
        "gemma":   art["jury_strong"]["gemma"],
        "qwen":    art["jury_strong"]["qwen"],
        "mistral": art["jury_strong"]["mistral"],
        "phi":     jury_phi_wrong["phi"],
    }
    return jury_wrong, jury_correct


def _make_closure(assignments, jury_wrong, jury_correct, unsuffixed: bool):
    def _closure(q_idx, item, wrong_idx, jury_data, tok):
        wrong_agents = assignments[q_idx]["wrong"]
        correct_agents = assignments[q_idx]["correct"]
        prompt = build_prompt_c6_self_framing(
            q_idx, item, wrong_idx,
            wrong_agents, correct_agents,
            jury_wrong, jury_correct,
            tokenizer=tok,
        )
        if unsuffixed:
            prompt = strip_priming_suffix(prompt)
        return prompt
    return _closure


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nosuffix", action="store_true",
                    help="Strip the priming suffix (unsuffixed protocol).")
    args = ap.parse_args()
    unsuffixed = args.nosuffix
    protocol = "UNSUFFIXED" if unsuffixed else "SUFFIXED"
    file_suffix = "_nosuffix" if unsuffixed else ""

    print("Loading artifacts and subject model...")
    art = load_artifacts()
    clean_lda = CleanLDA.fit_default()
    model, tokenizer = get_model_and_tokenizer()

    jury_wrong, jury_correct = _load_c6_corpora(art)
    n_questions = len(art["known_questions"])

    print(f"\n{'#' * 60}")
    print(f"C6 SELF-FRAMING SWEEP — {protocol} (5 gradient points)")
    print(f"{'#' * 60}")

    results = []
    for k_wrong, k_correct in GRADIENT_POINTS:
        label = f"c6_s_{k_wrong}v{k_correct}{file_suffix}"
        assignments = assign_agents_per_question(k_wrong, n_questions, seed=42)

        build_fn = _make_closure(assignments, jury_wrong, jury_correct, unsuffixed)

        print(f"\n{'=' * 60}")
        print(f"{'[NOSUFFIX] ' if unsuffixed else ''}C6 self-framing  {k_wrong}v{k_correct}  "
              f"({k_wrong} wrong / {k_correct} correct)")
        print(f"{'=' * 60}")

        if unsuffixed:
            result = run_experiment_unsuffixed(
                model=model,
                tokenizer=tokenizer,
                known_questions=art["known_questions"],
                final_probes=art["final_probes"],
                build_prompt_fn=build_fn,
                jury_data=art["jury_strong"],
                description=label,
                clean_lda=clean_lda,
            )
        else:
            result = run_experiment(
                build_fn,
                art["jury_strong"],
                model,
                tokenizer,
                description=label,
                clean_lda=clean_lda,
            )

        result["gradient_point"] = (k_wrong, k_correct)
        result["framing"] = "self"
        result["unsuffixed"] = unsuffixed
        result["agent_assignments"] = assignments

        out_path = RESULTS_DIR / f"{label}.pkl"
        with open(out_path, "wb") as f:
            pickle.dump(result, f)

        yield_pct = result["yield_rate"] * 100
        onset = result.get("onset", "N/A")
        print(f"  saved -> {out_path}")
        print(f"\n>>> {label} done  yield={yield_pct:.1f}%  onset={onset}\n")
        results.append(result)

    # Summary table.
    print(f"\n{'#' * 60}")
    print("SUMMARY — C6 self-framing [suffixed]")
    print(f"{'#' * 60}")
    print(f"  {'Point':<8}  {'Yield':>8}  {'Onset':>8}  {'Mean tokens':>12}")
    print("  " + "-" * 42)
    for res in results:
        kw, kc = res["gradient_point"]
        print(f"  {kw}v{kc:<5}  {res['yield_rate']*100:>7.1f}%  "
              f"{str(res.get('onset','N/A')):>8}  "
              f"{np.mean(res['token_counts']):>11.0f}")

    # Comparison with user-role suffixed.
    print(f"\n{'#' * 60}")
    print("COMPARISON: self-framing vs user-role (suffixed)")
    print(f"{'#' * 60}")
    print(f"  {'Point':<8}  {'User-role':>10}  {'Self':>8}  {'Δ (pp)':>8}")
    print("  " + "-" * 40)
    for res in results:
        kw, kc = res["gradient_point"]
        self_y = res["yield_rate"] * 100
        user_pkl = RESULTS_DIR / f"c6_u_{kw}v{kc}.pkl"
        if user_pkl.exists():
            with open(user_pkl, "rb") as f:
                user_y = pickle.load(f)["yield_rate"] * 100
            delta = self_y - user_y
            print(f"  {kw}v{kc:<5}  {user_y:>9.1f}%  {self_y:>7.1f}%  {delta:>+8.1f}")
        else:
            print(f"  {kw}v{kc:<5}  {'N/A':>9}  {self_y:>7.1f}%  {'N/A':>8}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
