#!/usr/bin/env python
"""Run the full C6 user-role sweep: suffixed then unsuffixed.

Runs all 5 gradient points (0v4, 1v3, 2v2, 3v1, 4v0) twice:
  1. Suffixed   — standard protocol (prompt ends with "The correct answer is (")
  2. Unsuffixed — priming suffix stripped via strip_priming_suffix()

Output files:
  results/c6_u_0v4.pkl  …  c6_u_4v0.pkl
  results/c6_u_0v4_nosuffix.pkl  …  c6_u_4v0_nosuffix.pkl

Progress is printed after each gradient point. A summary table is printed
when each sweep (suffixed / unsuffixed) completes.

Usage:
    python scripts/run_c6_user_sweep.py
"""
from __future__ import annotations

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
    build_prompt_c6_user_role,
)

GRADIENT_POINTS = [(0, 4), (1, 3), (2, 2), (3, 1), (4, 0)]


def _load_c6_corpora(art: dict) -> tuple[dict, dict]:
    """Load and merge all four wrong-arguing and correct-arguing corpora."""
    phi_wrong_path = RESULTS_DIR / "jury_responses_phi_wrong.json"
    correct_path = RESULTS_DIR / "jury_responses_correct.json"
    for p in (phi_wrong_path, correct_path):
        if not p.exists():
            raise SystemExit(
                f"Missing {p} — generate C6 jury corpora first "
                "(see scripts/generate_c6_jury.py)"
            )
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


def _make_c6_closure(assignments, jury_wrong, jury_correct, tokenizer, unsuffixed: bool):
    """Return a build_prompt_fn conforming to the standard run_experiment interface."""
    def _closure(q_idx, item, wrong_idx, jury_data, tok):
        wrong_agents = assignments[q_idx]["wrong"]
        correct_agents = assignments[q_idx]["correct"]
        prompt = build_prompt_c6_user_role(
            q_idx, item, wrong_idx,
            wrong_agents, correct_agents,
            jury_wrong, jury_correct,
            tokenizer=tok,
        )
        if unsuffixed:
            prompt = strip_priming_suffix(prompt)
        return prompt
    return _closure


def _run_one_point(
    k_wrong: int,
    k_correct: int,
    art: dict,
    jury_wrong: dict,
    jury_correct: dict,
    model,
    tokenizer,
    clean_lda: CleanLDA,
    unsuffixed: bool,
) -> dict:
    label = f"c6_u_{k_wrong}v{k_correct}" + ("_nosuffix" if unsuffixed else "")
    desc = f"C6_user_{k_wrong}v{k_correct}" + ("_nosuffix" if unsuffixed else "")
    n_questions = len(art["known_questions"])
    assignments = assign_agents_per_question(k_wrong, n_questions, seed=42)

    build_fn = _make_c6_closure(assignments, jury_wrong, jury_correct, tokenizer, unsuffixed)

    print(f"\n{'=' * 60}")
    print(f"{'[NOSUFFIX] ' if unsuffixed else ''}C6 user-role  {k_wrong}v{k_correct}  "
          f"({k_wrong} wrong / {k_correct} correct)")
    print(f"{'=' * 60}")

    if unsuffixed:
        result = run_experiment_unsuffixed(
            model=model,
            tokenizer=tokenizer,
            known_questions=art["known_questions"],
            final_probes=art["final_probes"],
            build_prompt_fn=build_fn,
            jury_data=art["jury_strong"],   # for wrong_idx lookup
            description=desc,
            clean_lda=clean_lda,
        )
    else:
        result = run_experiment(
            build_fn,
            art["jury_strong"],             # for wrong_idx lookup
            model,
            tokenizer,
            description=desc,
            clean_lda=clean_lda,
        )

    result["gradient_point"] = (k_wrong, k_correct)
    result["framing"] = "user"
    result["unsuffixed"] = unsuffixed
    result["agent_assignments"] = assignments

    out_path = RESULTS_DIR / f"{label}.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(result, f)
    print(f"  saved -> {out_path}")
    return result


def _print_sweep_summary(results: list[dict], protocol: str) -> None:
    print(f"\n{'#' * 60}")
    print(f"SUMMARY — C6 user-role  [{protocol}]")
    print(f"{'#' * 60}")
    header = f"  {'Point':<8}  {'Yield':>8}  {'Onset (binary)':>16}  {'Mean tokens':>12}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for res in results:
        kw, kc = res["gradient_point"]
        yield_pct = res["yield_rate"] * 100
        onset = res.get("onset", "N/A")
        mean_tok = np.mean(res["token_counts"])
        print(f"  {kw}v{kc:<5}  {yield_pct:>7.1f}%  {str(onset):>16}  {mean_tok:>11.0f}")
    print()


def main() -> int:
    print("Loading artifacts and subject model...")
    art = load_artifacts()
    clean_lda = CleanLDA.fit_default()
    model, tokenizer = get_model_and_tokenizer()

    jury_wrong, jury_correct = _load_c6_corpora(art)
    print(f"Loaded C6 corpora: {len(art['known_questions'])} questions, "
          f"{len(jury_wrong)} wrong-arguing models, "
          f"{len(jury_correct)} correct-arguing models")

    # ── Sweep 1: suffixed ────────────────────────────────────────────────────
    print(f"\n{'#' * 60}")
    print("SWEEP 1 of 2: SUFFIXED PROTOCOL (5 gradient points)")
    print(f"{'#' * 60}")

    suffixed_results = []
    for k_wrong, k_correct in GRADIENT_POINTS:
        res = _run_one_point(
            k_wrong, k_correct, art, jury_wrong, jury_correct,
            model, tokenizer, clean_lda, unsuffixed=False,
        )
        suffixed_results.append(res)
        print(f"\n>>> c6_u_{k_wrong}v{k_correct} done  "
              f"yield={res['yield_rate'] * 100:.1f}%\n")

    _print_sweep_summary(suffixed_results, "suffixed")

    # ── Sweep 2: unsuffixed ──────────────────────────────────────────────────
    print(f"\n{'#' * 60}")
    print("SWEEP 2 of 2: UNSUFFIXED PROTOCOL (5 gradient points)")
    print(f"{'#' * 60}")

    unsuffixed_results = []
    for k_wrong, k_correct in GRADIENT_POINTS:
        res = _run_one_point(
            k_wrong, k_correct, art, jury_wrong, jury_correct,
            model, tokenizer, clean_lda, unsuffixed=True,
        )
        unsuffixed_results.append(res)
        print(f"\n>>> c6_u_{k_wrong}v{k_correct}_nosuffix done  "
              f"yield={res['yield_rate'] * 100:.1f}%\n")

    _print_sweep_summary(unsuffixed_results, "unsuffixed")

    # ── Combined comparison table ────────────────────────────────────────────
    print(f"{'#' * 60}")
    print("C6 USER-ROLE: SUFFIXED vs UNSUFFIXED COMPARISON")
    print(f"{'#' * 60}")
    header = f"  {'Point':<8}  {'Suffixed':>9}  {'Unsuffixed':>11}  {'Δ (pp)':>8}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for suf, uns in zip(suffixed_results, unsuffixed_results):
        kw, kc = suf["gradient_point"]
        s_y = suf["yield_rate"] * 100
        u_y = uns["yield_rate"] * 100
        delta = u_y - s_y
        print(f"  {kw}v{kc:<5}  {s_y:>8.1f}%  {u_y:>10.1f}%  {delta:>+8.1f}")
    print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
