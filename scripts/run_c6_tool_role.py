#!/usr/bin/env python
"""Run C6 tool-role disagreement-gradient sweeps for N=4, 5, and 6.

Uses the N-generalized tool-role builder in :mod:`src.c6_tool_role`.
Reuses the corpora loading logic from ``scripts/run_c6_scaling.py``.

Output pickles: results/c6_scaling/c6_t_N{N}_{kw}v{kc}.pkl

Usage:
    python scripts/run_c6_tool_role.py
"""
from __future__ import annotations

import json
import os
import pickle
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.bootstrap import bootstrap_yield_ci
from src.c6_scaling import (
    agents_for_N,
    assign_agents_per_question_N,
)
from src.c6_tool_role import build_prompt_c6_tool_role_N
from src.config import LDA_LAYER, RESULTS_DIR
from src.data import load_artifacts
from src.experiment import run_experiment
from src.lda import CleanLDA
from src.model import get_model_and_tokenizer

OUT_DIR = RESULTS_DIR / "c6_scaling"
OUT_DIR.mkdir(exist_ok=True)


def _gradient_points_for_N(N: int) -> list[tuple[int, int]]:
    return [(k, N - k) for k in range(N + 1)]


def _load_corpora(N: int, art: dict) -> tuple[dict, dict]:
    agents = agents_for_N(N)

    phi_wrong_path = RESULTS_DIR / "jury_responses_phi_wrong.json"
    correct_path = RESULTS_DIR / "disagreement" / "jury_responses_correct.json"

    with open(phi_wrong_path) as f:
        phi_wrong = json.load(f)
    with open(correct_path) as f:
        jury_correct_4 = json.load(f)

    phi_correct_path = RESULTS_DIR / "jury_responses_phi_correct.json"
    with open(phi_correct_path) as f:
        phi_correct = json.load(f)

    jury_wrong = {
        "gemma":   art["jury_strong"]["gemma"],
        "qwen":    art["jury_strong"]["qwen"],
        "mistral": art["jury_strong"]["mistral"],
        "phi":     phi_wrong["phi"],
    }
    jury_correct = {
        "gemma":   jury_correct_4["gemma"],
        "qwen":    jury_correct_4["qwen"],
        "mistral": jury_correct_4["mistral"],
        "phi":     phi_correct["phi"],
    }

    if "llama32" in agents:
        with open(RESULTS_DIR / "jury_responses_llama32_wrong.json") as f:
            jury_wrong["llama32"] = json.load(f)["llama32"]
        with open(RESULTS_DIR / "jury_responses_llama32_correct.json") as f:
            jury_correct["llama32"] = json.load(f)["llama32"]

    if "yi15" in agents:
        with open(RESULTS_DIR / "jury_responses_yi15_wrong.json") as f:
            jury_wrong["yi15"] = json.load(f)["yi15"]
        with open(RESULTS_DIR / "jury_responses_yi15_correct.json") as f:
            jury_correct["yi15"] = json.load(f)["yi15"]

    return jury_wrong, jury_correct


def _make_closure(assignments, jury_wrong, jury_correct):
    def _closure(q_idx, item, wrong_idx, jury_data, tokenizer):
        a = assignments[q_idx]
        return build_prompt_c6_tool_role_N(
            q_idx, item, wrong_idx,
            a["wrong"], a["correct"],
            jury_wrong, jury_correct,
            tokenizer,
        )
    return _closure


def _run_one(
    N: int,
    k_wrong: int,
    k_correct: int,
    art: dict,
    jury_wrong: dict,
    jury_correct: dict,
    model,
    tokenizer,
    clean_lda: CleanLDA,
) -> dict:
    label = f"c6_t_N{N}_{k_wrong}v{k_correct}"
    out_path = OUT_DIR / f"{label}.pkl"
    if out_path.exists():
        print(f"  skip (exists) -> {out_path}")
        with open(out_path, "rb") as f:
            return pickle.load(f)

    n_questions = len(art["known_questions"])
    assignments = assign_agents_per_question_N(k_wrong, n_questions, N=N, seed=42)
    build_fn = _make_closure(assignments, jury_wrong, jury_correct)

    print(f"\n{'=' * 60}")
    print(f"N={N}  tool-framing  {k_wrong}v{k_correct}  "
          f"({k_wrong} wrong / {k_correct} correct)")
    print(f"{'=' * 60}")

    result = run_experiment(
        build_fn,
        art["jury_strong"],
        model,
        tokenizer,
        description=label,
        clean_lda=clean_lda,
    )

    correct_labels = np.array(
        [item["answer"] for item in art["known_questions"]], dtype=np.int64
    )
    boot = bootstrap_yield_ci(
        result["activations"][:, LDA_LAYER, :].astype(np.float32),
        correct_labels,
        result["wrong_indices"],
        clean_lda,
        n_iter=1000,
        ci=0.95,
        seed=42,
    )

    result["gradient_point"] = (k_wrong, k_correct)
    result["framing"] = "tool"
    result["unsuffixed"] = False
    result["N"] = N
    result["agent_assignments"] = assignments
    result["bootstrap_ci"] = {
        "mean": boot.mean, "lo": boot.lo, "hi": boot.hi,
        "se": boot.se, "n_iter": boot.n_iter,
    }

    with open(out_path, "wb") as f:
        pickle.dump(result, f)
    print(f"  saved -> {out_path}")
    print(f"  yield = {result['yield_rate']*100:.2f}%  "
          f"CI = [{boot.lo*100:.2f}, {boot.hi*100:.2f}]")
    return result


def _print_summary(N: int, results: list[dict]) -> None:
    print(f"\n{'#' * 60}")
    print(f"SUMMARY - N={N} tool-framing")
    print(f"{'#' * 60}")
    print(f"  {'Point':<8} {'Yield':>8} {'95% CI':>20} {'Mean tok':>10}")
    print("  " + "-" * 50)
    for r in results:
        kw, kc = r["gradient_point"]
        y = r["yield_rate"] * 100
        lo = r["bootstrap_ci"]["lo"] * 100
        hi = r["bootstrap_ci"]["hi"] * 100
        mt = float(np.mean(r["token_counts"]))
        print(f"  {kw}v{kc:<5} {y:>7.2f}%  [{lo:>5.2f}, {hi:>5.2f}]  {mt:>10.0f}")
    print()


def main() -> int:
    print("Loading artifacts and subject model...")
    art = load_artifacts()
    clean_lda = CleanLDA.fit_default()
    model, tokenizer = get_model_and_tokenizer()

    for N in [4, 5, 6]:
        jury_wrong, jury_correct = _load_corpora(N, art)
        print(f"\nLoaded corpora: N={N}, "
              f"wrong agents = {list(jury_wrong.keys())}, "
              f"correct agents = {list(jury_correct.keys())}")

        points = _gradient_points_for_N(N)
        results = []
        for k_wrong, k_correct in points:
            r = _run_one(
                N, k_wrong, k_correct,
                art, jury_wrong, jury_correct,
                model, tokenizer, clean_lda,
            )
            results.append(r)

        _print_summary(N, results)

    print("\nAll tool-role gradient points complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
