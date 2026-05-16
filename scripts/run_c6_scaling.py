#!/usr/bin/env python
"""Run C6 disagreement-gradient sweeps for N=5 and N=6 jury sizes.

Uses the generalized N-agent builders in :mod:`src.c6_scaling`. Suffixed
protocol only (the standard ``"The correct answer is ("`` priming); reuses
the project-wide CleanLDA basis fit on clean MMLU activations.

Per (N, framing) sweep produces N+1 gradient points:
    N=5: (0,5) (1,4) (2,3) (3,2) (4,1) (5,0)
    N=6: (0,6) (1,5) (2,4) (3,3) (4,2) (5,1) (6,0)

Output pickles are written to results/c6_scaling/c6_{u,s}_N{N}_{kw}v{kc}.pkl
with a schema identical to existing C6 pickles plus bootstrap_ci.

Usage:
    python scripts/run_c6_scaling.py --N 5 --framing user
    python scripts/run_c6_scaling.py --N 5 --framing self
    python scripts/run_c6_scaling.py --N 6 --framing user
    python scripts/run_c6_scaling.py --N 6 --framing self
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

from src.bootstrap import bootstrap_yield_ci  # noqa: E402
from src.c6_scaling import (  # noqa: E402
    AGENTS_N5, AGENTS_N6,
    agents_for_N,
    assign_agents_per_question_N,
    build_prompt_c6_user_role_N,
    build_prompt_c6_self_framing_N,
)
from src.config import DATA_DIR, LDA_LAYER, RESULTS_DIR  # noqa: E402
from src.data import load_artifacts  # noqa: E402
from src.experiment import run_experiment  # noqa: E402
from src.lda import CleanLDA  # noqa: E402
from src.model import get_model_and_tokenizer  # noqa: E402

OUT_DIR = RESULTS_DIR / "c6_scaling"
OUT_DIR.mkdir(exist_ok=True)


def _gradient_points_for_N(N: int) -> list[tuple[int, int]]:
    return [(k, N - k) for k in range(N + 1)]


def _load_corpora(N: int, art: dict) -> tuple[dict, dict]:
    """Load wrong-arguing and correct-arguing jury corpora for all N agents.

    Reuses the existing 4-agent corpora and adds llama32 (N>=5) and yi15 (N=6).
    """
    agents = agents_for_N(N)

    # Existing 4-agent wrong corpus is split across HF (gemma/qwen/mistral in
    # jury_strong) and the local phi corpus.
    phi_wrong_path = RESULTS_DIR / "jury_responses_phi_wrong.json"
    if not phi_wrong_path.exists():
        phi_wrong_path = DATA_DIR / "jury_responses_phi_wrong.json"
    if not phi_wrong_path.exists():
        sys.exit("Missing jury_responses_phi_wrong.json — run scripts/generate_c6_jury.py first")

    correct_path = RESULTS_DIR / "jury_responses_correct.json"
    if not correct_path.exists():
        correct_path = DATA_DIR / "jury_responses_correct.json"
    if not correct_path.exists():
        sys.exit("Missing jury_responses_correct.json — run scripts/generate_c6_jury.py first")

    with open(phi_wrong_path) as f:
        phi_wrong = json.load(f)
    with open(correct_path) as f:
        jury_correct_4 = json.load(f)

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
        "phi":     jury_correct_4["phi"],
    }

    if "llama32" in agents:
        wrong_p = RESULTS_DIR / "jury_responses_llama32_wrong.json"
        if not wrong_p.exists():
            wrong_p = DATA_DIR / "jury_responses_llama32_wrong.json"
        if not wrong_p.exists():
            sys.exit("Missing jury_responses_llama32_wrong.json — run scripts/generate_c6_jury.py first")

        cor_p = RESULTS_DIR / "jury_responses_llama32_correct.json"
        if not cor_p.exists():
            cor_p = DATA_DIR / "jury_responses_llama32_correct.json"
        if not cor_p.exists():
            sys.exit("Missing jury_responses_llama32_correct.json — run scripts/generate_c6_jury.py first")

        with open(wrong_p) as f:
            jury_wrong["llama32"] = json.load(f)["llama32"]
        with open(cor_p) as f:
            jury_correct["llama32"] = json.load(f)["llama32"]

    if "yi15" in agents:
        wrong_p = RESULTS_DIR / "jury_responses_yi15_wrong.json"
        if not wrong_p.exists():
            wrong_p = DATA_DIR / "jury_responses_yi15_wrong.json"
        if not wrong_p.exists():
            sys.exit("Missing jury_responses_yi15_wrong.json — run scripts/generate_c6_jury.py first")

        cor_p = RESULTS_DIR / "jury_responses_yi15_correct.json"
        if not cor_p.exists():
            cor_p = DATA_DIR / "jury_responses_yi15_correct.json"
        if not cor_p.exists():
            sys.exit("Missing jury_responses_yi15_correct.json — run scripts/generate_c6_jury.py first")

        with open(wrong_p) as f:
            jury_wrong["yi15"] = json.load(f)["yi15"]
        with open(cor_p) as f:
            jury_correct["yi15"] = json.load(f)["yi15"]

    return jury_wrong, jury_correct


def _make_closure(framing: str, assignments, jury_wrong, jury_correct):
    builder = (
        build_prompt_c6_user_role_N if framing == "user"
        else build_prompt_c6_self_framing_N
    )

    def _closure(q_idx, item, wrong_idx, jury_data, tokenizer):
        a = assignments[q_idx]
        return builder(
            q_idx, item, wrong_idx,
            a["wrong"], a["correct"],
            jury_wrong, jury_correct,
            tokenizer,
        )
    return _closure


def _run_one(
    N: int,
    framing: str,
    k_wrong: int,
    k_correct: int,
    art: dict,
    jury_wrong: dict,
    jury_correct: dict,
    model,
    tokenizer,
    clean_lda: CleanLDA,
) -> dict:
    label = f"c6_{'u' if framing == 'user' else 's'}_N{N}_{k_wrong}v{k_correct}"
    out_path = OUT_DIR / f"{label}.pkl"
    if out_path.exists():
        print(f"  skip (exists) -> {out_path}")
        with open(out_path, "rb") as f:
            return pickle.load(f)

    n_questions = len(art["known_questions"])
    assignments = assign_agents_per_question_N(k_wrong, n_questions, N=N, seed=42)
    build_fn = _make_closure(framing, assignments, jury_wrong, jury_correct)

    print(f"\n{'=' * 60}")
    print(f"N={N}  {framing}-framing  {k_wrong}v{k_correct}  "
          f"({k_wrong} wrong / {k_correct} correct)")
    print(f"{'=' * 60}")

    result = run_experiment(
        build_fn,
        art["jury_strong"],   # for wrong_idx lookup only
        model,
        tokenizer,
        description=label,
        clean_lda=clean_lda,
    )

    # Bootstrap CI on the LDA-layer activations.
    correct_labels = np.array([item["answer"] for item in art["known_questions"]], dtype=np.int64)
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
    result["framing"] = framing
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


def _print_summary(N: int, framing: str, results: list[dict]) -> None:
    print(f"\n{'#' * 60}")
    print(f"SUMMARY — N={N} {framing}-framing")
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, required=True, choices=[5, 6])
    ap.add_argument("--framing", required=True, choices=["user", "self"])
    args = ap.parse_args()

    print("Loading artifacts and subject model...")
    art = load_artifacts()
    clean_lda = CleanLDA.fit_default()
    model, tokenizer = get_model_and_tokenizer()

    jury_wrong, jury_correct = _load_corpora(args.N, art)
    print(f"Loaded corpora: N={args.N}, "
          f"wrong agents = {list(jury_wrong.keys())}, "
          f"correct agents = {list(jury_correct.keys())}")

    points = _gradient_points_for_N(args.N)
    results = []
    for k_wrong, k_correct in points:
        r = _run_one(
            args.N, args.framing, k_wrong, k_correct,
            art, jury_wrong, jury_correct,
            model, tokenizer, clean_lda,
        )
        results.append(r)

    _print_summary(args.N, args.framing, results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
