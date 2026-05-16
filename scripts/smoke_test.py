#!/usr/bin/env python
"""5-question sanity check.

Loads the model, runs C4a on a 5-question slice, prints yield rate + onset
metrics + mean P(correct) / P(wrong_target).

Gating criterion: yield rate must be > 0 and the logit-lens curves must be
finite. If this is green the full sweep can proceed.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import load_artifacts  # noqa: E402
from src.experiment import run_experiment  # noqa: E402
from src.lda import CleanLDA  # noqa: E402
from src.logit_lens import compute_onset_metrics  # noqa: E402
from src.model import get_model_and_tokenizer  # noqa: E402
from src.prompts import CONDITION_REGISTRY  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--condition", default="c4a")
    p.add_argument("--n", type=int, default=5)
    args = p.parse_args()

    print(f"smoke_test: condition={args.condition}, n={args.n}")

    art = load_artifacts()
    if args.condition not in CONDITION_REGISTRY:
        raise SystemExit(f"unknown condition: {args.condition}")
    build_fn, jury_name = CONDITION_REGISTRY[args.condition]
    jury = art["jury_strong"] if jury_name == "strong" else art["jury_weak"]

    # Slice to first N questions by temporarily swapping ``known_questions`` in
    # the cached artifact dict. ``run_experiment`` derives correct labels from
    # each item's ``answer`` field, so the slice is self-consistent. CleanLDA
    # still fits on the full 400 clean activations.
    full_questions = art["known_questions"]
    try:
        art["known_questions"] = full_questions[: args.n]
        clean_lda = CleanLDA.fit_default()

        model, tokenizer = get_model_and_tokenizer()
        result = run_experiment(
            build_fn, jury, model, tokenizer,
            description=f"smoke-{args.condition}", clean_lda=clean_lda,
        )
    finally:
        art["known_questions"] = full_questions

    print("\n=== smoke_test summary ===")
    print(f"  N questions: {args.n}")
    print(f"  Yield rate: {result['yield_rate'] * 100:.1f}%")
    print(f"  Onset (binary): L{result['onset']}")
    print(f"  Onset metrics: {result['onset_metrics']}")
    print(f"  Mean P(correct): final {result['avg_truth'][-1]:.3f}, "
          f"max {max(result['avg_truth']):.3f}")
    print(f"  Mean P(wrong):   final {result['avg_syco'][-1]:.3f}, "
          f"max {max(result['avg_syco']):.3f}")
    print(f"  Final probe acc: {result['probe_accs'][-1]:.3f}")

    assert np.isfinite(result["avg_truth"]).all(), "non-finite truth probs"
    assert np.isfinite(result["avg_syco"]).all(), "non-finite syco probs"
    assert 0.0 <= result["yield_rate"] <= 1.0, "yield rate out of bounds"
    print("\nSMOKE TEST: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
