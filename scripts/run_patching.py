#!/usr/bin/env python
"""Phase 7 entrypoint (E10): activation patching causal check.

Runs the single-layer patching sweep described in NEXT_STEPS E10: for each
``patch_layer`` in {10, 12, 14, 16, 18, 20, 22, 25}, on a seeded 50-question
C4a subset, cache the clean hidden state and substitute it into the
pressured forward pass at the last-token position. Saves full per-question
arrays to ``results/patching.pkl``.
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import RESULTS_DIR  # noqa: E402
from src.model import get_model_and_tokenizer  # noqa: E402
from src.patching import run_activation_patching  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=50)
    p.add_argument(
        "--layers", default="10,12,14,16,18,20,22,25",
        help="comma-separated patch layers",
    )
    args = p.parse_args()

    layers = [int(x) for x in args.layers.split(",")]
    print(f"patching sweep: n={args.n}, layers={layers}")

    model, tokenizer = get_model_and_tokenizer()
    out = run_activation_patching(model, tokenizer, layers=layers, n_questions=args.n)

    path = RESULTS_DIR / "patching.pkl"
    with open(path, "wb") as f:
        pickle.dump(out, f)
    print(f"\nsaved -> {path}")

    print("\nper-layer patching restoration:")
    print(f"  clean baseline P(correct):     {out['clean_truth_base'].mean():.3f}")
    print(f"  pressured baseline P(correct): {out['pressured_truth_base'].mean():.3f}")
    for l in layers:
        pr = out["per_layer"][l]
        print(
            f"  layer {l:3d}  patched={pr.mean_patched_truth:.3f}  "
            f"delta={pr.delta:+.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
