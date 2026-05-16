#!/usr/bin/env python
"""E14 — difference-in-means direction analysis across conditions.

Loads per-condition pickles, computes the empirical sycophantic direction at
layer 25 for each, builds the cosine-similarity matrix, and writes both a
pickle (`results/dim_analysis.pkl`) and a CSV heatmap
(`results/dim_cosine.csv`). Also generates a matplotlib heatmap figure at
`figures/dim_cosine_heatmap.png`.
"""
from __future__ import annotations

import csv
import os
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.config import FIGURES_DIR, LDA_LAYER, RESULTS_DIR  # noqa: E402
from src.data import load_artifacts  # noqa: E402
from src.dim_analysis import (  # noqa: E402
    build_all_directions,
    build_cosine_matrix,
    compute_alignments,
    compute_shared_axis,
)
from src.lda import CleanLDA  # noqa: E402


# Conditions to include, in a deliberate order (peer framings, then self/tool)
CONDITIONS = [
    "c4a",
    "c5a",
    "c4c",
    "c5c",
    "c4c_matched",
    "c5c_matched",
    "c4d",
    "c5d",
    "c4e",
    "c5e",
    "c4d_unmatched",
    "c5d_unmatched",
    "c4e_unmatched",
    "c5e_unmatched",
]


def main() -> int:
    art = load_artifacts()
    correct_labels = np.asarray(art["known_labels"])
    clean_lda = CleanLDA.fit_default()

    # Load per-condition pickles — skip missing ones, warn
    pkls: dict[str, dict] = {}
    for name in CONDITIONS:
        p = RESULTS_DIR / f"{name}.pkl"
        if not p.exists():
            print(f"skip {name}: {p} missing")
            continue
        with open(p, "rb") as f:
            pkls[name] = pickle.load(f)

    if not pkls:
        raise SystemExit("no condition pickles found; run run_all_conditions.py first")

    print(f"computing directions at layer {LDA_LAYER} for {len(pkls)} conditions...")
    directions = build_all_directions(
        pkls, clean_lda=clean_lda, correct_labels=correct_labels, layer=LDA_LAYER
    )

    for name, cd in directions.items():
        print(
            f"  {name:14s}  |direction|={cd.norm:8.3f}  "
            f"n_yielded={cd.n_yielded:4d}  n_unyielded={cd.n_unyielded:4d}"
        )

    # Degenerate conditions: c5e has n_unyielded=1 in principle; some tool-role
    # conditions will have very few unyielded questions. The direction is still
    # defined but its norm is noisy. We keep them in the matrix for completeness
    # but flag them in the writeup.

    dir_dict = {k: v.direction for k, v in directions.items()}
    names, cos_mat = build_cosine_matrix(dir_dict)
    shared = compute_shared_axis(dir_dict)
    alignments = compute_alignments(dir_dict, shared=shared)

    # Save pkl
    out_pkl = RESULTS_DIR / "dim_analysis.pkl"
    with open(out_pkl, "wb") as f:
        pickle.dump(
            {
                "layer": LDA_LAYER,
                "directions": dir_dict,
                "direction_norms": {k: v.norm for k, v in directions.items()},
                "n_yielded": {k: v.n_yielded for k, v in directions.items()},
                "n_unyielded": {k: v.n_unyielded for k, v in directions.items()},
                "cosine_matrix": cos_mat,
                "cosine_names": names,
                "shared_axis": shared,
                "alignments_with_shared": alignments,
            },
            f,
        )
    print(f"saved -> {out_pkl}")

    # CSV
    csv_path = RESULTS_DIR / "dim_cosine.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([""] + names)
        for i, name in enumerate(names):
            w.writerow([name] + [f"{cos_mat[i, j]:.4f}" for j in range(len(names))])
    print(f"saved -> {csv_path}")

    # Heatmap figure
    fig, ax = plt.subplots(figsize=(10, 9))
    im = ax.imshow(cos_mat, vmin=-1.0, vmax=1.0, cmap="RdBu_r", aspect="auto")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    for i in range(len(names)):
        for j in range(len(names)):
            val = cos_mat[i, j]
            color = "white" if abs(val) > 0.55 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    color=color, fontsize=8)
    ax.set_title(
        f"Sycophantic direction cosine similarity across conditions (layer {LDA_LAYER})"
    )
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("cosine similarity")
    fig.tight_layout()
    fig_path = FIGURES_DIR / "dim_cosine_heatmap.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved -> {fig_path}")

    # Alignment with shared axis
    print("\nalignment with shared sycophancy axis (cosine):")
    for k, v in sorted(alignments.items(), key=lambda kv: -kv[1]):
        print(f"  {k:14s}  {v:+.4f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
