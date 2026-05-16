"""Phase 6: Feature delta analysis across conditions.

Loads clean + per-condition feature tensors from ``results/sae/`` and produces:

1. Per-condition top-30 CSV tables with clean/pressured means and delta.
2. Cross-condition Jaccard overlap matrix (CSV + heatmap figure).
3. Source-conditional decomposition report (markdown fragment):
     C4a (peer) vs C4d (self)
     C4a vs C4c_matched (attribution vs consensus)
     C4d vs C4e (self vs tool)
4. An aggregate "delta summary" JSON for downstream use by the intervention
   script (which needs the top-N rising features for C4a).

Runs entirely on CPU. No GPU dependency.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from src.config import FIGURES_DIR, RESULTS_DIR  # noqa: E402
from src.sae import compute_feature_deltas, jaccard_overlap, sae_results_dir  # noqa: E402


# Conditions to analyze (must have a corresponding ``{cond}_features.pt`` file).
CONDITIONS = ["c4a", "c4d", "c4c_matched", "c4c", "c4e", "c5a", "c5d", "c5e"]

TOP_K = 30


def load_features(sae_dir: Path, cond: str) -> torch.Tensor:
    return torch.load(sae_dir / f"{cond}_features.pt")


def per_condition_analysis(
    clean: torch.Tensor,
    pressured: dict[str, torch.Tensor],
    top_k: int = TOP_K,
) -> dict[str, dict]:
    out = {}
    for cond, feats in pressured.items():
        out[cond] = compute_feature_deltas(clean, feats, top_k=top_k)
    return out


def write_top_feature_csv(cond: str, analysis: dict, csv_path: Path) -> None:
    top_idx = analysis["top_indices"].tolist()
    top_clean = analysis["top_clean"].tolist()
    top_pressured = analysis["top_pressured"].tolist()
    top_signed = analysis["top_signed"].tolist()
    with open(csv_path, "w") as f:
        f.write("rank,feature_idx,mean_clean,mean_pressured,delta,direction,description_placeholder\n")
        for rank, (idx, cl, pr, d) in enumerate(
            zip(top_idx, top_clean, top_pressured, top_signed), start=1
        ):
            direction = "rising" if d > 0 else "falling"
            f.write(f"{rank},{idx},{cl:.6f},{pr:.6f},{d:+.6f},{direction},\n")


def jaccard_matrix(analyses: dict[str, dict], top_k: int = TOP_K) -> tuple[list[str], np.ndarray]:
    conds = list(analyses.keys())
    n = len(conds)
    m = np.zeros((n, n), dtype=float)
    for i, a in enumerate(conds):
        ai = analyses[a]["top_indices"][:top_k].tolist()
        for j, b in enumerate(conds):
            bi = analyses[b]["top_indices"][:top_k].tolist()
            m[i, j] = jaccard_overlap(ai, bi)
    return conds, m


def plot_heatmap(conds: list[str], m: np.ndarray, out_path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 7), dpi=140)
    im = ax.imshow(m, cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(len(conds)))
    ax.set_yticks(range(len(conds)))
    ax.set_xticklabels(conds, rotation=45, ha="right", fontsize=11)
    ax.set_yticklabels(conds, fontsize=11)
    for i in range(len(conds)):
        for j in range(len(conds)):
            ax.text(
                j,
                i,
                f"{m[i, j]:.2f}",
                ha="center",
                va="center",
                color="white" if m[i, j] < 0.5 else "black",
                fontsize=10,
            )
    ax.set_title(
        f"Top-{TOP_K} SAE-feature Jaccard overlap across conditions\n"
        f"(Goodfire l19 SAE, 65536 features, layer 20)",
        fontsize=13,
    )
    fig.colorbar(im, ax=ax, label="Jaccard (|A∩B| / |A∪B|)")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_top_features_bar(
    cond: str,
    analysis: dict,
    out_path: Path,
    title: str,
    top_k: int = TOP_K,
) -> None:
    import matplotlib.pyplot as plt

    signed = analysis["top_signed"][:top_k].numpy()
    idxs = analysis["top_indices"][:top_k].tolist()

    # Sort by signed delta so rising/falling groups read naturally.
    order = np.argsort(signed)[::-1]
    signed = signed[order]
    idxs = [idxs[i] for i in order]

    colors = ["#1f77b4" if d > 0 else "#d62728" for d in signed]
    fig, ax = plt.subplots(figsize=(9, 6), dpi=140)
    ax.barh(range(top_k), signed, color=colors)
    ax.set_yticks(range(top_k))
    ax.set_yticklabels([f"f{i}" for i in idxs], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Δ mean activation (pressured − clean)", fontsize=11)
    ax.set_title(title, fontsize=13)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    sae_dir = sae_results_dir(RESULTS_DIR)
    FIGURES_DIR.mkdir(exist_ok=True)

    # Only analyze conditions whose feature tensors are on disk.
    available = [c for c in CONDITIONS if (sae_dir / f"{c}_features.pt").exists()]
    missing = [c for c in CONDITIONS if c not in available]
    print(f"Analyzing: {available}")
    if missing:
        print(f"Skipping (not yet encoded): {missing}")

    clean = load_features(sae_dir, "clean")
    print(f"clean features shape: {tuple(clean.shape)}")

    pressured = {c: load_features(sae_dir, c) for c in available}

    analyses = per_condition_analysis(clean, pressured, top_k=TOP_K)

    # 1. Per-condition top-30 CSVs.
    csv_dir = sae_dir / "top_features"
    csv_dir.mkdir(exist_ok=True)
    for cond, a in analyses.items():
        write_top_feature_csv(cond, a, csv_dir / f"{cond}_top{TOP_K}.csv")
    print(f"wrote top-{TOP_K} CSVs → {csv_dir}")

    # 2. Jaccard overlap matrix.
    conds, m = jaccard_matrix(analyses, top_k=TOP_K)
    np.savetxt(
        sae_dir / "jaccard_overlap.csv",
        m,
        delimiter=",",
        header=",".join(conds),
        comments="",
        fmt="%.4f",
    )
    plot_heatmap(conds, m, FIGURES_DIR / "sae_feature_overlap_matrix.png")
    print("wrote jaccard_overlap.csv + sae_feature_overlap_matrix.png")

    # Print overlap matrix to stdout.
    header = "".join(f"{c:>14}" for c in conds)
    print(f"\n{'':>14}{header}")
    for i, row in enumerate(m):
        print(
            f"{conds[i]:>14}"
            + "".join(f"{row[j]:>14.2f}" for j in range(len(conds)))
        )

    # 3. Bar charts for the canonical conditions.
    bar_conds = [c for c in ["c4a", "c4d", "c4c_matched", "c4e"] if c in analyses]
    for cond in bar_conds:
        plot_top_features_bar(
            cond,
            analyses[cond],
            FIGURES_DIR / f"sae_top_features_{cond}.png",
            title=f"{cond}: top-{TOP_K} features by |Δ activation| (pressured − clean)",
        )
    print(f"wrote top-feature bar charts for: {bar_conds}")

    # 4. Source-conditional decomposition markdown fragment.
    pairs = [
        ("c4a", "c4d", "peer user-role vs self-framing (attribution × quality axis)"),
        ("c4a", "c4c_matched", "attribution vs matched-no-attribution (attribution alone)"),
        ("c4d", "c4e", "self-framing vs tool-role (both high-trust)"),
        ("c4a", "c4e", "peer vs tool-role"),
        ("c4a", "c5a", "strong vs weak peer reasoning"),
        ("c4d", "c5d", "strong vs weak self-framing"),
    ]
    decomp_lines = ["# Source-conditional feature decomposition", ""]
    decomp_lines.append(
        f"Jaccard overlaps on the top-{TOP_K} |Δ| features for each pair.\n"
    )
    for a, b, label in pairs:
        if a in analyses and b in analyses:
            ov = jaccard_overlap(
                analyses[a]["top_indices"].tolist(),
                analyses[b]["top_indices"].tolist(),
            )
            shared = sorted(
                set(analyses[a]["top_indices"].tolist())
                & set(analyses[b]["top_indices"].tolist())
            )
            decomp_lines.append(f"## {a} vs {b} — {label}")
            decomp_lines.append(f"- Jaccard: {ov:.3f}")
            decomp_lines.append(
                f"- Shared features ({len(shared)}): "
                + ", ".join(f"f{i}" for i in shared[:20])
                + ("" if len(shared) <= 20 else f" ... (+{len(shared) - 20} more)")
            )
            decomp_lines.append("")

    with open(sae_dir / "source_conditional_decomposition.md", "w") as f:
        f.write("\n".join(decomp_lines))
    print("wrote source_conditional_decomposition.md")

    # 5. Export the top-N rising features for C4a for Phase 8 intervention.
    if "c4a" in analyses:
        c4a = analyses["c4a"]
        # Top rising (pressured > clean) features — use signed delta positive.
        signed = c4a["deltas"]
        rising_order = torch.argsort(signed, descending=True)
        top_rising = rising_order[:200].tolist()  # keep 200 for intervention sweep
        rising_summary = [
            dict(
                feature_idx=int(i),
                mean_clean=float(c4a["mean_clean"][i]),
                mean_pressured=float(c4a["mean_pressured"][i]),
                delta=float(c4a["deltas"][i]),
            )
            for i in top_rising
        ]
        with open(sae_dir / "c4a_top_rising.json", "w") as f:
            json.dump(rising_summary, f, indent=2)
        # Falling features (pressured < clean) for completeness.
        falling_order = torch.argsort(signed, descending=False)
        top_falling = falling_order[:200].tolist()
        falling_summary = [
            dict(
                feature_idx=int(i),
                mean_clean=float(c4a["mean_clean"][i]),
                mean_pressured=float(c4a["mean_pressured"][i]),
                delta=float(c4a["deltas"][i]),
            )
            for i in top_falling
        ]
        with open(sae_dir / "c4a_top_falling.json", "w") as f:
            json.dump(falling_summary, f, indent=2)
        print("wrote c4a_top_rising.json + c4a_top_falling.json")


if __name__ == "__main__":
    main()
