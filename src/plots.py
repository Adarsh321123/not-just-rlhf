"""Paper-ready figure helpers.

Shared styling: ~14pt axis labels, ~16pt titles, colorblind-safe ``tab10`` palette,
white background, tight layout.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from .config import CHOICES, LDA_LAYER, NUM_LAYERS
from .lda import CleanLDA

PAPER_RC = {
    "axes.labelsize": 14,
    "axes.titlesize": 16,
    "figure.titlesize": 18,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 11,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "axes.grid": True,
    "grid.alpha": 0.3,
}


def _apply_rc():
    plt.rcParams.update(PAPER_RC)


def plot_logit_lens_grid(
    conditions: list[tuple[str, dict]],
    suptitle: str,
    savepath: Path | None = None,
    cols: int | None = None,
):
    _apply_rc()
    n = len(conditions)
    cols = min(n, 4) if cols is None else cols
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows))
    axes = np.atleast_1d(axes).flatten()
    layers = np.arange(NUM_LAYERS)

    for idx, (label, res) in enumerate(conditions):
        ax = axes[idx]
        ax.plot(layers, res["avg_truth"], label="P(correct)", color="#2ca02c", lw=2.5)
        ax.plot(layers, res["avg_syco"], label="P(wrong target)", color="#d62728", lw=2.5)
        ax.plot(layers, res["probe_accs"], label="Probe acc", color="#1f77b4", ls="--", lw=2)
        onset = res.get("onset")
        if onset is not None:
            ax.axvline(onset, color="#ff7f0e", ls=":", lw=1.5, label=f"Onset L{onset}")
        ax.set_title(label, fontweight="bold")
        ax.set_xlabel("Layer")
        ax.set_ylabel("Prob / accuracy")
        ax.set_ylim(-0.05, 1.05)
        if idx == 0:
            ax.legend(loc="upper left")

    for i in range(len(conditions), len(axes)):
        axes[i].axis("off")

    fig.suptitle(suptitle, fontweight="bold")
    plt.tight_layout(rect=[0, 0.02, 1, 0.95])
    if savepath is not None:
        fig.savefig(savepath, dpi=200, bbox_inches="tight")
    return fig


def plot_lda_grid(
    conditions: list[tuple[str, dict]],
    suptitle: str,
    layer: int = LDA_LAYER,
    clean_lda: CleanLDA | None = None,
    savepath: Path | None = None,
):
    _apply_rc()
    if clean_lda is None:
        clean_lda = CleanLDA.fit(layer)

    from .data import load_artifacts

    art = load_artifacts()
    known_labels = art["known_labels"]

    clean_3d = clean_lda.project(art["known_acts"][:, layer, :])

    n_panels = 1 + len(conditions)
    fig = plt.figure(figsize=(6 * n_panels, 7))

    ax = fig.add_subplot(1, n_panels, 1, projection="3d")
    for c in range(4):
        mask = known_labels == c
        ax.scatter(
            clean_3d[mask, 0], clean_3d[mask, 1], clean_3d[mask, 2],
            alpha=0.35, s=15, label=f"Choice {CHOICES[c]}",
        )
    ax.set_title("Clean\n(no pressure)")
    ax.legend(fontsize=8)

    for panel_idx, (label, res) in enumerate(conditions, start=2):
        acts_l = res["activations"][:, layer, :].astype(np.float32)
        proj = clean_lda.project(acts_l)

        mask_yield = clean_lda.yield_mask(
            acts_l, known_labels, res["wrong_indices"]
        )
        held = proj[~mask_yield]
        yielded = proj[mask_yield]
        yr = 100 * mask_yield.mean()

        ax = fig.add_subplot(1, n_panels, panel_idx, projection="3d")
        if held.size:
            ax.scatter(
                held[:, 0], held[:, 1], held[:, 2],
                color="#2ca02c", alpha=0.6, s=15, label="Held",
            )
        if yielded.size:
            ax.scatter(
                yielded[:, 0], yielded[:, 1], yielded[:, 2],
                color="#d62728", alpha=0.6, s=15, label="Yielded",
            )
        ax.set_title(f"{label}\nyield={yr:.1f}%")
        ax.legend(fontsize=8)

    fig.suptitle(f"{suptitle} — LDA at layer {layer}", fontweight="bold")
    plt.tight_layout()
    if savepath is not None:
        fig.savefig(savepath, dpi=200, bbox_inches="tight")
    return fig


def plot_yield_bars(
    summary: list[dict[str, Any]],
    savepath: Path | None = None,
    title: str = "Yield rate by condition (L25)",
):
    """Grouped bar chart with CI error bars.

    ``summary`` items must have: ``label``, ``yield_rate``, ``ci_lo``, ``ci_hi``,
    ``group`` (colour group label).
    """
    _apply_rc()
    labels = [s["label"] for s in summary]
    yields = np.array([s["yield_rate"] for s in summary]) * 100
    lo = np.array([s["ci_lo"] for s in summary]) * 100
    hi = np.array([s["ci_hi"] for s in summary]) * 100
    errs = np.vstack([yields - lo, hi - yields])

    groups = [s.get("group", "") for s in summary]
    unique = list(dict.fromkeys(groups))
    cmap = plt.get_cmap("tab10")
    colors = [cmap(unique.index(g) % 10) for g in groups]

    fig, ax = plt.subplots(figsize=(max(8, 0.6 * len(labels) + 4), 6))
    x = np.arange(len(labels))
    ax.bar(x, yields, yerr=errs, color=colors, alpha=0.85, capsize=4, edgecolor="black")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylabel("Yield rate (%)")
    ax.set_ylim(0, 105)
    ax.set_title(title, fontweight="bold")

    # legend showing groups
    from matplotlib.patches import Patch

    handles = [Patch(facecolor=cmap(i % 10), label=g) for i, g in enumerate(unique)]
    ax.legend(handles=handles, loc="upper left")

    plt.tight_layout()
    if savepath is not None:
        fig.savefig(savepath, dpi=200, bbox_inches="tight")
    return fig


def plot_contamination_delta(
    clean_probe_accs: np.ndarray,
    per_condition_probe_accs: dict[str, np.ndarray],
    savepath: Path | None = None,
    title: str = "Per-layer probe accuracy delta (clean − pressured)",
):
    _apply_rc()
    layers = np.arange(NUM_LAYERS)
    fig, ax = plt.subplots(figsize=(10, 6))
    cmap = plt.get_cmap("tab10")
    for i, (label, accs) in enumerate(per_condition_probe_accs.items()):
        delta = np.asarray(clean_probe_accs) - np.asarray(accs)
        ax.plot(layers, delta, label=label, lw=2.2, color=cmap(i % 10))
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Probe accuracy delta (clean − pressured)")
    ax.set_title(title, fontweight="bold")
    ax.legend(loc="upper left", ncol=2)
    plt.tight_layout()
    if savepath is not None:
        fig.savefig(savepath, dpi=200, bbox_inches="tight")
    return fig


# axis labels 12 pt, ticks 11 pt, titles 14 pt bold,
# markers 10 pt.  Applied only inside plot_patching_restoration via rc_context
# so other figures are unaffected.
_PATCHING_RC = {
    **PAPER_RC,
    "axes.labelsize": 12,
    "axes.titlesize": 14,
    "figure.titlesize": 14,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 10,
    "lines.markersize": 10,
}


def plot_patching_restoration(
    per_layer_result,
    ci_bands=None,
    savepath: Path | None = None,
    title: str = "Activation patching: P(correct) restoration",
):
    """Patching restoration curve with optional shaded 95% CI bands.

    Parameters
    ----------
    per_layer_result : dict[int, PatchResult]
        The ``per_layer`` dict from ``run_activation_patching``.
    ci_bands : dict[int, BootstrapCI] | None
        Per-layer bootstrap CIs from ``scripts/bootstrap_patching.py``.
        Each CI is on the *delta* (patched − pressured); converted to absolute
        P(correct) via pressured_baseline + ci.{lo,hi} for the fill_between band.
    """
    layers = sorted(per_layer_result.keys())
    clean = per_layer_result[layers[0]].mean_clean_truth
    pressured = per_layer_result[layers[0]].mean_pressured_truth
    patched = [per_layer_result[l].mean_patched_truth for l in layers]

    with plt.rc_context(_PATCHING_RC):
        fig, ax = plt.subplots(figsize=(10, 6))

        if ci_bands is not None:
            ci_lo = [pressured + ci_bands[l].lo for l in layers]
            ci_hi = [pressured + ci_bands[l].hi for l in layers]
            ax.fill_between(
                layers, ci_lo, ci_hi,
                alpha=0.20, color="#1f77b4", label="95% CI (bootstrap, n=400)",
            )

        ax.plot(layers, patched, "o-", lw=2.5, color="#1f77b4", label="Patched P(correct)")
        ax.axhline(
            clean, color="#2ca02c", ls="--", lw=2,
            label=f"Clean baseline ({clean:.3f})",
        )
        ax.axhline(
            pressured, color="#d62728", ls="--", lw=2,
            label=f"Pressured baseline ({pressured:.3f})",
        )
        ax.set_xlabel("Patch layer")
        ax.set_ylabel("Final-layer P(correct)")
        ax.set_title(title, fontweight="bold")
        ax.legend(loc="best")
        ax.set_ylim(-0.02, 1.02)
        plt.tight_layout()
        if savepath is not None:
            fig.savefig(savepath, dpi=200, bbox_inches="tight")
        return fig
