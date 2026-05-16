#!/usr/bin/env python
"""Unified figure generator.

Produces Figures 2-12 (Figure 1 is a manually-composed schematic).

Figure  2: Wrong-agent count sweep at N=4
Figure  3: Activation patching restoration curve (Llama-3.1-8B-Instruct)
Figure  4: Base vs Instruct across four model families
Figure  5: Dissenter rescue across three framings
Figure  6: Yield vs fraction-wrong at N=4,5,6 (two panels)
Figure  7: Component decomposition at L14-L18
Figure  8: SAE feature clamping sweep
Figure  9: Mistral-7B replication (patching + component decomposition)
Figure 10: Cross-domain direct user assertion yield (humanities vs STEM)
Figure 11: Conditional activation patching grid (2x5x10)
Figure 12: User-role 4v0 yield vs calibration set size

Usage:
    python scripts/generate_figures.py            # generate all figures
    python scripts/generate_figures.py --fig 3    # generate only Figure 3
    python scripts/generate_figures.py --fig 2 5  # generate Figures 2 and 5

Each figure is saved as both .png (200 dpi) and .pdf in ``figures/``.
Figures whose data files are missing are skipped with a warning.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
import sys
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.config import FIGURES_DIR, RESULTS_DIR  # noqa: E402

# Ensure PatchResult is importable so pickle.load can resolve it.
from src.patching import PatchResult  # noqa: E402,F401

FIGURES_DIR.mkdir(exist_ok=True)

# ── Consistent paper style ──────────────────────────────────────────────────
PAPER_RC = {
    "axes.labelsize": 14,
    "axes.titlesize": 15,
    "figure.titlesize": 16,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "axes.grid": False,
    "font.size": 12,
}
plt.rcParams.update(PAPER_RC)

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


# ── Helpers ─────────────────────────────────────────────────────────────────

def save_both(fig: plt.Figure, basename: str) -> tuple[Path, Path]:
    """Save ``fig`` as ``.png`` and ``.pdf`` with identical content."""
    png_path = FIGURES_DIR / f"{basename}.png"
    pdf_path = FIGURES_DIR / f"{basename}.pdf"
    fig.savefig(png_path, dpi=200, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {png_path.name} + {pdf_path.name}")
    return png_path, pdf_path


def _load_pkl(path: Path) -> Any | None:
    if not path.exists():
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def _read_summary_csv(csv_path: Path) -> dict[str, dict[str, float]]:
    """Light-weight CSV reader that tolerates the two schemas used in the repo."""
    rows: dict[str, dict[str, float]] = {}
    with open(csv_path, "r") as f:
        lines = f.read().splitlines()
    header = [c.strip() for c in lines[0].split(",")]
    for line in lines[1:]:
        if not line.strip():
            continue
        values = [c.strip() for c in line.split(",")]
        record = dict(zip(header, values))
        cond = record["condition"]
        rows[cond] = {
            "yield": float(record["yield_pct"]),
            "ci_lo": float(record.get("ci_lo_pct", record.get("ci_lo", 0))),
            "ci_hi": float(record.get("ci_hi_pct", record.get("ci_hi", 0))),
        }
    return rows


# ════════════════════════════════════════════════════════════════════════════
# Figure 2: Wrong-agent count sweep at N=4
# ════════════════════════════════════════════════════════════════════════════

# Hardcoded from PAPER_NOTES canonical C6 measurement.
_USER_SUFFIXED = {
    "pts": [0, 1, 2, 3, 4],
    "yield": [0.00, 2.50, 3.75, 12.75, 80.25],
    "lo": [0.00, 1.24, 2.00, 9.50, 76.49],
    "hi": [0.00, 4.00, 5.75, 16.25, 84.25],
}
_SELF_SUFFIXED = {
    "pts": [0, 1, 2, 3, 4],
    "yield": [0.00, 0.25, 6.50, 60.25, 97.50],
    "lo": [0.00, 0.00, 4.25, 55.75, 95.75],
    "hi": [0.00, 0.75, 9.01, 65.00, 99.00],
}
_C4A_FULL_CONSENSUS_REF = 75.75


def fig2() -> None:
    """Fig 2: Wrong-agent count sweep at N=4 (yield vs k_wrong)."""
    fig, ax = plt.subplots(figsize=(6.4, 4.3))

    ax.axhline(
        _C4A_FULL_CONSENSUS_REF, color="#BBBBBB", linestyle=":", linewidth=0.8,
        alpha=0.55, zorder=1,
    )
    ax.axhline(0, color="#BBBBBB", linestyle=":", linewidth=0.8, alpha=0.55, zorder=1)

    x = np.array(_USER_SUFFIXED["pts"], dtype=float)
    X_LABELS = ["0/4", "1/3", "2/2", "3/1", "4/0"]
    DARK_BLUE = "#1565C0"
    LIGHT_BLUE = "#64B5F6"

    # User-role suffixed.
    y_u = np.array(_USER_SUFFIXED["yield"])
    lo_u = np.array(_USER_SUFFIXED["lo"])
    hi_u = np.array(_USER_SUFFIXED["hi"])
    ax.errorbar(
        x, y_u, yerr=[y_u - lo_u, hi_u - y_u],
        color=DARK_BLUE, marker="o", markersize=7, linewidth=2.0,
        capsize=3.5, capthick=1.0, elinewidth=0.9,
        label="User-role (suffixed)", zorder=5,
    )

    # Assistant-role suffixed.
    y_s = np.array(_SELF_SUFFIXED["yield"])
    lo_s = np.array(_SELF_SUFFIXED["lo"])
    hi_s = np.array(_SELF_SUFFIXED["hi"])
    ax.errorbar(
        x, y_s, yerr=[y_s - lo_s, hi_s - y_s],
        color=LIGHT_BLUE, marker="s", markersize=7, linewidth=2.0,
        capsize=3.5, capthick=1.0, elinewidth=0.9,
        label="Assistant-role (suffixed)", zorder=5,
    )

    # 47.5 pp gap annotation at 3v1.
    x_arrow = 3.0
    y_low = _USER_SUFFIXED["yield"][3]
    y_high = _SELF_SUFFIXED["yield"][3]
    ax.annotate(
        "", xy=(x_arrow, y_high), xytext=(x_arrow, y_low),
        arrowprops=dict(
            arrowstyle="<->", color="#333333", linewidth=2.2, shrinkA=0, shrinkB=0,
        ),
        zorder=6,
    )
    ax.text(
        x_arrow + 0.15, (y_low + y_high) / 2,
        r"$\Delta = 47.5$ pp",
        fontsize=16, fontweight="bold", color="#1a1a1a",
        ha="left", va="center", zorder=7,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                  edgecolor="#888888", alpha=0.9, linewidth=0.8),
    )

    ax.set_xlabel("Wrong-arguing agents (of 4)", labelpad=6)
    ax.set_ylabel("Yield (%)", labelpad=6)
    ax.set_xticks(x)
    ax.set_xticklabels(X_LABELS)
    ax.set_xlim(-0.3, 4.3)
    ax.set_ylim(-4, 104)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(20))
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%g"))
    ax.tick_params(axis="both", labelsize=11)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color("#666666")
    ax.spines["bottom"].set_color("#666666")

    ax.legend(loc="upper left", framealpha=0.95, edgecolor="#CCCCCC")

    fig.tight_layout()
    save_both(fig, "fig2_wrong_agent_sweep")


# ════════════════════════════════════════════════════════════════════════════
# Figure 3: Activation patching restoration curve (Llama-3.1-8B-Instruct)
# ════════════════════════════════════════════════════════════════════════════

def fig3() -> None:
    """Fig 3: Activation patching restoration curve with 95% bootstrap CI bands."""
    from src.plots import plot_patching_restoration

    p = RESULTS_DIR / "patching.pkl"
    bs_p = RESULTS_DIR / "patching_bootstrap.pkl"
    if not p.exists():
        print("  SKIP fig3: patching.pkl missing")
        return
    out = _load_pkl(p)
    ci_bands = _load_pkl(bs_p)
    if ci_bands is None:
        print("  WARNING fig3: patching_bootstrap.pkl missing, plotting without CI bands")

    for ext in ("png", "pdf"):
        savepath = FIGURES_DIR / f"fig3_patching_restoration.{ext}"
        plot_patching_restoration(
            out["per_layer"],
            ci_bands=ci_bands,
            savepath=savepath,
            title="",
        )
    print(f"  saved fig3_patching_restoration.png + .pdf")


# ════════════════════════════════════════════════════════════════════════════
# Figure 4: Base vs Instruct across four model families
# ════════════════════════════════════════════════════════════════════════════

def fig4() -> None:
    """Fig 4: Base vs Instruct grouped bar chart across four model families."""
    csv_path = RESULTS_DIR / "base_model" / "intersection_pool_analysis.csv"
    if not csv_path.exists():
        print(f"  SKIP fig4: {csv_path.name} not found")
        return

    rows = []
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            rows.append(r)

    FAMILIES = ["llama", "mistral", "gemma", "qwen"]
    CONDITIONS = ["c4a", "c4c_matched", "c4d"]

    instruct_data = {fam: {} for fam in FAMILIES}
    base_data = {fam: {} for fam in FAMILIES}
    for r in rows:
        fam = r["family"]
        cond = r["condition"]
        base_data[fam][cond] = float(r["base_yield"])
        instruct_data[fam][cond] = float(r["instruct_yield"])

    n_families = len(FAMILIES)
    n_conds = len(CONDITIONS)
    group_width = 0.7
    bar_width = group_width / (n_conds * 2)

    colors_inst = {"c4a": "#4C72B0", "c4c_matched": "#55A868", "c4d": "#8172B3"}
    colors_base = {"c4a": "#7AAAD4", "c4c_matched": "#88CC99", "c4d": "#B3A6D4"}

    x_family = np.arange(n_families)

    fig, ax = plt.subplots(figsize=(14, 6))

    legend_handles = []
    for ci, cond in enumerate(CONDITIONS):
        for ti, (mtype, cdict) in enumerate([("instruct", colors_inst), ("base", colors_base)]):
            offset = (ci * 2 + ti) * bar_width - group_width / 2 + bar_width / 2
            vals = []
            for family in FAMILIES:
                src = instruct_data[family] if mtype == "instruct" else base_data[family]
                vals.append(src.get(cond, 0))

            bars = ax.bar(
                x_family + offset, vals, bar_width,
                color=cdict[cond], edgecolor="white", linewidth=0.5,
            )
            legend_handles.append(bars)

            for bar in bars:
                h = bar.get_height()
                if h > 0:
                    ax.text(bar.get_x() + bar.get_width() / 2, h + 1.2,
                            f"{h:.0f}", ha="center", va="bottom", fontsize=10,
                            fontweight="bold")

    ax.set_xlabel("Model Family", fontsize=14)
    ax.set_ylabel("Yield (%)", fontsize=14)
    ax.set_xticks(x_family)
    ax.set_xticklabels([f.capitalize() for f in FAMILIES], fontsize=13)
    ax.tick_params(axis="y", labelsize=12)
    ax.set_ylim(0, 115)
    ax.axhline(y=25, color="gray", linestyle="--", alpha=0.7, linewidth=3.0)

    COND_DISPLAY = {"c4a": "Named peer jury", "c4c_matched": "Anon. jury", "c4d": "Assist.-role jury"}
    labels = []
    for cond in CONDITIONS:
        cond_label = COND_DISPLAY.get(cond, cond.upper().replace("_MATCHED", "-m"))
        labels.append(f"{cond_label} (Instruct)")
        labels.append(f"{cond_label} (Base)")
    ax.legend(legend_handles, labels, loc="upper center", fontsize=10, ncol=3,
              framealpha=0.95, edgecolor="#cccccc", bbox_to_anchor=(0.5, 1.02))

    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    save_both(fig, "fig4_base_vs_instruct")


# ════════════════════════════════════════════════════════════════════════════
# Figure 5: Dissenter rescue across three framings
# ════════════════════════════════════════════════════════════════════════════

def fig5() -> None:
    """Fig 5: Dissenter rescue — yield vs number of wrong-arguing agents, 3 framings."""
    csv_path = RESULTS_DIR / "disagreement_framings" / "summary.csv"
    if not csv_path.exists():
        print(f"  SKIP fig5: {csv_path.name} not found")
        return

    data = {}
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            if r["protocol"] != "suffixed":
                continue
            framing = r["framing"]
            if framing.startswith("self_def"):
                continue
            n_w = int(r["n_wrong"])
            y_pct = float(r["yield"])
            data.setdefault(framing, {})[n_w] = y_pct

    if not data:
        print("  SKIP fig5: no suffixed data in summary.csv")
        return

    colors = {"user": "#2196F3", "self": "#FF9800", "tool": "#4CAF50"}
    display_names = {"user": "user-role", "self": "assistant-role", "tool": "tool-role"}
    n_wrong = [0, 1, 2, 3]

    fig, ax = plt.subplots(figsize=(8, 5.5))

    for framing in ["user", "self", "tool"]:
        ys = [data.get(framing, {}).get(nw, float("nan")) for nw in n_wrong]
        ax.plot(n_wrong, ys, marker="o", linewidth=2.5, markersize=10,
                label=display_names.get(framing, framing), color=colors.get(framing, "gray"))

        for xv, y in zip(n_wrong, ys):
            if not np.isnan(y):
                if xv == 0:
                    offset_x, offset_y = 20, 5
                elif xv == 1:
                    offset_x = {"user": -30, "self": 0, "tool": 30}.get(framing, 0)
                    offset_y = {"user": 10, "self": 16, "tool": -8}.get(framing, 12)
                elif xv == 2:
                    offset_x = {"user": 25, "self": -10, "tool": -25}.get(framing, 0)
                    offset_y = {"user": -14, "self": 10, "tool": 10}.get(framing, 12)
                elif xv == 3:
                    offset_x = {"user": 20, "self": 0, "tool": 25}.get(framing, 0)
                    offset_y = {"user": -16, "self": -35, "tool": 10}.get(framing, 12)
                else:
                    offset_y = 12
                    offset_x = 0
                ax.annotate(
                    f"{y:.1f}%", (xv, y),
                    textcoords="offset points", xytext=(offset_x, offset_y),
                    ha="center", fontsize=10, color=colors.get(framing, "gray"),
                    fontweight="bold",
                )

    ax.set_xlabel("Number of wrong-arguing agents", fontsize=14)
    ax.set_ylabel("Yield (%)", fontsize=14)
    ax.set_xticks(n_wrong)
    ax.set_xticklabels(["0\n(all correct)", "1", "2", "3\n(all wrong)"], fontsize=12)
    ax.tick_params(axis="y", labelsize=12)
    ax.set_ylim(-5, 110)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=13, loc="center left")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    fig.tight_layout()
    save_both(fig, "fig5_dissenter_rescue")


# ════════════════════════════════════════════════════════════════════════════
# Figure 6: Yield vs fraction-wrong at N=4,5,6 (two panels)
# ════════════════════════════════════════════════════════════════════════════

def fig6() -> None:
    """Fig 6: Yield vs fraction-wrong at N=4,5,6 (user-role + assistant-role)."""
    points_path = RESULTS_DIR / "c6_scaling" / "all_points.json"
    if not points_path.exists():
        print(f"  SKIP fig6: {points_path.name} not found")
        return

    with open(points_path) as f:
        all_pts_raw = json.load(f)

    colors = {4: "#1f77b4", 5: "#ff7f0e", 6: "#2ca02c"}
    markers = {4: "o", 5: "s", 6: "^"}

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    for ax, framing in zip(axes, ["user", "self"]):
        for N in (4, 5, 6):
            key = f"N{N}_{framing}"
            pts = all_pts_raw.get(key, [])
            if not pts:
                continue
            x_frac = [p["k_wrong"] / N for p in pts]
            y = [p["yield"] * 100 for p in pts]
            lo = [(p["yield"] - p["ci_lo"]) * 100 for p in pts]
            hi = [(p["ci_hi"] - p["yield"]) * 100 for p in pts]
            ax.errorbar(
                x_frac, y, yerr=[lo, hi],
                marker=markers[N], color=colors[N], linewidth=2.5, markersize=9,
                capsize=5, label=f"N={N}",
            )

        ax.axhline(50, color="gray", linestyle="--", alpha=0.5, linewidth=1.2)
        ax.set_xlabel("Fraction wrong-arguing (k/N)", fontsize=14)
        if framing == "user":
            ax.set_ylabel("Yield (%)", fontsize=14)
        framing_display = {"user": "User-role framing", "self": "Assistant-role framing"}
        ax.set_title(framing_display[framing], fontsize=15)
        ax.set_ylim(-5, 105)
        ax.set_xlim(-0.05, 1.05)
        ax.tick_params(axis="both", labelsize=12)
        ax.grid(alpha=0.3)
        ax.legend(loc="upper left", fontsize=13, framealpha=0.95)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

    fig.tight_layout()
    save_both(fig, "fig6_scaling_cliff")


# ════════════════════════════════════════════════════════════════════════════
# Figure 7: Component decomposition at L14-L18
# ════════════════════════════════════════════════════════════════════════════

def fig7() -> None:
    """Fig 7: Component decomposition — MLP-only, attention-only, both, residual."""
    from src.bootstrap import BootstrapCI

    pkl_path = RESULTS_DIR / "component_patching.pkl"
    if not pkl_path.exists():
        print(f"  SKIP fig7: component_patching.pkl not found")
        return

    out = _load_pkl(pkl_path)
    layers = out["layers"]
    pressured = out["pressured_truth_base"]
    pressured_mean = float(pressured.mean())

    COMP_ORDER = ["mlp", "attn", "both", "residual"]
    have_both = (layers[0], "both") in out["patched_truth"]
    components = COMP_ORDER if have_both else ["mlp", "attn", "residual"]
    components = [c for c in components if (layers[0], c) in out["patched_truth"]]

    # Bootstrap CIs
    def _bootstrap_delta(patched, press, n_iter=1000, ci_level=0.95, seed=42):
        rng = np.random.default_rng(seed)
        delta = patched - press
        n = len(delta)
        samples = np.empty(n_iter)
        for j in range(n_iter):
            idx = rng.integers(0, n, size=n)
            samples[j] = delta[idx].mean()
        alpha = (1.0 - ci_level) / 2.0
        return BootstrapCI(
            mean=float(delta.mean()),
            lo=float(np.quantile(samples, alpha)),
            hi=float(np.quantile(samples, 1.0 - alpha)),
            se=float(samples.std(ddof=1)),
            n_iter=n_iter,
        )

    ci_dict = {}
    lookup = {}
    for l in layers:
        for comp in components:
            pt = out["patched_truth"][(l, comp)]
            ci = _bootstrap_delta(pt, pressured)
            ci_dict[(l, comp)] = ci
            lookup[(l, comp)] = {"mean_delta": ci.mean}

    bar_comps = ["mlp", "attn", "both"] if have_both else ["mlp", "attn"]
    bar_comps = [c for c in bar_comps if c in components]
    comp_labels = {
        "mlp": "MLP-only",
        "attn": "Attention-only",
        "both": "Both (layer-local baseline)",
        "residual": "Residual (full upstream)",
    }
    colors = {"mlp": "#1f77b4", "attn": "#ff7f0e", "both": "#9467bd", "residual": "#2ca02c"}

    with plt.rc_context(_PATCHING_RC):
        fig, ax = plt.subplots(figsize=(9.5, 5.5))

        n_bars = len(bar_comps)
        bar_width = 0.22
        x = np.arange(len(layers))

        for bi, comp in enumerate(bar_comps):
            offset = (bi - (n_bars - 1) / 2.0) * bar_width
            means = np.array([lookup[(l, comp)]["mean_delta"] for l in layers])
            err_lo = means - np.array([ci_dict[(l, comp)].lo for l in layers])
            err_hi = np.array([ci_dict[(l, comp)].hi for l in layers]) - means
            ax.bar(
                x + offset, means, width=bar_width,
                label=comp_labels[comp], color=colors[comp], alpha=0.85,
                yerr=[err_lo, err_hi],
                error_kw={"elinewidth": 1.4, "capsize": 3, "ecolor": "black"},
            )

        # Per-layer dashed residual segment.
        if "residual" in components:
            res_means = np.array([lookup[(l, "residual")]["mean_delta"] for l in layers])
            half_span = n_bars * bar_width / 2.0 + 0.06
            ax.hlines(
                res_means, x - half_span, x + half_span,
                colors=colors["residual"], linestyles="dashed",
                linewidth=2.0, label=comp_labels["residual"], alpha=0.9,
            )

        ax.set_xticks(x)
        ax.set_xticklabels([f"L{l}" for l in layers])
        ax.set_xlabel("Patched layer")
        ax.set_ylabel("P(correct) restoration delta")
        ax.legend(loc="upper left", framealpha=0.95, edgecolor="#CCCCCC")
        ax.axhline(0, color="black", linewidth=0.8, alpha=0.4)
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)

        fig.tight_layout()

    save_both(fig, "fig7_component_decomp")


# ════════════════════════════════════════════════════════════════════════════
# Figure 8: SAE feature clamping sweep
# ════════════════════════════════════════════════════════════════════════════

def fig8() -> None:
    """Fig 8: SAE feature clamping sweep (delta-P(wrong) and delta-P(correct))."""
    sae_dir = RESULTS_DIR / "sae"
    sweep_path = sae_dir / "intervention_sweep.json"
    recon_path = sae_dir / "reconstruction_baseline.json"

    if not sweep_path.exists() or not recon_path.exists():
        print(f"  SKIP fig8: SAE intervention files not found")
        return

    with open(sweep_path) as f:
        sweep = json.load(f)
    with open(recon_path) as f:
        recon = json.load(f)

    ks = sweep["k_values"]
    summary = sweep["summary"]
    recon_summary = recon["summary"]

    unhooked_pc = recon_summary["mean_unhooked_pcorrect"]
    unhooked_pw = recon_summary["mean_unhooked_pwrong"]
    recon_pc = recon_summary["mean_recon_pcorrect"]
    recon_pw = recon_summary["mean_recon_pwrong"]

    rising_pc = [summary[f"rising_clean_k{k}"]["mean_pcorrect"] for k in ks]
    falling_pc = [summary[f"falling_clean_k{k}"]["mean_pcorrect"] for k in ks]
    both_pc = [summary[f"both_clean_k{k}"]["mean_pcorrect"] for k in ks]
    rising_pw = [summary[f"rising_clean_k{k}"]["mean_pwrong"] for k in ks]
    falling_pw = [summary[f"falling_clean_k{k}"]["mean_pwrong"] for k in ks]
    both_pw = [summary[f"both_clean_k{k}"]["mean_pwrong"] for k in ks]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    panels = [
        (ax1, "P(correct)",
         [("Rising", rising_pc, "#1f77b4", "o"),
          ("Falling", falling_pc, "#2ca02c", "s"),
          ("Both", both_pc, "#9467bd", "^")]),
        (ax2, "P(wrong target)",
         [("Rising", rising_pw, "#1f77b4", "o"),
          ("Falling", falling_pw, "#2ca02c", "s"),
          ("Both", both_pw, "#9467bd", "^")]),
    ]

    for ax, title, traces in panels:
        ref_val = unhooked_pc if "correct" in title else unhooked_pw
        recon_val = recon_pc if "correct" in title else recon_pw

        ax.axhline(ref_val, color="#222222", linestyle="-", linewidth=1.5,
                   label=f"No hook = {ref_val:.3f}")
        ax.axhline(recon_val, color="#888888", linestyle="--", linewidth=1.5,
                   label=f"Recon-only = {recon_val:.3f}")

        for lbl, vals, color, marker in traces:
            ax.plot(ks, vals, f"{marker}-", color=color, linewidth=2.2,
                    markersize=8, label=f"{lbl} -> clean")

        ax.set_xlabel("Features clamped (k)", fontsize=14)
        ax.set_ylabel(title, fontsize=14)
        ax.set_xscale("log")
        ax.set_xticks(ks)
        ax.set_xticklabels([str(k) for k in ks], fontsize=11)
        ax.tick_params(axis="y", labelsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=11, framealpha=0.95)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

    fig.tight_layout()
    save_both(fig, "fig8_sae_clamping")


# ════════════════════════════════════════════════════════════════════════════
# Figure 9: Mistral-7B replication (patching + component decomposition)
# ════════════════════════════════════════════════════════════════════════════

_LAYER_BAND_LO = 14
_LAYER_BAND_HI = 18


def _load_patching_data(pkl_path: Path) -> dict[str, Any] | None:
    """Load a patching.pkl and extract layers / patched / clean / pressured."""
    data = _load_pkl(pkl_path)
    if data is None:
        return None
    layers = sorted(data["per_layer"].keys())
    patched = np.array([data["per_layer"][l].mean_patched_truth for l in layers])
    clean = float(np.mean(data["clean_truth_base"]))
    pressured = float(np.mean(data["pressured_truth_base"]))
    return {
        "layers": np.array(layers),
        "patched": patched,
        "clean": clean,
        "pressured": pressured,
    }


def _patching_panel(ax: plt.Axes, data: dict[str, Any], title: str, show_ylabel: bool) -> None:
    """Draw a single patching panel (shared between Llama and Mistral)."""
    layers = data["layers"]
    patched = data["patched"]
    clean = data["clean"]
    pressured = data["pressured"]

    ax.axvspan(_LAYER_BAND_LO, _LAYER_BAND_HI, color="#D0D0D0", alpha=0.35, zorder=0)

    ax.axhline(
        clean, color="#2ca02c", linestyle="--", linewidth=2.0,
        label=f"Clean baseline ({clean:.2f})", zorder=2,
    )
    ax.axhline(
        pressured, color="#d62728", linestyle="--", linewidth=2.0,
        label=f"Pressured baseline ({pressured:.2f})", zorder=2,
    )

    ax.plot(
        layers, patched, "o-", color="#1f77b4", linewidth=2.5, markersize=7,
        label="Patched P(correct)", zorder=5,
    )

    ax.set_xlabel("Patch layer")
    if show_ylabel:
        ax.set_ylabel("P(correct)")
    ax.set_title(title, fontweight="bold", pad=6)
    ax.set_xlim(9.5, 25.5)
    ax.set_ylim(-0.02, 1.05)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(5))
    ax.xaxis.set_minor_locator(mticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(mticker.MultipleLocator(0.2))
    ax.grid(True, color="#E0E0E0", linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color("#666666")
    ax.spines["bottom"].set_color("#666666")
    ax.legend(loc="lower right", fontsize=10, framealpha=0.95, edgecolor="#CCCCCC")


def fig9() -> None:
    """Fig 9: Mistral-7B replication — patching + component decomposition, 2 panels."""
    mistral_patch_path = RESULTS_DIR / "cross_model_patching" / "mistral_patching.pkl"
    mistral_comp_path = RESULTS_DIR / "cross_model_patching" / "mistral_component_patching.pkl"

    mistral_patch = _load_patching_data(mistral_patch_path)
    mistral_comp = _load_pkl(mistral_comp_path)

    if mistral_patch is None and mistral_comp is None:
        print(f"  SKIP fig9: no Mistral data files found")
        return

    n_panels = (1 if mistral_patch is not None else 0) + (1 if mistral_comp is not None else 0)

    if n_panels == 2:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

        # Panel 1: patching curve
        _patching_panel(ax1, mistral_patch, "Mistral-7B-Instruct: Patching", show_ylabel=True)

        # Panel 2: component decomposition
        out = mistral_comp
        layers = out["layers"]
        pressured = out["pressured_truth_base"]

        COMP_ORDER = ["mlp", "attn", "both", "residual"]
        have_both = (layers[0], "both") in out["patched_truth"]
        components = COMP_ORDER if have_both else ["mlp", "attn", "residual"]
        components = [c for c in components if (layers[0], c) in out["patched_truth"]]

        bar_comps = [c for c in ["mlp", "attn", "both"] if c in components]
        comp_labels = {
            "mlp": "MLP-only", "attn": "Attn-only",
            "both": "Both", "residual": "Residual",
        }
        colors = {"mlp": "#1f77b4", "attn": "#ff7f0e", "both": "#9467bd", "residual": "#2ca02c"}

        n_bars = len(bar_comps)
        bar_width = 0.22
        x = np.arange(len(layers))

        for bi, comp in enumerate(bar_comps):
            offset = (bi - (n_bars - 1) / 2.0) * bar_width
            means = np.array([
                float(out["patched_truth"][(l, comp)].mean()) - float(pressured.mean())
                for l in layers
            ])
            ax2.bar(
                x + offset, means, width=bar_width,
                label=comp_labels[comp], color=colors[comp], alpha=0.85,
            )

        if "residual" in components:
            res_means = np.array([
                float(out["patched_truth"][(l, "residual")].mean()) - float(pressured.mean())
                for l in layers
            ])
            half_span = n_bars * bar_width / 2.0 + 0.06
            ax2.hlines(
                res_means, x - half_span, x + half_span,
                colors=colors["residual"], linestyles="dashed",
                linewidth=2.0, label=comp_labels["residual"], alpha=0.9,
            )

        ax2.set_xticks(x)
        ax2.set_xticklabels([f"L{l}" for l in layers])
        ax2.set_xlabel("Patched layer")
        ax2.set_ylabel("P(correct) restoration delta")
        ax2.set_title("Mistral-7B-Instruct: Component Decomp.", fontweight="bold", pad=6)
        ax2.legend(loc="upper left", fontsize=10, framealpha=0.95)
        ax2.axhline(0, color="black", linewidth=0.8, alpha=0.4)
        for spine in ("top", "right"):
            ax2.spines[spine].set_visible(False)

    elif mistral_patch is not None:
        # Only patching data available
        fig, ax1 = plt.subplots(figsize=(7, 5))
        _patching_panel(ax1, mistral_patch, "Mistral-7B-Instruct: Patching", show_ylabel=True)
    else:
        # Only component data available -- unlikely but handle it
        print("  SKIP fig9: only component data without patching; not enough for figure")
        return

    fig.tight_layout()
    save_both(fig, "fig9_mistral_replication")


# ════════════════════════════════════════════════════════════════════════════
# Figure 10: Cross-domain (humanities vs STEM) bar chart
# ════════════════════════════════════════════════════════════════════════════

def fig10() -> None:
    """Fig 10: Cross-domain direct user assertion yield (humanities vs STEM)."""
    stem_csv = RESULTS_DIR / "stem" / "summary.csv"
    if not stem_csv.exists():
        print(f"  SKIP fig10: {stem_csv} not found")
        return

    stem_data = {}
    hum_data = {}
    with open(stem_csv) as f:
        for r in csv.DictReader(f):
            cond = r["condition"]
            proto = r.get("protocol", "suffixed")
            if proto != "suffixed":
                continue
            try:
                stem_data[cond] = float(r["stem_yield_pct"])
            except (ValueError, KeyError):
                pass
            try:
                hum_data[cond] = float(r["hum_yield_pct"])
            except (ValueError, KeyError):
                pass

    # Fall back to main summary for humanities if not in stem CSV.
    hum_csv = RESULTS_DIR / "summary.csv"
    if hum_csv.exists():
        with open(hum_csv) as f:
            for r in csv.DictReader(f):
                cond = r["condition"]
                if cond not in hum_data:
                    try:
                        hum_data[cond] = float(r["yield_pct"])
                    except (ValueError, KeyError):
                        pass

    compare = ["c1", "c4a", "c4c_matched", "c4d"]
    xlabels_map = {
        "c1": "Direct assertion", "c4a": "Named peer jury",
        "c4c_matched": "Anon. jury", "c4d": "Assist.-role jury",
    }

    s_vals, h_vals, xlabels = [], [], []
    for c in compare:
        s_vals.append(stem_data.get(c, 0))
        h_vals.append(hum_data.get(c, 0))
        xlabels.append(xlabels_map.get(c, c))

    x = np.arange(len(xlabels))
    bw = 0.35

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.bar(x - bw / 2, h_vals, bw, label="Humanities", color="#4C72B0", alpha=0.9,
           edgecolor="white", linewidth=0.8)
    ax.bar(x + bw / 2, s_vals, bw, label="STEM", color="#DD8452", alpha=0.9,
           edgecolor="white", linewidth=0.8)

    for i, (h, s) in enumerate(zip(h_vals, s_vals)):
        ax.text(i - bw / 2, h + 1.5, f"{h:.0f}%", ha="center", va="bottom",
                fontsize=11, fontweight="bold")
        ax.text(i + bw / 2, s + 1.5, f"{s:.0f}%", ha="center", va="bottom",
                fontsize=11, fontweight="bold")

    ax.axhline(25, color="gray", ls="--", alpha=0.7, lw=3.0)
    ax.set_xticks(x)
    ax.set_xticklabels(xlabels, fontsize=13)
    ax.set_xlabel("Condition", fontsize=14)
    ax.set_ylabel("Yield (%)", fontsize=14)
    ax.tick_params(axis="y", labelsize=12)
    ax.legend(fontsize=13)
    ax.set_ylim(0, 115)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    save_both(fig, "fig10_stem_vs_humanities")


# ════════════════════════════════════════════════════════════════════════════
# Figure 11: Conditional activation patching grid (2 x 5 x 10)
# ════════════════════════════════════════════════════════════════════════════

def fig11() -> None:
    """Fig 11: Conditional activation patching grid (2 framings x 5 k_wrong x 10 layers)."""
    final_path = RESULTS_DIR / "c6_patching" / "c6_conditional_patching_final.pkl"
    if not final_path.exists():
        print(f"  SKIP fig11: {final_path.name} not found")
        return

    final = _load_pkl(final_path)

    FRAMINGS = ["user", "self"]
    K_WRONG_VALUES = [0, 1, 2, 3, 4]
    LAYERS = [10, 12, 14, 15, 16, 17, 18, 20, 22, 25]

    framing_labels = {"user": "User-role", "self": "Assistant-role"}

    fig, axes = plt.subplots(1, 2, figsize=(16, 5), sharey=True)

    for ax, framing in zip(axes, FRAMINGS):
        grid = np.full((len(K_WRONG_VALUES), len(LAYERS)), np.nan)
        for ki, k_wrong in enumerate(K_WRONG_VALUES):
            key = (framing, k_wrong)
            if key not in final:
                continue
            cell = final[key]
            for li, layer in enumerate(LAYERS):
                if layer in cell["per_layer"]:
                    grid[ki, li] = cell["per_layer"][layer]["pct_restored"]

        im = ax.imshow(
            grid, aspect="auto", cmap="RdYlGn", vmin=-20, vmax=120,
            origin="lower",
        )

        # Annotate cells.
        for ki in range(len(K_WRONG_VALUES)):
            for li in range(len(LAYERS)):
                val = grid[ki, li]
                if not np.isnan(val):
                    color = "white" if abs(val) > 80 or val < 10 else "black"
                    ax.text(li, ki, f"{val:.0f}%", ha="center", va="center",
                            fontsize=8, color=color, fontweight="bold")

        ax.set_xticks(range(len(LAYERS)))
        ax.set_xticklabels([f"L{l}" for l in LAYERS], fontsize=10)
        ax.set_yticks(range(len(K_WRONG_VALUES)))
        ax.set_yticklabels([f"{k}v{4-k}" for k in K_WRONG_VALUES], fontsize=10)
        ax.set_xlabel("Patch layer", fontsize=12)
        if framing == "user":
            ax.set_ylabel("Wrong-arguing agents (k)", fontsize=12)
        ax.set_title(framing_labels[framing], fontsize=14, fontweight="bold")

    fig.colorbar(im, ax=axes, label="% gap restored", shrink=0.8)
    fig.tight_layout()
    save_both(fig, "fig11_conditional_patching_grid")


# ════════════════════════════════════════════════════════════════════════════
# Figure 12: User-role 4v0 yield vs calibration set size
# ════════════════════════════════════════════════════════════════════════════

def fig12() -> None:
    """Fig 12: User-role 4v0 yield vs calibration set size.

    Falls back to the free-text vs logit-lens comparison if the calibration
    data is not available.
    """
    # Primary: calibration set size sweep
    calib_path = RESULTS_DIR / "calibration_sweep.json"
    if calib_path.exists():
        with open(calib_path) as f:
            calib = json.load(f)

        sizes = calib["set_sizes"]
        yields = calib["yield_pct"]
        ci_lo = calib.get("ci_lo", [0] * len(sizes))
        ci_hi = calib.get("ci_hi", [100] * len(sizes))

        fig, ax = plt.subplots(figsize=(8, 5.5))
        y = np.array(yields)
        lo = np.array(ci_lo)
        hi = np.array(ci_hi)
        ax.errorbar(
            sizes, y, yerr=[y - lo, hi - y],
            marker="o", linewidth=2.5, markersize=9, capsize=5,
            color="#1f77b4",
        )
        ax.set_xlabel("Calibration set size", fontsize=14)
        ax.set_ylabel("Yield (%)", fontsize=14)
        ax.set_ylim(-5, 105)
        ax.grid(alpha=0.3)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        fig.tight_layout()
        save_both(fig, "fig12_calibration_sweep")
        return

    # Fallback: free-text vs logit-lens comparison
    cmp_path = RESULTS_DIR / "freetext_robustness" / "comparison.csv"
    if not cmp_path.exists():
        print(f"  SKIP fig12: neither calibration_sweep.json nor freetext comparison.csv found")
        return

    rows = []
    with open(cmp_path) as f:
        for r in csv.DictReader(f):
            rows.append(r)

    COND_DISPLAY_FT = {
        "c1": "Direct assertion", "c4a": "Named peer jury",
        "c4c_matched": "Anon. jury", "c4c-m": "Anon. jury",
        "c4d": "Assist.-role jury", "c6": "Wrong-agent\ncount sweep",
        "clean": "Clean",
    }
    labels = [COND_DISPLAY_FT.get(r["condition"], r["condition"]) for r in rows]
    logit = [float(r["logit_lens_yield_pct"]) for r in rows]
    judge = [float(r["judge_argues_target_pct"]) for r in rows]

    n = len(labels)
    xs = np.arange(n)
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.bar(xs - width / 2, logit, width=width, color="#4C72B0",
           label="Logit-lens yield (%)", edgecolor="white", linewidth=0.8)
    ax.bar(xs + width / 2, judge, width=width, color="#DD8452",
           label="Free-text argues_for_target (%)", edgecolor="white", linewidth=0.8)

    for i, (lv, jv) in enumerate(zip(logit, judge)):
        ax.text(i - width / 2, lv + 1.5, f"{lv:.0f}", ha="center", va="bottom",
                fontsize=12, fontweight="bold")
        ax.text(i + width / 2, jv + 1.5, f"{jv:.0f}", ha="center", va="bottom",
                fontsize=12, fontweight="bold")

    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=13)
    ax.set_ylim(0, 110)
    ax.set_ylabel("Rate (%)", fontsize=14)
    ax.set_xlabel("Condition", fontsize=14)
    ax.tick_params(axis="y", labelsize=12)
    ax.legend(loc="upper left", fontsize=12)
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    fig.tight_layout()
    save_both(fig, "fig12_calibration_sweep")


# ════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════

FIGURE_FUNCS = {
    2: ("Wrong-agent count sweep", fig2),
    3: ("Activation patching restoration", fig3),
    4: ("Base vs Instruct cross-family", fig4),
    5: ("Dissenter rescue", fig5),
    6: ("Yield vs fraction-wrong (scaling)", fig6),
    7: ("Component decomposition", fig7),
    8: ("SAE feature clamping", fig8),
    9: ("Mistral-7B replication", fig9),
    10: ("STEM vs humanities", fig10),
    11: ("Conditional patching grid", fig11),
    12: ("Calibration set size", fig12),
}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate NeurIPS paper figures 2-12."
    )
    parser.add_argument(
        "--fig", type=int, nargs="+", default=None,
        help="Generate only the specified figure number(s), e.g. --fig 3 5",
    )
    args = parser.parse_args()

    targets = args.fig if args.fig else sorted(FIGURE_FUNCS.keys())

    print(f"figures -> {FIGURES_DIR}\n")
    for n in targets:
        if n not in FIGURE_FUNCS:
            print(f"[fig{n}] unknown figure number, skipping")
            continue
        desc, func = FIGURE_FUNCS[n]
        print(f"[fig{n}] {desc} ...")
        try:
            func()
        except Exception as e:
            print(f"  ERROR fig{n}: {e}")

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
