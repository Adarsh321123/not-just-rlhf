#!/usr/bin/env python
"""Experiment #2 driver: MLP-only vs attention-only activation patching at L14-L18.

Modes
-----
Default          : run full experiment (mlp / attn / residual patches) from scratch.
--from-pkl       : reload saved pkl, skip patching, go straight to bootstrap + table.
--extend-both    : load pkl, run only the 'both' (simultaneous MLP+attn) patches,
                   merge back into pkl, then bootstrap + table.
--table-only     : print table and additivity check; skip figure generation.

Outputs
-------
results/component_patching.pkl          raw per-question patched_truth arrays
results/component_patching_summary.csv  bootstrap summary table (pct_of_both_delta)
figures/fig_component_decomp.png        grouped bar chart
figures/fig_component_decomp.pdf        same, vector
"""
from __future__ import annotations

import argparse
import csv
import os
import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.bootstrap import BootstrapCI  # noqa: E402
from src.component_patching import (  # noqa: E402
    run_both_extension,
    run_component_patching,
)
from src.config import FIGURES_DIR, RESULTS_DIR  # noqa: E402
from src.model import get_model_and_tokenizer  # noqa: E402


PAPER_RC = {
    "axes.labelsize": 14,
    "axes.titlesize": 16,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 11,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "axes.grid": True,
    "grid.alpha": 0.3,
}

# axis 12pt, ticks 11pt, title 14pt bold
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

# Display order for components in table / figure
COMP_ORDER = ["mlp", "attn", "both", "residual"]


# ──────────────────────────────────────────────────────────────────────────────
# Bootstrap
# ──────────────────────────────────────────────────────────────────────────────

def _bootstrap_delta_ci(
    patched: np.ndarray,
    pressured: np.ndarray,
    n_iter: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> BootstrapCI:
    rng = np.random.default_rng(seed)
    delta = patched - pressured
    n = len(delta)
    samples = np.empty(n_iter, dtype=np.float64)
    for j in range(n_iter):
        idx = rng.integers(0, n, size=n)
        samples[j] = delta[idx].mean()
    alpha = (1.0 - ci) / 2.0
    return BootstrapCI(
        mean=float(delta.mean()),
        lo=float(np.quantile(samples, alpha)),
        hi=float(np.quantile(samples, 1.0 - alpha)),
        se=float(samples.std(ddof=1)),
        n_iter=n_iter,
    )


def compute_bootstrap(
    out: dict, n_iter: int = 1000
) -> tuple[list[dict], dict[tuple, BootstrapCI], float, float,
           dict[int, float], dict[int, float]]:
    """Bootstrap CIs for all (layer, component) pairs.

    Uses 'both' delta as the baseline for pct_of_baseline when 'both'
    is present in patched_truth; otherwise falls back to 'residual'.

    Returns
    -------
    rows           : list of dicts for CSV output
    ci_dict        : {(layer, component): BootstrapCI}
    clean_mean     : float
    pressured_mean : float
    both_delta     : {layer: mean_delta from 'both' patch}  (None if unavailable)
    residual_delta : {layer: mean_delta from 'residual' patch}
    """
    layers = out["layers"]
    pressured = out["pressured_truth_base"]
    clean_mean = float(out["clean_truth_base"].mean())
    pressured_mean = float(pressured.mean())

    have_both = (layers[0], "both") in out["patched_truth"]
    components = COMP_ORDER if have_both else ["mlp", "attn", "residual"]
    components = [c for c in components if (layers[0], c) in out["patched_truth"]]

    both_delta: dict[int, float] = {}
    residual_delta: dict[int, float] = {}
    for l in layers:
        if have_both:
            both_delta[l] = (
                float(out["patched_truth"][(l, "both")].mean()) - pressured_mean
            )
        residual_delta[l] = (
            float(out["patched_truth"][(l, "residual")].mean()) - pressured_mean
        )

    baseline_delta = both_delta if have_both else residual_delta

    rows: list[dict] = []
    ci_dict: dict[tuple, BootstrapCI] = {}

    for l in layers:
        for comp in components:
            pt = out["patched_truth"][(l, comp)]
            ci = _bootstrap_delta_ci(pt, pressured, n_iter=n_iter, seed=42)
            ci_dict[(l, comp)] = ci
            mean_patched = float(pt.mean())
            base = baseline_delta[l]

            if comp == "both":
                pct = 100.0
            elif abs(base) > 1e-6:
                pct = round(ci.mean / base * 100, 2)
            else:
                pct = float("nan")

            rows.append({
                "layer": l,
                "component": comp,
                "mean_patched": round(mean_patched, 6),
                "mean_delta": round(ci.mean, 6),
                "ci_lo": round(ci.lo, 6),
                "ci_hi": round(ci.hi, 6),
                "ci_width": round(ci.hi - ci.lo, 6),
                "pct_of_both_delta": pct,
            })

    return rows, ci_dict, clean_mean, pressured_mean, both_delta, residual_delta


# ──────────────────────────────────────────────────────────────────────────────
# Output helpers
# ──────────────────────────────────────────────────────────────────────────────

def print_table(
    rows: list[dict],
    clean_mean: float,
    pressured_mean: float,
    n: int,
    both_delta: dict[int, float],
    residual_delta: dict[int, float],
) -> None:
    total_gap = clean_mean - pressured_mean
    have_both = bool(both_delta)
    print(
        f"\nn={n}  |  clean={clean_mean:.4f}  |  pressured={pressured_mean:.4f}  "
        f"|  full-restoration target={total_gap:+.4f}"
    )
    print()

    ref_col = "pct_of_both%" if have_both else "pct_of_res%"
    print(
        f"{'Layer':>6}  {'Component':>9}  {'mean_patched':>12}  "
        f"{'mean_delta':>11}  {'ci_lo':>8}  {'ci_hi':>8}  "
        f"{'ci_width':>9}  {ref_col:>11}"
    )
    print("-" * 93)

    prev_layer = None
    for row in rows:
        if prev_layer is not None and row["layer"] != prev_layer:
            print()
        prev_layer = row["layer"]
        pct = row["pct_of_both_delta"]
        pct_str = f"{pct:11.1f}" if isinstance(pct, float) and pct == pct else "         --"
        print(
            f"  L{row['layer']:2d}  {row['component']:>9}  "
            f"{row['mean_patched']:12.4f}  {row['mean_delta']:+11.4f}  "
            f"{row['ci_lo']:+8.4f}  {row['ci_hi']:+8.4f}  "
            f"{row['ci_width']:9.4f}  {pct_str}"
        )

    if have_both:
        print()
        print("── Upstream residual contribution (residual_delta − both_delta) ──")
        print(f"{'Layer':>6}  {'both_delta':>11}  {'residual_delta':>15}  {'upstream_pp':>12}")
        print("-" * 50)
        for l in sorted(both_delta):
            b = both_delta[l]
            r = residual_delta[l]
            up = (r - b) * 100
            print(f"  L{l:2d}  {b:+11.4f}  {r:+15.4f}  {up:+12.1f} pp")


def print_additivity_check(
    rows: list[dict],
    both_delta: dict[int, float],
    residual_delta: dict[int, float],
    layers: list[int],
) -> None:
    """Print additivity check.

    Primary check (when 'both' available): |attn_delta + mlp_delta - both_delta| ≤ 5 pp
    Secondary (always): |attn_delta + mlp_delta - residual_delta|, for reference.
    """
    have_both = bool(both_delta)
    lookup = {(r["layer"], r["component"]): r for r in rows}

    print()
    if have_both:
        print("── Additivity check: attn_delta + mlp_delta  vs  both_delta ──")
        print(
            f"{'Layer':>6}  {'attn_Δ':>9}  {'mlp_Δ':>8}  "
            f"{'sum':>8}  {'both_Δ':>8}  {'gap_pp':>8}  {'flag':>6}"
        )
        print("-" * 64)
        any_flag = False
        for l in layers:
            a  = lookup[(l, "attn")]["mean_delta"]
            m  = lookup[(l, "mlp")]["mean_delta"]
            b  = both_delta[l]
            r  = residual_delta[l]
            s  = a + m
            gap_b = (s - b) * 100
            gap_r = (s - r) * 100
            flag = "***" if abs(gap_b) > 5 else ""
            if flag:
                any_flag = True
            print(
                f"  L{l:2d}  {a:+9.4f}  {m:+8.4f}  {s:+8.4f}  "
                f"{b:+8.4f}  {gap_b:+8.1f}  {flag}"
            )
            print(
                f"{'':>50}  (vs residual: {gap_r:+.1f} pp)"
            )
        if any_flag:
            print(
                "\nNOTE: remaining gap vs 'both' is the MLP-adaptation term "
                "mlp(LN(h_in+clean_attn)) − mlp(LN(h_in+press_attn)), i.e.\n"
                "how much MLP output shifts when attn is patched at that layer."
            )
        else:
            print("\nAll layers within 5 pp of 'both' — additivity holds.")
    else:
        print("── Additivity check: attn_delta + mlp_delta  vs  residual_delta ──")
        print(
            f"{'Layer':>6}  {'attn_Δ':>9}  {'mlp_Δ':>8}  "
            f"{'sum':>8}  {'res_Δ':>8}  {'gap_pp':>8}  {'flag':>6}"
        )
        print("-" * 64)
        for l in layers:
            a = lookup[(l, "attn")]["mean_delta"]
            m = lookup[(l, "mlp")]["mean_delta"]
            r = residual_delta[l]
            s = a + m
            gap = (s - r) * 100
            flag = "***" if abs(gap) > 10 else ""
            print(
                f"  L{l:2d}  {a:+9.4f}  {m:+8.4f}  {s:+8.4f}  "
                f"{r:+8.4f}  {gap:+8.1f}  {flag}"
            )
        print("\n(run --extend-both to get the layer-local 'both' baseline)")


def save_csv(
    rows: list[dict],
    path: Path,
    both_delta: dict[int, float] | None = None,
    residual_delta: dict[int, float] | None = None,
) -> None:
    """Write CSV; adds upstream_delta = (residual_delta - both_delta) * 100 pp per layer."""
    fieldnames = [
        "layer", "component", "mean_patched", "mean_delta",
        "ci_lo", "ci_hi", "ci_width", "pct_of_both_delta", "upstream_delta",
    ]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            l = row["layer"]
            if (
                both_delta and residual_delta
                and l in both_delta and l in residual_delta
            ):
                upstream = round((residual_delta[l] - both_delta[l]) * 100, 4)
            else:
                upstream = float("nan")
            w.writerow({**row, "upstream_delta": upstream})


# ──────────────────────────────────────────────────────────────────────────────
# Figure
# ──────────────────────────────────────────────────────────────────────────────

def make_figure(
    rows: list[dict],
    ci_dict: dict[tuple, BootstrapCI],
    layers: list[int],
    out_stem: Path,
) -> None:
    have_both = any(r["component"] == "both" for r in rows)

    bar_comps = ["mlp", "attn", "both"] if have_both else ["mlp", "attn"]
    comp_labels = {
        "mlp": "MLP-only",
        "attn": "Attention-only",
        "both": "Both (layer-local baseline)",
        "residual": "Residual (full upstream)",
    }
    colors = {"mlp": "#1f77b4", "attn": "#ff7f0e", "both": "#9467bd", "residual": "#2ca02c"}

    lookup: dict[tuple, dict] = {(r["layer"], r["component"]): r for r in rows}

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

        # Per-layer dashed residual segment (one short dash per layer group)
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
        ax.set_title(
            "Component decomposition: attention carries the L14–L18 restoration; MLP is null.",
            fontweight="bold",
        )
        ax.legend(loc="upper left", framealpha=0.95, edgecolor="#CCCCCC")
        ax.axhline(0, color="black", linewidth=0.8, alpha=0.4)
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)

        caption = (
            '"Both" = layer-local baseline (patch attn + MLP simultaneously, pressured '
            'upstream h_in).  Residual exceeds "both" when upstream h_in carries the '
            'restoration signal.\n'
            'L14 and L17 are layer-local loci (attention ≈95–115 % of "both" baseline);  '
            'L15 / L16 / L18 are upstream-dominated (both ≈ 0 or negative; residual large).'
        )
        fig.text(
            0.02, -0.02, caption,
            fontsize=9, va="top", ha="left",
            style="italic", color="#444444",
        )

        fig.tight_layout(rect=[0, 0.08, 1, 1])

        png = out_stem.with_suffix(".png")
        pdf = out_stem.with_suffix(".pdf")
        fig.savefig(png, dpi=200, bbox_inches="tight")
        fig.savefig(pdf, bbox_inches="tight")
        plt.close(fig)
        print(f"saved -> {png}")
        print(f"saved -> {pdf}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n", type=int, default=400)
    p.add_argument("--layers", default="14,15,16,17,18")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--B", type=int, default=1000, help="bootstrap iterations")
    p.add_argument("--table-only", action="store_true",
                   help="print table + additivity check, skip figure")
    p.add_argument("--from-pkl", action="store_true",
                   help="load pkl without re-running patching")
    p.add_argument("--extend-both", action="store_true",
                   help="load pkl, run 'both' patches, merge and save")
    args = p.parse_args()

    layers = [int(x) for x in args.layers.split(",")]
    pkl_path = RESULTS_DIR / "component_patching.pkl"

    # ── Load or run ──────────────────────────────────────────────────────────
    if args.extend_both or args.from_pkl:
        if not pkl_path.exists():
            print(f"ERROR: {pkl_path} not found — run without --from-pkl first")
            return 1
        with open(pkl_path, "rb") as f:
            out = pickle.load(f)
        print(f"loaded from {pkl_path}")

        if args.extend_both:
            if (out["layers"][0], "both") in out["patched_truth"]:
                print("'both' already in pkl — skipping rerun, using cached data")
            else:
                print(f"running 'both' extension: {len(out['question_indices'])}q × {len(out['layers'])}L")
                model, tokenizer = get_model_and_tokenizer()
                pt_both, ps_both = run_both_extension(model, tokenizer, out)
                out["patched_truth"].update(pt_both)
                out["patched_syco"].update(ps_both)
                if "both" not in out["components"]:
                    out["components"] = out["components"] + ["both"]
                with open(pkl_path, "wb") as f:
                    pickle.dump(out, f)
                print(f"merged 'both' into {pkl_path}")
    else:
        print(f"component patching: n={args.n}, layers={layers}, seed={args.seed}")
        model, tokenizer = get_model_and_tokenizer()
        out = run_component_patching(
            model, tokenizer, layers=layers, n_questions=args.n, seed=args.seed
        )
        with open(pkl_path, "wb") as f:
            pickle.dump(out, f)
        print(f"\nsaved -> {pkl_path}")

    # ── Bootstrap + output ───────────────────────────────────────────────────
    rows, ci_dict, clean_mean, pressured_mean, both_delta, residual_delta = (
        compute_bootstrap(out, n_iter=args.B)
    )
    n = len(out["pressured_truth_base"])
    print_table(rows, clean_mean, pressured_mean, n, both_delta, residual_delta)
    print_additivity_check(rows, both_delta, residual_delta, out["layers"])

    csv_path = RESULTS_DIR / "component_patching_summary.csv"
    save_csv(rows, csv_path, both_delta=both_delta, residual_delta=residual_delta)
    print(f"\nsaved -> {csv_path}")

    if args.table_only:
        print("\n(figure skipped — re-run without --table-only to generate)")
        return 0

    make_figure(rows, ci_dict, out["layers"], FIGURES_DIR / "fig_component_decomp")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
