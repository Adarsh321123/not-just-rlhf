#!/usr/bin/env python
"""Cross-family base vs instruct comparison analysis.

Loads base model results from results/base_model/{family}/summary.csv and
instruct results from results/{family}/summary.csv for all four families
(llama, mistral, gemma, qwen). Produces a combined summary table, grouped
bar chart, and text analysis.

Usage::

    python scripts/analyze_base_vs_instruct.py
"""
from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import csv
import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import RESULTS_DIR, FIGURES_DIR

FAMILIES = ["llama", "mistral", "gemma", "qwen"]
CONDITIONS = ["c4a", "c4c_matched", "c4d"]
COND_LABELS = {"c4a": "C4a", "c4c_matched": "C4c-matched", "c4d": "C4d"}

FAMILY_DISPLAY = {
    "llama": "Llama-3.1-8B",
    "mistral": "Mistral-7B",
    "gemma": "Gemma-2-9B",
    "qwen": "Qwen-2.5-7B",
}


def _load_from_pickle(pkl_path: Path, cond: str) -> dict | None:
    if not pkl_path.exists():
        return None
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    n_q = data.get("n_questions")
    if n_q is None:
        if "yielded" in data:
            n_q = len(data["yielded"])
        elif "token_counts" in data:
            n_q = len(data["token_counts"])
    return {
        "yield_pct": data["yield_rate"] * 100,
        "onset": data.get("onset"),
        "n_questions": n_q,
        "final_probe": data["probe_accs"][-1] if "probe_accs" in data and data["probe_accs"] else None,
    }


def load_instruct(family: str) -> dict[str, dict]:
    rows = {}
    for cond in CONDITIONS:
        if family == "llama":
            pkl_path = RESULTS_DIR / f"{cond}.pkl"
        else:
            pkl_path = RESULTS_DIR / family / f"{cond}.pkl"
        entry = _load_from_pickle(pkl_path, cond)
        if entry is not None:
            rows[cond] = entry
    return rows


def load_base(family: str) -> dict[str, dict]:
    rows = {}
    for cond in CONDITIONS:
        if family == "llama":
            pkl_path = RESULTS_DIR / "base_model" / f"{cond}.pkl"
        else:
            pkl_path = RESULTS_DIR / "base_model" / family / f"{cond}.pkl"
        entry = _load_from_pickle(pkl_path, cond)
        if entry is not None:
            # Also grab CI from the CSV if available
            if family == "llama":
                csv_path = RESULTS_DIR / "base_model" / "summary.csv"
            else:
                csv_path = RESULTS_DIR / "base_model" / family / "summary.csv"
            if csv_path.exists():
                with open(csv_path) as f:
                    for r in csv.DictReader(f):
                        if r["condition"] == cond:
                            entry["ci_lo"] = float(r.get("ci_lo", 0))
                            entry["ci_hi"] = float(r.get("ci_hi", 0))
                            break
            rows[cond] = entry
    return rows


def make_per_family_figure(family: str, inst: dict, base: dict):
    """Produce a per-family base-vs-instruct grouped bar chart matching fig4 style."""
    conds = [c for c in CONDITIONS if c in inst and c in base]
    if not conds:
        return

    fig, ax = plt.subplots(figsize=(8, 5.5))
    x = np.arange(len(conds))
    width = 0.35

    inst_vals = [inst[c]["yield_pct"] for c in conds]
    base_vals = [base[c]["yield_pct"] for c in conds]

    # CIs for base (if available)
    base_ci_lo = [base[c].get("ci_lo", base[c]["yield_pct"]) for c in conds]
    base_ci_hi = [base[c].get("ci_hi", base[c]["yield_pct"]) for c in conds]
    base_err_lo = [base[c]["yield_pct"] - lo for lo, c in zip(base_ci_lo, conds)]
    base_err_hi = [hi - base[c]["yield_pct"] for hi, c in zip(base_ci_hi, conds)]

    display = FAMILY_DISPLAY[family]
    bars_inst = ax.bar(x - width / 2, inst_vals, width,
                       label=f"{display} Instruct", color="#55A868", alpha=0.9)
    bars_base = ax.bar(x + width / 2, base_vals, width,
                       yerr=[base_err_lo, base_err_hi], capsize=4,
                       label=f"{display} Base", color="#DD8452", alpha=0.9)

    ax.set_ylabel("Yield rate (%)", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels([COND_LABELS[c] for c in conds], fontsize=11)
    ax.set_ylim(0, 115)
    ax.axhline(y=25, color="gray", linestyle="--", alpha=0.4)
    ax.text(len(conds) - 0.5, 26, "chance (25%)", color="gray", fontsize=9,
            style="italic", ha="right")
    ax.legend(fontsize=10, loc="upper left")

    for bar in list(bars_inst) + list(bars_base):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 2,
                f"{h:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")

    plt.tight_layout()
    fig_path = FIGURES_DIR / f"fig_base_vs_instruct_{family}.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved per-family figure -> {fig_path}")


def main() -> int:
    print("=" * 70)
    print("CROSS-FAMILY BASE vs INSTRUCT ANALYSIS")
    print("=" * 70)

    all_rows = []
    instruct_data = {}
    base_data = {}

    for family in FAMILIES:
        print(f"\n--- {family.upper()} ---")
        inst = load_instruct(family)
        base = load_base(family)
        instruct_data[family] = inst
        base_data[family] = base

        make_per_family_figure(family, inst, base)

        for cond in CONDITIONS:
            if cond in inst:
                all_rows.append({
                    "family": family,
                    "model_type": "instruct",
                    "condition": cond,
                    "yield_pct": round(inst[cond]["yield_pct"], 2),
                    "onset": inst[cond].get("onset"),
                    "final_probe": inst[cond].get("final_probe"),
                    "n_questions": inst[cond]["n_questions"],
                })
            if cond in base:
                all_rows.append({
                    "family": family,
                    "model_type": "base",
                    "condition": cond,
                    "yield_pct": round(base[cond]["yield_pct"], 2),
                    "onset": base[cond].get("onset"),
                    "final_probe": base[cond].get("final_probe"),
                    "n_questions": base[cond]["n_questions"],
                })

    # Save combined CSV
    csv_path = RESULTS_DIR / "base_model" / "cross_family_summary.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["family", "model_type", "condition", "yield_pct", "onset", "final_probe", "n_questions"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(all_rows)
    print(f"\nSaved combined CSV -> {csv_path}")

    # Print combined table
    print(f"\n{'Family':<10} {'Type':<10} {'Condition':<15} {'Yield%':>8} {'Onset':>7} {'Probe':>8} {'N':>5}")
    print("-" * 70)
    for r in all_rows:
        onset_str = str(r["onset"]) if r["onset"] is not None else "-"
        fp_str = f"{r['final_probe']:.4f}" if r["final_probe"] is not None else "-"
        n_str = str(r["n_questions"]) if r["n_questions"] is not None else "-"
        print(f"{r['family']:<10} {r['model_type']:<10} {r['condition']:<15} "
              f"{r['yield_pct']:>8.1f} {onset_str:>7} {fp_str:>8} {n_str:>5}")

    # Figure: 4 families x 3 conditions x 2 types = 24 bars
    fig, ax = plt.subplots(figsize=(16, 7))

    n_families = len(FAMILIES)
    n_conds = len(CONDITIONS)
    group_width = 0.7
    bar_width = group_width / (n_conds * 2)

    colors_inst = {"c4a": "#4C72B0", "c4c_matched": "#55A868", "c4d": "#8172B3"}
    colors_base = {"c4a": "#7AAAD4", "c4c_matched": "#88CC99", "c4d": "#B3A6D4"}

    x_family = np.arange(n_families)
    legend_handles = []

    for ci, cond in enumerate(CONDITIONS):
        for ti, (mtype, cdict) in enumerate([("instruct", colors_inst), ("base", colors_base)]):
            offset = (ci * 2 + ti) * bar_width - group_width / 2 + bar_width / 2
            vals = []
            for family in FAMILIES:
                src = instruct_data[family] if mtype == "instruct" else base_data[family]
                vals.append(src[cond]["yield_pct"] if cond in src else 0)

            bars = ax.bar(
                x_family + offset, vals, bar_width,
                color=cdict[cond],
                edgecolor="white", linewidth=0.5,
                label=f"{cond} ({mtype})" if True else None,
            )
            legend_handles.append(bars)

            for bar in bars:
                h = bar.get_height()
                if h > 0:
                    ax.text(bar.get_x() + bar.get_width() / 2, h + 1,
                            f"{h:.0f}", ha="center", va="bottom", fontsize=7)

    ax.set_xlabel("Model Family", fontsize=12)
    ax.set_ylabel("Yield Rate (%)", fontsize=12)
    ax.set_title("Base vs Instruct: Sycophancy Yield Across Families and Conditions", fontsize=13)
    ax.set_xticks(x_family)
    ax.set_xticklabels([f.capitalize() for f in FAMILIES], fontsize=11)
    ax.set_ylim(0, 115)
    ax.axhline(y=25, color="gray", linestyle="--", alpha=0.3, label="chance (25%)")

    labels = []
    for cond in CONDITIONS:
        labels.append(f"{cond} (Instruct)")
        labels.append(f"{cond} (Base)")
    ax.legend(legend_handles, labels, loc="upper left", fontsize=8, ncol=2)

    plt.tight_layout()
    fig_path_png = FIGURES_DIR / "fig_base_vs_instruct_cross_family.png"
    fig_path_pdf = FIGURES_DIR / "fig_base_vs_instruct_cross_family.pdf"
    plt.savefig(fig_path_png, dpi=150, bbox_inches="tight")
    plt.savefig(fig_path_pdf, bbox_inches="tight")
    plt.close()
    print(f"\nSaved figure -> {fig_path_png}")
    print(f"Saved figure -> {fig_path_pdf}")

    # Text analysis
    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)

    diffs = []
    for family in FAMILIES:
        print(f"\n--- {family.upper()} ---")
        inst = instruct_data[family]
        base = base_data[family]

        if not base:
            print("  (no base results)")
            continue

        has_substitution = False
        for cond in CONDITIONS:
            if cond in base and cond in inst:
                b_yield = base[cond]["yield_pct"]
                i_yield = inst[cond]["yield_pct"]
                diff = b_yield - i_yield
                diffs.append(diff)
                marker = "YES" if b_yield > 25 else "no"
                print(f"  {cond}: base={b_yield:.1f}%, instruct={i_yield:.1f}%, diff={diff:+.1f}%  [{marker}]")
                if b_yield > 25:
                    has_substitution = True

        if has_substitution:
            print(f"  -> BASE model shows substitution pattern")
        else:
            print(f"  -> BASE model does NOT show substitution (all yields <= 25%)")

    # Cross-condition ordering check
    print("\n--- CROSS-CONDITION ORDERING (C4d > C4c_matched > C4a expected) ---")
    for family in FAMILIES:
        base = base_data[family]
        if not base or not all(c in base for c in CONDITIONS):
            print(f"  {family}: incomplete data")
            continue
        c4a_y = base["c4a"]["yield_pct"]
        c4c_y = base["c4c_matched"]["yield_pct"]
        c4d_y = base["c4d"]["yield_pct"]
        ordering_ok = c4d_y >= c4c_y >= c4a_y
        print(f"  {family}: C4d={c4d_y:.1f}% {'>' if c4d_y > c4c_y else '<='} "
              f"C4c_m={c4c_y:.1f}% {'>' if c4c_y > c4a_y else '<='} "
              f"C4a={c4a_y:.1f}%  {'PRESERVED' if ordering_ok else 'VIOLATED'}")

    # Onset comparison
    print("\n--- ONSET LAYER COMPARISON ---")
    for family in FAMILIES:
        base = base_data[family]
        inst = instruct_data[family]
        if not base:
            continue
        for cond in CONDITIONS:
            b_onset = base.get(cond, {}).get("onset")
            i_onset = inst.get(cond, {}).get("onset")
            print(f"  {family}/{cond}: base={b_onset}, instruct={i_onset}")

    # Mean yield difference
    if diffs:
        mean_diff = np.mean(diffs)
        print(f"\n--- MEAN YIELD DIFFERENCE (base - instruct) ---")
        print(f"  across all families x conditions: {mean_diff:+.1f}%")
        print(f"  (positive = base higher, negative = instruct higher)")

    # Verdict
    print("\n" + "=" * 70)
    families_with_base = [f for f in FAMILIES if base_data[f]]
    families_with_pattern = []
    for f in families_with_base:
        base = base_data[f]
        if any(base.get(c, {}).get("yield_pct", 0) > 25 for c in CONDITIONS):
            families_with_pattern.append(f)

    print(f"VERDICT: {len(families_with_pattern)}/{len(families_with_base)} families show "
          f"substitution in base models ({', '.join(families_with_pattern)})")
    if len(families_with_pattern) == len(families_with_base):
        print("=> The RLHF-causal account is REJECTED across all tested families.")
        print("   Base models exhibit the same sycophancy pattern as their Instruct variants.")
    elif len(families_with_pattern) >= 3:
        print("=> Strong evidence against the RLHF-causal account (>=3 families).")
    elif len(families_with_pattern) >= 2:
        print("=> Moderate evidence. The pattern generalizes but not universally.")
    else:
        print("=> Weak evidence. The pattern may be family-specific.")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
