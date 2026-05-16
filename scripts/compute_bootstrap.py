#!/usr/bin/env python
"""Phase 5 entrypoint (E8): bootstrap yield-rate CIs over the 400 questions.

Loads every ``results/<cond>.pkl`` saved by ``run_all_conditions.py``,
bootstraps 1000 iterations over question indices per condition, and writes
``results/bootstrap.pkl`` + ``results/summary.csv``.
"""
from __future__ import annotations

import csv
import os
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.bootstrap import BootstrapCI, bootstrap_yield_ci  # noqa: E402
from src.config import LDA_LAYER, RESULTS_DIR  # noqa: E402
from src.data import load_artifacts  # noqa: E402
from src.lda import CleanLDA  # noqa: E402
from src.prompts import CONDITION_REGISTRY  # noqa: E402


def main() -> int:
    art = load_artifacts()
    known_labels = art["known_labels"]
    clean_lda = CleanLDA.fit_default()

    cis: dict[str, BootstrapCI] = {}
    rows: list[dict] = []

    for name in CONDITION_REGISTRY:
        p = RESULTS_DIR / f"{name}.pkl"
        if not p.exists():
            print(f"skip {name}: {p} missing")
            continue
        with open(p, "rb") as f:
            res = pickle.load(f)

        acts = res["activations"][:, LDA_LAYER, :].astype(np.float32)
        ci = bootstrap_yield_ci(
            acts, known_labels, res["wrong_indices"], clean_lda,
            n_iter=1000, ci=0.95, seed=42,
        )
        cis[name] = ci
        print(
            f"{name:14s}  yield={ci.mean*100:5.1f}%  "
            f"[{ci.lo*100:5.1f}, {ci.hi*100:5.1f}]  se={ci.se*100:4.2f}"
        )

        om = res.get("onset_metrics", {})
        rows.append(
            {
                "condition": name,
                "yield_pct": round(ci.mean * 100, 2),
                "ci_lo_pct": round(ci.lo * 100, 2),
                "ci_hi_pct": round(ci.hi * 100, 2),
                "se_pct": round(ci.se * 100, 3),
                "onset_binary": res.get("onset"),
                "onset_gap_0.01": om.get("onset_gap_0.01"),
                "onset_gap_0.03": om.get("onset_gap_0.03"),
                "onset_gap_0.05": om.get("onset_gap_0.05"),
                "onset_gap_0.1": om.get("onset_gap_0.1"),
                "half_final_onset": om.get("half_final_onset"),
                "final_gap": round(om.get("final_gap", 0.0), 4),
                "mean_probe_acc_L25": round(float(res["probe_accs"][LDA_LAYER]), 4),
                "final_probe_acc": round(float(res["probe_accs"][-1]), 4),
                "token_count_mean": round(float(np.mean(res["token_counts"])), 1),
                "token_count_std": round(float(np.std(res["token_counts"])), 1),
            }
        )

    with open(RESULTS_DIR / "bootstrap.pkl", "wb") as f:
        pickle.dump(cis, f)
    print(f"\nsaved bootstrap CIs -> {RESULTS_DIR / 'bootstrap.pkl'}")

    if rows:
        csv_path = RESULTS_DIR / "summary.csv"
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"saved summary -> {csv_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
