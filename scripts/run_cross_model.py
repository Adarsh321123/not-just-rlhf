#!/usr/bin/env python
"""E13 — cross-model replication runner (full version).

For a new subject model, this script:

1. Runs the subject on the 400 clean MMLU prompts, computes
   ``P(correct)`` and saves per-layer activations.
2. Filters to questions with clean ``P(correct) > threshold`` (default 0.8).
3. Trains **33 per-layer linear probes** on the subject's clean activations
   using the ``setup.ipynb`` methodology (5-fold CV + retrain on 100%).
   Saves to ``results/probes/{subject}/final_probes.joblib``.
4. Fits a subject-specific CleanLDA at the subject's equivalent of layer 25.
5. Runs the 8 core conditions on the subject — the 4-framing × 2-jury grid
   that the task's replication figure calls for:

       c4a, c5a — user-role peer jury
       c4c_matched, c5c_matched — no-attribution matched-consensus
       c4d, c5d — self-framing
       c4e_xmodel, c5e_xmodel — subject-agnostic tool-role port

6. Saves per-condition pickles to ``results/{subject}/{cond}.pkl`` and a
   per-subject summary CSV to ``results/{subject}/summary.csv``.

Importantly, for subjects that are themselves one of the existing jury
models (Qwen, Mistral, Gemma), the jury entry for the subject is removed
and replaced with a duplicate of one of the remaining jury entries so the
subject never sees its own wrong-arguing response. This corrects a
self-attack confound from the first cross-model pass.

Usage::

    CUDA_VISIBLE_DEVICES=0 python scripts/run_cross_model.py --subject qwen
"""
from __future__ import annotations

import os

# Limit BLAS threads before importing numpy/sklearn — otherwise each
# LogisticRegression fit saturates all cores and two parallel runs thrash.
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "4")

import argparse
import json
import pickle
import sys
from pathlib import Path

import joblib
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import LDA_LAYER, RESULTS_DIR  # noqa: E402
from src.cross_model import (  # noqa: E402
    collect_clean,
    filter_jury_exclude_subject,
    load_subject_model,
    run_cross_experiment,
    run_cross_experiment_full,
    train_subject_probes,
)
from src.data import load_artifacts  # noqa: E402
from src.prompts import (  # noqa: E402
    build_prompt_c1_single_user,
    build_prompt_c3_token_matched,
    build_prompt_no_attribution,
    build_prompt_no_attribution_matched,
    build_prompt_self_framing,
    build_prompt_tool_role_xmodel,
    build_prompt_user_role_jury,
    set_c4a_token_counts,
)


SUBJECT_IDS = {
    "qwen": "Qwen/Qwen2.5-7B-Instruct",
    "mistral": "mistralai/Mistral-7B-Instruct-v0.3",
    "gemma": "google/gemma-2-9b-it",
}

# Conditions run on each cross-model subject. The 12 base conditions from
# the Llama sweep, with tool-role replaced by the subject-agnostic
# build_prompt_tool_role_xmodel variant (Llama-specific special-token
# version is not portable).
#
# Note: c3 depends on c4a token counts. ``_condition_order`` below
# promotes c4a before c3 when both are requested, and the per-subject
# runner captures c4a token counts via ``set_c4a_token_counts``.
CONDITIONS: dict[str, tuple[object, str]] = {
    "c1":           (build_prompt_c1_single_user,           "strong"),
    "c3":           (build_prompt_c3_token_matched,         "strong"),
    "c4a":          (build_prompt_user_role_jury,           "strong"),
    "c5a":          (build_prompt_user_role_jury,           "weak"),
    "c4c":          (build_prompt_no_attribution,           "strong"),
    "c5c":          (build_prompt_no_attribution,           "weak"),
    "c4c_matched":  (build_prompt_no_attribution_matched,   "strong"),
    "c5c_matched":  (build_prompt_no_attribution_matched,   "weak"),
    "c4d":          (build_prompt_self_framing,             "strong"),
    "c5d":          (build_prompt_self_framing,             "weak"),
    "c4e_xmodel":   (build_prompt_tool_role_xmodel,         "strong"),
    "c5e_xmodel":   (build_prompt_tool_role_xmodel,         "weak"),
}


def _order_conditions(names: list[str]) -> list[str]:
    """If c3 is requested, make sure c4a runs first (c3 uses its token counts)."""
    if "c3" in names and "c4a" in names:
        head = ["c4a", "c3"]
        return head + [n for n in names if n not in head]
    return list(names)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--subject", required=True, choices=list(SUBJECT_IDS.keys()))
    p.add_argument("--threshold", type=float, default=0.8)
    p.add_argument("--max-questions", type=int, default=None)
    p.add_argument(
        "--skip-probes",
        action="store_true",
        help="Skip 33-layer probe training (fast smoke-test mode)",
    )
    p.add_argument(
        "--conditions",
        default=None,
        help="Comma-separated subset of conditions to run. Default: all 12.",
    )
    p.add_argument(
        "--full-schema",
        action="store_true",
        help="Use run_cross_experiment_full to save Llama-compatible full "
             "pickles (per-layer truth/syco probs + activations + probe_accs + "
             "onset metrics). Slower but matches the task spec's "
             "'same pickle schema as the main Llama results' requirement.",
    )
    args = p.parse_args()

    subject_key = args.subject
    subject_id = SUBJECT_IDS[subject_key]

    out_dir = RESULTS_DIR / subject_key
    out_dir.mkdir(parents=True, exist_ok=True)
    probes_dir = RESULTS_DIR / "probes" / subject_key
    probes_dir.mkdir(parents=True, exist_ok=True)
    print(f"subject={subject_key}  writing to {out_dir}/")

    # ── artifacts + jury correction ────────────────────────────────────────
    art = load_artifacts()
    questions = art["known_questions"]
    if args.max_questions is not None:
        questions = questions[: args.max_questions]
        print(f"[smoke] limited to {len(questions)} questions")

    jury_strong_raw = art["jury_strong"]
    jury_weak_raw = art["jury_weak"]
    jury_strong = filter_jury_exclude_subject(jury_strong_raw, subject_key)
    jury_weak = filter_jury_exclude_subject(jury_weak_raw, subject_key)
    jury_strong_keys = list(jury_strong.keys())
    print(f"  jury lineup (subject removed, duplicate fill):  {jury_strong_keys}")

    # ── model load ─────────────────────────────────────────────────────────
    model, tok = load_subject_model(subject_id)
    n_layers = len(model.model.layers) + 1
    print(f"  num_layers_incl_embed={n_layers}  hidden_dim={model.config.hidden_size}")

    # ── Step 1: clean pass + filter + LDA ──────────────────────────────────
    print("\n[1] clean pass...")
    subject_art = collect_clean(
        model, tok, questions, threshold=args.threshold
    )
    n_pass = int(subject_art.passing_mask.sum())
    print(f"  filter threshold {args.threshold}: {n_pass}/{len(questions)} pass")
    print(f"  subject LDA layer: {subject_art.lda_layer}")

    # ── Step 2: 33-layer probe training (setup.ipynb methodology) ──────────
    if args.skip_probes:
        print("\n[2] skipping probe training (--skip-probes)")
        avg_probe_accs = None
    else:
        print("\n[2] probe training...")
        passing_idx_np = np.where(subject_art.passing_mask)[0]
        pass_labels = np.array(
            [questions[i]["answer"] for i in passing_idx_np], dtype=np.int64
        )
        pass_acts = subject_art.clean_activations[passing_idx_np].astype(np.float32)
        # setup.ipynb always trains on all layers. For subjects with != 33 layers
        # we use the subject's own layer count.
        n_layers_train = pass_acts.shape[1]
        final_probes, avg_probe_accs = train_subject_probes(
            pass_acts, pass_labels, n_layers=n_layers_train
        )
        joblib.dump(final_probes, probes_dir / "final_probes.joblib")
        joblib.dump(avg_probe_accs, probes_dir / "avg_probe_accs.joblib")
        print(
            f"  saved probes -> {probes_dir}/final_probes.joblib  "
            f"({n_layers_train} layers)"
        )

    # Save per-subject clean artifacts (without the big activations array)
    with open(out_dir / "clean.pkl", "wb") as f:
        pickle.dump(
            {
                "subject_id": subject_id,
                "num_layers": subject_art.num_layers,
                "hidden_dim": subject_art.hidden_dim,
                "clean_truth_probs": subject_art.clean_truth_probs,
                "clean_answer_probs": subject_art.clean_answer_probs,
                "passing_mask": subject_art.passing_mask,
                "n_pass": n_pass,
                "lda_layer": subject_art.lda_layer,
                "lda_centroids": subject_art.lda_centroids,
                "threshold": args.threshold,
                "avg_probe_accs": avg_probe_accs,
                "jury_lineup_after_subject_removed": list(jury_strong.keys()),
            },
            f,
        )
    with open(out_dir / "lda.pkl", "wb") as f:
        pickle.dump(subject_art.lda, f)
    print(f"  saved clean metadata -> {out_dir}/clean.pkl")

    passing_indices = np.where(subject_art.passing_mask)[0].astype(np.int64)
    if len(passing_indices) < 50:
        print(
            f"WARNING: only {len(passing_indices)} questions pass — "
            "results will be noisy but we run anyway"
        )

    # ── Step 3: conditions ─────────────────────────────────────────────────
    if args.conditions:
        wanted = [n.strip() for n in args.conditions.split(",")]
        conditions_to_run = {k: CONDITIONS[k] for k in wanted if k in CONDITIONS}
    else:
        conditions_to_run = dict(CONDITIONS)

    # c3 depends on c4a's token counts. If c3 is requested but c4a is not,
    # try to load cached c4a token counts from a previous run; if c4a is
    # also requested, ensure c4a runs first.
    ordered_names = _order_conditions(list(conditions_to_run.keys()))
    if "c3" in ordered_names and "c4a" not in ordered_names:
        tokens_cache = out_dir / "c4a_token_counts.npy"
        if tokens_cache.exists():
            set_c4a_token_counts(np.load(tokens_cache).tolist())
            print(f"  loaded cached c4a token counts from {tokens_cache}")
        else:
            raise SystemExit(
                f"c3 requires c4a token counts. Either include c4a in "
                f"--conditions or pre-run c4a so {tokens_cache} exists."
            )

    # Load subject probes for full-schema runs (needed for per-layer probe_accs)
    probes_for_full = None
    if args.full_schema:
        probes_path = probes_dir / "final_probes.joblib"
        if probes_path.exists():
            probes_for_full = joblib.load(probes_path)
            print(f"  loaded {len(probes_for_full)} subject probes for full-schema runs")
        else:
            print(
                f"  WARNING: --full-schema requested but no probes at {probes_path}; "
                "probe_accs will be NaN"
            )

    summary: dict[str, dict] = {}
    for name in ordered_names:
        build_fn, jury_name = conditions_to_run[name]
        jury = jury_strong if jury_name == "strong" else jury_weak
        print("\n" + "=" * 60)
        print(f"CONDITION {name}   (subject={subject_key}, jury={jury_name}, n={len(passing_indices)})")
        print("=" * 60)
        if args.full_schema:
            result = run_cross_experiment_full(
                build_fn, jury, model, tok,
                passing_indices=passing_indices,
                questions=questions,
                lda=subject_art.lda,
                centroids=subject_art.lda_centroids,
                lda_layer=subject_art.lda_layer,
                subject_probes=probes_for_full,
                description=f"{subject_key}/{name}",
            )
        else:
            result = run_cross_experiment(
                build_fn, jury, model, tok,
                passing_indices=passing_indices,
                questions=questions,
                lda=subject_art.lda,
                centroids=subject_art.lda_centroids,
                lda_layer=subject_art.lda_layer,
                description=f"{subject_key}/{name}",
            )
        cond_path = out_dir / f"{name}.pkl"
        with open(cond_path, "wb") as f:
            pickle.dump(result, f)
        print(f"  saved -> {cond_path}")
        summary[name] = {"yield_rate": result["yield_rate"], "n": result["n_questions"]}

        # Capture c4a token counts so the subsequent c3 run can length-match.
        # c3's prompt builder indexes ``_c4a_token_counts[q_idx]`` using the
        # *original* question index (0..len(questions)-1). We only run c4a on
        # the filtered passing subset, so we need to pad the list to the full
        # length of ``questions`` and only fill the passing positions.
        if name == "c4a":
            token_counts_arr = np.asarray(result["token_counts"], dtype=np.int64)
            padded = np.full(len(questions), fill_value=500, dtype=np.int64)
            padded[passing_indices] = token_counts_arr
            set_c4a_token_counts(padded.tolist())
            tokens_cache = out_dir / "c4a_token_counts.npy"
            np.save(tokens_cache, padded)
            print(
                f"  cached c4a token counts -> {tokens_cache} "
                f"(length {len(padded)}, {len(token_counts_arr)} filled)"
            )

    # ── per-subject summary CSV ────────────────────────────────────────────
    summary_csv = out_dir / "summary.csv"
    with open(summary_csv, "w") as f:
        f.write("condition,yield_pct,n\n")
        for name, info in summary.items():
            f.write(f"{name},{info['yield_rate'] * 100:.2f},{info['n']}\n")
    print(f"\nsaved summary CSV -> {summary_csv}")

    with open(out_dir / "summary.pkl", "wb") as f:
        pickle.dump({"subject": subject_id, "conditions": summary}, f)

    print("\n=== FINAL PER-CONDITION YIELD (subject=" + subject_key + ") ===")
    for name, info in summary.items():
        print(f"  {name:13s}  yield={info['yield_rate'] * 100:5.1f}%  n={info['n']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
