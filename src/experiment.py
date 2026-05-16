"""Per-question experiment loop + summary metrics.

The ported ``run_experiment`` is functionally identical to the notebook version
except that:

* ``model``, ``tokenizer`` are passed as arguments instead of living at module
  scope.
* Onset metrics: both the original binary ``onset`` *and* the continuous E6
  metrics dict ``onset_metrics`` are reported.
* Yield rate is computed against a shared clean LDA basis (at ``LDA_LAYER``)
  that is constructed once and reused across conditions.
"""
from __future__ import annotations

from typing import Any, Callable

import numpy as np
import torch
from tqdm.auto import tqdm

from .config import LDA_LAYER, NUM_LAYERS
from .data import load_artifacts
from .lda import CleanLDA
from .logit_lens import compute_onset_metrics, find_suppression_onset, run_logit_lens


def run_experiment(
    build_prompt_fn: Callable,
    jury_data: dict,
    model,
    tokenizer,
    description: str = "experiment",
    clean_lda: CleanLDA | None = None,
) -> dict[str, Any]:
    """Run one experimental condition over all 400 questions.

    Returns a dict with keys:
        truth_probs, syco_probs, activations, wrong_indices,
        avg_truth, avg_syco, probe_accs, onset, onset_metrics,
        token_counts, yield_rate
    """
    art = load_artifacts()
    questions = art["known_questions"]
    final_probes = art["final_probes"]

    if clean_lda is None:
        clean_lda = CleanLDA.fit_default()

    # Derive per-question labels from the items themselves so this function
    # works on any slice of ``known_questions`` (smoke test, subsets, etc.).
    correct_labels = np.array([item["answer"] for item in questions], dtype=np.int64)

    all_truth, all_syco, all_acts = [], [], []
    all_wrong_indices: list[int] = []
    token_counts: list[int] = []

    for q_idx, item in enumerate(tqdm(questions, desc=description)):
        ans = item["answer"]
        wrong_idx = jury_data["gemma"][q_idx]["wrong_idx"]
        all_wrong_indices.append(wrong_idx)

        prompt = build_prompt_fn(q_idx, item, wrong_idx, jury_data, tokenizer)
        token_counts.append(len(tokenizer.encode(prompt)))

        truth_p, syco_p, hidden = run_logit_lens(prompt, ans, wrong_idx, model, tokenizer)
        all_truth.append(truth_p)
        all_syco.append(syco_p)
        all_acts.append(
            torch.stack([s[0, -1, :].half().cpu() for s in hidden]).numpy()
        )

    acts_arr = np.array(all_acts)
    avg_truth = np.mean(all_truth, axis=0)
    avg_syco = np.mean(all_syco, axis=0)

    probe_accs = [
        final_probes[l].score(acts_arr[:, l, :], correct_labels)
        for l in range(NUM_LAYERS)
    ]

    onset = find_suppression_onset(avg_truth, avg_syco)
    onset_metrics = compute_onset_metrics(avg_truth, avg_syco)

    yield_rate = clean_lda.compute_yield_rate(
        acts_arr[:, LDA_LAYER, :].astype(np.float32),
        correct_labels,
        all_wrong_indices,
    )

    print(f"  Suppression onset (binary): layer {onset}")
    print(f"  Onset metrics: {onset_metrics}")
    print(f"  Yield @ L{LDA_LAYER}: {yield_rate * 100:.1f}%")
    print(
        f"  Token counts — mean: {np.mean(token_counts):.0f}, "
        f"std: {np.std(token_counts):.0f}"
    )

    return {
        "truth_probs": all_truth,
        "syco_probs": all_syco,
        "activations": acts_arr,
        "wrong_indices": all_wrong_indices,
        "avg_truth": avg_truth,
        "avg_syco": avg_syco,
        "probe_accs": probe_accs,
        "onset": onset,
        "onset_metrics": onset_metrics,
        "token_counts": token_counts,
        "yield_rate": yield_rate,
    }
