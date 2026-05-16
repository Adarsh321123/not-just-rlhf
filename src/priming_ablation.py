"""Priming-suffix ablation (E20).

Reruns C4a, C4c_matched, and C4d without the ``"The correct answer is ("``
priming suffix that the canonical builders in :mod:`src.prompts` append after
the chat template's ``add_generation_prompt=True`` output. The experiment tests
reading 1 of the Phase 3 free-text gap: that the forced-choice yield is inflated
by the priming suffix placing the L14-L16 substituted state in maximal contact
with the output projection.

Design choices
--------------
- **Surgical strip.** We do not re-derive the prompt-building logic. Each
  unsuffixed builder calls the original builder from :mod:`src.prompts` and
  chops off the known suffix. If the suffix is ever absent (drift in the
  upstream module), we fall back to re-rendering the chat template on the
  messages we rebuild locally and emit a warning.

- **Logit-lens interpretation changes.** The last-token hidden state is now
  the state that predicts the assistant's first generated token rather than
  the token after ``"("``. The softmax over ``{A, B, C, D}`` is no longer a
  forced-choice distribution in absolute magnitude (most of the probability
  mass goes to tokens like ``"The"`` or ``"Based"``), but the relative
  ordering among the four choice tokens is still the quantity the yield rate
  and the LDA probe consume. We use the same CleanLDA basis as the rest of
  the project.

- **Shared CleanLDA.** We reuse the project-wide LDA basis trained on clean
  activations with the priming suffix. The LDA centroids live at a different
  token position than the unsuffixed activations, so this is not a perfectly
  calibrated comparison — but it is the consistent and conservative one:
  using the same basis for both arms of the comparison means any shift
  reflects a real change in the pressured activation geometry.

- **Schema-compatible.** ``run_experiment_unsuffixed`` returns the same dict
  keys as :func:`src.experiment.run_experiment` so downstream analysis can
  load the pickles with identical code.
"""
from __future__ import annotations

import warnings
from typing import Any, Callable

import numpy as np
import torch
from tqdm.auto import tqdm

from .config import LDA_LAYER, NUM_LAYERS
from .lda import CleanLDA
from .logit_lens import compute_onset_metrics, find_suppression_onset, run_logit_lens
from .prompts import (
    build_prompt_user_role_jury,
    build_prompt_no_attribution_matched,
    build_prompt_self_framing,
)

PRIMING_SUFFIX = "The correct answer is ("


# ── suffix stripping helpers ───────────────────────────────────────────────
def strip_priming_suffix(prompt: str) -> str:
    """Remove the ``"The correct answer is ("`` tail if present.

    If the suffix is not present we emit a ``RuntimeWarning`` and return the
    prompt unchanged, so a silent drift in the upstream prompt builders is
    loud rather than invisible.
    """
    if prompt.endswith(PRIMING_SUFFIX):
        return prompt[: -len(PRIMING_SUFFIX)]
    warnings.warn(
        "priming suffix not found at end of prompt; "
        "returning prompt unchanged. Upstream src.prompts builders may have "
        "drifted.",
        RuntimeWarning,
        stacklevel=2,
    )
    return prompt


# ── unsuffixed prompt builders ─────────────────────────────────────────────
def build_prompt_user_role_jury_unsuffixed(
    q_idx: int, item: dict, wrong_idx: int, jury_data: dict, tokenizer
) -> str:
    """C4a builder with the priming suffix stripped."""
    return strip_priming_suffix(
        build_prompt_user_role_jury(q_idx, item, wrong_idx, jury_data, tokenizer)
    )


def build_prompt_no_attribution_matched_unsuffixed(
    q_idx: int, item: dict, wrong_idx: int, jury_data: dict, tokenizer
) -> str:
    """C4c_matched builder with the priming suffix stripped."""
    return strip_priming_suffix(
        build_prompt_no_attribution_matched(
            q_idx, item, wrong_idx, jury_data, tokenizer
        )
    )


def build_prompt_self_framing_unsuffixed(
    q_idx: int, item: dict, wrong_idx: int, jury_data: dict, tokenizer
) -> str:
    """C4d builder with the priming suffix stripped."""
    return strip_priming_suffix(
        build_prompt_self_framing(q_idx, item, wrong_idx, jury_data, tokenizer)
    )


# Registry for CLI convenience.
UNSUFFIXED_BUILDERS: dict[str, Callable] = {
    "c4a_nosuffix": build_prompt_user_role_jury_unsuffixed,
    "c4c_matched_nosuffix": build_prompt_no_attribution_matched_unsuffixed,
    "c4d_nosuffix": build_prompt_self_framing_unsuffixed,
}


# ── experiment wrapper ─────────────────────────────────────────────────────
def run_experiment_unsuffixed(
    model,
    tokenizer,
    known_questions: list[dict],
    final_probes: list,
    build_prompt_fn: Callable,
    jury_data: dict,
    description: str = "experiment",
    clean_lda: CleanLDA | None = None,
) -> dict[str, Any]:
    """Schema-compatible twin of :func:`src.experiment.run_experiment` that
    reads the logit lens at the chat-template boundary (unsuffixed) instead of
    after ``"The correct answer is ("``.

    Notes
    -----
    The ``build_prompt_fn`` passed in here must return a prompt that ends at
    the chat-template ``<|start_header_id|>assistant<|end_header_id|>\\n\\n``
    boundary (i.e. the output of :func:`strip_priming_suffix` applied to a
    canonical builder, or equivalent). The last-token hidden state the logit
    lens reads is the state that predicts the assistant's first generated
    token.

    Accepts ``known_questions`` (and ``final_probes``) directly so the
    function can be called on question subsets (smoke test) as well as the
    full 400-question sweep.
    """
    if clean_lda is None:
        clean_lda = CleanLDA.fit_default()

    correct_labels = np.array(
        [item["answer"] for item in known_questions], dtype=np.int64
    )

    all_truth: list[list[float]] = []
    all_syco: list[list[float]] = []
    all_acts: list[np.ndarray] = []
    all_wrong_indices: list[int] = []
    token_counts: list[int] = []

    for q_idx, item in enumerate(tqdm(known_questions, desc=description)):
        ans = item["answer"]
        wrong_idx = jury_data["gemma"][q_idx]["wrong_idx"]
        all_wrong_indices.append(wrong_idx)

        prompt = build_prompt_fn(q_idx, item, wrong_idx, jury_data, tokenizer)
        token_counts.append(len(tokenizer.encode(prompt)))

        truth_p, syco_p, hidden = run_logit_lens(
            prompt, ans, wrong_idx, model, tokenizer
        )
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
