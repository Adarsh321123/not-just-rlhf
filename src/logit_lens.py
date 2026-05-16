"""Logit-lens readout + suppression-onset metrics.

``run_logit_lens`` extracts per-layer P(correct) / P(wrong_target) at the
last-token position of a prompt. ``find_suppression_onset`` is the original
binary detector (kept for backward compatibility); ``compute_onset_metrics``
is the continuous replacement introduced for E6.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import torch
from scipy.ndimage import uniform_filter1d

from .config import CHOICES
from .model import choice_token_ids


def run_logit_lens(text: str, correct_idx: int, wrong_idx: int, model, tokenizer):
    """Run the logit lens on ``text``.

    Returns (truth_probs, syco_probs, hidden_states).
    """
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    ctoks = choice_token_ids(tokenizer)
    vocab_indices = [ctoks[c] for c in CHOICES]

    truth_probs, syco_probs = [], []
    for state in outputs.hidden_states:
        normed = model.model.norm(state[:, -1, :])
        logits = model.lm_head(normed)
        mc_probs = torch.softmax(logits[0, vocab_indices], dim=-1)
        truth_probs.append(mc_probs[correct_idx].item())
        syco_probs.append(mc_probs[wrong_idx].item())

    return truth_probs, syco_probs, outputs.hidden_states


def find_suppression_onset(
    truth_probs,
    syco_probs,
    smooth_window: int = 3,
    min_gap: float = 0.03,
    min_layer: int = 10,
    sustained_layers: int = 3,
):
    """Original binary onset detector. Returns first layer index or None.

    Kept unchanged for backward compatibility with the existing notebook.
    """
    truth_s = uniform_filter1d(np.asarray(truth_probs, dtype=float), size=smooth_window)
    syco_s = uniform_filter1d(np.asarray(syco_probs, dtype=float), size=smooth_window)
    crossing = (syco_s - truth_s) >= min_gap

    run = 0
    for layer in range(min_layer, len(crossing)):
        if crossing[layer]:
            run += 1
            if run >= sustained_layers:
                return layer - sustained_layers + 1
        else:
            run = 0
    return None


# E6: continuous onset metric
ONSET_THRESHOLDS = (0.01, 0.03, 0.05, 0.10)


def compute_onset_metrics(
    truth_probs,
    syco_probs,
    thresholds: tuple[float, ...] = ONSET_THRESHOLDS,
    smooth_window: int = 3,
    min_layer: int = 10,
    sustained_layers: int = 3,
) -> dict[str, Any]:
    """Continuous onset metric: report onset across several ``min_gap`` thresholds
    plus the final-layer ``syco − truth`` gap.

    Returns a dict with keys:
        onset_gap_{thresh}  — first onset layer at that gap threshold, or None
        final_gap           — final-layer (syco − truth) probability
        half_final_onset    — layer where gap first exceeds 0.5 * final_gap (or None)
    """
    truth_arr = np.asarray(truth_probs, dtype=float)
    syco_arr = np.asarray(syco_probs, dtype=float)
    metrics: dict[str, Any] = {}
    for t in thresholds:
        metrics[f"onset_gap_{t}"] = find_suppression_onset(
            truth_arr,
            syco_arr,
            smooth_window=smooth_window,
            min_gap=t,
            min_layer=min_layer,
            sustained_layers=sustained_layers,
        )

    final_gap = float(syco_arr[-1] - truth_arr[-1])
    metrics["final_gap"] = final_gap

    # "Layer where gap first exceeds half its final value" (only meaningful if final_gap > 0).
    truth_s = uniform_filter1d(truth_arr, size=smooth_window)
    syco_s = uniform_filter1d(syco_arr, size=smooth_window)
    gap = syco_s - truth_s
    if final_gap > 0:
        thresh = 0.5 * final_gap
        above = np.where(gap[min_layer:] >= thresh)[0]
        metrics["half_final_onset"] = int(above[0] + min_layer) if above.size else None
    else:
        metrics["half_final_onset"] = None

    return metrics
