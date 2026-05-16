"""Bootstrap confidence intervals over the 400 questions (E8).

Resamples question indices with replacement and recomputes the yield rate on
each resample, giving a CPU-only estimate of how much per-condition yield
could swing purely from question-sampling variance.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .lda import CleanLDA


@dataclass
class BootstrapCI:
    mean: float
    lo: float
    hi: float
    se: float
    n_iter: int


def bootstrap_yield_ci(
    activations_at_layer: np.ndarray,
    correct_labels: np.ndarray,
    wrong_indices,
    clean_lda: CleanLDA,
    n_iter: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> BootstrapCI:
    """Bootstrap a yield-rate confidence interval.

    Parameters
    ----------
    activations_at_layer : (n, d_model) float array — pressured activations at the LDA layer.
    correct_labels, wrong_indices : per-question ground-truth and wrong-target labels.
    clean_lda : fitted CleanLDA.
    """
    rng = np.random.default_rng(seed)

    yielded_mask = clean_lda.yield_mask(
        activations_at_layer, correct_labels, wrong_indices
    ).astype(np.float64)

    n = len(yielded_mask)
    samples = np.empty(n_iter, dtype=np.float64)
    for i in range(n_iter):
        idx = rng.integers(0, n, size=n)
        samples[i] = yielded_mask[idx].mean()

    alpha = (1.0 - ci) / 2.0
    lo = float(np.quantile(samples, alpha))
    hi = float(np.quantile(samples, 1.0 - alpha))
    return BootstrapCI(
        mean=float(yielded_mask.mean()),
        lo=lo,
        hi=hi,
        se=float(samples.std(ddof=1)),
        n_iter=n_iter,
    )
