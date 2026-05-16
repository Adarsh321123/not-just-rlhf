"""Clean LDA basis + yield-rate computation.

The notebook version baked this into ``plot_lda_grid``. Here it's factored
into a reusable class so ``run_experiment`` (and downstream scripts) can
compute yield rates without touching matplotlib.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from .config import LDA_LAYER
from .data import load_artifacts


@dataclass
class CleanLDA:
    """A 3-component LDA fit on clean activations at a chosen layer."""

    layer: int
    lda: LinearDiscriminantAnalysis
    centroids: np.ndarray  # (4, 3)

    @classmethod
    def fit(cls, layer: int = LDA_LAYER) -> "CleanLDA":
        art = load_artifacts()
        clean = art["known_acts"][:, layer, :].astype(np.float32)
        labels = art["known_labels"]
        lda = LinearDiscriminantAnalysis(n_components=3)
        lda.fit(clean, labels)
        centroids = lda.transform(lda.means_)
        return cls(layer=layer, lda=lda, centroids=centroids)

    @classmethod
    def fit_default(cls) -> "CleanLDA":
        return cls.fit(LDA_LAYER)

    def project(self, acts: np.ndarray) -> np.ndarray:
        """Project ``(n, d_model)`` activations to the 3-D LDA subspace."""
        return self.lda.transform(acts.astype(np.float32))

    def yield_mask(
        self,
        acts: np.ndarray,
        correct_labels: np.ndarray,
        wrong_indices,
    ) -> np.ndarray:
        """Return a boolean ``(n,)`` array: True where the projection is closer
        to the wrong-answer centroid than to the correct-answer centroid.
        """
        proj = self.project(acts)
        wrong_indices = np.asarray(wrong_indices)
        correct_labels = np.asarray(correct_labels)

        d_cor = np.linalg.norm(proj - self.centroids[correct_labels], axis=1)
        d_wrg = np.linalg.norm(proj - self.centroids[wrong_indices], axis=1)
        return d_wrg < d_cor

    def compute_yield_rate(
        self,
        acts: np.ndarray,
        correct_labels: np.ndarray,
        wrong_indices,
    ) -> float:
        mask = self.yield_mask(acts, correct_labels, wrong_indices)
        return float(mask.mean())
