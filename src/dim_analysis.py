"""Difference-in-means sycophantic-direction analysis (E14).

Applies the OpenReview `d24zTCznJu` "Sycophancy Is Not One Thing" methodology
to the cleanup pass's saved per-condition activations:

1. For each condition, compute the **empirical sycophantic direction** at a
   chosen layer as

       direction[cond] = mean(activations[yield_mask]) − mean(activations[~yield_mask])

   where ``yield_mask`` is the boolean output of ``CleanLDA.yield_mask`` —
   the LDA-based classification of "did this question's L25 activation
   land closer to the wrong-answer centroid than to the correct one."
2. Compute the cosine-similarity matrix across conditions.
3. Compute a "shared sycophancy axis" (mean direction across conditions)
   and each condition's alignment with it.

Zero GPU. Pure numpy on the saved ``activations`` arrays in ``results/*.pkl``.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import LDA_LAYER
from .lda import CleanLDA


@dataclass
class ConditionDirection:
    name: str
    direction: np.ndarray  # (d_model,)
    n_yielded: int
    n_unyielded: int
    norm: float


def compute_sycophantic_direction(
    activations: np.ndarray,
    yielded_mask: np.ndarray,
    layer: int = LDA_LAYER,
) -> np.ndarray:
    """Return ``mean(act[yielded]) - mean(act[~yielded])`` at ``layer``.

    Parameters
    ----------
    activations : (n, 33, d_model) float array
    yielded_mask : (n,) bool array — True for questions where the L25
        activation landed closer to the wrong-answer centroid.
    layer : which transformer layer to extract (default LDA_LAYER=25)
    """
    acts_layer = np.asarray(activations)[:, layer, :].astype(np.float64)
    yielded_mask = np.asarray(yielded_mask, dtype=bool)

    n_y = int(yielded_mask.sum())
    n_u = int((~yielded_mask).sum())
    if n_y == 0 or n_u == 0:
        raise ValueError(
            f"cannot compute direction: {n_y} yielded, {n_u} unyielded "
            "(need both groups to be non-empty)"
        )

    mean_yielded = acts_layer[yielded_mask].mean(axis=0)
    mean_unyielded = acts_layer[~yielded_mask].mean(axis=0)
    return mean_yielded - mean_unyielded


def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """Standard cosine similarity between two 1-D vectors."""
    v1 = np.asarray(v1, dtype=np.float64)
    v2 = np.asarray(v2, dtype=np.float64)
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 == 0.0 or n2 == 0.0:
        return 0.0
    return float(np.dot(v1, v2) / (n1 * n2))


def build_cosine_matrix(directions: dict[str, np.ndarray]) -> tuple[list[str], np.ndarray]:
    """Return ``(names, NxN cosine matrix)`` for a dict of condition directions."""
    names = list(directions.keys())
    n = len(names)
    mat = np.zeros((n, n), dtype=np.float64)
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            mat[i, j] = cosine_similarity(directions[a], directions[b])
    return names, mat


def compute_shared_axis(directions: dict[str, np.ndarray]) -> np.ndarray:
    """Return the mean (element-wise) direction across conditions."""
    vecs = np.stack([directions[k] for k in directions], axis=0)
    return vecs.mean(axis=0)


def compute_alignments(
    directions: dict[str, np.ndarray],
    shared: np.ndarray | None = None,
) -> dict[str, float]:
    """Return each condition's cosine alignment with the shared axis."""
    if shared is None:
        shared = compute_shared_axis(directions)
    return {k: cosine_similarity(v, shared) for k, v in directions.items()}


def build_all_directions(
    per_condition_pkls: dict[str, dict],
    clean_lda: CleanLDA,
    correct_labels: np.ndarray,
    layer: int = LDA_LAYER,
) -> dict[str, ConditionDirection]:
    """For each condition, derive the yield mask from LDA and compute the direction.

    ``per_condition_pkls`` maps condition name -> loaded pickle dict (the
    schema saved by ``scripts/run_all_conditions.py``).
    """
    out: dict[str, ConditionDirection] = {}
    for name, res in per_condition_pkls.items():
        acts = np.asarray(res["activations"])
        wrong_indices = np.asarray(res["wrong_indices"])
        acts_l25 = acts[:, LDA_LAYER, :].astype(np.float32)
        yielded = clean_lda.yield_mask(acts_l25, correct_labels, wrong_indices)
        d = compute_sycophantic_direction(acts, yielded, layer=layer)
        out[name] = ConditionDirection(
            name=name,
            direction=d,
            n_yielded=int(yielded.sum()),
            n_unyielded=int((~yielded).sum()),
            norm=float(np.linalg.norm(d)),
        )
    return out
