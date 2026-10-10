"""Small, frozen v2.0.1 numerical contracts (not the full algorithm)."""

from __future__ import annotations

import numpy as np


SPLIT_KMEANS_CONFIG = {
    "n_clusters": 2,
    "n_init": 10,
    "max_iter": 100,
    "random_state": 0,
}


def can_attempt_split(size: int, min_parent_size: int, min_child_size: int) -> bool:
    """Return whether a parent is large enough to produce two valid children."""
    return size >= max(min_parent_size, 2 * min_child_size)


def strict_relation_probabilities(relations, eps: float = 1e-12):
    """Normalize non-negative relation rows to strictly positive probabilities."""
    values = np.asarray(relations, dtype=float)
    if values.ndim != 2 or values.shape[1] < 1:
        raise ValueError("relations must be a 2D matrix with at least one column")
    if not np.all(np.isfinite(values)) or np.any(values < 0):
        raise ValueError("relations must be finite and non-negative")
    numerator = values + eps
    return numerator / numerator.sum(axis=1, keepdims=True)


def normalized_entropy(probabilities):
    """Return row-wise normalized entropy, with H=0 for a single ball."""
    probabilities = np.asarray(probabilities, dtype=float)
    if probabilities.ndim != 2 or probabilities.shape[1] < 1:
        raise ValueError("probabilities must be a non-empty 2D matrix")
    if probabilities.shape[1] == 1:
        return np.zeros(probabilities.shape[0], dtype=float)
    row_error = np.abs(probabilities.sum(axis=1) - 1.0)
    if np.any(row_error >= 1e-10) or np.any(probabilities <= 0):
        raise ValueError("probability rows must be positive and sum to one")
    entropy = -(probabilities * np.log(probabilities)).sum(axis=1)
    entropy /= np.log(probabilities.shape[1])
    if np.any(entropy < -1e-10) or np.any(entropy > 1.0 + 1e-10):
        raise ArithmeticError("entropy materially outside [0, 1]")
    return np.clip(entropy, 0.0, 1.0)


def view_weights(entropies, eps: float = 1e-12):
    """Normalize sample-level certainty scores across views."""
    entropies = np.asarray(entropies, dtype=float)
    if entropies.ndim != 2 or entropies.shape[1] < 1:
        raise ValueError("entropies must be a sample-by-view matrix")
    if np.any(entropies < 0) or np.any(entropies > 1):
        raise ValueError("entropies must lie in [0, 1]")
    scores = 1.0 - entropies + eps
    return scores / scores.sum(axis=1, keepdims=True)


def boundary_relation(
    distances,
    radius: float,
    tau: float = 0.5,
    eps: float = 1e-12,
    radius_eps: float = 1e-12,
    relation_floor: float = 1e-12,
):
    """Compute stable boundary relations, including the zero-radius rule."""
    distances = np.asarray(distances, dtype=float)
    if radius <= radius_eps:
        return np.where(distances <= radius_eps, 1.0, relation_floor)
    delta = (distances - radius) / (radius + eps)
    z = -delta / tau
    result = np.empty_like(z)
    positive = z >= 0
    result[positive] = 1.0 / (1.0 + np.exp(-z[positive]))
    exp_z = np.exp(z[~positive])
    result[~positive] = exp_z / (1.0 + exp_z)
    return result


def population_radial_std(distances) -> float:
    """Population radial standard deviation (frozen ddof=0)."""
    return float(np.std(np.asarray(distances, dtype=float), ddof=0))

