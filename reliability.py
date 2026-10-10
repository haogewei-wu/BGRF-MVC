"""Strict relation probabilities, entropy, and sample-specific view weights."""

from __future__ import annotations

import numpy as np

from .numerics import normalized_entropy, strict_relation_probabilities, view_weights


def relation_probability_and_entropy(corrected_relations, eps=1e-12):
    probabilities = strict_relation_probabilities(corrected_relations, eps=eps)
    entropy = normalized_entropy(probabilities)
    return probabilities, entropy


def sample_specific_view_weights(entropies_by_view, eps=1e-12):
    """Stack view entropy vectors and normalize scores over views."""
    if not entropies_by_view:
        raise ValueError("at least one view entropy vector is required")
    entropy_matrix = np.column_stack([np.asarray(item, dtype=float) for item in entropies_by_view])
    return view_weights(entropy_matrix, eps=eps)

