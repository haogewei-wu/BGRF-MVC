"""Deterministic, label-free recursive granular-ball construction."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.cluster import KMeans

from .numerics import can_attempt_split


@dataclass(frozen=True)
class GranularBall:
    indices: np.ndarray
    center: np.ndarray
    radius: float
    size: int
    depth: int
    radial_std: float
    quality_q: float


def make_granular_ball(X, indices, depth, eps=1e-12, radius_eps=1e-12):
    indices = np.asarray(indices, dtype=int)
    members = X[indices]
    center = members.mean(axis=0)
    distances = np.linalg.norm(members - center, axis=1)
    radius = float(distances.mean())
    radial_std = float(np.std(distances, ddof=0))
    if radius <= radius_eps and radial_std <= radius_eps:
        quality_q = 1.0
    else:
        quality_q = float(np.exp(-(radial_std / (radius + eps))))
    return GranularBall(
        indices=indices,
        center=np.asarray(center, dtype=float),
        radius=radius,
        size=int(indices.size),
        depth=int(depth),
        radial_std=radial_std,
        quality_q=quality_q,
    )


def build_granular_balls(X, method_config, granular_config):
    """Return terminal balls whose indices partition X exactly once."""
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or X.shape[0] < 2 or X.shape[1] < 1:
        raise ValueError("X must be a 2D matrix with at least two samples")
    if not np.all(np.isfinite(X)):
        raise ValueError("X must be finite")

    eps = float(method_config["eps"])
    radius_eps = float(method_config.get("radius_eps", eps))
    eta = float(method_config["eta"])
    min_parent = int(method_config["min_parent_size"])
    min_child = int(method_config["min_child_size"])
    max_depth = int(method_config["max_depth"])

    root = make_granular_ball(X, np.arange(X.shape[0]), 0, eps, radius_eps)
    pending = [root]
    terminal = []
    while pending:
        parent = pending.pop()
        if parent.depth >= max_depth or not can_attempt_split(parent.size, min_parent, min_child):
            terminal.append(parent)
            continue
        splitter = KMeans(
            n_clusters=2,
            n_init=int(granular_config["split_n_init"]),
            max_iter=int(granular_config["split_max_iter"]),
            random_state=int(granular_config["split_random_state"]),
        )
        assignments = splitter.fit_predict(X[parent.indices])
        child_indices = [parent.indices[assignments == label] for label in (0, 1)]
        if any(indices.size < min_child for indices in child_indices):
            terminal.append(parent)
            continue
        children = [
            make_granular_ball(X, indices, parent.depth + 1, eps, radius_eps)
            for indices in child_indices
        ]
        weighted_radius = sum(child.size * child.radius for child in children) / parent.size
        gain = (parent.radius - weighted_radius) / (parent.radius + eps)
        if gain > eta:
            # Reverse push keeps deterministic KMeans label order in the output traversal.
            pending.extend(reversed(children))
        else:
            terminal.append(parent)
    terminal.sort(key=lambda ball: int(ball.indices.min()))
    return terminal

