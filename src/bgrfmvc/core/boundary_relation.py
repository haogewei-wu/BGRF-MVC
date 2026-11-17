"""Boundary-aware sample-to-granular-ball relations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .numerics import boundary_relation


@dataclass(frozen=True)
class BoundaryRelationResult:
    distances: np.ndarray
    deltas: np.ndarray
    relations: np.ndarray
    corrected_relations: np.ndarray


def compute_boundary_relations(
    X,
    balls,
    tau=0.5,
    eps=1e-12,
    radius_eps=1e-12,
    relation_floor=1e-12,
    use_ball_reliability=True,
):
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or not balls:
        raise ValueError("X must be 2D and balls must be non-empty")
    centers = np.vstack([ball.center for ball in balls])
    if X.shape[1] != centers.shape[1]:
        raise ValueError("feature dimension differs from ball centers")
    distances = np.linalg.norm(X[:, None, :] - centers[None, :, :], axis=2)
    deltas = np.empty_like(distances)
    relations = np.empty_like(distances)
    for index, ball in enumerate(balls):
        if ball.radius <= radius_eps:
            # Delta is diagnostic only for a degenerate ball; keep a finite value
            # while the relation itself follows the explicit zero-radius rule.
            deltas[:, index] = (distances[:, index] - ball.radius) / (ball.radius + eps)
        else:
            deltas[:, index] = (distances[:, index] - ball.radius) / (ball.radius + eps)
        relations[:, index] = boundary_relation(
            distances[:, index], ball.radius, tau=tau, eps=eps,
            radius_eps=radius_eps, relation_floor=relation_floor,
        )
    quality = np.array([ball.quality_q for ball in balls]) if use_ball_reliability else np.ones(len(balls))
    corrected = relations * quality[None, :]
    if (not np.all(np.isfinite(distances)) or not np.all(np.isfinite(deltas))
            or not np.all(np.isfinite(relations)) or not np.all(np.isfinite(corrected))):
        raise ArithmeticError("boundary relations must be finite")
    return BoundaryRelationResult(distances, deltas, relations, corrected)
