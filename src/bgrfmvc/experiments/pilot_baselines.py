"""Frozen Stage 5 baseline representations without label access."""

from __future__ import annotations

import time

import numpy as np

from bgrfmvc.core.boundary_relation import compute_boundary_relations
from bgrfmvc.core.fusion import fuse_relation_blocks
from bgrfmvc.core.granular_ball import build_granular_balls
from bgrfmvc.core.preprocessing import MultiViewPreprocessor
from bgrfmvc.core.reliability import relation_probability_and_entropy, sample_specific_view_weights


METHODS = (
    "B0_RawConcat_KM",
    "B1_HardGB_KM",
    "B2_GaussianGB_KM",
    "B3_BoundaryEqual_KM",
    "B4_BoundaryEntropy_KM",
    "B5_BGRFMVC_Full",
)


def _equal_weights(n_samples, n_views):
    return np.full((n_samples, n_views), 1.0 / n_views, dtype=float)


def _mean_entropy(blocks, eps):
    entropies = [relation_probability_and_entropy(block, eps=eps)[1] for block in blocks]
    return entropies, [float(values.mean()) for values in entropies]


def build_pilot_representations(X_views, config):
    """Build B0-B5 while sharing preprocessing and one granularization."""
    started = time.perf_counter()
    preprocessor = MultiViewPreprocessor(
        standardize=config["preprocess"]["standardize"],
        use_pca=config["preprocess"]["use_pca"],
    )
    standardized = preprocessor.fit_transform(X_views)
    preprocessing_seconds = time.perf_counter() - started
    ball_started = time.perf_counter()
    balls_by_view = [
        build_granular_balls(view, config["method"], config["granular_ball"])
        for view in standardized
    ]
    ball_build_seconds = time.perf_counter() - ball_started
    method = config["method"]
    eps = float(method["eps"])
    n_samples = standardized[0].shape[0]
    n_views = len(standardized)
    equal_alpha = _equal_weights(n_samples, n_views)
    results = {}

    def store(name, representation, entropy_means, alpha, representation_seconds):
        if not np.all(np.isfinite(representation)):
            raise ArithmeticError("%s representation contains NaN or Inf" % name)
        results[name] = {
            "representation": representation,
            "mean_entropy_per_view": entropy_means,
            "mean_alpha_per_view": [float(value) for value in alpha.mean(axis=0)],
            "representation_seconds": float(representation_seconds),
            "fused_dimension": int(representation.shape[1]),
        }

    tick = time.perf_counter()
    raw = fuse_relation_blocks(standardized, equal_alpha, eps=eps)
    store(METHODS[0], raw, None, equal_alpha, time.perf_counter() - tick)

    tick = time.perf_counter()
    hard_blocks = []
    for balls in balls_by_view:
        block = np.zeros((n_samples, len(balls)), dtype=float)
        for ball_index, ball in enumerate(balls):
            block[ball.indices, ball_index] = 1.0
        hard_blocks.append(block)
    hard = fuse_relation_blocks(hard_blocks, equal_alpha, eps=eps)
    store(METHODS[1], hard, None, equal_alpha, time.perf_counter() - tick)

    tick = time.perf_counter()
    boundary_no_q = [
        compute_boundary_relations(
            view, balls, tau=method["tau"], eps=eps,
            radius_eps=method["radius_eps"], relation_floor=method["relation_floor"],
            use_ball_reliability=False,
        )
        for view, balls in zip(standardized, balls_by_view)
    ]
    gaussian_blocks = []
    for relation_result, balls in zip(boundary_no_q, balls_by_view):
        radii = np.array([ball.radius for ball in balls], dtype=float)
        denominator = 2.0 * np.square(radii + eps)
        gaussian_blocks.append(np.exp(-np.square(relation_result.distances) / denominator[None, :]))
    gaussian_entropies, gaussian_means = _mean_entropy(gaussian_blocks, eps)
    gaussian = fuse_relation_blocks(gaussian_blocks, equal_alpha, eps=eps)
    store(METHODS[2], gaussian, gaussian_means, equal_alpha, time.perf_counter() - tick)

    raw_boundary_blocks = [item.relations for item in boundary_no_q]
    boundary_entropies, boundary_means = _mean_entropy(raw_boundary_blocks, eps)
    tick = time.perf_counter()
    boundary_equal = fuse_relation_blocks(raw_boundary_blocks, equal_alpha, eps=eps)
    store(METHODS[3], boundary_equal, boundary_means, equal_alpha, time.perf_counter() - tick)

    tick = time.perf_counter()
    entropy_alpha = sample_specific_view_weights(boundary_entropies, eps=eps)
    boundary_entropy = fuse_relation_blocks(raw_boundary_blocks, entropy_alpha, eps=eps)
    store(METHODS[4], boundary_entropy, boundary_means, entropy_alpha, time.perf_counter() - tick)

    tick = time.perf_counter()
    boundary_with_q = [
        compute_boundary_relations(
            view, balls, tau=method["tau"], eps=eps,
            radius_eps=method["radius_eps"], relation_floor=method["relation_floor"],
            use_ball_reliability=True,
        )
        for view, balls in zip(standardized, balls_by_view)
    ]
    corrected_blocks = [item.corrected_relations for item in boundary_with_q]
    corrected_entropies, corrected_means = _mean_entropy(corrected_blocks, eps)
    full_alpha = sample_specific_view_weights(corrected_entropies, eps=eps)
    full = fuse_relation_blocks(corrected_blocks, full_alpha, eps=eps)
    store(METHODS[5], full, corrected_means, full_alpha, time.perf_counter() - tick)

    shared = {
        "n_samples": n_samples,
        "n_views": n_views,
        "ball_count_by_view": [len(balls) for balls in balls_by_view],
        "ball_count": int(sum(len(balls) for balls in balls_by_view)),
        "preprocessing_seconds": float(preprocessing_seconds),
        "ball_build_seconds": float(ball_build_seconds),
        "same_balls_for_B1_to_B5": True,
    }
    return results, shared

