"""Frozen Stage 5R-B representations built without label access."""

from __future__ import annotations

import time

import numpy as np

from bgrfmvc.core.boundary_relation import compute_boundary_relations
from bgrfmvc.core.fusion import fuse_relation_blocks
from bgrfmvc.core.granular_ball import build_granular_balls
from bgrfmvc.core.preprocessing import MultiViewPreprocessor
from bgrfmvc.core.reliability import relation_probability_and_entropy


METHODS = (
    "C0_RawConcat_KM",
    "C1_BoundaryEqual_RecoveredGB",
    "C2_BoundaryEntropy_Legacy",
    "C3_BoundaryEntropy_Corrected",
    "C4_BGRFMVC_v2.0.2_Full",
)

# Frozen before evaluation. Variances at or below this numerical floor are
# treated as zero for the protocol's degeneration diagnostic.
VARIANCE_TOLERANCE = 1.0e-15


def equal_weights(n_samples, n_views):
    return np.full((n_samples, n_views), 1.0 / n_views, dtype=float)


def legacy_entropy_weights(entropies, eps=1e-12):
    matrix = np.column_stack(entropies)
    scores = 1.0 - matrix + eps
    return scores / scores.sum(axis=1, keepdims=True)


def corrected_entropy_weights(entropies, ball_counts, eps=1e-12):
    """Correct normalized-entropy certainty for finite granular support.

    The purely structural factor (m-1)/m is zero for a single-ball view and
    approaches one as granular support grows. It is fixed before evaluation.
    """
    matrix = np.column_stack(entropies)
    counts = np.asarray(ball_counts, dtype=float)
    if matrix.shape[1] != counts.size or np.any(counts < 1):
        raise ValueError("one positive ball count is required per entropy view")
    support = (counts - 1.0) / counts
    scores = (1.0 - matrix) * support[None, :] + eps
    weights = scores / scores.sum(axis=1, keepdims=True)
    if np.any(weights < 0) or not np.allclose(weights.sum(axis=1), 1.0, atol=1e-10):
        raise ArithmeticError("corrected view weights violate the simplex")
    return weights


def representation_diagnostics(matrix):
    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim != 2 or not np.all(np.isfinite(matrix)):
        raise ArithmeticError("fused representation must be a finite 2D matrix")
    variances = np.var(matrix, axis=0, ddof=0)
    nonzero = int(np.count_nonzero(variances > VARIANCE_TOLERANCE))
    unique_rows = int(np.unique(matrix, axis=0).shape[0])
    return {
        "feature_variance_per_dimension": variances.tolist(),
        "mean_feature_variance": float(variances.mean()),
        "number_of_nonzero_variance_dimensions": nonzero,
        "unique_row_count": unique_rows,
        "relation_degenerate": bool(nonzero == 0 or unique_rows <= 1),
        "variance_tolerance": VARIANCE_TOLERANCE,
    }


def _entropy(blocks, eps):
    values = [relation_probability_and_entropy(block, eps=eps)[1] for block in blocks]
    return values


def build_stage5r_representations(X_views, config):
    """Build C0-C4 with one shared deterministic granularization."""
    started = time.perf_counter()
    preprocessor = MultiViewPreprocessor(
        standardize=bool(config["preprocess"]["standardize"]),
        use_pca=bool(config["preprocess"]["use_pca"]),
    )
    standardized = preprocessor.fit_transform(X_views)
    preprocessing_seconds = time.perf_counter() - started

    ball_started = time.perf_counter()
    balls_by_view = [
        build_granular_balls(view, config["method"], config["granular_ball"])
        for view in standardized
    ]
    ball_seconds = time.perf_counter() - ball_started
    counts = [len(items) for items in balls_by_view]
    n_samples = standardized[0].shape[0]
    n_views = len(standardized)
    eps = float(config["method"]["eps"])
    equal = equal_weights(n_samples, n_views)

    boundary_no_q_results = [
        compute_boundary_relations(
            view, balls, tau=config["method"]["tau"], eps=eps,
            radius_eps=config["method"]["radius_eps"],
            relation_floor=config["method"]["relation_floor"],
            use_ball_reliability=False,
        )
        for view, balls in zip(standardized, balls_by_view)
    ]
    boundary_blocks = [item.relations for item in boundary_no_q_results]
    entropy_no_q = _entropy(boundary_blocks, eps)
    legacy_alpha = legacy_entropy_weights(entropy_no_q, eps)
    corrected_alpha = corrected_entropy_weights(entropy_no_q, counts, eps)

    boundary_q_results = [
        compute_boundary_relations(
            view, balls, tau=config["method"]["tau"], eps=eps,
            radius_eps=config["method"]["radius_eps"],
            relation_floor=config["method"]["relation_floor"],
            use_ball_reliability=True,
        )
        for view, balls in zip(standardized, balls_by_view)
    ]
    q_blocks = [item.corrected_relations for item in boundary_q_results]
    entropy_q = _entropy(q_blocks, eps)
    full_alpha = corrected_entropy_weights(entropy_q, counts, eps)

    specifications = (
        (METHODS[0], standardized, equal, None),
        (METHODS[1], boundary_blocks, equal, entropy_no_q),
        (METHODS[2], boundary_blocks, legacy_alpha, entropy_no_q),
        (METHODS[3], boundary_blocks, corrected_alpha, entropy_no_q),
        (METHODS[4], q_blocks, full_alpha, entropy_q),
    )
    results = {}
    for name, blocks, alpha, entropies in specifications:
        tick = time.perf_counter()
        fused = fuse_relation_blocks(blocks, alpha, eps=eps)
        diag = representation_diagnostics(fused)
        diag.update({
            "representation_seconds": float(time.perf_counter() - tick),
            "fused_dimension": int(fused.shape[1]),
            "mean_entropy_per_view": None if entropies is None else [float(x.mean()) for x in entropies],
            "std_entropy_per_view": None if entropies is None else [float(x.std(ddof=0)) for x in entropies],
            "mean_alpha_per_view": [float(x) for x in alpha.mean(axis=0)],
            "std_alpha_per_view": [float(x) for x in alpha.std(axis=0, ddof=0)],
            "alpha_min": float(alpha.min()),
            "alpha_max_row_sum_error": float(np.max(np.abs(alpha.sum(axis=1) - 1.0))),
            "entropy_min": None if entropies is None else float(min(x.min() for x in entropies)),
            "entropy_max": None if entropies is None else float(max(x.max() for x in entropies)),
        })
        results[name] = {"representation": fused, "diagnostics": diag}

    ball_statistics = []
    for view_id, balls in enumerate(balls_by_view):
        sizes = np.asarray([ball.size for ball in balls], dtype=float)
        ball_statistics.append({
            "view_id": view_id,
            "ball_count": len(balls),
            "mean_ball_size": float(sizes.mean()),
            "median_ball_size": float(np.median(sizes)),
            "min_ball_size": int(sizes.min()),
            "max_ball_size": int(sizes.max()),
        })
    shared = {
        "n_samples": n_samples,
        "n_views": n_views,
        "raw_dimensions_by_view": [int(x.shape[1]) for x in standardized],
        "raw_total_dimension": int(sum(x.shape[1] for x in standardized)),
        "ball_count_by_view": counts,
        "relation_dimension": int(sum(counts)),
        "preprocessing_seconds": float(preprocessing_seconds),
        "granularization_seconds": float(ball_seconds),
        "ball_statistics": ball_statistics,
        "same_balls_for_C1_to_C4": True,
        "labels_loaded": False,
    }
    return results, shared
