"""Stage 5R-C equal-view raw/granular hybrid representations."""

from __future__ import annotations

import time

import numpy as np

from bgrfmvc.core.boundary_relation import compute_boundary_relations
from bgrfmvc.core.fusion import l2_normalize_rows, fuse_relation_blocks
from bgrfmvc.core.granular_ball import build_granular_balls
from bgrfmvc.core.preprocessing import MultiViewPreprocessor


METHODS = (
    "D0_RawConcat_KM",
    "D1_BoundaryOnly_Equal_KM",
    "D2_RawGaussianHybrid_KM",
    "D3_RawBoundaryHybrid_KM",
)
VARIANCE_TOLERANCE = 1.0e-15


def _equal_alpha(n, v):
    return np.full((n, v), 1.0 / v, dtype=float)


def _hybrid_fusion(raw_blocks, relation_blocks, eps):
    v = len(raw_blocks)
    pieces = []
    for raw, relation in zip(raw_blocks, relation_blocks):
        raw_bar = l2_normalize_rows(raw, eps=eps)
        relation_bar = l2_normalize_rows(relation, eps=eps)
        z = np.concatenate((raw_bar / np.sqrt(2.0), relation_bar / np.sqrt(2.0)), axis=1)
        pieces.append(z / np.sqrt(float(v)))
    return l2_normalize_rows(np.concatenate(pieces, axis=1), eps=eps)


def _diagnostics(matrix):
    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim != 2 or not np.all(np.isfinite(matrix)):
        raise ArithmeticError("representation must be finite and 2D")
    variance = np.var(matrix, axis=0, ddof=0)
    nonzero = int(np.count_nonzero(variance > VARIANCE_TOLERANCE))
    unique = int(np.unique(matrix, axis=0).shape[0])
    return {
        "unique_row_count": unique,
        "nonzero_variance_dimensions": nonzero,
        "mean_feature_variance": float(variance.mean()),
        "variance_tolerance": VARIANCE_TOLERANCE,
        "relation_degenerate": bool(nonzero == 0 or unique <= 1),
        "finite": True,
    }


def build_hybrid_representations(X_views, config):
    tick = time.perf_counter()
    standardized = MultiViewPreprocessor(
        standardize=bool(config["preprocess"]["standardize"]),
        use_pca=bool(config["preprocess"]["use_pca"]),
    ).fit_transform(X_views)
    preprocess_seconds = time.perf_counter() - tick
    tick = time.perf_counter()
    balls_by_view = [build_granular_balls(x, config["method"], config["granular_ball"])
                     for x in standardized]
    granulation_seconds = time.perf_counter() - tick
    method = config["method"]
    eps = float(method["eps"])
    boundary_results = [
        compute_boundary_relations(
            x, balls, tau=method["tau"], eps=eps,
            radius_eps=method["radius_eps"], relation_floor=method["relation_floor"],
            use_ball_reliability=False,
        ) for x, balls in zip(standardized, balls_by_view)
    ]
    boundary = [item.relations for item in boundary_results]
    gaussian = []
    for result, balls in zip(boundary_results, balls_by_view):
        radii = np.asarray([ball.radius for ball in balls], dtype=float)
        gaussian.append(np.exp(-np.square(result.distances) /
                               (2.0 * np.square(radii[None, :] + eps))))
    n = standardized[0].shape[0]
    v = len(standardized)
    equal = _equal_alpha(n, v)
    builders = (
        (METHODS[0], lambda: fuse_relation_blocks(standardized, equal, eps=eps)),
        (METHODS[1], lambda: fuse_relation_blocks(boundary, equal, eps=eps)),
        (METHODS[2], lambda: _hybrid_fusion(standardized, gaussian, eps)),
        (METHODS[3], lambda: _hybrid_fusion(standardized, boundary, eps)),
    )
    raw_dim = int(sum(x.shape[1] for x in standardized))
    relation_dim = int(sum(len(x) for x in balls_by_view))
    results = {}
    for name, builder in builders:
        started = time.perf_counter()
        representation = builder()
        diagnostic = _diagnostics(representation)
        diagnostic.update({
            "raw_dimension": raw_dim,
            "relation_dimension": relation_dim,
            "hybrid_dimension": int(representation.shape[1]),
            "relation_to_raw_ratio": relation_dim / float(raw_dim),
            "representation_seconds": float(time.perf_counter() - started),
            "labels_loaded": False,
        })
        results[name] = {"representation": representation, "diagnostics": diagnostic}
    shared = {
        "n_samples": n, "n_views": v,
        "raw_dimension": raw_dim, "relation_dimension": relation_dim,
        "ball_count_per_view": [len(x) for x in balls_by_view],
        "preprocess_seconds": float(preprocess_seconds),
        "granulation_seconds": float(granulation_seconds),
        "same_balls_for_D1_to_D3": True,
        "ball_reliability_enabled": False,
        "all_views_equal_weight": True,
        "labels_loaded": False,
    }
    return results, shared
