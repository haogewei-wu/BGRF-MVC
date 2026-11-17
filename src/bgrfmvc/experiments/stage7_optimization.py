"""Stage 7 strictly equivalent D3 implementation and safe in-memory cache.

This module contains engineering changes only.  It deliberately reuses the
frozen preprocessor, granular-ball constructor, Boundary formula, hybrid
weights, and KMeans settings from Stage 5R-C.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass

import numpy as np

from bgrfmvc.core.boundary_relation import BoundaryRelationResult, compute_boundary_relations
from bgrfmvc.core.fusion import fuse_relation_blocks, l2_normalize_rows
from bgrfmvc.core.granular_ball import build_granular_balls
from bgrfmvc.core.preprocessing import MultiViewPreprocessor
from bgrfmvc.experiments.stage5r_hybrid import _hybrid_fusion


@dataclass
class D3Build:
    representation: np.ndarray
    standardized: list
    balls: list
    boundary_results: list
    timings: dict
    memory: dict


def _array_digest(array):
    value = np.ascontiguousarray(np.asarray(array, dtype=np.float64))
    digest = hashlib.sha256()
    digest.update(str(value.shape).encode("ascii"))
    digest.update(value.dtype.str.encode("ascii"))
    digest.update(value.view(np.uint8))
    return digest.hexdigest()


def cache_key(dataset_identity, X_views, config):
    """Content/config/version cache key; filenames alone are never sufficient."""
    relevant = {
        "method": config["method"], "granular_ball": config["granular_ball"],
        "preprocess": config["preprocess"], "implementation": "D3-OPT-v1",
    }
    payload = {
        "dataset_identity": str(dataset_identity),
        "views": [_array_digest(view) for view in X_views],
        "config": relevant,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


class D3RepresentationCache:
    """Explicit process-local cache; callers can always bypass it with use_cache=False."""

    def __init__(self):
        self._values = {}
        self.build_count = 0
        self.hit_count = 0

    def get_or_build(self, dataset_identity, X_views, config, use_cache=True):
        key = cache_key(dataset_identity, X_views, config)
        if use_cache and key in self._values:
            self.hit_count += 1
            return self._values[key], True, key
        value = build_d3_optimized(X_views, config)
        self.build_count += 1
        if use_cache:
            self._values[key] = value
        return value, False, key


def _sample_ball_distances(X, centers):
    """Euclidean distances without the reference n-by-m-by-d temporary tensor."""
    x2 = np.einsum("ij,ij->i", X, X)[:, None]
    c2 = np.einsum("ij,ij->i", centers, centers)[None, :]
    squared = x2 + c2 - 2.0 * np.dot(X, centers.T)
    np.maximum(squared, 0.0, out=squared)
    np.sqrt(squared, out=squared)
    return squared


def _vectorized_boundary(X, balls, method):
    centers = np.vstack([ball.center for ball in balls])
    radii = np.asarray([ball.radius for ball in balls], dtype=float)
    eps = float(method["eps"]); tau = float(method["tau"])
    radius_eps = float(method["radius_eps"]); floor = float(method["relation_floor"])
    distances = _sample_ball_distances(np.asarray(X, dtype=float), centers)
    deltas = (distances - radii[None, :]) / (radii[None, :] + eps)
    relations = np.empty_like(distances)
    normal = radii > radius_eps
    if np.any(normal):
        z = -deltas[:, normal] / tau
        positive = z >= 0
        block = np.empty_like(z)
        block[positive] = 1.0 / (1.0 + np.exp(-z[positive]))
        ez = np.exp(z[~positive])
        block[~positive] = ez / (1.0 + ez)
        relations[:, normal] = block
    if np.any(~normal):
        relations[:, ~normal] = np.where(distances[:, ~normal] <= radius_eps, 1.0, floor)
    if not np.all(np.isfinite(relations)):
        raise ArithmeticError("optimized Boundary relation is non-finite")
    # D3 has q disabled, so corrected_relations is exactly R.
    return BoundaryRelationResult(distances, deltas, relations, relations)


def _preallocated_hybrid(raw_blocks, relation_blocks, eps):
    n = raw_blocks[0].shape[0]; views = len(raw_blocks)
    total = sum(raw.shape[1] + rel.shape[1] for raw, rel in zip(raw_blocks, relation_blocks))
    fused = np.empty((n, total), dtype=float)
    scale = np.sqrt(2.0 * float(views)); offset = 0
    for raw, relation in zip(raw_blocks, relation_blocks):
        raw_bar = l2_normalize_rows(raw, eps=eps)
        rel_bar = l2_normalize_rows(relation, eps=eps)
        width = raw.shape[1]; fused[:, offset:offset + width] = raw_bar / scale; offset += width
        width = relation.shape[1]; fused[:, offset:offset + width] = rel_bar / scale; offset += width
    return l2_normalize_rows(fused, eps=eps)


def _memory(standardized, boundary_results, representation, optimized=False):
    relation_bytes = sum(item.relations.nbytes for item in boundary_results)
    distance_bytes = sum(item.distances.nbytes for item in boundary_results)
    return {
        "representation_bytes": int(representation.nbytes),
        "relation_matrix_bytes": int(relation_bytes),
        "distance_matrix_bytes": int(distance_bytes),
        "temporary_array_estimated_peak_bytes": int(max(
            [representation.nbytes] + ([distance_bytes * 4] if optimized else
             [x.shape[0] * r.relations.shape[1] * x.shape[1] * 8
              for x, r in zip(standardized, boundary_results)])
        )),
    }


def build_d3_reference(X_views, config):
    total = time.perf_counter(); tick = time.perf_counter()
    standardized = MultiViewPreprocessor(
        standardize=config["preprocess"]["standardize"], use_pca=config["preprocess"]["use_pca"]
    ).fit_transform(X_views)
    preprocess = time.perf_counter() - tick; tick = time.perf_counter()
    balls = [build_granular_balls(x, config["method"], config["granular_ball"]) for x in standardized]
    granulation = time.perf_counter() - tick; tick = time.perf_counter()
    m = config["method"]
    results = [compute_boundary_relations(x, b, tau=m["tau"], eps=m["eps"],
               radius_eps=m["radius_eps"], relation_floor=m["relation_floor"],
               use_ball_reliability=False) for x, b in zip(standardized, balls)]
    relation = time.perf_counter() - tick; tick = time.perf_counter()
    representation = _hybrid_fusion(standardized, [r.relations for r in results], float(m["eps"]))
    fusion = time.perf_counter() - tick
    timings = {"standardization_time": preprocess, "granularization_time": granulation,
               "distance_computation_time": relation, "boundary_relation_time": relation,
               "relation_normalization_time": fusion, "hybrid_fusion_time": fusion,
               "representation_build_time": time.perf_counter() - total}
    return D3Build(representation, standardized, balls, results, timings,
                   _memory(standardized, results, representation, optimized=False))


def build_d3_optimized(X_views, config):
    total = time.perf_counter(); tick = time.perf_counter()
    standardized = MultiViewPreprocessor(
        standardize=config["preprocess"]["standardize"], use_pca=config["preprocess"]["use_pca"]
    ).fit_transform(X_views)
    preprocess = time.perf_counter() - tick; tick = time.perf_counter()
    balls = [build_granular_balls(x, config["method"], config["granular_ball"]) for x in standardized]
    granulation = time.perf_counter() - tick; tick = time.perf_counter()
    results = [_vectorized_boundary(x, b, config["method"]) for x, b in zip(standardized, balls)]
    relation = time.perf_counter() - tick; tick = time.perf_counter()
    representation = _preallocated_hybrid(
        standardized, [r.relations for r in results], float(config["method"]["eps"]))
    fusion = time.perf_counter() - tick
    timings = {"standardization_time": preprocess, "granularization_time": granulation,
               "distance_computation_time": relation, "boundary_relation_time": relation,
               "relation_normalization_time": fusion, "hybrid_fusion_time": fusion,
               "representation_build_time": time.perf_counter() - total}
    return D3Build(representation, standardized, balls, results, timings,
                   _memory(standardized, results, representation, optimized=True))


def build_d0(X_views, config):
    tick = time.perf_counter()
    standardized = MultiViewPreprocessor(
        standardize=config["preprocess"]["standardize"], use_pca=config["preprocess"]["use_pca"]
    ).fit_transform(X_views)
    standardization = time.perf_counter() - tick; tick = time.perf_counter()
    n = standardized[0].shape[0]; v = len(standardized)
    representation = fuse_relation_blocks(standardized, np.full((n, v), 1.0 / v), config["method"]["eps"])
    fusion = time.perf_counter() - tick
    return representation, {"standardization_time": standardization,
                            "hybrid_fusion_time": fusion,
                            "representation_build_time": standardization + fusion}


def equivalence(reference, optimized):
    ball_ok = True; ball_max = 0.0
    for ref_view, opt_view in zip(reference.balls, optimized.balls):
        ball_ok = ball_ok and len(ref_view) == len(opt_view)
        for a, b in zip(ref_view, opt_view):
            ball_ok = ball_ok and np.array_equal(a.indices, b.indices)
            values = [np.max(np.abs(a.center - b.center)), abs(a.radius - b.radius),
                      abs(a.radial_std - b.radial_std), abs(a.quality_q - b.quality_q)]
            ball_max = max(ball_max, *values)
    relation_diffs = [np.abs(a.relations - b.relations)
                      for a, b in zip(reference.boundary_results, optimized.boundary_results)]
    relation_max = max(float(x.max()) for x in relation_diffs)
    relation_mean = float(np.mean(np.concatenate([x.ravel() for x in relation_diffs])))
    rep_diff = np.abs(reference.representation - optimized.representation)
    return {"ball_structure_equal": bool(ball_ok), "ball_max_abs_diff": float(ball_max),
            "relation_allclose": bool(all(np.allclose(a.relations, b.relations, atol=1e-12, rtol=1e-10)
                                           for a, b in zip(reference.boundary_results, optimized.boundary_results))),
            "relation_max_abs_diff": relation_max, "relation_mean_abs_diff": relation_mean,
            "representation_allclose": bool(np.allclose(reference.representation, optimized.representation,
                                                          atol=1e-12, rtol=1e-10)),
            "representation_max_abs_diff": float(rep_diff.max()),
            "representation_mean_abs_diff": float(rep_diff.mean()),
            "finite": bool(np.all(np.isfinite(optimized.representation)))}
