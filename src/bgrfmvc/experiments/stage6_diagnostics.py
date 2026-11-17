"""Label-free numerical diagnostics for Stage 6."""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.neighbors import NearestNeighbors


VAR_TOL = 1e-15


def effective_rank(matrix):
    singular = np.linalg.svd(np.asarray(matrix, dtype=float), compute_uv=False)
    threshold = singular.max() * max(matrix.shape) * np.finfo(float).eps if singular.size else 0.0
    rank = int(np.count_nonzero(singular > threshold))
    total = singular.sum()
    if total <= 0:
        return rank, 0.0
    p = singular[singular > 0] / total
    return rank, float(np.exp(-np.sum(p * np.log(p))))


def representation_stats(matrix, d0_dim):
    x = np.asarray(matrix, dtype=float)
    variance = np.var(x, axis=0, ddof=0)
    unique = int(np.unique(x, axis=0).shape[0])
    rank, erank = effective_rank(x)
    nnz = int(np.count_nonzero(x))
    return {
        "n_samples": int(x.shape[0]), "representation_dim": int(x.shape[1]),
        "dim_ratio_vs_D0": float(x.shape[1] / d0_dim),
        "unique_row_count": unique, "unique_row_ratio": float(unique / x.shape[0]),
        "duplicate_row_count": int(x.shape[0] - unique),
        "collapsed_row_ratio": float((x.shape[0] - unique) / x.shape[0]),
        "zero_variance_dim_count": int(np.count_nonzero(variance <= VAR_TOL)),
        "nonzero_variance_dim_count": int(np.count_nonzero(variance > VAR_TOL)),
        "matrix_rank": rank, "effective_rank": erank,
        "nnz": nnz, "density": float(nnz / x.size), "zero_fraction": float(1.0 - nnz / x.size),
        "representation_memory_MB": float(x.nbytes / 1024.0 ** 2),
        "finite": bool(np.all(np.isfinite(x))),
    }


def distance_preservation(reference, candidate):
    first = pdist(reference, metric="euclidean")
    second = pdist(candidate, metric="euclidean")
    return float(np.corrcoef(first, second)[0, 1]), float(spearmanr(first, second)[0])


def knn_preservation(reference, candidate, k):
    ref = NearestNeighbors(n_neighbors=k + 1, metric="euclidean").fit(reference).kneighbors(return_distance=False)[:, 1:]
    cand = NearestNeighbors(n_neighbors=k + 1, metric="euclidean").fit(candidate).kneighbors(return_distance=False)[:, 1:]
    overlaps, jaccards = [], []
    for a, b in zip(ref, cand):
        intersection = len(set(a).intersection(b))
        overlaps.append(intersection / float(k))
        jaccards.append(intersection / float(2 * k - intersection))
    return float(np.mean(overlaps)), float(np.mean(jaccards)), ref, cand


def aligned_agreement(first, second, subset=None):
    a = np.asarray(first); b = np.asarray(second)
    av, ai = np.unique(a, return_inverse=True); bv, bi = np.unique(b, return_inverse=True)
    table = np.zeros((av.size, bv.size), dtype=np.int64); np.add.at(table, (ai, bi), 1)
    rows, cols = linear_sum_assignment(table.max() - table)
    mapping = {bv[c]: av[r] for r, c in zip(rows, cols)}
    indices = np.arange(a.size) if subset is None else np.asarray(subset)
    aligned = np.asarray([mapping.get(value, None) for value in b[indices]], dtype=object)
    return float(np.mean(aligned == a[indices]))


def prediction_comparison(first, second):
    return {
        "prediction_agreement": aligned_agreement(first, second),
        "adjusted_rand_between_predictions": float(adjusted_rand_score(first, second)),
        "NMI_between_predictions": float(normalized_mutual_info_score(first, second)),
    }


def boundary_proximity(deltas_by_view):
    per_view = np.column_stack([np.min(np.abs(delta), axis=1) for delta in deltas_by_view])
    return per_view, np.mean(per_view, axis=1)


def boundary_strata(proximity):
    order = np.argsort(np.asarray(proximity), kind="mergesort")
    n = order.size; cut = int(np.floor(0.2 * n))
    labels = np.empty(n, dtype=object)
    labels[order[:cut]] = "near_boundary"
    labels[order[cut:n-cut]] = "middle"
    labels[order[n-cut:]] = "interior_far"
    return labels


def paired_stats(values, tie_tol=1e-12):
    x = np.asarray(values, dtype=float)
    return {"mean": float(x.mean()), "median": float(np.median(x)), "std": float(x.std(ddof=0)),
            "min": float(x.min()), "max": float(x.max()),
            "positive": int(np.sum(x > tie_tol)), "negative": int(np.sum(x < -tie_tol)),
            "ties": int(np.sum(np.abs(x) <= tie_tol)), "n": int(x.size)}
