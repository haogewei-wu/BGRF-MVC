"""Relation-block normalization and sample-specific multi-view fusion."""

from __future__ import annotations

import numpy as np


def l2_normalize_rows(matrix, eps=1e-12):
    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("matrix must be 2D")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    result = matrix / (norms + eps)
    if not np.all(np.isfinite(result)):
        raise ArithmeticError("row normalization produced NaN or Inf")
    return result


def fuse_relation_blocks(corrected_blocks, alpha, eps=1e-12):
    if not corrected_blocks:
        raise ValueError("at least one relation block is required")
    alpha = np.asarray(alpha, dtype=float)
    blocks = [np.asarray(block, dtype=float) for block in corrected_blocks]
    n_samples = blocks[0].shape[0]
    if any(block.ndim != 2 or block.shape[0] != n_samples for block in blocks):
        raise ValueError("all relation blocks must be aligned 2D matrices")
    if alpha.shape != (n_samples, len(blocks)):
        raise ValueError("alpha must have shape (n_samples, n_views)")
    if np.any(alpha < 0) or not np.allclose(alpha.sum(axis=1), 1.0, atol=1e-10):
        raise ValueError("alpha must be non-negative with row sums of one")
    normalized = [l2_normalize_rows(block, eps=eps) for block in blocks]
    weighted = [block * np.sqrt(alpha[:, [index]]) for index, block in enumerate(normalized)]
    return l2_normalize_rows(np.concatenate(weighted, axis=1), eps=eps)

