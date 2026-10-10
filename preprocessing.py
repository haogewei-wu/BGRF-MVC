"""Per-view, label-free preprocessing for BGRF-MVC."""

from __future__ import annotations

import numpy as np
from sklearn.preprocessing import StandardScaler


class MultiViewPreprocessor:
    """Fit and apply one independent StandardScaler per view."""

    def __init__(self, standardize=True, use_pca=False):
        if use_pca:
            raise ValueError("PCA is disabled in the frozen BGRF-MVC v2.0.1 prototype")
        self.standardize = bool(standardize)
        self.use_pca = False
        self.scalers_ = None

    @staticmethod
    def _validate(X_views):
        if not isinstance(X_views, (list, tuple)) or not X_views:
            raise ValueError("X_views must be a non-empty list or tuple")
        arrays = [np.asarray(view, dtype=float) for view in X_views]
        if any(view.ndim != 2 for view in arrays):
            raise ValueError("every view must be a 2D matrix")
        if any(view.shape[0] < 2 or view.shape[1] < 1 for view in arrays):
            raise ValueError("every view must contain at least two samples and one feature")
        if len({view.shape[0] for view in arrays}) != 1:
            raise ValueError("all views must have the same sample count")
        if any(not np.all(np.isfinite(view)) for view in arrays):
            raise ValueError("views must not contain NaN or Inf")
        return arrays

    def fit(self, X_views):
        arrays = self._validate(X_views)
        if self.standardize:
            self.scalers_ = [StandardScaler().fit(view) for view in arrays]
        else:
            self.scalers_ = [None for _ in arrays]
        return self

    def transform(self, X_views):
        arrays = self._validate(X_views)
        if self.scalers_ is None:
            raise RuntimeError("preprocessor is not fitted")
        if len(arrays) != len(self.scalers_):
            raise ValueError("view count differs from fitted data")
        transformed = []
        for view, scaler in zip(arrays, self.scalers_):
            transformed.append(np.asarray(scaler.transform(view) if scaler else view.copy(), dtype=float))
        return transformed

    def fit_transform(self, X_views):
        return self.fit(X_views).transform(X_views)

