"""Public BGRF-MVC v2.0.1 unsupervised API."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import yaml
from sklearn.cluster import KMeans

from .boundary_relation import compute_boundary_relations
from .fusion import fuse_relation_blocks
from .granular_ball import build_granular_balls
from .preprocessing import MultiViewPreprocessor
from .reliability import relation_probability_and_entropy, sample_specific_view_weights


def _default_config_path():
    return Path(__file__).resolve().parents[3] / "configs" / "default.yaml"


def _load_config(config):
    if config is None:
        path = _default_config_path()
        with path.open("r", encoding="utf-8") as stream:
            return yaml.safe_load(stream)
    if isinstance(config, (str, Path)):
        with Path(config).open("r", encoding="utf-8") as stream:
            return yaml.safe_load(stream)
    if isinstance(config, dict):
        return config
    raise TypeError("config must be None, a mapping, or a YAML path")


class BGRFMVC:
    """Boundary-aware granular relation fusion for multi-view clustering."""

    def __init__(self, config=None):
        self.config = _load_config(config)
        preprocess = self.config["preprocess"]
        self.preprocessor_ = MultiViewPreprocessor(
            standardize=preprocess.get("standardize", True),
            use_pca=preprocess.get("use_pca", False),
        )
        self.balls_ = None
        self.fused_ = None
        self.alpha_ = None
        self.entropies_ = None
        self.probabilities_ = None
        self.relations_ = None
        self.diagnostics_ = None

    def _represent(self, transformed_views):
        method = self.config["method"]
        relation_results = [
            compute_boundary_relations(
                view, balls,
                tau=method["tau"], eps=method["eps"],
                radius_eps=method["radius_eps"], relation_floor=method["relation_floor"],
                use_ball_reliability=method["use_ball_reliability"],
            )
            for view, balls in zip(transformed_views, self.balls_)
        ]
        probability_entropy = [
            relation_probability_and_entropy(result.corrected_relations, eps=method["eps"])
            for result in relation_results
        ]
        probabilities = [item[0] for item in probability_entropy]
        entropies = [item[1] for item in probability_entropy]
        alpha = sample_specific_view_weights(entropies, eps=method["eps"])
        fused = fuse_relation_blocks(
            [result.corrected_relations for result in relation_results], alpha, eps=method["eps"]
        )
        return fused, relation_results, probabilities, entropies, alpha

    def fit(self, X_views):
        started = time.perf_counter()
        transformed = self.preprocessor_.fit_transform(X_views)
        self.balls_ = [
            build_granular_balls(view, self.config["method"], self.config["granular_ball"])
            for view in transformed
        ]
        fused, relations, probabilities, entropies, alpha = self._represent(transformed)
        self.fused_ = fused
        self.relations_ = relations
        self.probabilities_ = probabilities
        self.entropies_ = entropies
        self.alpha_ = alpha
        elapsed = time.perf_counter() - started
        self.diagnostics_ = {
            "ball_count_by_view": [len(balls) for balls in self.balls_],
            "ball_sizes_by_view": [[ball.size for ball in balls] for balls in self.balls_],
            "ball_radii_by_view": [[ball.radius for ball in balls] for balls in self.balls_],
            "ball_quality_q_by_view": [[ball.quality_q for ball in balls] for balls in self.balls_],
            "entropy_mean_by_view": [float(values.mean()) for values in entropies],
            "entropy_std_by_view": [float(values.std(ddof=0)) for values in entropies],
            "alpha_mean_by_view": [float(value) for value in alpha.mean(axis=0)],
            "alpha_std_by_view": [float(value) for value in alpha.std(axis=0, ddof=0)],
            "fused_representation_dimension": int(fused.shape[1]),
            "runtime_seconds": float(elapsed),
        }
        return self

    def transform(self, X_views):
        if self.balls_ is None:
            raise RuntimeError("model is not fitted")
        transformed = self.preprocessor_.transform(X_views)
        fused, _, _, _, _ = self._represent(transformed)
        return fused

    def fit_transform(self, X_views):
        self.fit(X_views)
        return self.fused_.copy()

    def fit_predict(self, X_views, n_clusters, seed):
        if not isinstance(n_clusters, (int, np.integer)) or n_clusters < 1:
            raise ValueError("n_clusters must be a positive integer supplied explicitly")
        fused = self.fit_transform(X_views)
        kmeans = self.config["kmeans"]
        return KMeans(
            n_clusters=int(n_clusters),
            n_init=int(kmeans["n_init"]),
            max_iter=int(kmeans["max_iter"]),
            random_state=int(seed),
        ).fit_predict(fused)

