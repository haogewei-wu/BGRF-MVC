"""Run BGRF-MVC from feature views and a manifest-frozen K only.

This module deliberately has no dependency on evaluation or label loading.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import yaml

from bgrfmvc.core import BGRFMVC
from bgrfmvc.data.loader import load_multiview_features


def _read_yaml(path):
    with Path(path).open("r", encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def _json_ready(value):
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def run_dataset(dataset_name, dataset_config, method_config, output_dir, seed=0):
    """Run one enabled dataset without accessing its ground-truth array."""
    n_clusters = dataset_config.get("n_clusters")
    if not dataset_config.get("enabled", False):
        raise ValueError("dataset is not enabled in the frozen manifest")
    if not isinstance(n_clusters, int) or n_clusters < 1:
        raise ValueError("n_clusters must be pre-frozen in datasets.yaml")
    if not dataset_config.get("k_source"):
        raise ValueError("n_clusters requires a documented external k_source")
    # Whitelist only feature-related fields. Evaluation-only manifest fields are
    # never passed to the feature loader or the model.
    feature_config = {
        key: dataset_config[key]
        for key in ("path", "view_keys", "transpose_views")
        if key in dataset_config
    }
    started = time.perf_counter()
    X_views = load_multiview_features(feature_config)
    load_seconds = time.perf_counter() - started
    model = BGRFMVC(method_config)
    fit_started = time.perf_counter()
    predictions = model.fit_predict(X_views, n_clusters=n_clusters, seed=seed)
    fit_seconds = time.perf_counter() - fit_started
    total_seconds = time.perf_counter() - started

    n_samples = X_views[0].shape[0]
    if predictions.shape != (n_samples,) or not np.all(np.isfinite(model.fused_)):
        raise ArithmeticError("invalid prediction or fused representation output")
    if not np.all(np.isfinite(model.alpha_)) or not np.all(np.isfinite(np.column_stack(model.entropies_))):
        raise ArithmeticError("weights and entropy must be finite")
    if not np.allclose(model.alpha_.sum(axis=1), 1.0, atol=1e-10):
        raise ArithmeticError("view-weight rows do not sum to one")
    if any(len(balls) < 1 for balls in model.balls_):
        raise ArithmeticError("every view must produce at least one granular ball")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "predictions.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["predicted_label"])
        writer.writerows([[int(value)] for value in predictions])
    np.save(str(output_dir / "view_weights.npy"), model.alpha_)
    np.save(str(output_dir / "entropy.npy"), np.column_stack(model.entropies_))
    with (output_dir / "ball_statistics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=["view", "ball", "size", "depth", "radius", "radial_std", "quality_q"]
        )
        writer.writeheader()
        for view_index, balls in enumerate(model.balls_):
            for ball_index, ball in enumerate(balls):
                writer.writerow({
                    "view": view_index, "ball": ball_index, "size": ball.size, "depth": ball.depth,
                    "radius": ball.radius, "radial_std": ball.radial_std, "quality_q": ball.quality_q,
                })
    diagnostics = dict(model.diagnostics_)
    diagnostics.update({
        "dataset": dataset_name,
        "seed": int(seed),
        "n_clusters": int(n_clusters),
        "k_source": dataset_config["k_source"],
        "n_samples": int(n_samples),
        "n_views": len(X_views),
        "view_dimensions": [list(view.shape) for view in X_views],
        "fused_rows": int(model.fused_.shape[0]),
        "finite": True,
    })
    runtime = {"load_seconds": load_seconds, "fit_predict_seconds": fit_seconds, "total_seconds": total_seconds}
    for name, payload in (("diagnostics.json", diagnostics), ("runtime.json", runtime)):
        with (output_dir / name).open("w", encoding="utf-8") as stream:
            json.dump(_json_ready(payload), stream, indent=2, sort_keys=True)
    return predictions, diagnostics, runtime


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--datasets-config", default="configs/datasets.yaml")
    parser.add_argument("--method-config", default="configs/default.yaml")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    manifest = _read_yaml(args.datasets_config)["datasets"]
    if args.dataset not in manifest:
        raise KeyError("dataset not found in manifest: %s" % args.dataset)
    run_dataset(
        args.dataset, manifest[args.dataset], _read_yaml(args.method_config), args.output_dir, seed=args.seed
    )


if __name__ == "__main__":
    main()

