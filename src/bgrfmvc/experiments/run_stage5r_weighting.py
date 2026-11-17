"""Stage 5R-B Phase A: X plus frozen manifest K only."""

from __future__ import annotations

import argparse
import csv
import json
import re
import time
from pathlib import Path

import numpy as np
import yaml
from sklearn.cluster import KMeans

from bgrfmvc.data.loader import load_multiview_features
from bgrfmvc.experiments.stage5r_weighting import (
    METHODS, VARIANCE_TOLERANCE, build_stage5r_representations,
)


SEEDS = (0, 1, 2, 3, 4)
DATASET_ORDER = ("3sources", "MSRCV1", "Handwritten_numerals")
EXPECTED_COUNTS = {
    "3sources": [1, 1, 1],
    "MSRCV1": [11, 7, 14, 9, 12, 5],
    "Handwritten_numerals": [105, 116, 119, 75, 120, 112],
}
KMEANS_N_INIT = 20
KMEANS_MAX_ITER = 300


def _read_yaml(path):
    with Path(path).open("r", encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def _slug(dataset_key):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", dataset_key.split("__")[-1]).strip("_")


def _selected(manifest):
    result = []
    for slug in DATASET_ORDER:
        matches = [(key, value) for key, value in manifest.items()
                   if _slug(key) == slug and value.get("enabled")]
        if len(matches) != 1:
            raise RuntimeError("expected exactly one enabled manifest entry for %s" % slug)
        key, config = matches[0]
        if not isinstance(config.get("n_clusters"), int) or not config.get("k_verified_without_labels"):
            raise RuntimeError("dataset lacks externally frozen K: %s" % slug)
        result.append((key, slug, config))
    return result


def _write_predictions(path, predictions):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["predicted_label"])
        writer.writerows([[int(value)] for value in predictions])


def run_phase_a(output_root, datasets_config, candidate_config):
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    manifest = _read_yaml(datasets_config)["datasets"]
    config = _read_yaml(candidate_config)
    phase_manifest = {
        "stage": "5R-B", "phase": "A", "phase_a_complete": False,
        "labels_loaded": False, "seeds": list(SEEDS), "methods": list(METHODS),
        "datasets": list(DATASET_ORDER), "variance_tolerance": VARIANCE_TOLERANCE,
        "corrected_weight_formula": "((1-H)*(m-1)/m + eps), normalized across views",
        "frozen_granulation": {key: config["method"][key] for key in
                               ("eta", "min_parent_size", "min_child_size", "max_depth")},
    }
    manifest_path = root / "phase_a_manifest.json"
    manifest_path.write_text(json.dumps(phase_manifest, indent=2, sort_keys=True), encoding="utf-8")

    for dataset_key, slug, dataset in _selected(manifest):
        feature_config = {key: dataset[key] for key in ("path", "view_keys", "transpose_views") if key in dataset}
        load_tick = time.perf_counter()
        views = load_multiview_features(feature_config)
        load_seconds = time.perf_counter() - load_tick
        results, shared = build_stage5r_representations(views, config)
        actual = shared["ball_count_by_view"]
        if actual != EXPECTED_COUNTS[slug]:
            failure = {
                "dataset": slug, "expected": EXPECTED_COUNTS[slug], "actual": actual,
                "status": "REPRODUCIBILITY_FAILURE", "labels_loaded": False,
            }
            (root / "reproducibility_failure.json").write_text(
                json.dumps(failure, indent=2, sort_keys=True), encoding="utf-8")
            raise RuntimeError("granularization reproducibility failure for %s" % slug)

        shared.update({
            "dataset": slug, "dataset_key": dataset_key,
            "n_clusters": int(dataset["n_clusters"]), "k_source": dataset["k_source"],
            "load_seconds": float(load_seconds), "expected_ball_count_by_view": EXPECTED_COUNTS[slug],
            "reproducibility_check": "PASS",
        })
        dataset_root = root / "predictions" / slug
        dataset_root.mkdir(parents=True, exist_ok=True)
        (dataset_root / "shared_diagnostics.json").write_text(
            json.dumps(shared, indent=2, sort_keys=True), encoding="utf-8")

        for method, payload in results.items():
            representation = payload["representation"]
            method_diag = payload["diagnostics"]
            method_root = dataset_root / method
            method_root.mkdir(parents=True, exist_ok=True)
            (method_root / "representation_diagnostics.json").write_text(
                json.dumps(method_diag, indent=2, sort_keys=True), encoding="utf-8")
            for seed in SEEDS:
                tick = time.perf_counter()
                predictions = KMeans(
                    n_clusters=int(dataset["n_clusters"]),
                    n_init=KMEANS_N_INIT,
                    max_iter=KMEANS_MAX_ITER,
                    random_state=seed,
                ).fit_predict(representation)
                kmeans_seconds = time.perf_counter() - tick
                repeat = KMeans(
                    n_clusters=int(dataset["n_clusters"]),
                    n_init=KMEANS_N_INIT,
                    max_iter=KMEANS_MAX_ITER,
                    random_state=seed,
                ).fit_predict(representation)
                if not np.array_equal(predictions, repeat):
                    raise RuntimeError("KMeans reproducibility failure: %s %s %d" % (slug, method, seed))
                run_root = method_root / ("seed_%d" % seed)
                run_root.mkdir(parents=True, exist_ok=True)
                _write_predictions(run_root / "predictions.csv", predictions)
                run_diag = {
                    "dataset": slug, "method": method, "seed": seed,
                    "n_samples": int(shared["n_samples"]), "n_clusters": int(dataset["n_clusters"]),
                    "ball_count_by_view": actual if method != METHODS[0] else [],
                    "fused_dimension": method_diag["fused_dimension"],
                    "runtime_seconds": float(load_seconds + shared["preprocessing_seconds"]
                                             + (0.0 if method == METHODS[0] else shared["granularization_seconds"])
                                             + method_diag["representation_seconds"] + kmeans_seconds),
                    "kmeans_seconds": float(kmeans_seconds), "labels_loaded": False,
                    "finite": True, "reproducible": True,
                    "alpha_min": method_diag["alpha_min"],
                    "alpha_max_row_sum_error": method_diag["alpha_max_row_sum_error"],
                    "entropy_min": method_diag["entropy_min"], "entropy_max": method_diag["entropy_max"],
                }
                (run_root / "diagnostics.json").write_text(
                    json.dumps(run_diag, indent=2, sort_keys=True), encoding="utf-8")

    phase_manifest["phase_a_complete"] = True
    manifest_path.write_text(json.dumps(phase_manifest, indent=2, sort_keys=True), encoding="utf-8")
    return phase_manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", default="result/stage5r_weighting")
    parser.add_argument("--datasets-config", default="configs/datasets.yaml")
    parser.add_argument("--candidate-config", default="configs/stage5r_candidate.yaml")
    args = parser.parse_args()
    run_phase_a(args.output_root, args.datasets_config, args.candidate_config)


if __name__ == "__main__":
    main()
