"""Stage 5R-C Phase A; this module has no evaluation dependency."""

from __future__ import annotations

import argparse
import csv
import json
import re
import time
from pathlib import Path

import numpy as np
import yaml
from scipy.spatial.distance import pdist
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans

from bgrfmvc.data.loader import load_multiview_features
from bgrfmvc.experiments.stage5r_hybrid import METHODS, VARIANCE_TOLERANCE, build_hybrid_representations


SEEDS = (0, 1, 2, 3, 4)
DATASETS = ("3sources", "MSRCV1", "Handwritten_numerals")
EXPECTED = {
    "3sources": [1, 1, 1],
    "MSRCV1": [11, 7, 14, 9, 12, 5],
    "Handwritten_numerals": [105, 116, 119, 75, 120, 112],
}


def _yaml(path):
    with Path(path).open("r", encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def _slug(key):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", key.split("__")[-1]).strip("_")


def _select(manifest):
    selected = []
    for slug in DATASETS:
        matches = [(k, v) for k, v in manifest.items() if _slug(k) == slug and v.get("enabled")]
        if len(matches) != 1:
            raise RuntimeError("expected one enabled dataset for %s" % slug)
        key, cfg = matches[0]
        if not isinstance(cfg.get("n_clusters"), int) or not cfg.get("k_verified_without_labels"):
            raise RuntimeError("externally frozen K unavailable for %s" % slug)
        selected.append((key, slug, cfg))
    return selected


def _write_predictions(path, values):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream); writer.writerow(["predicted_label"])
        writer.writerows([[int(x)] for x in values])


def _partition_agreement(first, second):
    first_values, first_codes = np.unique(first, return_inverse=True)
    second_values, second_codes = np.unique(second, return_inverse=True)
    table = np.zeros((first_values.size, second_values.size), dtype=np.int64)
    np.add.at(table, (first_codes, second_codes), 1)
    rows, columns = linear_sum_assignment(table.max() - table)
    return float(table[rows, columns].sum() / first_codes.size)


def run_phase_a(output_root, datasets_config, candidate_config):
    root = Path(output_root); root.mkdir(parents=True, exist_ok=True)
    manifest = _yaml(datasets_config)["datasets"]
    config = _yaml(candidate_config)
    selected = _select(manifest)
    phase = {
        "stage": "5R-C", "phase": "A", "phase_a_complete": False,
        "labels_loaded": False, "methods": list(METHODS), "datasets": list(DATASETS),
        "seeds": list(SEEDS), "variance_tolerance": VARIANCE_TOLERANCE,
        "granulation": {k: config["method"][k] for k in
                         ("eta", "min_parent_size", "min_child_size", "max_depth")},
        "tau": config["method"]["tau"], "ball_reliability": False,
        "view_weighting": "equal", "hybrid_energy": "1/sqrt(2) raw + 1/sqrt(2) relation",
    }
    phase_path = root / "phase_a_manifest.json"
    phase_path.write_text(json.dumps(phase, indent=2, sort_keys=True), encoding="utf-8")

    # Build all representations and verify every ball count before any prediction is written.
    prepared = []
    for key, slug, dataset in selected:
        feature_cfg = {k: dataset[k] for k in ("path", "view_keys", "transpose_views") if k in dataset}
        load_tick = time.perf_counter()
        views = load_multiview_features(feature_cfg)
        load_seconds = time.perf_counter() - load_tick
        results, shared = build_hybrid_representations(views, config)
        if shared["ball_count_per_view"] != EXPECTED[slug]:
            failure = {"status": "GRANULATION_REPRODUCIBILITY_FAILURE", "dataset": slug,
                       "expected": EXPECTED[slug], "actual": shared["ball_count_per_view"],
                       "labels_loaded": False}
            (root / "reproducibility_failure.json").write_text(
                json.dumps(failure, indent=2, sort_keys=True), encoding="utf-8")
            raise RuntimeError("GRANULATION_REPRODUCIBILITY_FAILURE")
        prepared.append((key, slug, dataset, results, shared, load_seconds))

    safe = []
    for key, slug, dataset, results, shared, load_seconds in prepared:
        dataset_root = root / "predictions" / slug; dataset_root.mkdir(parents=True, exist_ok=True)
        shared.update({"dataset_key": key, "dataset": slug, "n_clusters": dataset["n_clusters"],
                       "load_seconds": load_seconds, "granulation_reproducibility": "PASS"})
        (dataset_root / "shared_diagnostics.json").write_text(
            json.dumps(shared, indent=2, sort_keys=True), encoding="utf-8")
        if slug == "3sources":
            d0_dist = pdist(results[METHODS[0]]["representation"], metric="euclidean")
            d3_dist = pdist(results[METHODS[3]]["representation"], metric="euclidean")
            distance_correlation = float(np.corrcoef(d0_dist, d3_dist)[0, 1])
        else:
            distance_correlation = None
        predictions_by_method = {}
        for method, payload in results.items():
            representation = payload["representation"]
            diagnostic = payload["diagnostics"]
            method_root = dataset_root / method; method_root.mkdir(parents=True, exist_ok=True)
            (method_root / "representation_diagnostics.json").write_text(
                json.dumps(diagnostic, indent=2, sort_keys=True), encoding="utf-8")
            predictions_by_seed = {}
            for seed in SEEDS:
                tick = time.perf_counter()
                predictions = KMeans(n_clusters=int(dataset["n_clusters"]), n_init=20,
                                     max_iter=300, random_state=seed).fit_predict(representation)
                kmeans_seconds = time.perf_counter() - tick
                repeat = KMeans(n_clusters=int(dataset["n_clusters"]), n_init=20,
                                max_iter=300, random_state=seed).fit_predict(representation)
                if not np.array_equal(predictions, repeat):
                    raise RuntimeError("KMeans reproducibility failure")
                predictions_by_seed[seed] = predictions
                run_root = method_root / ("seed_%d" % seed); run_root.mkdir(parents=True, exist_ok=True)
                _write_predictions(run_root / "predictions.csv", predictions)
                run_diag = dict(diagnostic)
                run_diag.update({
                    "dataset": slug, "method": method, "seed": seed,
                    "n_samples": shared["n_samples"], "n_clusters": dataset["n_clusters"],
                    "runtime": float(load_seconds + shared["preprocess_seconds"] +
                                     (0 if method == METHODS[0] else shared["granulation_seconds"]) +
                                     diagnostic["representation_seconds"] + kmeans_seconds),
                    "ball_count_per_view": [] if method == METHODS[0] else shared["ball_count_per_view"],
                    "labels_loaded": False, "reproducible": True,
                })
                (run_root / "diagnostics.json").write_text(
                    json.dumps(run_diag, indent=2, sort_keys=True), encoding="utf-8")
            predictions_by_method[method] = predictions_by_seed
        if slug == "3sources":
            for seed in SEEDS:
                a = predictions_by_method[METHODS[0]][seed]
                b = predictions_by_method[METHODS[3]][seed]
                safe.append({"dataset": slug, "seed": seed,
                             "prediction_agreement": _partition_agreement(a, b),
                             "pairwise_distance_correlation": distance_correlation,
                             "labels_loaded": False})
    (root / "safe_fallback_phase_a.json").write_text(json.dumps(safe, indent=2), encoding="utf-8")
    phase["phase_a_complete"] = True
    phase_path.write_text(json.dumps(phase, indent=2, sort_keys=True), encoding="utf-8")
    return phase


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", default="result/stage5r_hybrid")
    parser.add_argument("--datasets-config", default="configs/datasets.yaml")
    parser.add_argument("--candidate-config", default="configs/stage5r_candidate.yaml")
    args = parser.parse_args()
    run_phase_a(args.output_root, args.datasets_config, args.candidate_config)


if __name__ == "__main__":
    main()
