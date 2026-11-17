"""Generate all Stage 5 predictions before any evaluation label is loaded."""

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
from bgrfmvc.experiments.pilot_baselines import METHODS, build_pilot_representations


SEEDS = (0, 1, 2, 3, 4)


def _read_yaml(path):
    with Path(path).open("r", encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def _slug(dataset_key):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", dataset_key.split("__")[-1]).strip("_")


def select_pilot_datasets(eligible_csv, manifest, limit=5):
    """Deterministically select by X-derived sample count ascending."""
    with Path(eligible_csv).open("r", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    rows.sort(key=lambda row: (int(row["n_samples"]), row["dataset"]))
    selected = []
    for row in rows[:limit]:
        config = manifest[row["dataset"]]
        if not config.get("enabled") or not isinstance(config.get("n_clusters"), int):
            raise ValueError("eligible dataset lacks enabled frozen K: %s" % row["dataset"])
        selected.append(row)
    return selected


def run_pilot(output_root, datasets_config, method_config, eligible_csv):
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    manifest = _read_yaml(datasets_config)["datasets"]
    config = _read_yaml(method_config)
    selected = select_pilot_datasets(eligible_csv, manifest, limit=5)
    selection = {
        "rule": "eligible datasets sorted by X-derived n_samples ascending; take first 5",
        "dataset_keys": [row["dataset"] for row in selected],
        "dataset_slugs": [_slug(row["dataset"]) for row in selected],
        "seeds": list(SEEDS),
        "methods": list(METHODS),
        "labels_loaded": False,
    }
    with (output_root / "pilot_manifest.json").open("w", encoding="utf-8") as stream:
        json.dump(selection, stream, indent=2, sort_keys=True)

    for row in selected:
        dataset_key = row["dataset"]
        dataset = manifest[dataset_key]
        feature_config = {
            key: dataset[key] for key in ("path", "view_keys", "transpose_views") if key in dataset
        }
        loaded_at = time.perf_counter()
        X_views = load_multiview_features(feature_config)
        load_seconds = time.perf_counter() - loaded_at
        representations, shared = build_pilot_representations(X_views, config)
        shared["load_seconds"] = float(load_seconds)
        shared["dataset_key"] = dataset_key
        shared["dataset_slug"] = _slug(dataset_key)
        shared["n_clusters"] = int(dataset["n_clusters"])
        shared["k_source"] = dataset["k_source"]
        dataset_root = output_root / "predictions" / _slug(dataset_key)
        dataset_root.parent.mkdir(parents=True, exist_ok=True)
        with (dataset_root.parent / (dataset_root.name + "_shared.json")).open("w", encoding="utf-8") as stream:
            json.dump(shared, stream, indent=2, sort_keys=True)
        for method_name, payload in representations.items():
            representation = payload.pop("representation")
            for seed in SEEDS:
                tick = time.perf_counter()
                predictions = KMeans(
                    n_clusters=int(dataset["n_clusters"]),
                    n_init=int(config["kmeans"]["n_init"]),
                    max_iter=int(config["kmeans"]["max_iter"]),
                    random_state=seed,
                ).fit_predict(representation)
                kmeans_seconds = time.perf_counter() - tick
                run_dir = dataset_root / method_name / ("seed_%d" % seed)
                run_dir.mkdir(parents=True, exist_ok=True)
                with (run_dir / "predictions.csv").open("w", newline="", encoding="utf-8") as stream:
                    writer = csv.writer(stream)
                    writer.writerow(["predicted_label"])
                    writer.writerows([[int(value)] for value in predictions])
                diagnostics = dict(payload)
                diagnostics.update({
                    "dataset_key": dataset_key,
                    "dataset_slug": _slug(dataset_key),
                    "method": method_name,
                    "seed": seed,
                    "n_clusters": int(dataset["n_clusters"]),
                    "n_samples": int(shared["n_samples"]),
                    "ball_count": 0 if method_name == METHODS[0] else shared["ball_count"],
                    "ball_count_by_view": [] if method_name == METHODS[0] else shared["ball_count_by_view"],
                    "shared_preprocessing_seconds": shared["preprocessing_seconds"],
                    "shared_ball_build_seconds": 0.0 if method_name == METHODS[0] else shared["ball_build_seconds"],
                    "kmeans_seconds": float(kmeans_seconds),
                    "runtime_seconds": float(
                        load_seconds + shared["preprocessing_seconds"]
                        + (0.0 if method_name == METHODS[0] else shared["ball_build_seconds"])
                        + payload["representation_seconds"] + kmeans_seconds
                    ),
                    "finite": True,
                    "labels_loaded": False,
                })
                with (run_dir / "diagnostics.json").open("w", encoding="utf-8") as stream:
                    json.dump(diagnostics, stream, indent=2, sort_keys=True)
    return selection


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", default="result/stage5_pilot")
    parser.add_argument("--datasets-config", default="configs/datasets.yaml")
    parser.add_argument("--method-config", default="configs/default.yaml")
    parser.add_argument("--eligible-csv", default="result/stage2_data_audit/eligible_datasets.csv")
    args = parser.parse_args()
    run_pilot(args.output_root, args.datasets_config, args.method_config, args.eligible_csv)


if __name__ == "__main__":
    main()
