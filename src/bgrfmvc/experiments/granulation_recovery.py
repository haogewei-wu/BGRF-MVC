"""Label-free Stage 5R-A granulation structure grid and selection."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import time
from pathlib import Path

import numpy as np
import yaml

from bgrfmvc.core.granular_ball import build_granular_balls
from bgrfmvc.core.preprocessing import MultiViewPreprocessor
from bgrfmvc.data.loader import load_multiview_features


DATASET_KEYS = (
    "Sources__Consensus One-step Multi-view Subspace Clustering__3sources",
    "MSRC__Consensus One-step Multi-view Subspace Clustering__MSRCV1",
    "Handwritten__Consensus One-step Multi-view Subspace Clustering__Handwritten_numerals",
)
DATASET_NAMES = ("3sources", "MSRCV1", "Handwritten_numerals")
ETA_GRID = (0.050, 0.030, 0.020, 0.010, 0.005)
MIN_CHILD_GRID = (5, 10)
RAW_FIELDS = (
    "dataset", "eta", "min_parent_size", "min_child_size", "view_id",
    "n_samples", "n_features", "ball_count", "single_ball_view",
    "mean_ball_size", "median_ball_size", "min_ball_size", "max_ball_size",
    "mean_radius", "median_radius", "radius_std", "mean_q", "median_q",
    "max_depth_reached", "granulation_runtime", "dataset_total_balls",
    "dataset_single_ball_views", "fused_relation_dimension", "ball_count_per_sample",
)
SELECTION_RULE = (
    "minimize collapse_count; then minimize total_ball_count; then prefer larger eta; "
    "then prefer larger min_child_size"
)


def candidate_grid():
    return [
        {"eta": eta, "min_child_size": child, "min_parent_size": max(10, 2 * child)}
        for eta in ETA_GRID for child in MIN_CHILD_GRID
    ]


def rank_candidates(rows):
    """Apply the frozen structure-only priority order."""
    return sorted(
        rows,
        key=lambda row: (
            int(row["candidate_collapse_count"]),
            int(row["total_ball_count"]),
            -float(row["eta"]),
            -int(row["min_child_size"]),
        ),
    )


def _validate_partition(X, balls):
    if not balls or len(balls) > X.shape[0] or any(ball.size <= 0 for ball in balls):
        raise ArithmeticError("invalid or empty granular-ball collection")
    indices = np.concatenate([ball.indices for ball in balls])
    if indices.size != X.shape[0] or not np.array_equal(np.sort(indices), np.arange(X.shape[0])):
        raise ArithmeticError("terminal balls do not partition samples exactly once")
    values = np.array([
        value for ball in balls
        for value in (ball.radius, ball.radial_std, ball.quality_q)
    ])
    if not np.all(np.isfinite(values)) or any(ball.radius < 0 for ball in balls):
        raise ArithmeticError("non-finite or negative granular-ball statistic")
    if any(not (0 < ball.quality_q <= 1) for ball in balls):
        raise ArithmeticError("granular-ball reliability outside (0,1]")


def _view_row(dataset, candidate, view_id, X, balls, runtime):
    sizes = [ball.size for ball in balls]
    radii = [ball.radius for ball in balls]
    qualities = [ball.quality_q for ball in balls]
    return {
        "dataset": dataset,
        "eta": candidate["eta"],
        "min_parent_size": candidate["min_parent_size"],
        "min_child_size": candidate["min_child_size"],
        "view_id": view_id,
        "n_samples": X.shape[0],
        "n_features": X.shape[1],
        "ball_count": len(balls),
        "single_ball_view": len(balls) <= 1,
        "mean_ball_size": float(np.mean(sizes)),
        "median_ball_size": float(statistics.median(sizes)),
        "min_ball_size": min(sizes),
        "max_ball_size": max(sizes),
        "mean_radius": float(np.mean(radii)),
        "median_radius": float(statistics.median(radii)),
        "radius_std": float(np.std(radii, ddof=0)),
        "mean_q": float(np.mean(qualities)),
        "median_q": float(statistics.median(qualities)),
        "max_depth_reached": max(ball.depth for ball in balls),
        "granulation_runtime": float(runtime),
        "ball_count_per_sample": float(len(balls) / X.shape[0]),
    }


def _write_csv(path, rows, fields):
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def run_recovery(project_root):
    project_root = Path(project_root)
    output = project_root / "result" / "stage5r_granulation"
    output.mkdir(parents=True, exist_ok=True)
    with (project_root / "configs" / "default.yaml").open("r", encoding="utf-8") as stream:
        default = yaml.safe_load(stream)
    with (project_root / "configs" / "datasets.yaml").open("r", encoding="utf-8") as stream:
        manifest = yaml.safe_load(stream)["datasets"]

    loaded = []
    for name, key in zip(DATASET_NAMES, DATASET_KEYS):
        entry = manifest[key]
        feature_config = {
            field: entry[field] for field in ("path", "view_keys", "transpose_views") if field in entry
        }
        views = load_multiview_features(feature_config)
        preprocessor = MultiViewPreprocessor(
            standardize=default["preprocess"]["standardize"],
            use_pca=default["preprocess"]["use_pca"],
        )
        loaded.append((name, preprocessor.fit_transform(views)))

    raw = []
    summaries = []
    for candidate in candidate_grid():
        candidate_rows = []
        for dataset, views in loaded:
            dataset_rows = []
            for view_id, X in enumerate(views):
                method_config = dict(default["method"])
                method_config.update(candidate)
                tick = time.perf_counter()
                balls = build_granular_balls(X, method_config, default["granular_ball"])
                runtime = time.perf_counter() - tick
                _validate_partition(X, balls)
                dataset_rows.append(_view_row(dataset, candidate, view_id, X, balls, runtime))
            dataset_total = sum(row["ball_count"] for row in dataset_rows)
            dataset_single = sum(bool(row["single_ball_view"]) for row in dataset_rows)
            for row in dataset_rows:
                row["dataset_total_balls"] = dataset_total
                row["dataset_single_ball_views"] = dataset_single
                row["fused_relation_dimension"] = dataset_total
            candidate_rows.extend(dataset_rows)
        raw.extend(candidate_rows)
        summaries.append({
            "eta": candidate["eta"],
            "min_parent_size": candidate["min_parent_size"],
            "min_child_size": candidate["min_child_size"],
            "candidate_collapse_count": sum(bool(row["single_ball_view"]) for row in candidate_rows),
            "total_ball_count": sum(row["ball_count"] for row in candidate_rows),
            "mean_ball_count_per_sample": float(np.mean([row["ball_count_per_sample"] for row in candidate_rows])),
            "max_ball_count_per_sample": float(np.max([row["ball_count_per_sample"] for row in candidate_rows])),
            "total_granulation_runtime": float(sum(row["granulation_runtime"] for row in candidate_rows)),
            "all_structure_checks_pass": True,
        })
    ranked = rank_candidates(summaries)
    for rank, row in enumerate(ranked, 1):
        row["rank"] = rank
        row["selected"] = rank == 1
    selected = ranked[0]
    selected_rows = [
        row for row in raw
        if float(row["eta"]) == float(selected["eta"])
        and int(row["min_child_size"]) == int(selected["min_child_size"])
    ]

    summary_fields = (
        "eta", "min_parent_size", "min_child_size", "candidate_collapse_count",
        "total_ball_count", "mean_ball_count_per_sample", "max_ball_count_per_sample",
        "total_granulation_runtime", "all_structure_checks_pass",
    )
    ranking_fields = ("rank", "selected") + summary_fields
    _write_csv(output / "granulation_grid_raw.csv", raw, RAW_FIELDS)
    _write_csv(output / "granulation_grid_summary.csv", summaries, summary_fields)
    _write_csv(output / "candidate_ranking.csv", ranked, ranking_fields)
    _write_csv(output / "view_ball_counts.csv", selected_rows, RAW_FIELDS)

    complete = int(selected["candidate_collapse_count"]) == 0
    selected_payload = {
        "eta": float(selected["eta"]),
        "min_parent_size": int(selected["min_parent_size"]),
        "min_child_size": int(selected["min_child_size"]),
        "max_depth": int(default["method"]["max_depth"]),
        "tau": float(default["method"]["tau"]),
        "eps": float(default["method"]["eps"]),
        "radius_eps": float(default["method"]["radius_eps"]),
        "relation_floor": float(default["method"]["relation_floor"]),
        "use_ball_reliability": bool(default["method"]["use_ball_reliability"]),
        "split_n_init": int(default["granular_ball"]["split_n_init"]),
        "split_max_iter": int(default["granular_ball"]["split_max_iter"]),
        "split_random_state": int(default["granular_ball"]["split_random_state"]),
        "standardize": bool(default["preprocess"]["standardize"]),
        "use_pca": bool(default["preprocess"]["use_pca"]),
        "collapse_count": int(selected["candidate_collapse_count"]),
        "total_ball_count": int(selected["total_ball_count"]),
        "selection_rule": SELECTION_RULE,
        "labels_loaded": False,
        "status": "PASS" if complete else "PARTIAL_RECOVERY",
    }
    with (output / "selected_granulation_config.json").open("w", encoding="utf-8") as stream:
        json.dump(selected_payload, stream, indent=2, sort_keys=True)
    candidate_yaml = {
        "experimental_recovery_candidate": True,
        "method": {
            "eta": selected_payload["eta"],
            "min_parent_size": selected_payload["min_parent_size"],
            "min_child_size": selected_payload["min_child_size"],
            "max_depth": selected_payload["max_depth"],
            "tau": selected_payload["tau"],
            "eps": selected_payload["eps"],
            "radius_eps": selected_payload["radius_eps"],
            "relation_floor": selected_payload["relation_floor"],
            "use_ball_reliability": selected_payload["use_ball_reliability"],
        },
        "granular_ball": {
            "split_n_init": selected_payload["split_n_init"],
            "split_max_iter": selected_payload["split_max_iter"],
            "split_random_state": selected_payload["split_random_state"],
        },
        "preprocess": {
            "standardize": selected_payload["standardize"],
            "use_pca": selected_payload["use_pca"],
        },
        "selection": {
            "collapse_count": selected_payload["collapse_count"],
            "total_ball_count": selected_payload["total_ball_count"],
            "selection_rule": SELECTION_RULE,
            "labels_loaded": False,
            "status": selected_payload["status"],
        },
    }
    with (project_root / "configs" / "stage5r_candidate.yaml").open("w", encoding="utf-8", newline="\n") as stream:
        yaml.safe_dump(candidate_yaml, stream, allow_unicode=True)
    print(json.dumps(selected_payload, sort_keys=True))
    return selected_payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    args = parser.parse_args()
    run_recovery(Path(args.project_root).resolve())


if __name__ == "__main__":
    main()
