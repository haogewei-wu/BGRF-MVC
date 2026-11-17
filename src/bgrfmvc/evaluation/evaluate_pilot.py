"""Evaluate a fully written Stage 5 prediction set without fitting models."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from bgrfmvc.evaluation.evaluate_predictions import compute_metrics
from bgrfmvc.evaluation.labels import load_evaluation_labels


METRICS = ("ACC", "NMI", "ARI", "PUR", "F_score")
COMPARISONS = (
    ("B0_RawConcat_KM", "B1_HardGB_KM", "granular-ball representation"),
    ("B2_GaussianGB_KM", "B3_BoundaryEqual_KM", "boundary relation vs Gaussian relation"),
    ("B3_BoundaryEqual_KM", "B4_BoundaryEntropy_KM", "sample-specific entropy weighting"),
    ("B4_BoundaryEntropy_KM", "B5_BGRFMVC_Full", "ball reliability"),
)


def _write_csv(path, rows, fields):
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def evaluate_pilot(output_root, datasets_config):
    output_root = Path(output_root)
    with (output_root / "pilot_manifest.json").open("r", encoding="utf-8") as stream:
        pilot = json.load(stream)
    with Path(datasets_config).open("r", encoding="utf-8") as stream:
        datasets = yaml.safe_load(stream)["datasets"]
    raw = []
    # This evaluation begins only after pilot_manifest and all predictions exist.
    for dataset_key, slug in zip(pilot["dataset_keys"], pilot["dataset_slugs"]):
        y_true = load_evaluation_labels(datasets[dataset_key])
        for method in pilot["methods"]:
            for seed in pilot["seeds"]:
                run_dir = output_root / "predictions" / slug / method / ("seed_%d" % seed)
                prediction_path = run_dir / "predictions.csv"
                diagnostic_path = run_dir / "diagnostics.json"
                if not prediction_path.is_file() or not diagnostic_path.is_file():
                    raise FileNotFoundError("prediction phase is incomplete: %s" % run_dir)
                predictions = pd.read_csv(str(prediction_path))["predicted_label"].values
                diagnostics = json.load(diagnostic_path.open("r", encoding="utf-8"))
                metrics = compute_metrics(y_true, predictions)
                row = {
                    "dataset": slug, "dataset_key": dataset_key, "method": method, "seed": seed,
                    "n_samples": diagnostics["n_samples"], "n_clusters": diagnostics["n_clusters"],
                    "runtime": diagnostics["runtime_seconds"], "ball_count": diagnostics["ball_count"],
                    "fused_dimension": diagnostics["fused_dimension"],
                    "mean_entropy": json.dumps(diagnostics["mean_entropy_per_view"]),
                    "mean_alpha_per_view": json.dumps(diagnostics["mean_alpha_per_view"]),
                }
                row.update(metrics)
                raw.append(row)
    raw_fields = [
        "dataset", "dataset_key", "method", "seed", "n_samples", "n_clusters",
        "ACC", "NMI", "ARI", "PUR", "F_score", "runtime", "ball_count",
        "fused_dimension", "mean_entropy", "mean_alpha_per_view",
    ]
    _write_csv(output_root / "pilot_raw.csv", raw, raw_fields)

    grouped = defaultdict(list)
    for row in raw:
        grouped[(row["dataset"], row["method"])].append(row)
    summary = []
    for (dataset, method), rows in sorted(grouped.items()):
        item = {"dataset": dataset, "method": method, "runs": len(rows)}
        for metric in METRICS + ("runtime",):
            values = np.array([float(row[metric]) for row in rows])
            item[metric + "_mean"] = float(values.mean())
            item[metric + "_std"] = float(values.std(ddof=0))
        item.update({
            "ball_count": rows[0]["ball_count"],
            "fused_dimension": rows[0]["fused_dimension"],
            "mean_entropy": rows[0]["mean_entropy"],
            "mean_alpha_per_view": rows[0]["mean_alpha_per_view"],
        })
        summary.append(item)
    summary_fields = ["dataset", "method", "runs"]
    for metric in METRICS + ("runtime",):
        summary_fields.extend([metric + "_mean", metric + "_std"])
    summary_fields.extend(["ball_count", "fused_dimension", "mean_entropy", "mean_alpha_per_view"])
    _write_csv(output_root / "pilot_summary.csv", summary, summary_fields)

    ranking = []
    methods = pilot["methods"]
    method_values = {}
    for method in methods:
        method_rows = [row for row in raw if row["method"] == method]
        method_values[method] = {metric: float(np.mean([row[metric] for row in method_rows])) for metric in METRICS}
    ranks = {method: [] for method in methods}
    for metric in METRICS:
        ordered = sorted(methods, key=lambda method: (-method_values[method][metric], method))
        for rank, method in enumerate(ordered, 1):
            ranks[method].append(rank)
    for method in methods:
        item = {"method": method}
        item.update({metric + "_mean": method_values[method][metric] for metric in METRICS})
        item["composite_metric_mean"] = float(np.mean(list(method_values[method].values())))
        item["average_rank"] = float(np.mean(ranks[method]))
        ranking.append(item)
    ranking.sort(key=lambda row: (row["average_rank"], -row["composite_metric_mean"]))
    for index, item in enumerate(ranking, 1):
        item["overall_rank"] = index
    ranking_fields = ["overall_rank", "method"] + [metric + "_mean" for metric in METRICS] + [
        "composite_metric_mean", "average_rank"
    ]
    _write_csv(output_root / "method_ranking.csv", ranking, ranking_fields)

    keyed = {(row["dataset"], row["method"], row["seed"]): row for row in raw}
    analyses = []
    for before, after, mechanism in COMPARISONS:
        deltas = []
        by_dataset = defaultdict(list)
        for dataset in pilot["dataset_slugs"]:
            for seed in pilot["seeds"]:
                old = keyed[(dataset, before, seed)]
                new = keyed[(dataset, after, seed)]
                delta = float(np.mean([new[m] - old[m] for m in METRICS]))
                deltas.append(delta)
                by_dataset[dataset].append(delta)
        mean_delta = float(np.mean(deltas))
        analyses.append({
            "comparison": "%s -> %s" % (before, after),
            "mechanism": mechanism,
            "mean_composite_delta": mean_delta,
            "positive_pairs": int(np.sum(np.array(deltas) > 0)),
            "total_pairs": len(deltas),
            "dataset_mean_deltas": {key: float(np.mean(value)) for key, value in by_dataset.items()},
            "useful": "YES" if mean_delta > 0 else "NO",
        })
    positive_mechanisms = sum(item["mean_composite_delta"] > 0 for item in analyses)
    raw_mean = next(row["composite_metric_mean"] for row in ranking if row["method"] == "B0_RawConcat_KM")
    full_mean = next(row["composite_metric_mean"] for row in ranking if row["method"] == "B5_BGRFMVC_Full")
    if positive_mechanisms >= 3 and full_mean > raw_mean:
        gate = "PROMISING"
    elif positive_mechanisms <= 1 and full_mean <= raw_mean:
        gate = "NOT_PROMISING"
    else:
        gate = "MIXED"
    analysis_payload = {
        "selection_rule": pilot["rule"],
        "parameter_tuning": False,
        "comparisons": analyses,
        "B5_minus_B0_composite": float(full_mean - raw_mean),
        "positive_mechanism_comparisons": positive_mechanisms,
        "gate_rule": "PROMISING iff >=3/4 mechanism deltas are positive and B5>B0; NOT_PROMISING iff <=1/4 and B5<=B0; otherwise MIXED",
        "gate": gate,
    }
    with (output_root / "pilot_analysis.json").open("w", encoding="utf-8") as stream:
        json.dump(analysis_payload, stream, indent=2, sort_keys=True)
    return raw, summary, ranking, analysis_payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", default="result/stage5_pilot")
    parser.add_argument("--datasets-config", default="configs/datasets.yaml")
    args = parser.parse_args()
    evaluate_pilot(args.output_root, args.datasets_config)


if __name__ == "__main__":
    main()

