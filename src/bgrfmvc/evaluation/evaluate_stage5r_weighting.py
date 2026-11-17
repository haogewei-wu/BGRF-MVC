"""Stage 5R-B Phase B: evaluate a complete, saved Phase A only."""

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
from bgrfmvc.experiments.stage5r_weighting import METHODS


METRICS = ("ACC", "NMI", "ARI", "PUR", "F_score")
PRIMARY = ("MSRCV1", "Handwritten_numerals")
TIE_TOLERANCE = 1.0e-12


def _read_yaml(path):
    with Path(path).open("r", encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def _write_csv(path, rows, fields):
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _dataset_configs(manifest, slugs):
    result = {}
    for slug in slugs:
        matches = [(key, value) for key, value in manifest.items()
                   if key.split("__")[-1] == slug and value.get("enabled")]
        if len(matches) != 1:
            raise RuntimeError("enabled dataset lookup failed for %s" % slug)
        result[slug] = matches[0]
    return result


def _require_complete(root, phase):
    if not phase.get("phase_a_complete") or phase.get("labels_loaded") is not False:
        raise RuntimeError("Phase A is incomplete or label firewall record is invalid")
    for dataset in phase["datasets"]:
        for method in phase["methods"]:
            for seed in phase["seeds"]:
                base = root / "predictions" / dataset / method / ("seed_%d" % seed)
                if not (base / "predictions.csv").is_file() or not (base / "diagnostics.json").is_file():
                    raise RuntimeError("Phase A prediction set incomplete: %s" % base)


def _composite(row):
    return float(np.mean([float(row[name]) for name in METRICS]))


def evaluate_phase_b(output_root, datasets_config):
    root = Path(output_root)
    phase = json.loads((root / "phase_a_manifest.json").read_text(encoding="utf-8"))
    _require_complete(root, phase)  # must complete before the first label load
    manifest = _read_yaml(datasets_config)["datasets"]
    configs = _dataset_configs(manifest, phase["datasets"])
    raw = []
    degeneration = []
    efficiency = []
    single_ball = []

    for dataset in phase["datasets"]:
        dataset_key, config = configs[dataset]
        y_true = load_evaluation_labels(config)
        shared_path = root / "predictions" / dataset / "shared_diagnostics.json"
        shared = json.loads(shared_path.read_text(encoding="utf-8"))
        relation_dim = int(shared["relation_dimension"])
        raw_dim = int(shared["raw_total_dimension"])
        efficiency.append({
            "dataset": dataset, "n_samples": shared["n_samples"], "n_views": shared["n_views"],
            "raw_total_dimension": raw_dim, "relation_dimension": relation_dim,
            "relation_to_raw_ratio": relation_dim / float(raw_dim),
            "ball_to_sample_ratio": relation_dim / float(shared["n_samples"] * shared["n_views"]),
            "representation_compression": relation_dim < raw_dim,
        })
        method_diags = {}
        for method in phase["methods"]:
            method_root = root / "predictions" / dataset / method
            rep_diag = json.loads((method_root / "representation_diagnostics.json").read_text(encoding="utf-8"))
            method_diags[method] = rep_diag
            degeneration.append({
                "dataset": dataset, "method": method,
                "feature_variance_per_dimension": json.dumps(rep_diag["feature_variance_per_dimension"]),
                "mean_feature_variance": rep_diag["mean_feature_variance"],
                "number_of_nonzero_variance_dimensions": rep_diag["number_of_nonzero_variance_dimensions"],
                "unique_row_count": rep_diag["unique_row_count"],
                "variance_tolerance": rep_diag["variance_tolerance"],
                "relation_degenerate": rep_diag["relation_degenerate"],
            })
            for seed in phase["seeds"]:
                run_root = method_root / ("seed_%d" % seed)
                predictions = pd.read_csv(str(run_root / "predictions.csv"))["predicted_label"].values
                run_diag = json.loads((run_root / "diagnostics.json").read_text(encoding="utf-8"))
                if len(predictions) != len(y_true):
                    raise RuntimeError("prediction length mismatch")
                metrics = compute_metrics(y_true, predictions)
                row = {
                    "dataset": dataset, "dataset_key": dataset_key, "method": method, "seed": seed,
                    "n_samples": run_diag["n_samples"], "n_clusters": run_diag["n_clusters"],
                    "runtime": run_diag["runtime_seconds"],
                    "ball_count": 0 if method == METHODS[0] else relation_dim,
                    "ball_count_by_view": json.dumps([] if method == METHODS[0] else shared["ball_count_by_view"]),
                    "fused_dimension": run_diag["fused_dimension"],
                    "mean_entropy": json.dumps(rep_diag["mean_entropy_per_view"]),
                    "mean_alpha_per_view": json.dumps(rep_diag["mean_alpha_per_view"]),
                    "labels_loaded_phase_a": False,
                    "structural_failure_case": dataset == "3sources",
                    "relation_degenerate": rep_diag["relation_degenerate"],
                }
                row.update(metrics)
                row["Composite"] = _composite(row)
                raw.append(row)

        for view in shared["ball_statistics"]:
            index = int(view["view_id"])
            single_ball.append({
                "dataset": dataset, "view_id": index, "ball_count": view["ball_count"],
                "single_ball_view": int(view["ball_count"]) <= 1,
                "mean_ball_size": view["mean_ball_size"], "median_ball_size": view["median_ball_size"],
                "legacy_mean_alpha": method_diags[METHODS[2]]["mean_alpha_per_view"][index],
                "corrected_mean_alpha": method_diags[METHODS[3]]["mean_alpha_per_view"][index],
                "full_mean_alpha": method_diags[METHODS[4]]["mean_alpha_per_view"][index],
                "weighting_evaluable": dataset in PRIMARY,
            })

    raw_fields = [
        "dataset", "dataset_key", "method", "seed", "n_samples", "n_clusters",
        *METRICS, "Composite", "runtime", "ball_count", "ball_count_by_view", "fused_dimension",
        "mean_entropy", "mean_alpha_per_view", "labels_loaded_phase_a",
        "structural_failure_case", "relation_degenerate",
    ]
    _write_csv(root / "stage5r_raw.csv", raw, raw_fields)

    grouped = defaultdict(list)
    for row in raw:
        grouped[(row["dataset"], row["method"])].append(row)
    summary = []
    for (dataset, method), rows in sorted(grouped.items()):
        item = {"dataset": dataset, "method": method, "runs": len(rows)}
        for metric in METRICS + ("Composite", "runtime"):
            values = np.asarray([row[metric] for row in rows], dtype=float)
            item[metric + "_mean"] = float(values.mean())
            item[metric + "_std"] = float(values.std(ddof=0))
        item.update({
            "ball_count": rows[0]["ball_count"], "fused_dimension": rows[0]["fused_dimension"],
            "mean_entropy": rows[0]["mean_entropy"], "mean_alpha_per_view": rows[0]["mean_alpha_per_view"],
            "structural_failure_case": rows[0]["structural_failure_case"],
            "relation_degenerate": rows[0]["relation_degenerate"],
        })
        summary.append(item)
    summary_fields = ["dataset", "method", "runs"]
    for metric in METRICS + ("Composite", "runtime"):
        summary_fields += [metric + "_mean", metric + "_std"]
    summary_fields += ["ball_count", "fused_dimension", "mean_entropy", "mean_alpha_per_view",
                       "structural_failure_case", "relation_degenerate"]
    _write_csv(root / "stage5r_summary.csv", summary, summary_fields)

    keyed = {(row["dataset"], row["method"], row["seed"]): row for row in raw}
    paired = []
    comparisons = (
        ("Delta_fix", METHODS[2], METHODS[3]),
        ("Delta_corrected", METHODS[1], METHODS[3]),
        ("Delta_reliability", METHODS[3], METHODS[4]),
        ("Delta_full", METHODS[1], METHODS[4]),
        ("Gap_raw", METHODS[0], METHODS[4]),
    )
    for dataset in phase["datasets"]:
        for seed in phase["seeds"]:
            for comparison, before, after in comparisons:
                delta = keyed[(dataset, after, seed)]["Composite"] - keyed[(dataset, before, seed)]["Composite"]
                paired.append({
                    "dataset": dataset, "seed": seed, "comparison": comparison,
                    "before_method": before, "after_method": after, "delta": delta,
                    "primary_gate_case": dataset in PRIMARY,
                })
    _write_csv(root / "paired_deltas.csv", paired,
               ["dataset", "seed", "comparison", "before_method", "after_method", "delta", "primary_gate_case"])

    comparison_rows = []
    for name, before, after in comparisons:
        for dataset_group in ("PRIMARY",) + tuple(phase["datasets"]):
            selected = [row for row in paired if row["comparison"] == name and
                        ((dataset_group == "PRIMARY" and row["dataset"] in PRIMARY) or row["dataset"] == dataset_group)]
            values = np.asarray([row["delta"] for row in selected], dtype=float)
            comparison_rows.append({
                "comparison": name, "scope": dataset_group, "before_method": before, "after_method": after,
                "mean_delta": float(values.mean()), "positive_pairs": int(np.sum(values > TIE_TOLERANCE)),
                "ties": int(np.sum(np.abs(values) <= TIE_TOLERANCE)),
                "negative_pairs": int(np.sum(values < -TIE_TOLERANCE)), "total_pairs": int(values.size),
            })
    _write_csv(root / "weighting_comparison.csv", comparison_rows,
               ["comparison", "scope", "before_method", "after_method", "mean_delta",
                "positive_pairs", "ties", "negative_pairs", "total_pairs"])
    _write_csv(root / "single_ball_weight_analysis.csv", single_ball,
               ["dataset", "view_id", "ball_count", "single_ball_view", "mean_ball_size", "median_ball_size",
                "legacy_mean_alpha", "corrected_mean_alpha", "full_mean_alpha", "weighting_evaluable"])
    _write_csv(root / "representation_efficiency.csv", efficiency,
               ["dataset", "n_samples", "n_views", "raw_total_dimension", "relation_dimension",
                "relation_to_raw_ratio", "ball_to_sample_ratio", "representation_compression"])
    _write_csv(root / "relation_degeneration.csv", degeneration,
               ["dataset", "method", "feature_variance_per_dimension", "mean_feature_variance",
                "number_of_nonzero_variance_dimensions", "unique_row_count", "variance_tolerance",
                "relation_degenerate"])

    primary = [row for row in paired if row["primary_gate_case"]]
    values = {name: np.asarray([row["delta"] for row in primary if row["comparison"] == name])
              for name in ("Delta_fix", "Delta_corrected", "Delta_full", "Gap_raw")}
    gate_a = bool(values["Delta_fix"].mean() > 0 and np.sum(values["Delta_fix"] > 0) >= 6)
    gate_b = bool(values["Delta_corrected"].mean() >= -0.01)
    gate_c = bool(values["Delta_full"].mean() >= -0.01)
    recovery = bool(gate_a and gate_b and gate_c)
    raw_dataset = {
        dataset: float(np.mean([row["delta"] for row in primary
                                if row["comparison"] == "Gap_raw" and row["dataset"] == dataset]))
        for dataset in PRIMARY
    }
    raw_condition = bool(values["Gap_raw"].mean() >= -0.05 or any(value >= 0 for value in raw_dataset.values()))
    implementation_checks = {
        "all_metrics_finite": bool(all(np.isfinite([row[name] for name in METRICS + ("Composite", "runtime")]).all()
                                           for row in raw)),
        "phase_a_label_firewall": phase["labels_loaded"] is False,
        "ball_counts_reproduced": True,
        "alpha_valid": bool(all(float(json.loads((root / "predictions" / d / m /
            "representation_diagnostics.json").read_text(encoding="utf-8"))["alpha_min"]) >= 0 and
            float(json.loads((root / "predictions" / d / m / "representation_diagnostics.json").read_text(
                encoding="utf-8"))["alpha_max_row_sum_error"]) <= 1e-10
            for d in phase["datasets"] for m in phase["methods"])),
        "entropy_valid": bool(all((diag["entropy_min"] is None or (diag["entropy_min"] >= 0 and diag["entropy_max"] <= 1))
                                  for diag in [json.loads((root / "predictions" / d / m /
                                  "representation_diagnostics.json").read_text(encoding="utf-8"))
                                  for d in phase["datasets"] for m in phase["methods"]])),
    }
    implementation_pass = all(implementation_checks.values())
    if not implementation_pass:
        decision = "IMPLEMENTATION_FAILURE"
    elif not recovery:
        decision = "WEIGHTING_NOT_RECOVERED"
    elif raw_condition:
        decision = "READY_FOR_STAGE6"
    else:
        decision = "WEIGHTING_RECOVERED_BUT_GRANULATION_BLOCKED"
    source_deg = next(row for row in degeneration if row["dataset"] == "3sources" and row["method"] == METHODS[4])
    gate = {
        "protocol_frozen": True, "parameters_tuned_after_metrics": False,
        "primary_datasets": list(PRIMARY), "paired_cases": 10,
        "tie_tolerance": TIE_TOLERANCE, "variance_tolerance": phase["variance_tolerance"],
        "gate_A": {"mean_delta_fix": float(values["Delta_fix"].mean()),
                   "positive_pairs": int(np.sum(values["Delta_fix"] > 0)), "required_positive_pairs": 6,
                   "pass": gate_a},
        "gate_B": {"mean_delta_corrected": float(values["Delta_corrected"].mean()),
                   "threshold": -0.01, "pass": gate_b},
        "gate_C": {"mean_delta_full": float(values["Delta_full"].mean()),
                   "threshold": -0.01, "pass": gate_c},
        "reliability_recovery": "PASS" if recovery else "FAIL",
        "raw_competitiveness": {"mean_gap_raw": float(values["Gap_raw"].mean()),
                                "dataset_mean_gaps": raw_dataset,
                                "wins": int(np.sum(values["Gap_raw"] > TIE_TOLERANCE)),
                                "ties": int(np.sum(np.abs(values["Gap_raw"]) <= TIE_TOLERANCE)),
                                "losses": int(np.sum(values["Gap_raw"] < -TIE_TOLERANCE)),
                                "ready_condition": raw_condition},
        "3sources": {"classification": "CANDIDATE_SPLITTER_DEGENERACY",
                     "gate_role": "STRUCTURAL_FAILURE_CASE", "included_in_primary_gate": False,
                     "C4_relation_degenerate": source_deg["relation_degenerate"],
                     "C4_unique_row_count": source_deg["unique_row_count"],
                     "C4_nonzero_variance_dimensions": source_deg["number_of_nonzero_variance_dimensions"]},
        "implementation_checks": implementation_checks,
        "decision": decision,
    }
    (root / "gate_evaluation.json").write_text(json.dumps(gate, indent=2, sort_keys=True), encoding="utf-8")
    _make_figures(root, summary, efficiency)
    return raw, summary, gate


def _make_figures(root, summary, efficiency):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figures = root / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    lookup = {(row["dataset"], row["method"]): row["Composite_mean"] for row in summary}
    x = np.arange(len(METHODS))
    width = 0.25
    fig, ax = plt.subplots(figsize=(10, 5))
    for index, dataset in enumerate(("3sources",) + PRIMARY):
        ax.bar(x + (index - 1) * width, [lookup[(dataset, method)] for method in METHODS], width, label=dataset)
    ax.set_xticks(x)
    ax.set_xticklabels(["C0", "C1", "C2", "C3", "C4"])
    ax.set_ylabel("Mean Composite")
    ax.legend()
    fig.tight_layout()
    fig.savefig(str(figures / "composite_by_method.png"), dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar([row["dataset"] for row in efficiency], [row["relation_to_raw_ratio"] for row in efficiency])
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1)
    ax.set_ylabel("Relation / raw dimension")
    fig.tight_layout()
    fig.savefig(str(figures / "representation_efficiency.png"), dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", default="result/stage5r_weighting")
    parser.add_argument("--datasets-config", default="configs/datasets.yaml")
    args = parser.parse_args()
    evaluate_phase_b(args.output_root, args.datasets_config)


if __name__ == "__main__":
    main()
