"""Stage 5R-C Phase B: evaluate only after all predictions are on disk."""

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
from bgrfmvc.experiments.stage5r_hybrid import METHODS


METRICS = ("ACC", "NMI", "ARI", "PUR", "F_score")
PRIMARY = ("MSRCV1", "Handwritten_numerals")


def _write(path, rows, fields):
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerows(rows)


def _complete(root, phase):
    if phase.get("labels_loaded") is not False or not phase.get("phase_a_complete"):
        raise RuntimeError("Phase A incomplete")
    for d in phase["datasets"]:
        for m in phase["methods"]:
            for s in phase["seeds"]:
                if not (root / "predictions" / d / m / ("seed_%d" % s) / "predictions.csv").is_file():
                    raise RuntimeError("prediction set incomplete")


def evaluate(output_root, datasets_config):
    root = Path(output_root)
    phase = json.loads((root / "phase_a_manifest.json").read_text(encoding="utf-8"))
    _complete(root, phase)
    with Path(datasets_config).open("r", encoding="utf-8") as stream:
        manifest = yaml.safe_load(stream)["datasets"]
    configs = {}
    for slug in phase["datasets"]:
        matches = [(k, v) for k, v in manifest.items() if k.split("__")[-1] == slug and v.get("enabled")]
        if len(matches) != 1: raise RuntimeError("manifest mismatch")
        configs[slug] = matches[0]
    raw, diagnostics = [], []
    for dataset in phase["datasets"]:
        key, config = configs[dataset]
        y = load_evaluation_labels(config)
        shared = json.loads((root / "predictions" / dataset / "shared_diagnostics.json").read_text(encoding="utf-8"))
        for method in phase["methods"]:
            rep = json.loads((root / "predictions" / dataset / method /
                              "representation_diagnostics.json").read_text(encoding="utf-8"))
            diagnostics.append({
                "dataset": dataset, "method": method, "raw_dimension": rep["raw_dimension"],
                "relation_dimension": rep["relation_dimension"], "hybrid_dimension": rep["hybrid_dimension"],
                "relation_to_raw_ratio": rep["relation_to_raw_ratio"],
                "unique_row_count": rep["unique_row_count"],
                "nonzero_variance_dimensions": rep["nonzero_variance_dimensions"],
                "mean_feature_variance": rep["mean_feature_variance"],
                "ball_count_per_view": json.dumps([] if method == METHODS[0] else shared["ball_count_per_view"]),
                "labels_loaded_phase_a": False,
            })
            for seed in phase["seeds"]:
                run_root = root / "predictions" / dataset / method / ("seed_%d" % seed)
                pred = pd.read_csv(str(run_root / "predictions.csv"))["predicted_label"].values
                run_diag = json.loads((run_root / "diagnostics.json").read_text(encoding="utf-8"))
                metrics = compute_metrics(y, pred)
                row = {"dataset": dataset, "dataset_key": key, "method": method, "seed": seed,
                       "n_samples": len(y), "n_clusters": config["n_clusters"],
                       "runtime": run_diag["runtime"], "raw_dimension": rep["raw_dimension"],
                       "relation_dimension": rep["relation_dimension"], "hybrid_dimension": rep["hybrid_dimension"],
                       "relation_to_raw_ratio": rep["relation_to_raw_ratio"],
                       "unique_row_count": rep["unique_row_count"],
                       "nonzero_variance_dimensions": rep["nonzero_variance_dimensions"],
                       "mean_feature_variance": rep["mean_feature_variance"],
                       "ball_count_per_view": json.dumps([] if method == METHODS[0] else shared["ball_count_per_view"]),
                       "labels_loaded_phase_a": False}
                row.update(metrics); row["Composite"] = float(np.mean([metrics[m] for m in METRICS])); raw.append(row)
    fields = ["dataset", "dataset_key", "method", "seed", "n_samples", "n_clusters", *METRICS,
              "Composite", "runtime", "raw_dimension", "relation_dimension", "hybrid_dimension",
              "relation_to_raw_ratio", "unique_row_count", "nonzero_variance_dimensions",
              "mean_feature_variance", "ball_count_per_view", "labels_loaded_phase_a"]
    _write(root / "hybrid_raw.csv", raw, fields)
    grouped = defaultdict(list)
    for row in raw: grouped[(row["dataset"], row["method"])].append(row)
    summary = []
    for (dataset, method), rows in sorted(grouped.items()):
        item = {"dataset": dataset, "method": method, "runs": len(rows)}
        for metric in METRICS + ("Composite", "runtime"):
            values = np.asarray([r[metric] for r in rows]); item[metric + "_mean"] = float(values.mean()); item[metric + "_std"] = float(values.std(ddof=0))
        item.update({k: rows[0][k] for k in ("raw_dimension", "relation_dimension", "hybrid_dimension", "relation_to_raw_ratio")})
        summary.append(item)
    sf = ["dataset", "method", "runs"]
    for metric in METRICS + ("Composite", "runtime"): sf += [metric + "_mean", metric + "_std"]
    sf += ["raw_dimension", "relation_dimension", "hybrid_dimension", "relation_to_raw_ratio"]
    _write(root / "hybrid_summary.csv", summary, sf)
    keyed = {(r["dataset"], r["method"], r["seed"]): r for r in raw}
    comparisons = (("Delta_hybrid_raw", METHODS[0], METHODS[3]),
                   ("Delta_boundary_gaussian", METHODS[2], METHODS[3]),
                   ("Delta_hybrid_boundaryonly", METHODS[1], METHODS[3]))
    paired = []
    for dataset in phase["datasets"]:
        for seed in phase["seeds"]:
            for name, before, after in comparisons:
                paired.append({"dataset": dataset, "seed": seed, "comparison": name,
                               "before_method": before, "after_method": after,
                               "delta": keyed[(dataset, after, seed)]["Composite"] - keyed[(dataset, before, seed)]["Composite"],
                               "primary_gate_case": dataset in PRIMARY})
    _write(root / "paired_deltas.csv", paired,
           ["dataset", "seed", "comparison", "before_method", "after_method", "delta", "primary_gate_case"])
    _write(root / "representation_diagnostics.csv", diagnostics,
           ["dataset", "method", "raw_dimension", "relation_dimension", "hybrid_dimension",
            "relation_to_raw_ratio", "unique_row_count", "nonzero_variance_dimensions",
            "mean_feature_variance", "ball_count_per_view", "labels_loaded_phase_a"])
    safe = json.loads((root / "safe_fallback_phase_a.json").read_text(encoding="utf-8"))
    for item in safe:
        item["composite_difference_D3_minus_D0"] = keyed[("3sources", METHODS[3], item["seed"])]["Composite"] - keyed[("3sources", METHODS[0], item["seed"])]["Composite"]
    _write(root / "safe_fallback_analysis.csv", safe,
           ["dataset", "seed", "prediction_agreement", "pairwise_distance_correlation",
            "composite_difference_D3_minus_D0", "labels_loaded"])
    primary = [r for r in paired if r["primary_gate_case"]]
    vals = {name: np.asarray([r["delta"] for r in primary if r["comparison"] == name])
            for name, _, _ in comparisons}
    dataset_d3d0 = {d: float(np.mean([r["delta"] for r in primary if r["dataset"] == d and r["comparison"] == "Delta_hybrid_raw"])) for d in PRIMARY}
    criterion_a = bool(vals["Delta_hybrid_raw"].mean() >= -0.01)
    criterion_b = bool(any(v > 0 for v in dataset_d3d0.values()))
    criterion_c = bool(vals["Delta_boundary_gaussian"].mean() > 0 or np.sum(vals["Delta_boundary_gaussian"] > 0) >= 6)
    d3_source = next(r for r in diagnostics if r["dataset"] == "3sources" and r["method"] == METHODS[3])
    criterion_d = bool(int(d3_source["unique_row_count"]) > 1 and int(d3_source["nonzero_variance_dimensions"]) > 0)
    implementation = bool(all(np.isfinite([r[m] for m in METRICS + ("Composite", "runtime")]).all() for r in raw) and criterion_d)
    if not implementation: decision = "IMPLEMENTATION_FAILURE"
    elif not criterion_a or not criterion_b: decision = "HYBRID_NOT_EFFECTIVE"
    elif not criterion_c: decision = "HYBRID_WORKS_BOUNDARY_NOT_DISTINCT"
    else: decision = "READY_FOR_STAGE6"
    gate = {
        "criterion_A": {"mean_D3_minus_D0": float(vals["Delta_hybrid_raw"].mean()), "threshold": -0.01, "pass": criterion_a},
        "criterion_B": {"dataset_mean_D3_minus_D0": dataset_d3d0, "pass": criterion_b},
        "criterion_C": {"mean_D3_minus_D2": float(vals["Delta_boundary_gaussian"].mean()),
                        "positive_pairs": int(np.sum(vals["Delta_boundary_gaussian"] > 0)), "pass": criterion_c},
        "criterion_D": {"3sources_D3_unique_row_count": d3_source["unique_row_count"],
                        "3sources_D3_nonzero_variance_dimensions": d3_source["nonzero_variance_dimensions"], "pass": criterion_d},
        "mean_D3_minus_D1": float(vals["Delta_hybrid_boundaryonly"].mean()),
        "labels_loaded_phase_a": False, "parameter_search": False,
        "implementation_validation": "PASS" if implementation else "FAIL", "decision": decision,
    }
    (root / "gate_evaluation.json").write_text(json.dumps(gate, indent=2, sort_keys=True), encoding="utf-8")
    _figures(root, summary)
    return gate


def _figures(root, summary):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figures = root / "figures"; figures.mkdir(parents=True, exist_ok=True)
    lookup = {(r["dataset"], r["method"]): r for r in summary}
    datasets = ("3sources",) + PRIMARY
    specs = (("D0_vs_D3_metrics", (METHODS[0], METHODS[3])),
             ("D2_vs_D3", (METHODS[2], METHODS[3])))
    for filename, methods in specs:
        fig, ax = plt.subplots(figsize=(8, 4)); x = np.arange(len(datasets)); width = .35
        for i, method in enumerate(methods):
            ax.bar(x + (i-.5)*width, [lookup[(d, method)]["Composite_mean"] for d in datasets], width, label=method[:2])
        ax.set_xticks(x); ax.set_xticklabels(datasets); ax.set_ylabel("Mean Composite"); ax.legend(); fig.tight_layout(); fig.savefig(str(figures / (filename + ".png")), dpi=160); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4)); x=np.arange(len(datasets));
    ax.bar(x-.2, [lookup[(d,METHODS[0])]["raw_dimension"] for d in datasets], .2, label="raw")
    ax.bar(x, [lookup[(d,METHODS[3])]["relation_dimension"] for d in datasets], .2, label="relation")
    ax.bar(x+.2, [lookup[(d,METHODS[3])]["hybrid_dimension"] for d in datasets], .2, label="hybrid")
    ax.set_yscale("log"); ax.set_xticks(x); ax.set_xticklabels(datasets); ax.legend(); fig.tight_layout(); fig.savefig(str(figures / "representation_dimension.png"), dpi=160); plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 4)); x=np.arange(len(METHODS)); width=.25
    for i,d in enumerate(datasets): ax.bar(x+(i-1)*width,[lookup[(d,m)]["Composite_mean"] for m in METHODS],width,label=d)
    ax.set_xticks(x); ax.set_xticklabels(["D0","D1","D2","D3"]); ax.set_ylabel("Mean Composite"); ax.legend(); fig.tight_layout(); fig.savefig(str(figures / "dataset_composite.png"), dpi=160); plt.close(fig)


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--output-root",default="result/stage5r_hybrid"); parser.add_argument("--datasets-config",default="configs/datasets.yaml"); args=parser.parse_args(); evaluate(args.output_root,args.datasets_config)


if __name__ == "__main__": main()
