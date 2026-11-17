"""Evaluate saved predictions after unsupervised model execution has ended."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, f1_score, normalized_mutual_info_score

from bgrfmvc.evaluation.labels import load_evaluation_labels


def _aligned_predictions(y_true, y_pred):
    true_values, true_inverse = np.unique(y_true, return_inverse=True)
    pred_values, pred_inverse = np.unique(y_pred, return_inverse=True)
    contingency = np.zeros((len(true_values), len(pred_values)), dtype=np.int64)
    np.add.at(contingency, (true_inverse, pred_inverse), 1)
    rows, columns = linear_sum_assignment(contingency.max() - contingency)
    mapping = {pred_values[column]: true_values[row] for row, column in zip(rows, columns)}
    # Unmatched predicted clusters remain distinct and therefore incorrect.
    fallback = object()
    aligned = np.array([mapping.get(value, fallback) for value in y_pred], dtype=object)
    return aligned, contingency


def compute_metrics(y_true, y_pred):
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    if y_true.shape != y_pred.shape:
        raise ValueError("ground truth and predictions must have equal length")
    aligned, contingency = _aligned_predictions(y_true, y_pred)
    accuracy = float(np.mean(aligned == y_true))
    purity = float(np.sum(np.max(contingency, axis=0)) / y_true.size)
    # Macro F-score is calculated after the same one-to-one assignment used for ACC.
    aligned_strings = np.array([str(item) for item in aligned])
    true_strings = np.array([str(item) for item in y_true])
    return {
        "ACC": accuracy,
        "NMI": float(normalized_mutual_info_score(y_true, y_pred)),
        "ARI": float(adjusted_rand_score(y_true, y_pred)),
        "PUR": purity,
        "F_score": float(f1_score(true_strings, aligned_strings, average="macro")),
    }


def evaluate_saved_predictions(dataset_config, predictions_path, metrics_path):
    """Load evaluation-only labels and saved predictions; never fit or transform."""
    y_true = load_evaluation_labels(dataset_config)
    frame = pd.read_csv(str(predictions_path))
    if list(frame.columns) != ["predicted_label"]:
        raise ValueError("predictions.csv must contain only predicted_label")
    metrics = compute_metrics(y_true, frame["predicted_label"].values)
    with Path(metrics_path).open("w", encoding="utf-8") as stream:
        json.dump(metrics, stream, indent=2, sort_keys=True)
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--datasets-config", default="configs/datasets.yaml")
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    with Path(args.datasets_config).open("r", encoding="utf-8") as stream:
        manifest = yaml.safe_load(stream)["datasets"]
    evaluate_saved_predictions(manifest[args.dataset], args.predictions, args.output)


if __name__ == "__main__":
    main()
