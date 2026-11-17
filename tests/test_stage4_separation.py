import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from bgrfmvc.evaluation.evaluate_predictions import compute_metrics, evaluate_saved_predictions
from bgrfmvc.experiments.run_unsupervised import run_dataset


ROOT = Path(__file__).resolve().parents[1]


def test_unsupervised_runner_has_no_evaluation_or_label_loader_dependency():
    path = ROOT / "src/bgrfmvc/experiments/run_unsupervised.py"
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            imported.append(node.module or "")
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
    assert not any("evaluation" in name for name in imported)
    assert "load_labels_for_evaluation" not in source


def test_evaluator_does_not_fit_or_transform():
    path = ROOT / "src/bgrfmvc/evaluation/evaluate_predictions.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    assert not any(
        isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        and node.func.attr in {"fit", "transform", "fit_transform", "fit_predict"}
        for node in ast.walk(tree)
    )


def test_metric_values_for_permuted_perfect_partition():
    metrics = compute_metrics(np.array([0, 0, 1, 1]), np.array([7, 7, 3, 3]))
    for name in ("ACC", "NMI", "ARI", "PUR", "F_score"):
        assert metrics[name] == 1.0


def test_saved_prediction_evaluation_is_legacy_pandas_compatible(tmpdir):
    data_path = str(tmpdir.join("evaluation.npz"))
    prediction_path = str(tmpdir.join("predictions.csv"))
    metrics_path = str(tmpdir.join("metrics.json"))
    np.savez(data_path, X=np.arange(8).reshape(4, 2), y=np.array([0, 0, 1, 1]))
    pd.DataFrame({"predicted_label": [4, 4, 9, 9]}).to_csv(prediction_path, index=False)
    metrics = evaluate_saved_predictions(
        {"path": data_path, "label_key": "y"}, prediction_path, metrics_path
    )
    assert metrics["ACC"] == 1.0
    assert Path(metrics_path).is_file()


def test_unsupervised_artifacts_and_reproducibility(tmpdir, synthetic_views):
    data_path = str(tmpdir.join("features.npz"))
    np.savez(data_path, X1=synthetic_views[0], X2=synthetic_views[1])
    dataset_config = {
        "path": data_path,
        "view_keys": ["X1", "X2"],
        "transpose_views": False,
        "n_clusters": 2,
        "k_source": "synthetic test fixture",
        "enabled": True,
    }
    method_config = yaml.safe_load((ROOT / "configs/default.yaml").read_text(encoding="utf-8"))
    first_dir = Path(str(tmpdir.mkdir("first")))
    second_dir = Path(str(tmpdir.mkdir("second")))
    first, diagnostics, runtime = run_dataset("synthetic", dataset_config, method_config, first_dir, seed=0)
    second, _, _ = run_dataset("synthetic", dataset_config, method_config, second_dir, seed=0)
    np.testing.assert_array_equal(first, second)
    assert diagnostics["n_clusters"] == 2
    assert diagnostics["fused_rows"] == synthetic_views[0].shape[0]
    assert runtime["total_seconds"] >= 0
    for name in ("predictions.csv", "diagnostics.json", "runtime.json", "view_weights.npy",
                 "entropy.npy", "ball_statistics.csv"):
        assert (first_dir / name).is_file()
    weights = np.load(str(first_dir / "view_weights.npy"))
    entropy = np.load(str(first_dir / "entropy.npy"))
    np.testing.assert_allclose(weights.sum(axis=1), 1.0, atol=1e-10)
    assert np.all(np.isfinite(weights)) and np.all(np.isfinite(entropy))
    assert len(pd.read_csv(str(first_dir / "predictions.csv"))) == synthetic_views[0].shape[0]
    json.loads((first_dir / "diagnostics.json").read_text(encoding="utf-8"))


def test_runner_rejects_unfrozen_or_disabled_k(tmpdir, synthetic_views):
    data_path = str(tmpdir.join("features.npz"))
    np.savez(data_path, X=synthetic_views[0])
    config = yaml.safe_load((ROOT / "configs/default.yaml").read_text(encoding="utf-8"))
    for dataset_config in (
        {"path": data_path, "view_keys": ["X"], "enabled": False, "n_clusters": 2, "k_source": "x"},
        {"path": data_path, "view_keys": ["X"], "enabled": True, "n_clusters": None, "k_source": None},
    ):
        try:
            run_dataset("blocked", dataset_config, config, str(tmpdir.mkdir("out" + str(id(dataset_config)))), 0)
        except ValueError:
            pass
        else:
            raise AssertionError("runner accepted a blocked/unfrozen dataset")
