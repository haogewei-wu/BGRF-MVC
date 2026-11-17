import ast
from pathlib import Path

import numpy as np

from bgrfmvc.experiments.pilot_baselines import METHODS, build_pilot_representations


ROOT = Path(__file__).resolve().parents[1]


def test_all_pilot_representations_are_finite_and_share_balls(synthetic_views):
    import yaml
    config = yaml.safe_load((ROOT / "configs/default.yaml").read_text(encoding="utf-8"))
    results, shared = build_pilot_representations(synthetic_views, config)
    assert tuple(results) == METHODS
    assert shared["same_balls_for_B1_to_B5"] is True
    assert all(count >= 1 for count in shared["ball_count_by_view"])
    for name, payload in results.items():
        representation = payload["representation"]
        assert representation.shape[0] == synthetic_views[0].shape[0]
        assert np.all(np.isfinite(representation)), name
        assert payload["fused_dimension"] == representation.shape[1]
    shared_dimension = sum(shared["ball_count_by_view"])
    for method in METHODS[1:]:
        assert results[method]["fused_dimension"] == shared_dimension


def test_pilot_unsupervised_runner_has_no_label_dependency():
    path = ROOT / "src/bgrfmvc/experiments/run_pilot_unsupervised.py"
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imports = [node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
    assert not any("evaluation" in name for name in imports)
    assert "load_labels_for_evaluation" not in source


def test_pilot_evaluator_never_fits_or_transforms():
    path = ROOT / "src/bgrfmvc/evaluation/evaluate_pilot.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    forbidden = {"fit", "transform", "fit_transform", "fit_predict"}
    assert not any(
        isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in forbidden
        for node in ast.walk(tree)
    )

