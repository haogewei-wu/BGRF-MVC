import ast
from pathlib import Path

import numpy as np

from bgrfmvc.experiments.stage5r_weighting import (
    corrected_entropy_weights, legacy_entropy_weights, representation_diagnostics,
)


ROOT = Path(__file__).resolve().parents[1]


def test_corrected_weight_penalizes_single_ball_without_labels():
    entropy = [np.zeros(4), np.full(4, 0.25), np.full(4, 0.5)]
    corrected = corrected_entropy_weights(entropy, [1, 5, 10])
    legacy = legacy_entropy_weights(entropy)
    assert np.all(corrected >= 0)
    assert np.allclose(corrected.sum(axis=1), 1.0, atol=1e-10)
    assert np.all(corrected[:, 0] < legacy[:, 0])


def test_relation_degeneration_rule():
    same = np.ones((7, 3))
    diag = representation_diagnostics(same)
    assert diag["relation_degenerate"] is True
    assert diag["number_of_nonzero_variance_dimensions"] == 0
    assert diag["unique_row_count"] == 1


def test_phase_a_has_no_evaluation_or_label_loader_import():
    path = ROOT / "src" / "bgrfmvc" / "experiments" / "run_stage5r_weighting.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.append(node.module or "")
    assert not any("evaluation" in name for name in imports)
    assert not any("labels" in name for name in imports)


def test_frozen_stage5r_parameters_and_counts_are_literal():
    import yaml
    candidate = yaml.safe_load((ROOT / "configs" / "stage5r_candidate.yaml").read_text(encoding="utf-8"))
    assert {k: candidate["method"][k] for k in
            ("eta", "min_parent_size", "min_child_size", "max_depth")} == {
                "eta": 0.02, "min_parent_size": 20, "min_child_size": 10, "max_depth": 20,
            }
    from bgrfmvc.experiments.run_stage5r_weighting import EXPECTED_COUNTS
    assert EXPECTED_COUNTS == {
        "3sources": [1, 1, 1],
        "MSRCV1": [11, 7, 14, 9, 12, 5],
        "Handwritten_numerals": [105, 116, 119, 75, 120, 112],
    }


def test_evaluator_does_not_fit_or_transform():
    path = ROOT / "src" / "bgrfmvc" / "evaluation" / "evaluate_stage5r_weighting.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    called = {node.func.attr for node in ast.walk(tree)
              if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)}
    assert not ({"fit", "fit_predict", "fit_transform", "transform"} & called)
