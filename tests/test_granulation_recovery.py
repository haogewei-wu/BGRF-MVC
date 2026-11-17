import ast
import inspect
from pathlib import Path

import numpy as np

from bgrfmvc.core.granular_ball import build_granular_balls
from bgrfmvc.experiments.granulation_recovery import candidate_grid, rank_candidates


ROOT = Path(__file__).resolve().parents[1]


def test_recovery_script_has_no_evaluation_data_import():
    path = ROOT / "src/bgrfmvc/experiments/granulation_recovery.py"
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imports = [node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
    assert not any("evaluation" in name for name in imports)
    assert "load_labels_for_evaluation" not in source
    assert "load_evaluation_labels" not in source


def test_selection_rule_uses_only_structure_statistics():
    source = inspect.getsource(rank_candidates)
    assert "collapse_count" in source and "total_ball_count" in source
    assert "eta" in source and "min_child_size" in source
    assert "metric" not in source.lower()
    rows = [
        {"candidate_collapse_count": 0, "total_ball_count": 20, "eta": 0.01, "min_child_size": 5},
        {"candidate_collapse_count": 0, "total_ball_count": 10, "eta": 0.005, "min_child_size": 5},
        {"candidate_collapse_count": 1, "total_ball_count": 3, "eta": 0.05, "min_child_size": 10},
    ]
    assert rank_candidates(rows)[0]["total_ball_count"] == 10


def test_grid_is_exact_and_parent_guard_is_derived():
    grid = candidate_grid()
    assert len(grid) == 10
    assert {row["eta"] for row in grid} == {0.05, 0.03, 0.02, 0.01, 0.005}
    assert {row["min_child_size"] for row in grid} == {5, 10}
    assert all(row["min_parent_size"] == max(10, 2 * row["min_child_size"]) for row in grid)


def test_same_parameters_repeat_identical_ball_counts(synthetic_views):
    method = {
        "eta": 0.02, "min_parent_size": 10, "min_child_size": 5,
        "max_depth": 20, "eps": 1e-12, "radius_eps": 1e-12,
    }
    split = {"split_n_init": 10, "split_max_iter": 100, "split_random_state": 0}
    first = [len(build_granular_balls(view, method, split)) for view in synthetic_views]
    second = [len(build_granular_balls(view, method, split)) for view in synthetic_views]
    assert first == second
    assert split["split_random_state"] == 0

