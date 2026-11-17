from pathlib import Path

import numpy as np
import yaml

from bgrfmvc.core.numerics import (
    SPLIT_KMEANS_CONFIG,
    boundary_relation,
    can_attempt_split,
    normalized_entropy,
    population_radial_std,
    strict_relation_probabilities,
    view_weights,
)


ROOT = Path(__file__).resolve().parents[1]


def test_strict_probability_normalization_and_positivity():
    relations = np.random.RandomState(0).random_sample((20, 7))
    probabilities = strict_relation_probabilities(relations)
    assert np.max(np.abs(probabilities.sum(axis=1) - 1.0)) < 1e-10
    assert np.all(probabilities > 0)


def test_entropy_single_ball_and_general_range():
    single = strict_relation_probabilities(np.array([[0.0], [2.0]]))
    np.testing.assert_array_equal(normalized_entropy(single), np.zeros(2))
    probabilities = strict_relation_probabilities(np.array([[1.0, 2.0, 3.0], [1.0, 1.0, 1.0]]))
    entropy = normalized_entropy(probabilities)
    assert np.all((entropy >= 0) & (entropy <= 1))


def test_view_weights_uniform_fallback_and_row_sum():
    uniform = view_weights(np.ones((2, 4)))
    np.testing.assert_allclose(uniform, np.full((2, 4), 0.25))
    weights = view_weights(np.array([[0.0, 0.3, 1.0], [0.8, 0.2, 0.1]]))
    np.testing.assert_allclose(weights.sum(axis=1), 1.0, atol=1e-12)
    assert np.all(weights >= 0)


def test_population_radial_std_uses_ddof_zero():
    distances = np.array([0.0, 1.0, 4.0, 8.0])
    assert population_radial_std(distances) == np.std(distances, ddof=0)
    assert population_radial_std(distances) != np.std(distances, ddof=1)


def test_zero_radius_rule():
    relations = boundary_relation(np.array([0.0, 1e-13, 1e-6]), radius=0.0)
    np.testing.assert_array_equal(relations, np.array([1.0, 1.0, 1e-12]))


def test_boundary_relation_is_numerically_stable():
    relations = boundary_relation(np.array([0.0, 1.0, 1e300]), radius=1.0)
    assert np.all(np.isfinite(relations))
    assert relations[0] > 0.5 and relations[1] == 0.5 and relations[2] < 0.5


def test_split_configuration_is_frozen():
    assert SPLIT_KMEANS_CONFIG == {"n_clusters": 2, "n_init": 10, "max_iter": 100, "random_state": 0}
    config = yaml.safe_load((ROOT / "configs/default.yaml").read_text(encoding="utf-8"))
    assert config["granular_ball"] == {
        "split_n_init": 10, "split_max_iter": 100,
        "split_random_state": 0, "radial_std_ddof": 0,
    }


def test_parent_child_size_guard():
    threshold = max(10, 2 * 6)
    assert not can_attempt_split(threshold - 1, 10, 6)
    assert can_attempt_split(threshold, 10, 6)


def test_frozen_spec_contains_v201_revisions():
    text = (ROOT / "docs/METHOD_SPEC.md").read_text(encoding="utf-8")
    for phrase in ("v2.0.1", "characteristic radius", "ddof = 0", "relation_floor", "random_state=0"):
        assert phrase in text

