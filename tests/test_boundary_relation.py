import numpy as np

from bgrfmvc.core.boundary_relation import compute_boundary_relations
from bgrfmvc.core.granular_ball import build_granular_balls, make_granular_ball


METHOD = {
    "eta": 0.05, "min_parent_size": 10, "min_child_size": 5,
    "max_depth": 20, "eps": 1e-12, "radius_eps": 1e-12,
}
GRANULAR = {"split_n_init": 10, "split_max_iter": 100, "split_random_state": 0}


def test_relation_shapes_and_finite_values(synthetic_views):
    X = synthetic_views[0]
    balls = build_granular_balls(X, METHOD, GRANULAR)
    result = compute_boundary_relations(X, balls)
    expected = (X.shape[0], len(balls))
    assert result.distances.shape == expected
    assert result.deltas.shape == expected
    assert result.relations.shape == expected
    assert result.corrected_relations.shape == expected
    assert np.all(np.isfinite(result.relations))
    assert np.all(np.isfinite(result.corrected_relations))
    assert np.all(np.isfinite(result.distances))
    assert np.all(np.isfinite(result.deltas))
    assert np.all((result.relations > 0) & (result.relations <= 1))


def test_zero_radius_relation_rule_is_exact():
    members = np.repeat([[1.0, 2.0]], repeats=5, axis=0)
    ball = make_granular_ball(members, np.arange(5), depth=0)
    query = np.array([[1.0, 2.0], [1.0, 2.001]])
    result = compute_boundary_relations(query, [ball], relation_floor=1e-12)
    np.testing.assert_array_equal(result.relations[:, 0], np.array([1.0, 1e-12]))
    assert np.all(np.isfinite(result.deltas))
    assert np.all(np.isfinite(result.corrected_relations))
