import numpy as np

from bgrfmvc.core.granular_ball import build_granular_balls, make_granular_ball


METHOD = {
    "eta": 0.05, "min_parent_size": 10, "min_child_size": 5,
    "max_depth": 20, "eps": 1e-12, "radius_eps": 1e-12,
}
GRANULAR = {"split_n_init": 10, "split_max_iter": 100, "split_random_state": 0}


def test_terminal_balls_partition_all_samples_once(synthetic_views):
    X = synthetic_views[0]
    balls = build_granular_balls(X, METHOD, GRANULAR)
    combined = np.concatenate([ball.indices for ball in balls])
    assert combined.size == X.shape[0]
    np.testing.assert_array_equal(np.sort(combined), np.arange(X.shape[0]))
    assert np.unique(combined).size == X.shape[0]
    assert all(ball.size == len(ball.indices) for ball in balls)
    assert all(np.isfinite(ball.radius) and np.isfinite(ball.quality_q) for ball in balls)


def test_granularization_is_deterministic(synthetic_views):
    first = build_granular_balls(synthetic_views[0], METHOD, GRANULAR)
    second = build_granular_balls(synthetic_views[0], METHOD, GRANULAR)
    assert len(first) == len(second)
    for left, right in zip(first, second):
        np.testing.assert_array_equal(left.indices, right.indices)
        np.testing.assert_allclose(left.center, right.center)


def test_repeated_points_make_valid_zero_radius_ball():
    X = np.repeat([[2.0, -1.0]], repeats=5, axis=0)
    ball = make_granular_ball(X, np.arange(5), depth=2)
    assert ball.radius == 0.0
    assert ball.radial_std == 0.0
    assert ball.quality_q == 1.0

