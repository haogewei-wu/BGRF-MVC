import numpy as np

from bgrfmvc.core.reliability import (
    relation_probability_and_entropy,
    sample_specific_view_weights,
)


def test_probability_and_entropy_contracts():
    relations = np.array([[0.0, 0.5, 1.0], [1e-12, 1e-6, 100.0]])
    probabilities, entropy = relation_probability_and_entropy(relations)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=1e-12)
    assert np.all(probabilities > 0)
    assert np.all(np.isfinite(entropy))
    assert np.all((entropy >= 0) & (entropy <= 1))


def test_single_ball_entropy_is_zero():
    _, entropy = relation_probability_and_entropy(np.ones((8, 1)))
    np.testing.assert_array_equal(entropy, np.zeros(8))


def test_sample_specific_alpha_is_valid():
    alpha = sample_specific_view_weights([
        np.array([0.0, 0.3, 1.0]),
        np.array([0.8, 0.2, 1.0]),
    ])
    assert np.all(alpha >= 0)
    np.testing.assert_allclose(alpha.sum(axis=1), 1.0, atol=1e-12)
    np.testing.assert_allclose(alpha[2], np.array([0.5, 0.5]))

