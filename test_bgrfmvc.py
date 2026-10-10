import inspect
import json

import numpy as np

from bgrfmvc.core import BGRFMVC


def test_public_api_has_no_y_parameter():
    for name in ("fit", "transform", "fit_transform"):
        assert list(inspect.signature(getattr(BGRFMVC, name)).parameters) == ["self", "X_views"]
    assert list(inspect.signature(BGRFMVC.fit_predict).parameters) == [
        "self", "X_views", "n_clusters", "seed"
    ]


def test_end_to_end_synthetic_representation(synthetic_views):
    model = BGRFMVC()
    fused = model.fit_transform(synthetic_views)
    n_samples = synthetic_views[0].shape[0]
    expected_dimension = sum(len(balls) for balls in model.balls_)
    assert fused.shape == (n_samples, expected_dimension)
    assert expected_dimension < n_samples  # relation blocks, never an n-by-n sample graph
    assert np.all(np.isfinite(fused))
    assert np.all(model.alpha_ >= 0)
    np.testing.assert_allclose(model.alpha_.sum(axis=1), 1.0, atol=1e-12)
    assert not hasattr(model, "y") and not hasattr(model, "y_")
    json.dumps(model.diagnostics_)


def test_fit_predict_returns_one_label_per_sample(synthetic_views):
    labels = BGRFMVC().fit_predict(synthetic_views, n_clusters=2, seed=7)
    assert labels.shape == (synthetic_views[0].shape[0],)
    assert np.issubdtype(labels.dtype, np.integer)


def test_permuting_external_leakage_labels_cannot_change_fusion(synthetic_views):
    n_samples = synthetic_views[0].shape[0]
    y1 = np.arange(n_samples) % 2
    y2 = np.random.RandomState(9).permutation(y1)
    assert not np.array_equal(y1, y2)
    # Neither label vector can be supplied to the core API.
    first = BGRFMVC().fit_transform(synthetic_views)
    second = BGRFMVC().fit_transform([view.copy() for view in synthetic_views])
    np.testing.assert_array_equal(first, second)


def test_model_does_not_store_n_by_n_graph(synthetic_views):
    model = BGRFMVC().fit(synthetic_views)
    n_samples = synthetic_views[0].shape[0]
    for value in vars(model).values():
        if isinstance(value, np.ndarray):
            assert value.shape != (n_samples, n_samples)

