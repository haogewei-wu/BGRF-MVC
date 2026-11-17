import inspect

import numpy as np

from bgrfmvc.data.loader import load_labels_for_evaluation, load_multiview_features
from bgrfmvc.data.preprocessing import standardize_views


def test_feature_loader_returns_features_only(tmpdir):
    path = str(tmpdir.join("sample.npz"))
    X1 = np.arange(12, dtype=float).reshape(4, 3)
    X2 = np.arange(8, dtype=float).reshape(4, 2)
    np.savez(path, X1=X1, X2=X2, y=np.array([0, 1, 0, 1]))
    loaded = load_multiview_features({"path": str(path), "view_keys": ["X1", "X2"]})
    assert isinstance(loaded, list) and len(loaded) == 2
    assert all(isinstance(view, np.ndarray) for view in loaded)
    np.testing.assert_array_equal(loaded[0], X1)
    np.testing.assert_array_equal(loaded[1], X2)


def test_shuffled_labels_cannot_change_loaded_features(tmpdir):
    X = np.arange(20, dtype=float).reshape(5, 4)
    first = str(tmpdir.join("first.npz"))
    second = str(tmpdir.join("second.npz"))
    np.savez(first, X=X, y=np.array([0, 1, 2, 3, 4]))
    np.savez(second, X=X, y=np.array([3, 0, 4, 1, 2]))
    loaded1 = load_multiview_features({"path": str(first), "view_keys": ["X"]})
    loaded2 = load_multiview_features({"path": str(second), "view_keys": ["X"]})
    np.testing.assert_array_equal(loaded1[0], loaded2[0])


def test_standardizer_api_accepts_only_features():
    assert list(inspect.signature(standardize_views).parameters) == ["X_views"]
    result = standardize_views([np.array([[1.0], [2.0], [3.0]])])
    assert result[0].shape == (3, 1)


def test_loader_rejects_label_key_as_view(tmpdir):
    path = str(tmpdir.join("sample.npz"))
    np.savez(path, y=np.array([[0], [1]]))
    try:
        load_multiview_features({"path": str(path), "view_keys": ["y"]})
    except ValueError as exc:
        assert "label-like" in str(exc)
    else:
        raise AssertionError("label key was accepted as a feature")


def test_label_loader_rejects_non_evaluation_caller(tmpdir):
    path = str(tmpdir.join("sample.npz"))
    np.savez(path, y=np.array([0, 1]))
    try:
        load_labels_for_evaluation({"path": path, "label_key": "y"})
    except PermissionError as exc:
        assert "evaluation" in str(exc)
    else:
        raise AssertionError("non-evaluation caller loaded ground-truth labels")
