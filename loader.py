"""Strictly separated feature and evaluation-label loading."""

from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.io import loadmat


_LABEL_TOKENS = ("label", "target", "class", "truth")


def _is_label_name(key):
    name = str(key).lower()
    return name in {"y", "gnd", "gt"} or any(token in name for token in _LABEL_TOKENS)


def _path(config):
    path = Path(config["path"])
    if not path.is_absolute():
        path = Path(__file__).resolve().parents[3] / path
    return path


def _numeric_2d(value, key, transpose=False):
    if sparse.issparse(value):
        value = value.toarray()
    array = np.asarray(value)
    if array.dtype == object:
        raise TypeError("object/cell value requires an explicit cell view key")
    if array.ndim != 2:
        raise ValueError("view %s must be two-dimensional" % key)
    if not np.issubdtype(array.dtype, np.number):
        try:
            array = array.astype(float)
        except (TypeError, ValueError):
            raise TypeError("view %s is not numeric" % key)
    array = np.asarray(array.T if transpose else array, dtype=float)
    if array.shape[0] < 2 or array.shape[1] < 1:
        raise ValueError("view %s has invalid shape %r" % (key, array.shape))
    return array


def _expand_value(value, key, transpose=False):
    array = np.asarray(value)
    if array.dtype != object and array.ndim == 2:
        return [_numeric_2d(value, key, transpose)]
    if array.dtype != object and array.ndim == 3:
        view_axis = int(np.argmin(array.shape))
        return [
            _numeric_2d(item, "%s[%d]" % (key, index), transpose)
            for index, item in enumerate(np.moveaxis(array, view_axis, 0))
        ]
    if array.dtype != object:
        raise ValueError("view %s must be 2D or a 3D stack of 2D views" % key)
    views = []
    for index, item in enumerate(array.ravel()):
        views.append(_numeric_2d(item, "%s[%d]" % (key, index), transpose))
    if not views:
        raise ValueError("cell view %s is empty" % key)
    return views


def load_multiview_features(dataset_config):
    """Return only X_views; label-like keys are rejected and never loaded."""
    path = _path(dataset_config)
    view_keys = list(dataset_config.get("view_keys") or [])
    if not view_keys:
        raise ValueError("dataset_config must explicitly provide view_keys")
    if any(_is_label_name(key) for key in view_keys):
        raise ValueError("label-like keys cannot be feature view keys")
    transpose = dataset_config.get("transpose_views", False)
    suffix = path.suffix.lower()
    values = {}
    if suffix == ".mat":
        try:
            values = loadmat(str(path), variable_names=view_keys)
        except NotImplementedError:
            import h5py
            with h5py.File(str(path), "r") as container:
                values = {key: np.asarray(container[key]).T for key in view_keys}
    elif suffix == ".npz":
        with np.load(str(path), allow_pickle=False) as archive:
            values = {key: archive[key] for key in view_keys}
    elif suffix == ".npy":
        if len(view_keys) != 1:
            raise ValueError("a .npy dataset must declare exactly one view key")
        values = {view_keys[0]: np.load(str(path), allow_pickle=False)}
    elif suffix == ".csv":
        if len(view_keys) != 1:
            raise ValueError("a .csv dataset must declare exactly one view key")
        values = {view_keys[0]: pd.read_csv(str(path)).to_numpy()}
    else:
        raise ValueError("unsupported dataset format: %s" % suffix)
    missing = [key for key in view_keys if key not in values]
    if missing:
        raise KeyError("missing feature view keys: %s" % missing)
    X_views = []
    for index, key in enumerate(view_keys):
        do_transpose = transpose[index] if isinstance(transpose, list) else bool(transpose)
        X_views.extend(_expand_value(values[key], key, do_transpose))
    sample_counts = {view.shape[0] for view in X_views}
    if len(sample_counts) != 1:
        raise ValueError("all feature views must have the same sample count")
    return X_views


def load_labels_for_evaluation(dataset_config):
    """Load labels separately; only evaluation modules may call this function."""
    caller = inspect.currentframe().f_back
    caller_module = caller.f_globals.get("__name__", "") if caller is not None else ""
    if not caller_module.startswith("bgrfmvc.evaluation"):
        raise PermissionError("labels may only be loaded by bgrfmvc.evaluation modules")
    path = _path(dataset_config)
    label_key = dataset_config.get("label_key")
    if not label_key:
        raise ValueError("dataset_config must provide label_key for evaluation")
    suffix = path.suffix.lower()
    if suffix == ".mat":
        try:
            values = loadmat(str(path), variable_names=[label_key])
            if label_key not in values:
                raise KeyError("missing label key: %s" % label_key)
            labels = values[label_key]
        except NotImplementedError:
            import h5py
            with h5py.File(str(path), "r") as container:
                labels = np.asarray(container[label_key]).T
    elif suffix == ".npz":
        with np.load(str(path), allow_pickle=False) as archive:
            labels = archive[label_key]
    elif suffix == ".npy":
        raise ValueError("single-array .npy files cannot separately provide labels")
    elif suffix == ".csv":
        labels = pd.read_csv(str(path), usecols=[label_key]).to_numpy()
    else:
        raise ValueError("unsupported dataset format: %s" % suffix)
    return np.asarray(labels).reshape(-1)
