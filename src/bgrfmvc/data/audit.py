"""Read-only dataset inventory for Stage 2.

The auditor never loads label values. Label presence and shape come from container
metadata only, and all eligibility diagnostics are computed from feature arrays.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np
import yaml
from scipy import sparse
from scipy.io import loadmat, whosmat


SUPPORTED = {".mat", ".npz", ".npy", ".csv"}
LABEL_PATTERN = re.compile(r"(^y$|label|target|class|truth|^gnd$|^gt$)", re.I)
FEATURE_PATTERN = re.compile(r"^(x|xs|x\d+|fea|fea\d+|features?|views?|data|k.*|kernels?)$", re.I)
FIELDS = [
    "dataset", "file_path", "file_type", "n_samples", "n_views", "view_keys",
    "view_dimensions", "contains_label_key", "label_shape", "status", "notes",
]


def _dataset_id(root: Path, path: Path) -> str:
    relative = path.relative_to(root).with_suffix("")
    return "__".join(relative.parts)


def _expand_feature(value):
    if sparse.issparse(value):
        return [(value, True, False)]
    array = np.asarray(value)
    if array.dtype != object:
        return [(array, False, False)]
    expanded = []
    for item in array.ravel():
        expanded.append((item, sparse.issparse(item), True))
    return expanded


def _choose_orientation(arrays):
    shapes = [np.asarray(item.toarray() if sparse.issparse(item) else item).shape for item in arrays]
    if not shapes or any(len(shape) != 2 for shape in shapes):
        return False, "non-2D feature view"
    rows = {shape[0] for shape in shapes}
    columns = {shape[1] for shape in shapes}
    if len(rows) == 1:
        return False, "sample axis inferred from aligned rows"
    if len(columns) == 1:
        return True, "transpose required; sample axis inferred from aligned columns"
    return False, "unaligned view sample dimensions"


def _feature_stats(arrays, transpose):
    dimensions = []
    nan_count = 0
    inf_count = 0
    constant = 0
    feature_count = 0
    numeric = True
    for item in arrays:
        if sparse.issparse(item):
            item = item.toarray()
        array = np.asarray(item)
        if array.ndim != 2:
            return None, None, None, None, False, "view is not two-dimensional"
        if not np.issubdtype(array.dtype, np.number):
            try:
                array = array.astype(float)
            except (TypeError, ValueError):
                numeric = False
                break
        array = np.asarray(array.T if transpose else array, dtype=float)
        dimensions.append([int(array.shape[0]), int(array.shape[1])])
        nan_count += int(np.isnan(array).sum())
        inf_count += int(np.isinf(array).sum())
        finite = np.where(np.isfinite(array), array, np.nan)
        with np.errstate(all="ignore"):
            maxima = np.nanmax(finite, axis=0)
            minima = np.nanmin(finite, axis=0)
        constant += int(np.sum(maxima == minima))
        feature_count += array.shape[1]
    if not numeric:
        return None, None, None, None, False, "feature dtype is not numeric-convertible"
    counts = {shape[0] for shape in dimensions}
    aligned = len(counts) == 1
    n_samples = next(iter(counts)) if aligned else None
    ratio = float(constant / feature_count) if feature_count else None
    return n_samples, dimensions, nan_count, inf_count, aligned, "constant_column_ratio=%.8g" % ratio


def _mat_metadata(path):
    try:
        metadata = whosmat(str(path))
    except NotImplementedError:
        return _hdf_mat_metadata(path)
    labels = [(name, shape) for name, shape, _ in metadata if LABEL_PATTERN.search(name)]
    candidates = []
    for name, shape, class_name in metadata:
        if LABEL_PATTERN.search(name) or name.startswith("__"):
            continue
        if class_name == "cell" or FEATURE_PATTERN.match(name):
            candidates.append(name)
    if not candidates:
        numeric = [(name, shape) for name, shape, cls in metadata
                   if len(shape) == 2 and min(shape) > 1 and cls not in {"char", "struct", "cell"}
                   and not LABEL_PATTERN.search(name)]
        candidates = [item[0] for item in numeric]
    if not candidates:
        raise ValueError("no explicit feature/view key identified")
    # variable_names guarantees that label arrays are not read.
    payload = loadmat(str(path), variable_names=candidates)
    arrays = []
    keys = []
    sparse_seen = False
    cell_seen = False
    for name in candidates:
        if name not in payload:
            continue
        expanded = _expand_feature(payload[name])
        for index, (item, is_sparse, is_cell) in enumerate(expanded):
            arrays.append(item)
            keys.append(name if len(expanded) == 1 else "%s[%d]" % (name, index))
            sparse_seen |= is_sparse
            cell_seen |= is_cell
    return arrays, keys, labels, sparse_seen, cell_seen, candidates


def _hdf_mat_metadata(path):
    import h5py

    arrays = []
    expanded_keys = []
    manifest_keys = []
    labels = []
    with h5py.File(str(path), "r") as container:
        names = [name for name in container.keys() if name != "#refs#"]
        for name in names:
            item = container[name]
            if LABEL_PATTERN.search(name):
                labels.append((name, tuple(reversed(item.shape))))
                continue
            if not FEATURE_PATTERN.match(name):
                continue
            value = np.asarray(item)
            manifest_keys.append(name)
            if value.ndim == 2:
                arrays.append(value.T)
                expanded_keys.append(name)
            elif value.ndim == 3:
                # A stack of 2D kernels/views is expanded without mixing axes.
                view_axis = int(np.argmin(value.shape))
                moved = np.moveaxis(value, view_axis, 0)
                for index, view in enumerate(moved):
                    arrays.append(view.T)
                    expanded_keys.append("%s[%d]" % (name, index))
    if not arrays:
        raise ValueError("no explicit HDF5 feature/view key identified")
    return arrays, expanded_keys, labels, False, False, manifest_keys


def inspect_file(root: Path, path: Path):
    row = {field: "" for field in FIELDS}
    row.update(dataset=_dataset_id(root, path), file_path=path.relative_to(root.parent).as_posix(),
               file_type=path.suffix.lower().lstrip("."), contains_label_key=False)
    notes = []
    try:
        if path.suffix.lower() == ".mat":
            arrays, expanded_keys, labels, sparse_seen, cell_seen, manifest_keys = _mat_metadata(path)
            row["contains_label_key"] = bool(labels)
            row["label_shape"] = json.dumps({name: list(shape) for name, shape in labels})
            if cell_seen:
                notes.append("MATLAB object/cell feature container expanded")
            if sparse_seen:
                notes.append("sparse feature view convertible to dense")
        elif path.suffix.lower() == ".npz":
            with np.load(str(path), allow_pickle=False) as archive:
                names = list(archive.keys())
                label_names = [name for name in names if LABEL_PATTERN.match(name)]
                feature_names = [name for name in names if FEATURE_PATTERN.match(name)]
                arrays = [archive[name] for name in feature_names]
                expanded_keys = feature_names
                manifest_keys = feature_names
                row["contains_label_key"] = bool(label_names)
                row["label_shape"] = json.dumps({name: list(archive[name].shape) for name in label_names})
        elif path.suffix.lower() == ".npy":
            arrays = [np.load(str(path), mmap_mode="r", allow_pickle=False)]
            expanded_keys = manifest_keys = ["array"]
        else:
            import pandas as pd
            frame = pd.read_csv(str(path))
            label_names = [name for name in frame.columns if LABEL_PATTERN.match(str(name))]
            feature_names = [name for name in frame.columns if name not in label_names]
            arrays = [frame[feature_names].to_numpy()]
            expanded_keys = manifest_keys = ["features"]
            row["contains_label_key"] = bool(label_names)
            row["label_shape"] = json.dumps({name: [len(frame)] for name in label_names})
        if not arrays:
            raise ValueError("no feature arrays identified")
        transpose, orientation_note = _choose_orientation(arrays)
        notes.append(orientation_note)
        n, dimensions, nan_count, inf_count, aligned, constant_note = _feature_stats(arrays, transpose)
        notes.append(constant_note)
        notes.extend(["nan_count=%s" % nan_count, "inf_count=%s" % inf_count])
        row.update(n_samples=n if n is not None else "", n_views=len(arrays),
                   view_keys=json.dumps(expanded_keys), view_dimensions=json.dumps(dimensions or []))
        blockers = []
        if not aligned:
            blockers.append("views not sample-aligned")
        if n is None or n < 2:
            blockers.append("fewer than two aligned samples")
        if dimensions and any(shape[1] < 1 for shape in dimensions):
            blockers.append("empty feature dimension")
        if nan_count:
            blockers.append("NaN prevents algorithm run")
        if inf_count:
            blockers.append("Inf prevents algorithm run")
        # K is deliberately unknown until externally frozen in the manifest.
        blockers.append("n_clusters missing external benchmark source")
        row["status"] = "blocked"
        notes.extend(blockers)
        row["notes"] = "; ".join(notes)
        manifest = {
            "path": row["file_path"],
            "view_keys": manifest_keys,
            "label_key": labels[0][0] if path.suffix.lower() == ".mat" and labels else None,
            "transpose_views": transpose,
            "n_clusters": None,
            "k_source": None,
            "k_source_note": None,
            "enabled": False,
            "notes": row["notes"],
        }
    except Exception as exc:
        row["status"] = "blocked"
        row["notes"] = "parse/read blocked: %s" % exc
        manifest = {
            "path": row["file_path"], "view_keys": [], "label_key": None,
            "transpose_views": False, "n_clusters": None, "k_source": None,
            "k_source_note": None, "enabled": False, "notes": row["notes"],
        }
    return row, manifest


def run_audit(project_root: Path):
    data_root = project_root / "datasets-multiview"
    output_root = project_root / "result" / "stage2_data_audit"
    output_root.mkdir(parents=True, exist_ok=True)
    files = sorted(path for path in data_root.rglob("*")
                   if path.is_file() and not path.name.startswith("._") and path.suffix.lower() in SUPPORTED)
    rows = []
    manifest = {}
    for path in files:
        row, config = inspect_file(data_root, path)
        rows.append(row)
        manifest[row["dataset"]] = config
    for name, selected in (("dataset_inventory.csv", rows),
                           ("eligible_datasets.csv", [row for row in rows if row["status"] == "eligible"]),
                           ("blocked_datasets.csv", [row for row in rows if row["status"] != "eligible"])):
        with (output_root / name).open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(selected)
    with (project_root / "configs" / "datasets.yaml").open("w", encoding="utf-8", newline="\n") as stream:
        stream.write("# Stage 2 read-only audit manifest. K is never inferred from label values.\n")
        yaml.safe_dump({"datasets": manifest}, stream, allow_unicode=True)
    return len(rows), sum(row["status"] == "eligible" for row in rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    args = parser.parse_args()
    total, eligible = run_audit(args.project_root.resolve())
    print("audited=%d eligible=%d blocked=%d" % (total, eligible, total - eligible))
