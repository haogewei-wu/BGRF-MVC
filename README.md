# BGRF-MVC

BGRF-MVC is a research implementation of boundary-aware granular relation fusion for multi-view clustering. This GitHub-ready subset contains the method implementation, a reproducible synthetic demo dataset, and a Windows launcher.

## Quick start (Windows)

1. Install Python 3.9 or newer (select “Add Python to PATH”).
2. Double-click `run_demo.bat`. On its first run it creates a virtual environment, installs dependencies, and runs the demo.
3. Inspect `results/demo/` for cluster assignments and diagnostics.

Or run from a terminal:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -e .
python -m bgrfmvc.experiments.run_unsupervised --dataset demo --datasets-config configs/demo.yaml --method-config configs/default.yaml --output-dir results/demo --seed 0
```

## Included data

`datasets/demo_multiview.npz` contains two small, deterministic **synthetic** feature views (180 samples, two features per view). It is provided only to smoke-test the code and is not an empirical benchmark or a result reported in the paper. The archive did not contain the main `datasets-multiview` collection, so real benchmark data are not bundled here.

## Use your own data

The loader accepts `.mat`, `.npz`, `.npy`, and `.csv` feature files. All views must contain the same aligned samples. Create a YAML dataset manifest modeled on `configs/demo.yaml`, change `path`, `view_keys`, and `n_clusters`, and cite a source for the externally specified cluster count (`k_source`). Run the same module command with your manifest and a new dataset name. The unsupervised run loads only feature keys; ground-truth labels are not used by training or prediction.

Example for an NPZ containing arrays named `X1` and `X2`:

```yaml
datasets:
  my_data:
    enabled: true
    path: datasets/my_data.npz
    view_keys: [X1, X2]
    transpose_views: false
    n_clusters: 4
    k_source: "Please cite the external benchmark documentation here"
```

## Project contents

- `src/bgrfmvc/core/`: BGRF-MVC model and granular relation components.
- `src/bgrfmvc/data/loader.py`: multi-view feature loader, separated from evaluation labels.
- `configs/`: demo and method parameters.
- `datasets/`: synthetic smoke-test data only.
- `docs/METHOD_SPEC.md`: method specification included in the source archive.

## Notes before publication

This is a curated subset of the supplied research folder, not a claim that the complete original experiment environment is reproduced. Original third-party baseline repositories, their nested Git histories, large result dumps, intermediate artifacts, and benchmark datasets were omitted. Before publishing, add the appropriate license and dataset citations/terms for the code and any real datasets you decide to include.
