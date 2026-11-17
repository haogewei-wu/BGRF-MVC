# BGRF-MVC: Boundary-aware Granular Relation Fusion for Multi-View Clustering

Companion code for the manuscript **"Inspectable Multi-View Clustering via Continuous Granular-Ball Relations"**
(PeerJ Computer Science, manuscript ID-152667).

This README follows the section list requested by the editorial office: Title, Description,
Dataset Information, Code Information, Usage Instructions, Requirements, Methodology, Citations,
License & Contribution Guidelines. A closing section states the scope and known limitations of
this release.

---

## 1. Title

BGRF-MVC — Boundary-aware Granular Relation Fusion for Multi-View Clustering.

Package name: `bgrfmvc`. Current version: `2.0.1` (research prototype).

## 2. Description

BGRF-MVC is a multi-view clustering method that builds granular balls independently in each
view, describes every sample by its continuous relation to the ball it belongs to, and fuses the
per-view relations with reliability-derived weights before a final k-means partition.

The design goal is *inspectability*: every cluster assignment can be traced back to a ball, a
relation value and a per-view weight, rather than to an opaque embedding. Concretely, the
implementation exposes

- the granular balls built in each view (size, radius, quality `q`),
- the continuous boundary-relation value of every sample,
- the per-view fusion weight `alpha` and the per-sample relation entropy,
- a member-tracing record that maps cluster members back to source balls.

This repository contains the method implementation, its configuration files and its test suite.
It does **not** contain the third-party benchmark feature matrices or the frozen experiment
environment used to produce the published tables — see Section 3.

## 3. Dataset Information

**No benchmark feature matrices are redistributed in this repository.** The datasets used in the
manuscript are third-party benchmarks with their own distribution terms, and several are large;
redistributing them here would conflict with their terms of use.

- **Provenance of every dataset configuration** — sample counts, view counts, cluster counts, the
  feature-file path recorded in the project inventory, the identified public source (repository,
  file name or commit identifier) and a verification status — is documented in
  **Supplemental Data S2** of the submission (`Data_Provenance_Register.csv`). Rows marked
  `AUTHOR_CONFIRMATION_REQUIRED` are configurations whose original acquisition URL could not be
  verified and are marked as such rather than presented as confirmed sources.
- **Run-level results** behind the reported tables (30-run summaries, paired ablation
  differences, relation-contribution and margin-fidelity records, fixed-seed cases, manifest) are
  in **Supplemental Data S1** (`CSV` archive, plain UTF-8, no proprietary format).

### Synthetic smoke-test fixture

`datasets/demo_multiview.npz` holds a small synthetic fixture (180 samples, two views of two
features each, three generative groups) used to exercise the pipeline end to end and to
regression-test the numerics; `configs/demo.yaml` is its manifest.

**It is a smoke test only.** It is not one of the benchmark datasets, it is not used in any
reported experiment, and running it must not be presented as reproducing the empirical tables.
Any clustering metric computed on it is meaningless as evidence about method quality.

### Input format expected by the code

The loader expects a list of per-view matrices, each of shape `(n_samples, n_features)` with rows
in the same sample order across views:

```python
X_views = [X_view1, X_view2, ...]   # list of np.ndarray, float, n_samples identical
```

Optional HDF5-backed loading is available (`h5py` is a declared dependency) for datasets stored as
one array per view.

## 4. Code Information

```
src/bgrfmvc/
  core/
    model.py              BGRFMVC estimator: fit / transform / fit_transform / fit_predict
    granular_ball.py      ball construction, radius and quality q
    boundary_relation.py  continuous sample-to-ball relation
    reliability.py        per-view reliability and fusion weight alpha
    fusion.py             reliability-weighted relation fusion
    preprocessing.py      per-view standardisation (optional PCA)
    numerics.py           numerically guarded primitives
  data/
    loader.py             multi-view loading (arrays / HDF5)
    audit.py              dataset inventory checks
    preprocessing.py      dataset-level preparation helpers
  evaluation/
    labels.py             metric helpers
    evaluate_*.py         one evaluator per experiment stage
  experiments/
    run_*.py              stage runners
    plot_*.py             figure-generation scripts
    export_*.py           export of frozen result tables
configs/                  YAML configuration files (see Section 6); demo.yaml is the demo manifest
datasets/                 synthetic smoke-test fixture only (see Section 3)
tests/                    pytest suite (unit tests for each core component)
docs/                     method specification, protocol notes, changelog
```

`src/bgrfmvc/__init__.py` exports only `__version__`. The estimator is imported from the core
package:

```python
from bgrfmvc.core import BGRFMVC
```

## 5. Usage Instructions

### Install

```bash
git clone https://github.com/haogewei-wu/BGRF-MVC.git
cd BGRF-MVC
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e .
```

### Run the test suite

```bash
pytest -q
```

The suite covers ball construction, boundary relations, fusion, the data loader and the
label-firewall rule that keeps ground-truth labels out of the representation step. All 63 tests
pass on a clean install; the tests that consume the frozen benchmark output are not part of this
snapshot, because that output is not redistributed here.

### End-to-end demo

```bash
python -m bgrfmvc.experiments.run_unsupervised --dataset demo \
    --datasets-config configs/demo.yaml --method-config configs/default.yaml \
    --output-dir results/demo --seed 0
```

On Windows, `run_demo.bat` performs the same run inside a virtual environment. It writes
`ball_statistics.csv`, `diagnostics.json`, `predictions.csv`, `runtime.json`, `entropy.npy` and
`view_weights.npy` to `results/demo/`.

### Minimal example

```python
import numpy as np
from bgrfmvc.core import BGRFMVC

# one matrix per view, rows in the same sample order
X_views = [np.random.rand(200, 30), np.random.rand(200, 20)]

model = BGRFMVC()                       # loads configs/default.yaml
labels = model.fit_predict(X_views, n_clusters=5, seed=0)

print(labels.shape)                                  # (200,)
print(model.diagnostics_["ball_count_by_view"])       # balls per view
print(model.diagnostics_["alpha_mean_by_view"])       # per-view fusion weight
```

`fit_predict(X_views, n_clusters, seed)` requires the cluster count explicitly; the number of
clusters is never inferred from labels. To control behaviour, pass a configuration dictionary or a
path to a YAML file:

```python
model = BGRFMVC("configs/default.yaml")
```

`model.diagnostics_` also reports ball sizes, radii, quality `q`, per-view entropy statistics,
the fused representation dimension and the runtime.

### Reproduce the paper

Not from this repository alone. The reported tables were produced in a frozen environment that is
not part of this release; see Section 10.

## 6. Requirements

- Python >= 3.7
- `numpy >= 1.15`, `scipy >= 1.1`, `scikit-learn >= 0.19`
- `pandas >= 0.23`, `PyYAML >= 3.13`, `h5py >= 2.10`
- Tests: `pytest >= 3.8` (`pip install -e ".[test]"`)

Declared in `pyproject.toml`; `pip install -e .` installs everything except the test extras.

Default configuration (`configs/default.yaml`): `eta = 0.05`, `min_parent_size = 10`,
`min_child_size = 5`, `max_depth = 20`, `tau = 0.5`, ball reliability weighting on, per-view
standardisation on, PCA off, k-means `n_init = 20`.

## 7. Methodology

1. **Granulation.** Each view is standardised and recursively split into granular balls until a
   size or quality stopping rule is met. A ball is described by its centre, radius, size and
   quality `q`.
2. **Continuous boundary relation.** Instead of a hard member/outsider decision, every sample
   receives a continuous relation value to the ball it is assigned to, computed from its distance
   relative to the ball radius with a numerically guarded floor. Samples near a ball boundary are
   therefore described by an intermediate value rather than being forced to one side.
3. **View reliability.** Each view receives a reliability score derived from the dispersion of its
   ball structure; unreliable views are down-weighted rather than discarded.
4. **Fusion.** The per-view relations are combined into one representation with weights `alpha`
   obtained from the reliability scores. The fused representation preserves the per-view
   contribution, so a downstream inspection can attribute a cluster to specific views.
5. **Partition and tracing.** k-means is applied to the fused representation with an explicitly
   supplied cluster count. Members are then traced back to their source balls, which yields the
   inspectability record used in the manuscript's structural analyses.

`docs/METHOD_SPEC.md` states the formal specification; `docs/STAGE_5R_B_PROTOCOL.md` records the
evaluation protocol; `docs/LABEL_FIREWALL.md` states the rule that labels never enter the
representation step.

## 8. Citations

If you use this code, please cite the manuscript:

> He H, Xing K, Cai B, Wang W. Inspectable Multi-View Clustering via Continuous Granular-Ball
> Relations. PeerJ Computer Science. (in review)

The method builds on granular-ball computing and on multi-view clustering baselines whose original
implementations are **not redistributed here**; obtain them from their own public repositories and
cite those works according to their own instructions. The comparison results themselves are in
Supplemental Data S1.

## 9. License & Contribution Guidelines

**License.** The code in this repository is released for academic and research use in connection
with the associated manuscript. Third-party baseline implementations are not redistributed here
and are therefore not covered by any licence granted in this repository; the published results of
those baselines are cited in the manuscript and reproduced as records in Supplemental Data S1.

**Contributions.** This is a companion release for a specific manuscript, not a general-purpose
library, and it is not under active open development. Bug reports and questions about reproducing
a specific number are welcome as issues. Pull requests that change numerical behaviour will not be
merged without a corresponding change in the manuscript, because the code and the paper must not
diverge silently.

## 10. Scope and Limitations of This Release

These limitations are stated here explicitly rather than left implicit, and they match the
statements in the manuscript.

1. **This is a curated implementation release, not the environment that produced the published
   numbers.** The frozen benchmark environment, the intermediate array files and the full
   benchmarking harness are not part of this release. The code is therefore not a
   byte-for-byte reproduction path for the tables in the paper.
2. **Configuration versions differ.** The parameters in `configs/default.yaml` (for example
   `eta = 0.05`, `min_parent_size = 10`, reliability-weighted fusion) describe the released
   snapshot. They are not identical to the configuration described in the manuscript's Methods
   section. Numbers obtained here will not match the published tables exactly.
3. **No benchmark data is redistributed.** Provenance and public sources are documented in
   Supplemental Data S2; the values behind the reported results are in Supplemental Data S1.
4. **Six prototype-metadata reconstructions remain unresolved.** They are reported in the
   manuscript as reconstruction-pending instances and are excluded from the verified
   member-traceability claims.
5. **The synthetic fixture is a smoke test.** It must not be used to claim reproduction of the
   empirical results.

Please do not cite this repository as evidence that the published benchmark tables can be
regenerated from it.
