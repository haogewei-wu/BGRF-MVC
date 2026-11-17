# Label Firewall Protocol

## Rule

Ground-truth labels `y` may enter only the final `evaluation` module, after predicted cluster assignments have been produced. They exist solely for computing evaluation metrics.

The following operations must never access `y`:

- `fit` and `transform`;
- `StandardScaler`;
- granular-ball generation and 2-means splitting;
- center and radius calculation;
- population radial standard deviation calculation;
- ball-quality/reliability calculation;
- boundary-relation and probability-relation calculation;
- entropy calculation;
- sample-level view weighting;
- relation-level fusion;
- KMeans initialization and fitting;
- parameter selection;
- dataset selection.

## Core API boundary

Forbidden core signatures:

```python
fit(X, y)
fit_transform(X, y)
```

Required core signatures:

```python
fit(X_views)
transform(X_views)
fit_transform(X_views)
fit_predict(X_views, n_clusters, seed)
```

`n_clusters` is always an explicit argument sourced from `configs/datasets.yaml` or another pre-frozen dataset registry. The core package must not accept, retain, inspect, or infer anything from ground-truth labels. Evaluation code may accept `y_true` together with already-produced predictions, but it must not call back into preprocessing, fitting, transformation, fusion, or clustering.

## Enforcement

- Keep evaluation code under `src/bgrfmvc/evaluation/` and algorithm code under `src/bgrfmvc/core/`.
- Load labels separately from views at the experiment boundary.
- Only modules under `src/bgrfmvc/evaluation/` may import and call `load_labels_for_evaluation`.
- `src/bgrfmvc/core/` and unsupervised experiment runners must never import the evaluation label loader.
- Produce predictions before passing labels to evaluation.
- Freeze dataset identifiers, cluster counts, parameters, and seed lists before evaluation.
- Treat any label access before final evaluation as a protocol violation and invalidate that run.
