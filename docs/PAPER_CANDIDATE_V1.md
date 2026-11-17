# BGRF-MVC Paper Candidate v1.0

## Method Name

Boundary-aware Granular Relation Fusion for Multi-View Clustering (BGRF-MVC).
Boundary denotes a geometric modeling mechanism, not demonstrated superiority
over Gaussian relations.

## Core Research Problem

Preserve heterogeneous raw multi-view geometry while augmenting it with
continuous sample-to-granule structural relations in a label-free pipeline.

## Inputs

Aligned view matrices only. Cluster count K is externally frozen in the dataset
manifest. Ground-truth labels are not algorithm inputs.

## Preprocessing

Each view is independently standardized with StandardScaler. PCA is disabled.

## View-Specific Granularization

Deterministic recursive 2-means uses eta=0.02, min_parent_size=20,
min_child_size=10, max_depth=20, split n_init=10, split max_iter=100, and split
random_state=0.

## Granular Relation Representation

Each sample has a continuous relation to every terminal granule in its view.
This is retained as structural augmentation, not as a replacement for raw data.

## Boundary-Aware Relation

The frozen normalized radial position and stable sigmoid follow METHOD_SPEC.
tau=0.5, eps=radius_eps=relation_floor=1e-12, including the zero-radius rule.

## Ball Reliability

Ball reliability remains defined in METHOD_SPEC, but the frozen D3 paper
candidate uses the q-free Boundary block (`q=1`) established in Stage 5R-C.

## Safe Hybrid Representation

Within each view, independently L2-normalized raw and Boundary blocks receive
equal energy (each divided by sqrt(2)). A constant relation block therefore
cannot erase informative raw geometry.

## Multi-View Fusion

All views are equally weighted by 1/sqrt(V), concatenated, and row-L2-normalized.
No entropy or sample-specific weighting is used.

## Final KMeans

KMeans uses manifest K, n_init=20, max_iter=300, and experiment seed.

## Default Parameters

The paper candidate uses `configs/stage5r_candidate.yaml` for frozen D3
granulation and `configs/default.yaml` only for the unchanged KMeans settings
and seeds. Neither file was modified in Stage 7.

## Label Firewall

Phase A uses X, manifest K, and seeds only. Phase B reads saved predictions and
labels for evaluation and may not fit, transform, granularize, or rebuild a
representation.

## Supported Innovation 1

Granular Relation Representation — SUPPORTED as auxiliary continuous structure.

## Supported Innovation 2

Safe Hybrid Representation — SUPPORTED; 3sources exactly preserves D0 geometry
and predictions when its granular relations are non-informative.

## Interpretation-Only Mechanism

Boundary-aware geometry is `BOUNDARY_INTERPRETABILITY_ONLY`: geometry-aware,
boundary-sensitive, interpretable, and performance-comparable to Gaussian.

## Known Limitation 1

Boundary does not currently outperform Gaussian materially (D3-D2=+0.000334).

## Known Limitation 2

MSRCV1 exhibits negative transfer (D3-D0=-0.018574 in Stage 6).

## Known Limitation 3

Handwritten representation dimension expands from 649 to 1296.

## Known Limitation 4

Cold D3 construction remains expensive; deterministic granularization dominates.

## Optimized Implementation

D3-OPT uses content/config keyed process-local caching, seed-independent reuse,
matrix-form Euclidean distances, vectorized Boundary evaluation, and preallocated
hybrid fusion. `--no-cache`-equivalent rebuilding is available through
`use_cache=False`; cache state is never required for reproducibility.

## Mathematical Equivalence Statement

D3-OPT is an implementation of D3, not a new method. Ball membership and
statistics are identical, maximum observed relation difference is 3.97e-13,
maximum fused-representation difference is 7.18e-14, all predictions agree up
to permutation, and external metric maximum error is zero.

## Frozen Stage 8 Rules

Stage 8 may audit datasets, run main experiments and baselines, calculate
statistics, and visualize results. It must not change formulas, thresholds,
fallback, relations, weighting, granularization, D3 structure, default
parameters, or label isolation. A blocking scientific bug requires stopping the
formal experiment rather than silently revising the method.
