# Stage 5R-B Protocol (Revised and Frozen)

## Protocol Status

- Status: `FROZEN_NOT_EXECUTED`
- Basis: Stage 5R-A label-free structural diagnostics
- Revision type: pre-registered protocol revision, not metric-driven tuning
- Stage 5R-B execution: not started
- Stage 6 execution: not authorized

This document supersedes any earlier informal Stage 5R-B plan. It does not
change the BGRF-MVC core algorithm, the boundary relation, entropy, view
weighting, the deterministic 2-means candidate splitter, or the frozen default
configuration.

## 1. Frozen Stage 5R-A Facts

Stage 5R-A ended with `PARTIAL_RECOVERY`. Its selected granulation candidate is
frozen for Stage 5R-B:

```yaml
eta: 0.02
min_parent_size: 20
min_child_size: 10
max_depth: 20
```

No further search over these parameters is permitted in Stage 5R-B.

The final per-view ball counts are:

| Dataset | Per-view ball counts | Total balls |
|---|---:|---:|
| 3sources | `[1, 1, 1]` | 3 |
| MSRCV1 | `[11, 7, 14, 9, 12, 5]` | 58 |
| Handwritten_numerals | `[105, 116, 119, 75, 120, 112]` | 647 |
| **Total** | — | **708** |

Stage 5R-A loaded no labels (`labels_loaded=false`).

## 2. 3sources Failure Classification

For each of the three 3sources views, the root deterministic 2-means candidate
split is `1 / 168`. This is an extremely imbalanced candidate split and fails
the frozen minimum-child constraint. The failure is therefore classified as:

`CANDIDATE_SPLITTER_DEGENERACY`

It is not classified as `ETA_THRESHOLD_FAILURE`. Stage 5R-B must not:

- lower `eta` further to try to repair 3sources;
- lower `min_child_size` to 1 merely to force balls on this dataset;
- use labels to guide or validate a splitter change;
- modify or replace the deterministic 2-means candidate splitter.

## 3. Structural Validity

For dataset \(d\) and view \(v\):

\[
\operatorname{valid\_view}(d,v)=
\begin{cases}
1, & \operatorname{ball\_count}(d,v)>1\\
0, & \text{otherwise}
\end{cases}
\]

and

\[
\operatorname{valid\_view\_count}(d)=\sum_v
\operatorname{valid\_view}(d,v).
\]

`weighting_evaluable(d)` is true if and only if
`valid_view_count(d) >= 2`. This rule depends only on X-derived granular
structure.

| Dataset | valid_view_count | weighting_evaluable | Classification |
|---|---:|---|---|
| 3sources | 0 | false | `STRUCTURAL_FAILURE_CASE` |
| MSRCV1 | 6 | true | primary Gate dataset |
| Handwritten_numerals | 6 | true | primary Gate dataset |

## 4. Required Dataset Execution Scope

Stage 5R-B must execute C0, C1, C2, C3, and C4 for all three datasets, including
3sources, and must write predictions before the isolated evaluation phase reads
labels. Each method must produce predictions, diagnostics, and metrics.

3sources is excluded only from the primary weighting recovery Gate because it
has fewer than two structurally valid views. It must remain in all outputs and
be reported separately as a `STRUCTURAL_FAILURE_CASE`.

## 5. Relation Degeneration Diagnostics

For every dataset-method fused representation \(F\), Stage 5R-B must record:

- `feature_variance_per_dimension`;
- `mean_feature_variance`;
- `number_of_nonzero_variance_dimensions`;
- `unique_row_count`.

The variance tolerance used to call a dimension nonzero must be fixed before
examining metrics and recorded in the run configuration and report.

Define `relation_degenerate=true` when either:

```text
number_of_nonzero_variance_dimensions == 0
```

or:

```text
unique_row_count <= 1
```

Degeneration must be recorded rather than hidden. In particular, C1 through C4
on 3sources must be checked for identical fused rows.

## 6. Composite Score

The Composite score retains the Stage 5 convention and is not a tunable
quantity:

\[
\operatorname{Composite}=\frac{ACC+NMI+ARI+PUR+F\text{-score}}{5}.
\]

Metrics may be read only in the physically separate evaluation/analysis phase,
after predictions have been written. They must not affect fitting,
granularization, dataset eligibility, K, or parameter selection.

## 7. Primary Reliability Recovery Gate

The primary Gate uses only MSRCV1 and Handwritten_numerals across five frozen
seeds, for 10 paired cases in total. 3sources is not included.

### A. Legacy to Corrected

For every paired case:

\[
\Delta_{fix}=Composite(C3)-Composite(C2).
\]

Pass A requires both:

- mean \(\Delta_{fix}>0\);
- at least 6 of 10 paired deltas are positive.

### B. Corrected versus Equal

\[
\Delta_{corrected}=Composite(C3)-Composite(C1).
\]

Pass B requires mean \(\Delta_{corrected}\geq-0.01\).

### C. Full versus Equal

\[
\Delta_{full}=Composite(C4)-Composite(C1).
\]

Pass C requires mean \(\Delta_{full}\geq-0.01\).

`Reliability Recovery=PASS` if and only if A, B, and C all pass. Otherwise it is
`FAIL`. Paired deltas must be matched by dataset and seed.

## 8. Raw Feature Competitiveness

For MSRCV1 and Handwritten_numerals:

\[
Gap_{raw}=Composite(C4)-Composite(C0).
\]

Report the overall mean, each dataset-specific mean, and win/tie/loss counts.
A tie must use a predeclared numerical tolerance recorded in the analysis.

Report 3sources C4 versus C0 separately as granulation robustness failure
evidence. Its relation degeneration must not directly fail the primary
weighting recovery Gate.

## 9. Representation Efficiency

For each dataset, record:

\[
raw\_total\_dimension=\sum_v d_v,
\]

\[
relation\_dimension=\sum_v m_v,
\]

\[
relation\_to\_raw\_ratio=\frac{relation\_dimension}
{raw\_total\_dimension},
\]

and

\[
ball\_to\_sample\_ratio=\frac{\sum_v m_v}{n\,V}.
\]

Only a `relation_to_raw_ratio < 1` may be described as representation
compression. A small ratio does not imply a useful representation when the
relation is degenerate.

Frozen structural values from Stage 5R-A are:

| Dataset | raw total dimension | relation dimension | relation/raw | ball/(sample×view) |
|---|---:|---:|---:|---:|
| 3sources | 10,259 | 3 | 0.00029243 | 0.00591716 |
| MSRCV1 | 2,438 | 58 | 0.02378999 | 0.04603175 |
| Handwritten_numerals | 649 | 647 | 0.99691834 | 0.05391667 |

The 3sources ratio must be accompanied by the degeneration warning. For
Handwritten_numerals, Stage 5R-B must additionally report per-view mean and
median ball size. Its 647-dimensional relation representation is only slightly
smaller than its 649-dimensional raw representation and must not trigger a
parameter change in this stage.

## 10. Four-Way Final Decision

The final decision must be exactly one of the following.

### READY_FOR_STAGE6

All of the following must hold:

1. `Reliability Recovery=PASS`;
2. on MSRCV1 and Handwritten_numerals, either the 10-pair mean
   `Gap_raw >= -0.05`, or the five-seed mean Composite of C4 is at least the
   five-seed mean Composite of C0 on at least one dataset;
3. the 3sources failure has been explicitly identified as a splitter-specific
   structural limitation.

Any Stage 6 handoff must still mark the 3sources granulation limitation as
unresolved.

### WEIGHTING_RECOVERED_BUT_GRANULATION_BLOCKED

Use this decision when `Reliability Recovery=PASS`, but either C4 remains
clearly behind RawConcat on both evaluable datasets, or the complete 3sources
relation degeneration is judged unsuitable for direct inclusion in formal main
experiments. The next stage is Stage 5R-C: Candidate Splitter Robustness Study.

### WEIGHTING_NOT_RECOVERED

Use this decision when `Reliability Recovery=FAIL`. Stage 6 is prohibited; the
entropy-reliability assumption must be reviewed.

### IMPLEMENTATION_FAILURE

Use this decision if any NaN/Inf, view-weight sum error, granularization
inconsistency, label leakage, or test failure occurs. This outcome takes
precedence over analytical Gate results.

## 11. Required 3sources Report Section

The future `reports/STAGE5R_WEIGHTING_ANALYSIS.md` must contain a section named
`3sources Structural Failure Case` stating at least:

1. all three views have ball counts `[1, 1, 1]`;
2. all root 2-means candidates split `1 / 168`;
3. the failure is independent of further lowering `eta`;
4. the failure originates in the current deterministic 2-means candidate
   splitter;
5. block normalization of a single-ball relation can remove sample
   discrimination;
6. labels and a forced `min_child_size=1` are not acceptable repairs;
7. if weighting recovers elsewhere, the next appropriate study is an independent
   splitter robustness study.

## 12. Label Firewall and Stop Conditions

Granularization and fitting remain strictly label-free. Labels may enter only
the post-prediction evaluation module. Stage 5R-B must not automatically:

- modify the 2-means splitter;
- introduce balanced KMeans or farthest-point splitting;
- introduce PCA, spectral clustering, or multi-granularity;
- retune the frozen granulation candidate;
- execute Stage 5R-C or Stage 6.

Stage 5R-B must stop after its tests, analysis, decision, and handoff are
complete.
