# BGRF-MVC v2.0.1

Relative to v2.0:

1. Probability normalization now has a strict row sum of one.
2. A zero-radius granular-ball rule handles coincident and non-coincident samples explicitly.
3. Population radial standard deviation is fixed to `ddof=0`.
4. Candidate split 2-means parameters are fixed (`n_init=10`, `max_iter=100`, `random_state=0`).
5. Mean radius is explicitly interpreted as a characteristic/effective radius rather than an enclosing radius.
6. A robust parent/child size guard requires `size >= max(min_parent_size, 2 * min_child_size)` before attempting a split.

The overall algorithm framework is unchanged.

