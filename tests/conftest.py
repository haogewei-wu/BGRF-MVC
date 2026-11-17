import sys
from pathlib import Path

import numpy as np
import pytest


SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


@pytest.fixture
def synthetic_views():
    """Two clear clusters, two scales, and enough duplicates for a zero-radius ball."""
    rng = np.random.RandomState(42)
    left = rng.normal(loc=(-3.0, -3.0), scale=0.25, size=(24, 2))
    right = rng.normal(loc=(3.0, 3.0), scale=0.30, size=(24, 2))
    duplicates = np.repeat([[0.0, 0.0]], repeats=6, axis=0)
    view1 = np.vstack([left, right, duplicates])
    view2 = np.column_stack([
        100.0 * view1[:, 0] + rng.normal(0, 1.0, view1.shape[0]),
        0.01 * view1[:, 1],
        view1[:, 0] - view1[:, 1],
    ])
    return [view1, view2]

