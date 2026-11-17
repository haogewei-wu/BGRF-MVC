import numpy as np

from bgrfmvc.core.fusion import fuse_relation_blocks, l2_normalize_rows


def test_fusion_shape_weighting_and_finiteness():
    first = np.array([[1.0, 0.0], [1.0, 1.0], [0.0, 0.0]])
    second = np.array([[1.0], [2.0], [0.0]])
    alpha = np.array([[0.25, 0.75], [0.8, 0.2], [0.5, 0.5]])
    fused = fuse_relation_blocks([first, second], alpha)
    assert fused.shape == (3, 3)
    assert np.all(np.isfinite(fused))
    norms = np.linalg.norm(fused, axis=1)
    np.testing.assert_allclose(norms[:2], 1.0, atol=1e-10)
    assert norms[2] == 0.0


def test_zero_block_normalization_has_no_nan():
    normalized = l2_normalize_rows(np.zeros((4, 3)))
    np.testing.assert_array_equal(normalized, np.zeros((4, 3)))
    assert not np.isnan(normalized).any()

