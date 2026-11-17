import inspect

import numpy as np
from sklearn.cluster import KMeans

from bgrfmvc.experiments.stage7_optimization import (
    D3RepresentationCache, build_d3_optimized, build_d3_reference, cache_key, equivalence,
)


def _config():
    return {
        "method": {"eta": 0.02, "min_parent_size": 20, "min_child_size": 10,
                   "max_depth": 20, "tau": .5, "eps": 1e-12,
                   "radius_eps": 1e-12, "relation_floor": 1e-12},
        "granular_ball": {"split_n_init": 10, "split_max_iter": 100,
                          "split_random_state": 0},
        "preprocess": {"standardize": True, "use_pca": False},
    }


def _views(seed=0):
    rng = np.random.RandomState(seed)
    a = np.r_[rng.normal(-2, .3, (30, 4)), rng.normal(2, .3, (30, 4))]
    b = np.r_[rng.normal(-1, .4, (30, 3)), rng.normal(1, .4, (30, 3))]
    return [a, b]


def test_reference_optimized_ball_relation_and_fusion_equivalence():
    ref = build_d3_reference(_views(), _config())
    opt = build_d3_optimized(_views(), _config())
    result = equivalence(ref, opt)
    assert result["ball_structure_equal"]
    assert result["ball_max_abs_diff"] <= 1e-12
    assert result["relation_allclose"]
    assert result["representation_allclose"]
    assert result["representation_max_abs_diff"] <= 1e-10
    assert result["finite"]


def test_reference_optimized_prediction_equivalence_and_determinism():
    ref = build_d3_reference(_views(), _config()).representation
    opt = build_d3_optimized(_views(), _config()).representation
    for seed in (0, 1, 2):
        a = KMeans(n_clusters=2, n_init=20, max_iter=300, random_state=seed).fit_predict(ref)
        b = KMeans(n_clusters=2, n_init=20, max_iter=300, random_state=seed).fit_predict(opt)
        assert np.array_equal(a, b)


def test_multiseed_cache_builds_once_and_no_cache_rebuilds():
    cache = D3RepresentationCache(); views = _views(); config = _config()
    first, hit1, key1 = cache.get_or_build("A", views, config, use_cache=True)
    second, hit2, key2 = cache.get_or_build("A", views, config, use_cache=True)
    assert not hit1 and hit2 and first is second and key1 == key2
    assert cache.build_count == 1 and cache.hit_count == 1
    cache.get_or_build("A", views, config, use_cache=False)
    assert cache.build_count == 2


def test_cache_invalidates_dataset_content_and_preprocess_config():
    views = _views(); config = _config()
    assert cache_key("A", views, config) != cache_key("B", views, config)
    changed = _views(); changed[0] = changed[0].copy(); changed[0][0, 0] += 1
    assert cache_key("A", views, config) != cache_key("A", changed, config)
    config2 = _config(); config2["preprocess"]["standardize"] = False
    assert cache_key("A", views, config) != cache_key("A", views, config2)


def test_phase_a_has_no_evaluation_or_label_loader_dependency():
    import bgrfmvc.experiments.run_stage7_phase_a as phase_a
    source = inspect.getsource(phase_a)
    assert "load_evaluation_labels" not in source
    assert "bgrfmvc.evaluation" not in source
    assert '"labels_loaded": False' in source


def test_frozen_split_and_float64_finite_contract():
    config = _config(); assert config["granular_ball"]["split_random_state"] == 0
    result = build_d3_optimized(_views(), config)
    assert result.representation.dtype == np.float64
    assert np.isfinite(result.representation).all()


def test_phase_b_only_evaluates_saved_predictions():
    import bgrfmvc.evaluation.evaluate_stage7 as phase_b
    source = inspect.getsource(phase_b)
    for forbidden in ("fit_predict(", "fit_transform(", "build_granular_balls(",
                      "build_d3_reference(", "build_d3_optimized("):
        assert forbidden not in source
    assert '"models_refit":False' in source


def test_recorded_real_dataset_equivalence_if_stage7_exists():
    import pandas as pd
    from pathlib import Path
    path = Path("result/stage7_freeze/equivalence/numerical_equivalence.csv")
    if path.is_file():
        frame = pd.read_csv(str(path))
        assert frame.ball_structure_equal.all()
        assert frame.relation_allclose.all()
        assert frame.representation_allclose.all()
        assert frame.representation_max_abs_diff.max() <= 1e-10


def test_phase_a_prediction_manifest_is_label_free_if_stage7_exists():
    import pandas as pd
    from pathlib import Path
    path = Path("result/stage7_freeze/predictions/prediction_manifest.csv")
    if path.is_file():
        frame = pd.read_csv(str(path))
        assert len(frame) == 90
        assert not frame.labels_loaded.astype(bool).any()
