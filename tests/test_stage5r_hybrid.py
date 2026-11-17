import ast
from pathlib import Path

import numpy as np

from bgrfmvc.experiments.stage5r_hybrid import _hybrid_fusion


ROOT = Path(__file__).resolve().parents[1]


def test_hybrid_preserves_raw_variation_with_constant_relation():
    raw = [np.array([[1., 0.], [0., 1.], [-1., 0.]])]
    relation = [np.ones((3, 1))]
    fused = _hybrid_fusion(raw, relation, 1e-12)
    assert np.unique(fused, axis=0).shape[0] == 3
    assert np.count_nonzero(np.var(fused, axis=0) > 1e-15) > 0


def test_hybrid_blocks_have_equal_energy():
    raw = [np.array([[3., 4.], [4., 3.]])]
    relation = [np.array([[1., 0.], [0., 2.]])]
    fused = _hybrid_fusion(raw, relation, 1e-12)
    assert np.allclose(np.linalg.norm(fused[:, :2], axis=1), 1/np.sqrt(2), atol=1e-10)
    assert np.allclose(np.linalg.norm(fused[:, 2:], axis=1), 1/np.sqrt(2), atol=1e-10)


def test_phase_a_has_no_evaluation_or_label_loader_import():
    path = ROOT / "src" / "bgrfmvc" / "experiments" / "run_stage5r_hybrid.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imports=[]
    for node in ast.walk(tree):
        if isinstance(node,ast.Import): imports += [x.name for x in node.names]
        elif isinstance(node,ast.ImportFrom): imports.append(node.module or "")
    assert not any("evaluation" in x or "labels" in x for x in imports)


def test_phase_b_does_not_fit_or_transform():
    path=ROOT/"src"/"bgrfmvc"/"evaluation"/"evaluate_stage5r_hybrid.py"
    tree=ast.parse(path.read_text(encoding="utf-8"))
    calls={n.func.attr for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)}
    assert not ({"fit","fit_predict","fit_transform","transform"}&calls)
