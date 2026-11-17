import ast
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def _imports_label_loader(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if any(alias.name == "load_labels_for_evaluation" for alias in node.names):
                return True
        if isinstance(node, ast.Import):
            if any("load_labels_for_evaluation" in alias.name for alias in node.names):
                return True
    return False


def test_core_does_not_import_evaluation_label_loader():
    assert not any(_imports_label_loader(path) for path in (ROOT / "src/bgrfmvc/core").glob("*.py"))


def test_unsupervised_runner_does_not_import_label_loader():
    runner = ROOT / "src/bgrfmvc/experiments/run_unsupervised.py"
    assert not runner.exists() or not _imports_label_loader(runner)


def test_manifest_has_no_label_derived_k_expression():
    text = (ROOT / "configs/datasets.yaml").read_text(encoding="utf-8").lower().replace(" ", "")
    assert "unique(y)" not in text
    manifest = yaml.safe_load((ROOT / "configs/datasets.yaml").read_text(encoding="utf-8"))
    for config in manifest["datasets"].values():
        assert not config["enabled"] or (config["n_clusters"] is not None and config["k_source"])

