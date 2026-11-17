"""Stage 9 pre-experiment audit.  Stops before training if <4 fair baselines."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

import pandas as pd
import yaml


EXCLUDED = "20newsgroups"
CANDIDATES = (
    ("MGBCC", "Multi-view Granular-ball Contrastive Clustering", "AAAI", 2025, "https://github.com/poensu/mgbcc_main", "Python/PyTorch", "main.py"),
    ("AWMVC", "Auto-Weighted Multi-View Clustering for Large-Scale Data", "AAAI", 2023, "https://github.com/wanxinhang/AAAI-2023-AWMVC", "MATLAB", "demo/run script"),
    ("FastMICE", "Fast Multi-view Clustering via Ensembles", "IEEE TKDE", 2023, "https://github.com/huangdonghere/FastMICE", "MATLAB", "demo_1.m"),
    ("CAMVC", "Learning Cluster-Wise Anchors for Multi-View Clustering", "AAAI", 2024, "https://github.com/czhangnju/CAMVC", "MATLAB archive", "CAMVC.zip"),
    ("DIVIDE", "Decoupled Contrastive Multi-View Clustering with High-Order Random Walks", "AAAI", 2024, "https://github.com/XLearning-SCU/2024-AAAI-DIVIDE", "Python/PyTorch", "main_train.py"),
    ("MGCMVC", "Multi-Granularity Information Fused Contrastive Learning with Multi-View Clustering", "public repository", 2025, "https://github.com/Luyangabc/MGCMVC", "Python/PyTorch", "repository entrypoint"),
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build_eval_manifest(project_root="."):
    root = Path(project_root).resolve()
    source = yaml.safe_load((root / "configs/stage8r_selected_benchmark.yaml").read_text(encoding="utf-8"))
    audit = pd.read_csv(root / "result/stage8r_main_benchmark/dataset_recovery_audit.csv").set_index("dataset")
    datasets = {}
    for name, config in source["datasets"].items():
        if name == EXCLUDED:
            continue
        view_dimensions = []
        if name in audit.index:
            try:
                shapes = json.loads(audit.loc[name, "resolved_shapes"])
                view_dimensions = [int(shape[1]) for shape in shapes]
            except Exception:
                view_dimensions = []
        datasets[name] = {
            "dataset": name, "dataset_family": name, "path": config["path"],
            "n_samples": int(config["n_samples"]), "n_views": int(config["n_views"]),
            "view_dimensions": view_dimensions, "K": int(config["K"]),
            "K_source": config["K_source_reference"],
            "stage8r_selection_hash": "156e948e8e4a4fa979e30935d6f76194585199fa8f6bbea7e207c6cdd31e2989",
            "feature_hash": config["feature_hash"],
        }
    manifest = {"N_EVAL": len(datasets), "excluded_stage8r_data_failure": EXCLUDED,
                "source_selection_hash": "156e948e8e4a4fa979e30935d6f76194585199fa8f6bbea7e207c6cdd31e2989",
                "datasets": datasets}
    path = root / "configs/stage9_eval_11.yaml"
    path.write_text(yaml.safe_dump(manifest, allow_unicode=True), encoding="utf-8")
    return manifest, sha(path)


def run(project_root="."):
    root = Path(project_root).resolve(); out = root / "result/stage9_public_baselines"
    for name in ("audit", "sources", "patches", "environments", "sanity", "phase_a", "phase_b", "predictions", "raw", "summary", "statistics", "runtime", "memory", "figures", "tables", "supplementary", "failures"):
        (out / name).mkdir(parents=True, exist_ok=True)
    manifest, selection_hash = build_eval_manifest(root)
    if manifest["N_EVAL"] != 11 or EXCLUDED in manifest["datasets"]:
        raise RuntimeError("STAGE9_EVAL_SELECTION_INVALID")
    rows = []
    for method, title, venue, year, repository, framework, entrypoint in CANDIDATES:
        local = root / "external_baselines" / method / "original"
        available = local.exists() and any(local.iterdir())
        rows.append({"method": method, "paper_title": title, "venue": venue, "year": year,
                     "repository": repository, "repository_type": "official public GitHub",
                     "official_or_author_code": True, "commit_hash": "UNAVAILABLE_DOWNLOAD_FAILED",
                     "download_date": "2026-09-07", "language": framework.split("/")[0],
                     "framework": framework, "license_info": "not locally auditable",
                     "original_entrypoint": entrypoint,
                     "audit_status": "RUNTIME_UNAVAILABLE" if not available else "AUDIT_PENDING",
                     "audit_reason": "GitHub git and codeload failed; source absent, so commit, static label/K/parameter audit and sanity run are impossible" if not available else "local source requires audit"})
    sources = pd.DataFrame(rows); sources.to_csv(out / "audit/baseline_sources.csv", index=False)
    pd.DataFrame(columns=["method","file","line/function","patch_type","original_behavior","patched_behavior","scientific_semantics_changed","reason","test","status"]).to_csv(out / "audit/baseline_patch_manifest.csv", index=False)
    pd.DataFrame(columns=["method","dataset","parameter","value","source","source_type"]).to_csv(out / "audit/baseline_parameter_manifest.csv", index=False)
    accepted = sources[sources.audit_status == "ACCEPT_MAIN"].method.tolist()
    phase_a = {"stage": 9, "status": "NOT_STARTED_AUDIT_GATE", "labels_loaded": False,
               "accepted_main_baselines": accepted, "accepted_supplementary_baselines": [],
               "planned_external_runs": 0, "terminal_external_runs": 0,
               "stage9_selection_hash": selection_hash,
               "BGRF_paper_candidate_hash": sha(root / "docs/PAPER_CANDIDATE_V1.md")}
    (out / "phase_a_manifest.json").write_text(json.dumps(phase_a, indent=2), encoding="utf-8")
    phase_b = {"stage": 9, "status": "NOT_STARTED_AUDIT_GATE", "labels_loaded": False,
               "training_calls": 0, "baseline_fit_calls": 0, "representation_build_calls": 0}
    (out / "phase_b_manifest.json").write_text(json.dumps(phase_b, indent=2), encoding="utf-8")
    environment = {"python": sys.version, "os": platform.platform(), "matlab": "C:/Program Files/MATLAB/R2025b/bin/matlab.exe",
                   "pytorch": "NOT_INSTALLED_IN_FROZEN_PYTHON", "cuda": "UNAVAILABLE_NOT_AUDITABLE_WITHOUT_PYTORCH",
                   "git": subprocess.check_output(["git","--version"], text=True).strip(),
                   "source_network": "GitHub clone and codeload unavailable (connection reset/timeouts)"}
    (out / "environments/environment.json").write_text(json.dumps(environment, indent=2), encoding="utf-8")
    gate = {"final_gate": "EXTERNAL_EVIDENCE_WEAK", "blocking_status": "STAGE9_INSUFFICIENT_FAIR_BASELINES",
            "N_EVAL": 11, "stage9_eval_selection_hash": selection_hash, "baseline_candidates_audited": len(CANDIDATES),
            "ACCEPT_MAIN": accepted, "ACCEPT_MAIN_count": len(accepted), "required_ACCEPT_MAIN": 4,
            "MGBCC_status": sources.set_index("method").loc["MGBCC","audit_status"],
            "formal_phase_a_started": False, "formal_phase_b_started": False,
            "algorithm_modified": False, "parameters_modified": False,
            "BGRF_freeze_integrity": sha(root / "docs/PAPER_CANDIDATE_V1.md") == "3ed77b3254192f89e6c72178b1afbc692468eb150fee296d24fcb9698d35d6e6"}
    (out / "stage9_gate.json").write_text(json.dumps(gate, indent=2), encoding="utf-8")
    return gate


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
