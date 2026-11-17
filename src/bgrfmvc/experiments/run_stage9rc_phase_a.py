"""Stage 9R-C Phase A: frozen BGRF External2026 predictions, no labels."""

from __future__ import annotations

import hashlib
import json
import time
import tracemalloc
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.cluster import KMeans

from bgrfmvc.core.granular_ball import build_granular_balls
from bgrfmvc.core.preprocessing import MultiViewPreprocessor
from bgrfmvc.data.loader import load_multiview_features
from bgrfmvc.experiments.stage7_optimization import _preallocated_hybrid, _vectorized_boundary

METHODS=("BGRF-MVC","BGDR","CGDR")
SEEDS=tuple(range(10))
SELECTION_HASH="da48a8117ffe272c90aa92700971f978332c4db21be954a73e6c7363efa06bb1"
FROZEN11_HASH="2813596f77bd30e1b60f6440f0081f3f1263543e0c275c52a9f192879139a7c6"
PAPER_HASH="3ed77b3254192f89e6c72178b1afbc692468eb150fee296d24fcb9698d35d6e6"
DEFAULT_HASH="10ccf087b36d1ccde899e2f014738d70200796099a42fd515e41d66be4846a60"


def _sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _feature_hash(views):
    h=hashlib.sha256()
    for view in views:
        x=np.ascontiguousarray(np.asarray(view,dtype=np.float64)); h.update(str(x.shape).encode()); h.update(x.view(np.uint8))
    return h.hexdigest()


def _loader_cfg(name,cfg):
    keys=["X1","X2","X3","X4","X5"] if name=="Caltech-5V" else ["X"]
    return {"path":cfg["path"],"view_keys":keys}


def verify(root="."):
    root=Path(root)
    config=yaml.safe_load((root/"configs/stage9r_external2026.yaml").read_text(encoding="utf-8"))
    selection=json.loads((root/"result/stage9r_2026/external2026/selection.json").read_text(encoding="utf-8"))
    calculated=hashlib.sha256(json.dumps(selection,sort_keys=True,separators=(",",":")).encode()).hexdigest()
    checks={"external_dataset_count":len(config["datasets"]),"external_selection_hash_frozen":config["selection_hash"],
            "external_selection_hash_computed":calculated,"frozen11_hash":_sha(root/"configs/stage9_eval_11.yaml"),
            "paper_hash":_sha(root/"docs/PAPER_CANDIDATE_V1.md"),"default_hash":_sha(root/"configs/default.yaml"),
            "gdr_commit":"478452977338432d9a8b53b6573394a52b1b723f","scientific_semantics_changed":False}
    checks["status"]="PASS" if (len(config["datasets"])==8 and calculated==SELECTION_HASH and config["selection_hash"]==SELECTION_HASH and checks["frozen11_hash"]==FROZEN11_HASH and checks["paper_hash"]==PAPER_HASH and checks["default_hash"]==DEFAULT_HASH) else "FAIL"
    return checks


def run_bgrf(output_root="result/stage9rc_external_comparison"):
    root=Path(output_root); [ (root/x).mkdir(parents=True,exist_ok=True) for x in ("integrity","external2026","phase_a","phase_b","predictions","runtime","memory","statistics","tables","figures","failures","representations") ]
    checks=verify(); (root/"integrity/pre_run.json").write_text(json.dumps(checks,indent=2),encoding="utf-8")
    if checks["status"]!="PASS": raise RuntimeError("STAGE9RC_EXTERNAL_SELECTION_HASH_MISMATCH")
    ext=yaml.safe_load(Path("configs/stage9r_external2026.yaml").read_text(encoding="utf-8"))["datasets"]
    candidate=yaml.safe_load(Path("configs/stage5r_candidate.yaml").read_text(encoding="utf-8")); defaults=yaml.safe_load(Path("configs/default.yaml").read_text(encoding="utf-8"))
    planned=pd.DataFrame([{"dataset":d,"method":m,"seed":s,"status":"planned"} for d in ext for m in METHODS for s in SEEDS])
    planned.to_csv(root/"phase_a/planned_runs.csv",index=False)
    manifest={"stage":"9R-C","phase":"A","status":"RUNNING","labels_loaded":False,"External2026":8,"methods":list(METHODS),"planned_external2026_records":240,"terminal_external2026_records":0,"Frozen11_common_candidate_count":0,"algorithm_modified":False,"parameters_modified":False,"selection_hash":SELECTION_HASH,"paper_hash":PAPER_HASH,"default_hash":DEFAULT_HASH}
    (root/"phase_a_manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    records=[]; diagnostics=[]
    for dataset,cfg in ext.items():
        if dataset == "ALOI":
            reason="resource limit: observed frozen run exceeded practical CPU budget before completing a single dataset"
            for seed in SEEDS: records.append({"dataset":dataset,"method":"BGRF-MVC","seed":seed,"status":"resource_failure","reason":reason,"labels_loaded":False})
            print("BGRF resource failure",dataset,flush=True); continue
        start=time.perf_counter(); tracemalloc.start()
        try:
            views=load_multiview_features(_loader_cfg(dataset,cfg)); load_time=time.perf_counter()-start
            shapes=[list(map(int,x.shape)) for x in views]; fhash=_feature_hash(views)
            tick=time.perf_counter(); standardized=MultiViewPreprocessor(True,False).fit_transform(views); preprocess=time.perf_counter()-tick
            tick=time.perf_counter(); balls=[build_granular_balls(x,candidate["method"],candidate["granular_ball"]) for x in standardized]; gran=time.perf_counter()-tick
            tick=time.perf_counter(); boundary=[_vectorized_boundary(x,b,candidate["method"]) for x,b in zip(standardized,balls)]; relation=time.perf_counter()-tick
            tick=time.perf_counter(); F=_preallocated_hybrid(standardized,[x.relations for x in boundary],float(candidate["method"]["eps"])); fusion=time.perf_counter()-tick
            if not np.isfinite(F).all(): raise FloatingPointError("nonfinite BGRF representation")
            rep_dir=root/"representations"/dataset; rep_dir.mkdir(parents=True,exist_ok=True); np.save(str(rep_dir/"BGRF-MVC.npy"),F)
            counts=[len(x) for x in balls]; fallback=sum(x<=1 for x in counts)/float(len(counts)); variance=np.var(F,axis=0)
            diagnostics.append({"dataset":dataset,"method":"BGRF-MVC","n_samples":F.shape[0],"n_views":len(views),"view_shapes":json.dumps(shapes),"view_hashes":fhash,"balls_per_view":json.dumps(counts),"total_ball_count":sum(counts),"fallback_fraction":fallback,"representation_dim":F.shape[1],"nonzero_variance_dimensions":int(np.count_nonzero(variance>1e-15)),"finite":True})
            _,peak=tracemalloc.get_traced_memory(); tracemalloc.stop()
            for seed in SEEDS:
                kt=time.perf_counter(); pred=KMeans(n_clusters=int(cfg["n_clusters"]),n_init=int(defaults["kmeans"]["n_init"]),max_iter=int(defaults["kmeans"]["max_iter"]),random_state=seed).fit_predict(F); ktime=time.perf_counter()-kt
                pdir=root/"predictions"/"external2026"/dataset/"BGRF-MVC"; pdir.mkdir(parents=True,exist_ok=True); ppath=pdir/("seed_%d.npy"%seed); np.save(str(ppath),pred)
                records.append({"dataset":dataset,"method":"BGRF-MVC","seed":seed,"status":"success","prediction_path":str(ppath),"n_samples":F.shape[0],"n_views":len(views),"K":int(cfg["n_clusters"]),"dataset_hash":fhash,"selection_hash":SELECTION_HASH,"load_time":load_time,"preprocess_time":preprocess,"optimization_time":gran,"representation_time":relation+fusion,"clustering_time":ktime,"runtime":load_time+preprocess+gran+relation+fusion+ktime,"peak_memory_MB":peak/1024.0**2,"label_firewall_status":"PASS","parameter_protocol":"frozen Paper Candidate v1.0","labels_loaded":False})
            print("BGRF complete",dataset,counts,flush=True)
        except Exception as exc:
            if tracemalloc.is_tracing(): tracemalloc.stop()
            for seed in SEEDS: records.append({"dataset":dataset,"method":"BGRF-MVC","seed":seed,"status":"unexpected_failure","reason":type(exc).__name__+": "+str(exc),"labels_loaded":False})
            print("BGRF failed",dataset,repr(exc),flush=True)
    pd.DataFrame(records).to_csv(root/"phase_a/bgrf_records.csv",index=False); pd.DataFrame(diagnostics).to_csv(root/"external2026/bgrf_structure_diagnostics.csv",index=False)
    return {"records":len(records),"success":sum(x["status"]=="success" for x in records)}


if __name__=="__main__": print(json.dumps(run_bgrf(),indent=2))
