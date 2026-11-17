"""Stage 8R Phase A: frozen D0-D3 benchmark without label access."""

from __future__ import annotations

import hashlib
import json
import os
import time
import tracemalloc
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.cluster import KMeans

from bgrfmvc.core.fusion import fuse_relation_blocks
from bgrfmvc.core.granular_ball import build_granular_balls
from bgrfmvc.core.preprocessing import MultiViewPreprocessor
from bgrfmvc.core.reliability import relation_probability_and_entropy
from bgrfmvc.data.loader import load_multiview_features
from bgrfmvc.experiments.stage7_optimization import _preallocated_hybrid, _vectorized_boundary


METHODS=("D0_RawConcat_KM","D1_HardGB_KM","D2_RawGaussianHybrid_KM","D3_BGRFMVC_PaperCandidate_v1")
SEEDS=tuple(range(10)); VAR_TOL=1e-15


def _sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _stats(x):
    x=np.asarray(x,dtype=float); variance=np.var(x,axis=0); singular=np.linalg.svd(x,compute_uv=False)
    total=singular.sum()
    if total>0:
        p=singular[singular>0]/total; effective=float(np.exp(-np.sum(p*np.log(p))))
    else: effective=0.0
    unique=int(np.unique(x,axis=0).shape[0]); nonzero=int(np.count_nonzero(variance>VAR_TOL))
    return {"representation_dim":int(x.shape[1]),"representation_memory_MB":x.nbytes/1024.0**2,
      "unique_row_count":unique,"unique_row_ratio":unique/float(x.shape[0]),
      "zero_variance_dim_count":int(x.shape[1]-nonzero),"nonzero_variance_dim_count":nonzero,
      "matrix_rank":int(np.linalg.matrix_rank(x)),"effective_rank":effective,
      "zero_fraction":float(np.mean(x==0)),"finite":bool(np.isfinite(x).all())}


def _build(views,config):
    timing={}; tick=time.perf_counter(); standardized=MultiViewPreprocessor(True,False).fit_transform(views)
    timing["preprocess_time"]=time.perf_counter()-tick; tick=time.perf_counter()
    balls=[build_granular_balls(x,config["method"],config["granular_ball"]) for x in standardized]
    timing["granularization_time"]=time.perf_counter()-tick; tick=time.perf_counter()
    boundary=[_vectorized_boundary(x,b,config["method"]) for x,b in zip(standardized,balls)]
    gaussian=[]; eps=float(config["method"]["eps"])
    for result,view_balls in zip(boundary,balls):
        radii=np.asarray([b.radius for b in view_balls]); gaussian.append(
            np.exp(-np.square(result.distances)/(2*np.square(radii[None,:]+eps))))
    timing["relation_time"]=time.perf_counter()-tick; n=standardized[0].shape[0]; v=len(standardized)
    equal=np.full((n,v),1.0/v); reps={}; rep_times={}
    tick=time.perf_counter(); reps[METHODS[0]]=fuse_relation_blocks(standardized,equal,eps); rep_times[METHODS[0]]=time.perf_counter()-tick
    hard=[]
    for view_balls in balls:
        block=np.zeros((n,len(view_balls)))
        for j,ball in enumerate(view_balls): block[ball.indices,j]=1.0
        hard.append(block)
    tick=time.perf_counter(); reps[METHODS[1]]=fuse_relation_blocks(hard,equal,eps); rep_times[METHODS[1]]=time.perf_counter()-tick
    tick=time.perf_counter(); reps[METHODS[2]]=_preallocated_hybrid(standardized,gaussian,eps); rep_times[METHODS[2]]=time.perf_counter()-tick
    tick=time.perf_counter(); reps[METHODS[3]]=_preallocated_hybrid(standardized,[r.relations for r in boundary],eps); rep_times[METHODS[3]]=time.perf_counter()-tick
    timing["fusion_time"]=sum(rep_times.values()); timing["representation_times"]=rep_times
    return standardized,balls,boundary,gaussian,reps,timing


def run(root_path="result/stage8r_main_benchmark"):
    root=Path(root_path); selected=yaml.safe_load(Path("configs/stage8r_selected_benchmark.yaml").read_text(encoding="utf-8"))
    selection=json.loads((root/"dataset_selection.json").read_text(encoding="utf-8")); defaults=yaml.safe_load(Path("configs/default.yaml").read_text(encoding="utf-8"))
    candidate=yaml.safe_load(Path("configs/stage5r_candidate.yaml").read_text(encoding="utf-8"))
    if tuple(defaults["experiment"]["seeds"])!=SEEDS: raise RuntimeError("SEED_CONFIG_MISMATCH")
    config_hash=_sha("configs/stage5r_candidate.yaml")+":"+_sha("configs/default.yaml")
    paper_hash=_sha("docs/PAPER_CANDIDATE_V1.md"); code_hash=_sha("src/bgrfmvc/experiments/stage7_optimization.py")
    planned=len(selected["datasets"])*len(METHODS)*len(SEEDS)
    phase={"stage":"8R","phase":"A","status":"RUNNING","labels_loaded":False,"planned_runs":planned,
      "terminal_records":0,"selection_hash":selection["selection_hash"],"config_hash":config_hash,
      "paper_candidate_hash":paper_hash,"code_hash":code_hash,"seeds":list(SEEDS),"methods":list(METHODS),
      "algorithm_modified":False,"parameters_modified":False}
    (root/"phase_a_manifest.json").write_text(json.dumps(phase,indent=2),encoding="utf-8")
    records=[]; diagnostics=[]; granular_rows=[]; failures=[]; runtime_rows=[]
    for dataset,cfg in selected["datasets"].items():
        load_start=time.perf_counter()
        try: views=load_multiview_features({k:cfg[k] for k in ("path","view_keys","transpose_views")}); load_time=time.perf_counter()-load_start
        except Exception as error:
            for method in METHODS:
                for seed in SEEDS: records.append({"dataset":dataset,"method":method,"seed":seed,"status":"data_failure"})
            failures.append({"dataset":dataset,"method":"ALL","seed":"ALL","status":"data_failure","exception_type":type(error).__name__,"exception_message":str(error),"stage":"load","reproducible":True}); continue
        tracemalloc.start(); build_start=time.perf_counter()
        try: standardized,balls,boundary,gaussian,reps,timing=_build(views,candidate)
        except Exception as error:
            tracemalloc.stop()
            for method in METHODS:
                for seed in SEEDS: records.append({"dataset":dataset,"method":method,"seed":seed,"status":"structural_failure"})
            failures.append({"dataset":dataset,"method":"ALL","seed":"ALL","status":"structural_failure","exception_type":type(error).__name__,"exception_message":str(error),"stage":"representation","reproducible":True}); continue
        _,peak=tracemalloc.get_traced_memory(); tracemalloc.stop(); shared_build=time.perf_counter()-build_start
        counts=[len(x) for x in balls]; sizes=[b.size for x in balls for b in x]; radii=[b.radius for x in balls for b in x]; qs=[b.quality_q for x in balls for b in x]
        single=sum(c<=1 for c in counts); relation_blocks=[r.relations for r in boundary]
        entropy=[relation_probability_and_entropy(r,candidate["method"]["eps"])[1] for r in relation_blocks]
        granular={"dataset":dataset,"balls_per_view":json.dumps(counts),"total_ball_count":sum(counts),
          "mean_ball_count":float(np.mean(counts)),"min_ball_count":min(counts),"max_ball_count":max(counts),
          "mean_ball_size":float(np.mean(sizes)),"min_ball_size":min(sizes),"max_ball_size":max(sizes),
          "mean_radius":float(np.mean(radii)),"mean_reliability":float(np.mean(qs)),
          "safe_fallback_triggered":single>0,"fallback_count":single*len(views[0]),
          "fallback_fraction":single/float(len(counts)),"full_view_fallback_count":single,
          "full_dataset_fallback":single==len(counts),"relation_dimension":sum(counts),
          "relation_variance":float(np.mean([np.var(r) for r in relation_blocks])),
          "relation_effective_rank":float(np.mean([_stats(r)["effective_rank"] for r in relation_blocks])),
          "mean_relation_entropy":float(np.mean(np.column_stack(entropy)))}; granular_rows.append(granular)
        dataset_kmeans={m:0.0 for m in METHODS}
        for method,F in reps.items():
            info=_stats(F); info.update({"dataset":dataset,"method":method,"peak_memory_MB":peak/1024.0**2,
              "representation_build_time":shared_build if method!=METHODS[0] else timing["preprocess_time"]+timing["representation_times"][method],
              "preprocess_build_count":1,"granularization_build_count":0 if method==METHODS[0] else 1,
              "representation_build_count":1,"KMeans_call_count":10}); info.update(granular if method!=METHODS[0] else
              {"balls_per_view":"[]","total_ball_count":0,"fallback_count":0,"fallback_fraction":0.0})
            diagnostics.append(info); rep_dir=root/"representations"/dataset; rep_dir.mkdir(parents=True,exist_ok=True)
            np.save(str(rep_dir/(method+".npy")),F)
            rep_hash=hashlib.sha256(np.ascontiguousarray(F).view(np.uint8)).hexdigest()
            for seed in SEEDS:
                try:
                    tick=time.perf_counter(); pred=KMeans(n_clusters=int(cfg["K"]),n_init=20,max_iter=300,random_state=seed).fit_predict(F); kt=time.perf_counter()-tick
                    dataset_kmeans[method]+=kt; pred_dir=root/"predictions"/dataset/method; pred_dir.mkdir(parents=True,exist_ok=True)
                    pred_path=pred_dir/("seed_%d.npy"%seed); np.save(str(pred_path),pred)
                    meta={"dataset":dataset,"method":method,"seed":seed,"K":cfg["K"],"config_hash":config_hash,
                      "code_hash":code_hash,"selection_hash":selection["selection_hash"],"representation_hash":rep_hash,"status":"success","labels_loaded":False}
                    pred_path.with_suffix(".json").write_text(json.dumps(meta,indent=2),encoding="utf-8")
                    records.append({"dataset":dataset,"method":method,"seed":seed,"status":"success","prediction_path":str(pred_path),"kmeans_time":kt,
                      "representation_hash":rep_hash,"selection_hash":selection["selection_hash"],"config_hash":config_hash,"paper_candidate_hash":paper_hash})
                except Exception as error:
                    records.append({"dataset":dataset,"method":method,"seed":seed,"status":"unexpected_failure"})
                    failures.append({"dataset":dataset,"method":method,"seed":seed,"status":"unexpected_failure","exception_type":type(error).__name__,"exception_message":str(error),"stage":"KMeans","reproducible":True})
        for method in METHODS:
            rep_cost=next(x["representation_build_time"] for x in diagnostics if x["dataset"]==dataset and x["method"]==method)
            total=load_time+rep_cost+dataset_kmeans[method]
            runtime_rows.append({"dataset":dataset,"method":method,"data_load_time":load_time,"preprocess_time":timing["preprocess_time"],
              "granularization_time":"N/A" if method==METHODS[0] else timing["granularization_time"],
              "relation_time":"N/A" if method in METHODS[:2] else timing["relation_time"],"fusion_time":timing["representation_times"][method],
              "representation_build_time":rep_cost,"shared_cost":load_time+rep_cost,"kmeans_total_time":dataset_kmeans[method],
              "multi_seed_total_time":total,"amortized_time_per_seed":total/10.0})
        print("completed",dataset,"balls",counts,flush=True)
    pd.DataFrame(records).to_csv(root/"phase_a"/"run_records.csv",index=False); pd.DataFrame(diagnostics).to_csv(root/"diagnostics"/"representation_diagnostics.csv",index=False)
    pd.DataFrame(granular_rows).to_csv(root/"diagnostics"/"granular_diagnostics.csv",index=False); pd.DataFrame(runtime_rows).to_csv(root/"runtime"/"runtime.csv",index=False)
    pd.DataFrame(failures,columns=["dataset","method","seed","status","exception_type","exception_message","stage","reproducible"]).to_csv(root/"failures"/"failures.csv",index=False)
    phase["terminal_records"]=len(records); phase["success_records"]=sum(r["status"]=="success" for r in records)
    phase["status"]="COMPLETE" if len(records)==planned else "INCOMPLETE"; phase["phase_a_complete"]=len(records)==planned
    (root/"phase_a_manifest.json").write_text(json.dumps(phase,indent=2),encoding="utf-8")
    if len(records)!=planned: raise RuntimeError("planned run accounting failure")
    return phase


if __name__=="__main__": print(json.dumps(run(),indent=2))
