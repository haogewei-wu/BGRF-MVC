"""Stage 6 Phase B: evaluate frozen predictions without model learning."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, f1_score, normalized_mutual_info_score

from bgrfmvc.evaluation.evaluate_predictions import compute_metrics
from bgrfmvc.evaluation.labels import load_evaluation_labels
from bgrfmvc.experiments.stage5r_hybrid import METHODS
from bgrfmvc.experiments.stage6_diagnostics import paired_stats


METRICS=("ACC","NMI","ARI","PUR","F_score")
PRIMARY=("MSRCV1","Handwritten_numerals")


def _mapping(y,pred):
    truth,ti=np.unique(y,return_inverse=True); clusters,pi=np.unique(pred,return_inverse=True)
    table=np.zeros((truth.size,clusters.size),dtype=np.int64); np.add.at(table,(ti,pi),1)
    rows,cols=linear_sum_assignment(table.max()-table)
    return {clusters[c]:truth[r] for r,c in zip(rows,cols)}


def _subset_metrics(y,pred,indices,mapping):
    indices=np.asarray(indices,dtype=int); aligned=np.asarray([mapping.get(x,"__unmatched__") for x in pred[indices]],dtype=object)
    true=np.asarray([str(x) for x in y[indices]]); aligned=np.asarray([str(x) for x in aligned])
    return {"ACC":float(np.mean(aligned==true)),"F_score":float(f1_score(true,aligned,average="macro")),
            "NMI":float(normalized_mutual_info_score(y[indices],pred[indices])),
            "ARI":float(adjusted_rand_score(y[indices],pred[indices]))}


def _configs(manifest,datasets):
    out={}
    for slug in datasets:
        matches=[(k,v) for k,v in manifest.items() if k.split("__")[-1]==slug and v.get("enabled")]
        if len(matches)!=1: raise RuntimeError("manifest mismatch")
        out[slug]=matches[0]
    return out


def evaluate(root_path,datasets_config,stage5_root):
    root=Path(root_path); phase=json.loads((root/"phase_a_manifest.json").read_text(encoding="utf-8"))
    if phase.get("labels_loaded") is not False or not phase.get("phase_a_complete"): raise RuntimeError("Phase A incomplete")
    for d in phase["datasets"]:
        for m in list(METHODS)+["D3_DimCtrl"]:
            for s in phase["seeds"]:
                if not (root/"predictions"/d/m/("seed_%d"%s)/"predictions.csv").is_file(): raise RuntimeError("prediction set incomplete")
    manifest=yaml.safe_load(Path(datasets_config).read_text(encoding="utf-8"))["datasets"]
    configs=_configs(manifest,phase["datasets"]); rep=pd.read_csv(root/"diagnostics"/"representation_diagnostics.csv"); dim=pd.read_csv(root/"dimension_control"/"dimension_control_phase_a.csv")
    stage5=pd.read_csv(Path(stage5_root)/"hybrid_raw.csv")
    strata_assign=pd.read_csv(root/"boundary"/"boundary_strata_assignments.csv")
    raw=[]; subset=[]; loo=[]
    for dataset in phase["datasets"]:
        key,cfg=configs[dataset]; y=np.asarray(load_evaluation_labels(cfg)).reshape(-1)
        for method in list(METHODS)+["D3_DimCtrl"]:
            for seed in phase["seeds"]:
                path=root/"predictions"/dataset/method/("seed_%d"%seed)/"predictions.csv"
                pred=pd.read_csv(path)["predicted_label"].values; metrics=compute_metrics(y,pred)
                if method=="D3_DimCtrl": info=dim[(dim.dataset==dataset)&(dim.seed==seed)].iloc[0]
                else: info=rep[(rep.dataset==dataset)&(rep.method==method)].iloc[0]
                row={"dataset":dataset,"dataset_key":key,"method":method,"seed":seed,"n_samples":len(y),"n_clusters":cfg["n_clusters"],
                     "dimension":int(info.dimension if method=="D3_DimCtrl" else info.representation_dim),
                     "effective_rank":float(info.effective_rank),"labels_loaded_phase_a":False}
                row.update(metrics); row["Composite"]=float(np.mean([metrics[x] for x in METRICS])); raw.append(row)
                if method in (METHODS[2],METHODS[3]):
                    mapping=_mapping(y,pred); ds_strata=strata_assign[strata_assign.dataset==dataset]
                    for stratum in ("near_boundary","middle","interior_far"):
                        idx=ds_strata[ds_strata.stratum==stratum].sample_index.values
                        item={"dataset":dataset,"method":method,"seed":seed,"stratum":stratum,"n_samples":len(idx),"mapping_source":"full_dataset_once"}
                        item.update(_subset_metrics(y,pred,idx,mapping)); subset.append(item)
        if dataset=="MSRCV1":
            lm=pd.read_csv(root/"msrcv1_failure"/"leave_one_view_out_manifest.csv")
            for item in lm.itertuples():
                pred=pd.read_csv(item.prediction_path)["predicted_label"].values; metrics=compute_metrics(y,pred)
                row={"dataset":dataset,"omitted_view":item.omitted_view,"method":item.method,"seed":item.seed,"labels_loaded_phase_a":False}; row.update(metrics); row["Composite"]=float(np.mean([metrics[x] for x in METRICS])); loo.append(row)
    pd.DataFrame(raw).to_csv(root/"stage6_raw.csv",index=False)
    summary=pd.DataFrame(raw).groupby(["dataset","method"],as_index=False).agg({m:["mean","std"] for m in METRICS+("Composite",)})
    summary.columns=["_".join(x).rstrip("_") for x in summary.columns]; summary.to_csv(root/"stage6_summary.csv",index=False)
    pd.DataFrame(subset).to_csv(root/"boundary"/"boundary_strata_evaluation.csv",index=False)
    loo_df=pd.DataFrame(loo); loo_df.to_csv(root/"msrcv1_failure"/"msrcv1_leave_one_view_out_raw.csv",index=False)
    loo_summary=loo_df.groupby(["omitted_view","method"],as_index=False).agg({m:["mean","std"] for m in METRICS+("Composite",)})
    loo_summary.columns=["_".join(x).rstrip("_") for x in loo_summary.columns]; loo_summary.to_csv(root/"msrcv1_failure"/"msrcv1_leave_one_view_out_summary.csv",index=False)
    keyed={(r["dataset"],r["method"],r["seed"]):r for r in raw}; paired=[]
    for d in phase["datasets"]:
        for s in phase["seeds"]:
            for name,before,after in (("D3-D0",METHODS[0],METHODS[3]),("D3-D1",METHODS[1],METHODS[3]),("D3-D2",METHODS[2],METHODS[3]),("DimCtrl-D3",METHODS[3],"D3_DimCtrl"),("DimCtrl-D0",METHODS[0],"D3_DimCtrl")):
                paired.append({"dataset":d,"seed":s,"comparison":name,"delta":keyed[(d,after,s)]["Composite"]-keyed[(d,before,s)]["Composite"],"primary":d in PRIMARY})
    paired_df=pd.DataFrame(paired); paired_df.to_csv(root/"paired_performance.csv",index=False)
    stats=[]
    for comparison in paired_df.comparison.unique():
        for scope in ("all","primary","3sources","MSRCV1","Handwritten_numerals"):
            values=paired_df[paired_df.comparison==comparison]
            if scope=="primary": values=values[values.primary]
            elif scope!="all": values=values[values.dataset==scope]
            item={"comparison":comparison,"scope":scope}; item.update(paired_stats(values.delta.values)); stats.append(item)
    pd.DataFrame(stats).to_csv(root/"paired_statistics.csv",index=False)
    # Efficiency summary and ratios.
    eff=pd.read_csv(root/"efficiency"/"efficiency_raw.csv")
    eff_summary=eff.groupby(["dataset","method"],as_index=False).agg({"total_runtime":"mean","peak_memory_MB":"mean","representation_memory_MB":"mean","kmeans_time":"mean"})
    ratios=[]
    for d in phase["datasets"]:
        base=eff_summary[(eff_summary.dataset==d)&(eff_summary.method==METHODS[0])].iloc[0]
        full=eff_summary[(eff_summary.dataset==d)&(eff_summary.method==METHODS[3])].iloc[0]
        d0dim=int(rep[(rep.dataset==d)&(rep.method==METHODS[0])].representation_dim.iloc[0]); d3dim=int(rep[(rep.dataset==d)&(rep.method==METHODS[3])].representation_dim.iloc[0])
        ratios.append({"dataset":d,"dimension_ratio_D3_vs_D0":d3dim/d0dim,"runtime_ratio_D3_vs_D0":full.total_runtime/base.total_runtime,"memory_ratio_D3_vs_D0":full.peak_memory_MB/base.peak_memory_MB})
    eff_summary.to_csv(root/"efficiency"/"efficiency_summary.csv",index=False); pd.DataFrame(ratios).to_csv(root/"efficiency"/"efficiency_ratios.csv",index=False)
    phase_b={"stage":"6","phase":"B","phase_a_complete_before_labels":True,"models_refit":False,"labels_loaded":True,"outputs_complete":True}
    (root/"phase_b_manifest.json").write_text(json.dumps(phase_b,indent=2),encoding="utf-8")
    return raw,paired_df,subset,loo_df


def main():
    p=argparse.ArgumentParser(); p.add_argument("--root",default="result/stage6_validation"); p.add_argument("--datasets-config",default="configs/datasets.yaml"); p.add_argument("--stage5-root",default="result/stage5r_hybrid"); a=p.parse_args(); evaluate(a.root,a.datasets_config,a.stage5_root)


if __name__=="__main__": main()
