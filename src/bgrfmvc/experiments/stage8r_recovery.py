"""Stage 8R metadata-only eligibility recovery and deterministic freeze."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from bgrfmvc.data.loader import load_multiview_features


def _sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _feature_hash(views):
    h=hashlib.sha256()
    for view in views:
        a=np.ascontiguousarray(np.asarray(view,dtype=np.float64)); h.update(str(a.shape).encode()); h.update(a.view(np.uint8))
    return h.hexdigest()


def _reason_category(reason):
    text=str(reason)
    categories=[]
    mapping=(("K_UNAVAILABLE","K_UNAVAILABLE"),("AMBIGUOUS_ORIENTATION","AMBIGUOUS_ORIENTATION"),
      ("FEWER_THAN_TWO_VIEWS","VIEW_COUNT_MISMATCH"),("SAMPLE_ALIGNMENT","SAMPLE_COUNT_MISMATCH"),
      ("DATA_UNREADABLE","LOAD_FAILURE"),("NONFINITE","NAN_INF"),("EVALUATION_LABEL_ABSENT","LABEL_METADATA_UNRECOGNIZED"))
    for token,name in mapping:
        if token in text: categories.append(name)
    return categories or ["OTHER"]


def recover(project_root="."):
    root=Path(project_root).resolve(); out=root/"result"/"stage8r_main_benchmark"
    for name in ("recovery","audit","frozen_selection","phase_a","phase_b","representations","predictions",
                 "diagnostics","runtime","memory","statistics","figures","tables","failures"):
        (out/name).mkdir(parents=True,exist_ok=True)
    old=pd.read_csv(root/"result"/"stage8_main_benchmark"/"dataset_audit.csv")
    counts={}
    for reason in old.ineligible_reason.fillna(""):
        for category in _reason_category(reason): counts[category]=counts.get(category,0)+1
    pd.DataFrame([{"reason":k,"dataset_count":v} for k,v in sorted(counts.items())]).to_csv(
        out/"recovery"/"original_failure_reason_summary.csv",index=False)
    registry=yaml.safe_load((root/"configs"/"stage8r_dataset_registry.yaml").read_text(encoding="utf-8"))["datasets"]
    by_key={c["dataset_key"]:(name,c) for name,c in registry.items()}; rows=[]; eligible=[]
    for item in old.itertuples(index=False):
        old_reason="" if bool(item.eligible) else str(item.ineligible_reason)
        base={"dataset":item.dataset,"path":item.path,"old_eligible":bool(item.eligible),
              "old_failure_reason":old_reason,"format":Path(item.path).suffix.lower().lstrip("."),
              "container_type":"registered_multiview_container" if item.dataset in by_key else "metadata_audited",
              "candidate_view_variables":"","n_views":item.n_views,"raw_shapes":item.view_shapes,
              "resolved_shapes":item.view_shapes,"orientation_action":item.orientation_status,
              "n_samples":item.n_samples,"sum_dimensions":item.sum_raw_dimensions,
              "alignment_evidence":"STRUCTURAL" if item.sample_alignment_status=="aligned" else "UNKNOWN",
              "K":"","K_source_type":"","K_source_reference":"",
              "label_candidate_exists":bool(item.label_file_exists),"label_content_loaded":False,
              "nan_count":item.nan_count,"inf_count":item.inf_count,
              "constant_feature_count":item.constant_feature_count,"adapter_fix_applied":False,
              "adapter_fix_type":"","new_eligible":False,"new_failure_reason":old_reason,
              "recovery_class":"TRUE_INELIGIBLE","feature_hash":""}
        if item.dataset in by_key:
            canonical,cfg=by_key[item.dataset]
            feature_cfg={k:cfg[k] for k in ("path","view_keys","transpose_views")}
            try:
                views=load_multiview_features(feature_cfg); shapes=[list(x.shape) for x in views]
                finite=all(np.isfinite(x).all() for x in views); n=len(views[0]); k=int(cfg["K"])
                reasons=[]
                if len(views)<2: reasons.append("INELIGIBLE_NOT_MULTIVIEW")
                if len({x.shape[0] for x in views})!=1: reasons.append("INELIGIBLE_ALIGNMENT_UNKNOWN")
                if not finite: reasons.append("INELIGIBLE_NONFINITE_FEATURES")
                if k<2 or k>n: reasons.append("INELIGIBLE_K_SAMPLE_CONSTRAINT")
                base.update({"dataset":canonical,"candidate_view_variables":json.dumps(cfg["view_keys"]),
                  "n_views":len(views),"resolved_shapes":json.dumps(shapes),
                  "orientation_action":"transpose_all" if cfg["transpose_views"] else "keep",
                  "n_samples":n,"sum_dimensions":sum(x.shape[1] for x in views),"alignment_evidence":"STRUCTURAL",
                  "K":k,"K_source_type":cfg["K_source_type"],"K_source_reference":cfg["K_source_reference"],
                  "label_candidate_exists":bool(cfg.get("label_key")),"nan_count":sum(int(np.isnan(x).sum()) for x in views),
                  "inf_count":sum(int(np.isinf(x).sum()) for x in views),
                  "constant_feature_count":sum(int(np.sum(np.ptp(x,axis=0)==0)) for x in views),
                  "adapter_fix_applied":item.dataset.startswith("Caltech101-") or canonical=="100leaves",
                  "adapter_fix_type":"explicit X-only cell extraction" if item.dataset.startswith("Caltech101-") else
                     ("orientation resolved by independent K feasibility" if canonical=="100leaves" else ""),
                  "new_eligible":not reasons,"new_failure_reason":";".join(reasons),"feature_hash":_feature_hash(views)})
                if not reasons:
                    if bool(item.eligible): cls="UNCHANGED_ELIGIBLE"
                    elif base["adapter_fix_applied"]: cls="RECOVERED_LOADER" if canonical.startswith("Caltech") else "RECOVERED_ORIENTATION"
                    elif "K_UNAVAILABLE" in old_reason: cls="RECOVERED_K_METADATA"
                    else: cls="RECOVERED_ALIGNMENT_RULE"
                    base["recovery_class"]=cls; eligible.append((canonical,cfg,base))
            except Exception as error:
                base.update({"new_failure_reason":"INELIGIBLE_DATA_LOAD: %s"%error,"recovery_class":"STILL_BLOCKED_DATA"})
        elif "K_UNAVAILABLE" in old_reason:
            base["recovery_class"]="STILL_BLOCKED_K"
        elif "ALIGN" in old_reason or "ORIENTATION" in old_reason:
            base["recovery_class"]="STILL_BLOCKED_ALIGNMENT"
        else: base["recovery_class"]="STILL_BLOCKED_DATA"
        rows.append(base)
    frame=pd.DataFrame(rows); frame.to_csv(out/"recovery"/"dataset_recovery_audit.csv",index=False)
    frame.to_csv(out/"dataset_recovery_audit.csv",index=False)
    summary=frame.groupby("recovery_class",as_index=False).size().rename(columns={"size":"dataset_count"})
    summary.to_csv(out/"recovery"/"eligibility_recovery_summary.csv",index=False)
    summary.to_csv(out/"eligibility_recovery_summary.csv",index=False)
    # Registry contains exactly one representative per family; feature hashes prove no selected duplicates.
    hashes=[x[2]["feature_hash"] for x in eligible]
    if len(hashes)!=len(set(hashes)): raise RuntimeError("duplicate selected dataset content")
    E=len(eligible)
    if E<10: mode="NONE"; selected=[]; status="STAGE8R_INSUFFICIENT_ELIGIBLE_DATASETS"
    elif E<15: mode="REDUCED_E"; selected=sorted(eligible,key=lambda x:(x[2]["n_samples"],x[0])); status="REDUCED_BENCHMARK_AVAILABLE"
    else:
        ordered=sorted(eligible,key=lambda x:(x[2]["n_samples"],x[0])); indices=np.rint(np.linspace(0,E-1,15)).astype(int)
        selected=[ordered[i] for i in indices]; mode="FULL_15"; status="FULL_BENCHMARK_AVAILABLE"
    selection={"benchmark_mode":mode,"N_BENCH":len(selected),"eligible_dataset_count":E,
      "selection_rule":"all eligible if 10<=E<15; otherwise sorted (n_samples,name) linspace coverage",
      "labels_loaded":False,"datasets":{name:{**{k:cfg[k] for k in ("path","view_keys","label_key","transpose_views","K","K_source_type","K_source_reference","K_source_confidence")},
       "n_samples":int(row["n_samples"]),"n_views":int(row["n_views"]),"sum_dimensions":int(row["sum_dimensions"]),
       "alignment_evidence":row["alignment_evidence"],"feature_hash":row["feature_hash"]} for name,cfg,row in selected}}
    yaml_path=root/"configs"/"stage8r_selected_benchmark.yaml"
    yaml_path.write_text(yaml.safe_dump(selection,allow_unicode=True),encoding="utf-8")
    selection_hash=_sha(yaml_path); result={"status":status,"benchmark_mode":mode,"N_BENCH":len(selected),
      "eligible_dataset_count":E,"selected_datasets":[x[0] for x in selected],"selection_hash":selection_hash,
      "labels_loaded":False,"formal_predictions_started":False}
    (out/"dataset_selection.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    return result


if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser(); p.add_argument("--project-root",default=".")
    print(json.dumps(recover(p.parse_args().project_root),indent=2))
