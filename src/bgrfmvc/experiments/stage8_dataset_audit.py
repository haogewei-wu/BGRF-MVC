"""Stage 8 label-free dataset audit and deterministic selection freeze gate."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import yaml

from bgrfmvc.data.audit import SUPPORTED, inspect_file


AUDIT_COLUMNS = (
    "dataset", "path", "load_status", "n_samples", "n_views", "view_shapes",
    "raw_dimensions", "sum_raw_dimensions", "configured_K", "K_source",
    "K_source_file", "orientation_status", "sample_alignment_status",
    "nan_count", "inf_count", "constant_feature_count", "label_file_exists",
    "label_content_loaded", "eligible", "ineligible_reason",
)


def _manifest(root):
    return yaml.safe_load((root / "configs" / "datasets.yaml").read_text(encoding="utf-8"))["datasets"]


def _number(notes, prefix):
    for part in str(notes).split(";"):
        part = part.strip()
        if part.startswith(prefix):
            try: return int(float(part.split("=", 1)[1]))
            except (ValueError, IndexError): return None
    return None


def _decimal(notes, prefix):
    for part in str(notes).split(";"):
        part=part.strip()
        if part.startswith(prefix):
            try: return float(part.split("=",1)[1])
            except (ValueError,IndexError): return None
    return None


def audit(project_root):
    root=Path(project_root).resolve(); data_root=root/"datasets-multiview"
    output=root/"result"/"stage8_main_benchmark"; output.mkdir(parents=True,exist_ok=True)
    (output/"audit").mkdir(exist_ok=True); manifest=_manifest(root)
    files=sorted(p for p in data_root.rglob("*") if p.is_file() and
                 not p.name.startswith("._") and p.suffix.lower() in SUPPORTED)
    rows=[]
    for path in files:
        inspected,_=inspect_file(data_root,path); name=inspected["dataset"]
        cfg=manifest.get(name,{})
        try: shapes=json.loads(inspected.get("view_dimensions") or "[]")
        except ValueError: shapes=[]
        n_views=int(inspected.get("n_views") or 0); n=int(inspected.get("n_samples") or 0)
        raw=[int(x[1]) for x in shapes if isinstance(x,list) and len(x)==2]
        notes=str(inspected.get("notes") or "")
        parse_failure=notes.startswith("parse/read blocked")
        aligned=bool(n>=2 and shapes and all(int(x[0])==n for x in shapes))
        nan=_number(notes,"nan_count") or 0; inf=_number(notes,"inf_count") or 0
        constant_ratio=_decimal(notes,"constant_column_ratio")
        constant_count=(int(round(constant_ratio*sum(raw))) if constant_ratio is not None and raw else "")
        k=cfg.get("n_clusters"); k_source=cfg.get("k_source")
        label_exists=bool(inspected.get("contains_label_key"))
        reasons=[]
        if parse_failure: reasons.append("INELIGIBLE_DATA_UNREADABLE")
        if n_views<2: reasons.append("INELIGIBLE_FEWER_THAN_TWO_VIEWS")
        if not aligned: reasons.append("INELIGIBLE_SAMPLE_ALIGNMENT")
        if nan or inf: reasons.append("INELIGIBLE_NONFINITE_FEATURES")
        if not isinstance(k,int) or not k_source: reasons.append("INELIGIBLE_K_UNAVAILABLE")
        if not label_exists or not cfg.get("label_key"): reasons.append("INELIGIBLE_EVALUATION_LABEL_ABSENT")
        orientation="transposed" if "transpose required" in notes else (
                    "aligned_rows" if "sample axis inferred from aligned rows" in notes else "ambiguous")
        if orientation=="ambiguous": reasons.append("INELIGIBLE_AMBIGUOUS_ORIENTATION")
        rows.append({"dataset":name,"path":inspected["file_path"],
          "load_status":"failure" if parse_failure else "success","n_samples":n or "",
          "n_views":n_views,"view_shapes":json.dumps(shapes),"raw_dimensions":json.dumps(raw),
          "sum_raw_dimensions":sum(raw) if raw else "","configured_K":k if isinstance(k,int) else "",
          "K_source":k_source or "","K_source_file":"configs/datasets.yaml" if k_source else "",
          "orientation_status":orientation,"sample_alignment_status":"aligned" if aligned else "not_aligned",
          "nan_count":nan,"inf_count":inf,"constant_feature_count":constant_count,
          "label_file_exists":label_exists,"label_content_loaded":False,
          "eligible":not reasons,"ineligible_reason":";".join(dict.fromkeys(reasons))})
    frame=pd.DataFrame(rows,columns=AUDIT_COLUMNS); frame.to_csv(output/"dataset_audit.csv",index=False)
    eligible=frame[frame.eligible==True].sort_values(["n_samples","dataset"]).reset_index(drop=True)  # noqa:E712
    eligible.to_csv(output/"audit"/"eligible_datasets.csv",index=False)
    frame[frame.eligible==False].to_csv(output/"audit"/"ineligible_datasets.csv",index=False)  # noqa:E712
    status="DATASET_SELECTION_FROZEN" if len(eligible)>=15 else "STAGE8_INSUFFICIENT_ELIGIBLE_DATASETS"
    selection={"status":status,"selection_rule":"eligible sorted by (n_samples,dataset), then 15 evenly spaced positions",
      "candidate_files_scanned":len(frame),"eligible_dataset_count":len(eligible),
      "required_dataset_count":15,"selected_dataset_names":[],"selected_dataset_paths":[],
      "label_content_loaded":False,"selection_manifest_hash":None,
      "formal_predictions_started":False,"blocker":None if len(eligible)>=15 else
      "Only %d datasets have readable aligned multi-view X, independent frozen K, and evaluation-label presence metadata."%len(eligible)}
    if len(eligible)>=15:
        raise NotImplementedError("Selection execution is intentionally separated from audit Gate")
    (output/"dataset_selection.json").write_text(json.dumps(selection,indent=2),encoding="utf-8")
    audit_hash=hashlib.sha256((output/"dataset_audit.csv").read_bytes()).hexdigest()
    (output/"audit"/"audit_manifest.json").write_text(json.dumps({"status":status,
      "dataset_audit_sha256":audit_hash,"labels_loaded":False,"files_scanned":len(frame)},indent=2),encoding="utf-8")
    return selection


if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser(); p.add_argument("--project-root",default=".")
    print(json.dumps(audit(p.parse_args().project_root),indent=2))
