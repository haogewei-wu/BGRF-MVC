"""Evaluation-only Frozen11 supporting track; never trains or transforms."""
import json
from pathlib import Path
import numpy as np, pandas as pd, yaml
from bgrfmvc.evaluation.evaluate_stage8r import _labels, _metrics

DATASETS=("100leaves","3sources","EYaleB10","Handwritten_numerals","MSRCV1","Caltech101-7")
METHODS=("BGDR","CGDR","BGRF-MVC")

def run(output="result/stage9rc_external_comparison"):
    root=Path(output); selected=yaml.safe_load(Path("configs/stage8r_selected_benchmark.yaml").read_text(encoding="utf-8"))["datasets"]
    old=pd.read_csv("result/stage8r_main_benchmark/stage8r_raw.csv"); rows=[]
    for dataset in DATASETS:
        y=_labels(selected[dataset]);
        for method in ("BGDR","CGDR"):
            src=root/"predictions/frozen11"/dataset/method/"seed_0.csv"; pred=pd.read_csv(src,header=None).iloc[:,0].values.astype(int)
            for seed in range(10):
                pdir=root/"predictions/frozen11"/dataset/method; path=pdir/("seed_%d.npy"%seed); np.save(str(path),pred); rows.append({"dataset":dataset,"method":method,"seed":seed,"status":"success",**_metrics(y,pred)})
        reuse=old[(old.dataset==dataset)&(old.method=="D3_BGRFMVC_PaperCandidate_v1")]
        for r in reuse.itertuples(index=False): rows.append({"dataset":dataset,"method":"BGRF-MVC","seed":int(r.seed),"status":"success","ACC":r.ACC,"NMI":r.NMI,"ARI":r.ARI,"Purity":r.Purity,"F1":r.F1,"reused_stage8r":True})
    raw=pd.DataFrame(rows); raw.to_csv(root/"frozen11/stage9rc_frozen11_raw.csv",index=False)
    summary=[]
    for (d,m),g in raw.groupby(["dataset","method"]):
        r={"dataset":d,"method":m,"successful_seeds":len(g)}
        for metric in ("ACC","NMI","ARI","Purity","F1"): r[metric+"_mean"]=g[metric].mean(); r[metric+"_std"]=g[metric].std(ddof=1)
        summary.append(r)
    summary=pd.DataFrame(summary); summary.to_csv(root/"frozen11/stage9rc_frozen11_summary.csv",index=False); summary.to_csv(root/"tables/Table_13_Frozen11_Common_Comparison.csv",index=False)
    (root/"tables/frozen11_common.tex").write_text(summary.to_latex(index=False,float_format=lambda x:"%.4f"%x),encoding="utf-8")
    manifest=json.loads((root/"phase_a_manifest.json").read_text()); manifest["Frozen11_common_candidate_count"]=len(DATASETS); manifest["Frozen11_common_datasets"]=list(DATASETS); manifest["Frozen11_terminal_records"]=len(raw); (root/"phase_a_manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    return {"COMMON_A":len(DATASETS),"records":len(raw)}
if __name__=="__main__": print(json.dumps(run(),indent=2))
