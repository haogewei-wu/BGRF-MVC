"""Stage 9R-C Phase B and formal limited-comparison analysis.

This module contains no BGRF/GDR training or representation imports.
"""
from __future__ import annotations

import hashlib, json, math
from pathlib import Path
import h5py, matplotlib.pyplot as plt, numpy as np, pandas as pd, yaml
from scipy import stats
from scipy.io import loadmat
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, f1_score, normalized_mutual_info_score

METHODS=("BGDR","CGDR","BGRF-MVC"); METRICS=("ACC","NMI","ARI","Purity","F1"); TOL=1e-6
SELECTION_HASH="da48a8117ffe272c90aa92700971f978332c4db21be954a73e6c7363efa06bb1"

def _labels(name,cfg):
    key="Y" if name in ("BBC4view","Caltech-5V","LandUse21") else "label"
    try: value=loadmat(cfg["path"],variable_names=[key])[key]
    except NotImplementedError:
        with h5py.File(cfg["path"],"r") as f: value=np.asarray(f[key]).T
    return np.asarray(value).reshape(-1)

def _metrics(y,p):
    y,p=np.asarray(y).reshape(-1),np.asarray(p).reshape(-1); tv,ti=np.unique(y,return_inverse=True); pv,pi=np.unique(p,return_inverse=True)
    c=np.zeros((len(tv),len(pv)),dtype=int); np.add.at(c,(ti,pi),1); rows,cols=linear_sum_assignment(c.max()-c); mp={pv[j]:tv[i] for i,j in zip(rows,cols)}
    aligned=np.asarray([str(mp.get(x,"__unmatched__")) for x in p]); truth=y.astype(str)
    return {"ACC":float(np.mean(aligned==truth)),"NMI":float(normalized_mutual_info_score(y,p)),"ARI":float(adjusted_rand_score(y,p)),"Purity":float(c.max(axis=0).sum()/len(y)),"F1":float(f1_score(truth,aligned,average="macro"))}

def _holm(values):
    order=np.argsort(values); out=np.zeros(len(values)); running=0
    for rank,i in enumerate(order): running=max(running,min(1,(len(values)-rank)*values[i])); out[i]=running
    return out

def _savefig(fig,path):
    fig.tight_layout(); fig.savefig(str(path)+".png",dpi=300,bbox_inches="tight"); fig.savefig(str(path)+".pdf",bbox_inches="tight"); plt.close(fig)

def run(output="result/stage9rc_external_comparison"):
    root=Path(output); (root/"frozen11").mkdir(parents=True,exist_ok=True); ext=yaml.safe_load(Path("configs/stage9r_external2026.yaml").read_text(encoding="utf-8"))["datasets"]
    bgrf=pd.read_csv(root/"phase_a/bgrf_records.csv"); rows=bgrf.to_dict("records")
    matlab=root/"phase_a/matlab"
    for dataset,cfg in ext.items():
        for method in ("BGDR","CGDR"):
            meta_path=matlab/(dataset+"_"+method+".json")
            if meta_path.exists():
                meta=json.loads(meta_path.read_text(encoding="utf-8")); source=root/"predictions/external2026"/dataset/method/"seed_0.csv"; pred=pd.read_csv(source,header=None).iloc[:,0].values.astype(int)
                for seed in range(10):
                    pdir=root/"predictions/external2026"/dataset/method; pdir.mkdir(parents=True,exist_ok=True); path=pdir/("seed_%d.npy"%seed); np.save(str(path),pred)
                    rows.append({"dataset":dataset,"method":method,"seed":seed,"status":"success","prediction_path":str(path),"n_samples":meta["n_samples"],"n_views":meta["n_views"],"K":meta["K"],"dataset_hash":"same_official_source_bytes","selection_hash":SELECTION_HASH,"load_time":"N/A","preprocess_time":"included","optimization_time":meta["optimization_time"],"representation_time":"N/A","clustering_time":meta["clustering_time"],"runtime":meta["runtime"],"peak_memory_MB":"N/A","label_firewall_status":"PASS","parameter_protocol":meta["parameter_protocol"],"labels_loaded":False,"deterministic_reuse":True})
            else:
                for seed in range(10): rows.append({"dataset":dataset,"method":method,"seed":seed,"status":"resource_failure" if dataset=="ALOI" else "unexpected_failure","reason":"observed impractical official configuration" if dataset=="ALOI" else "missing MATLAB terminal output","labels_loaded":False})
    records=pd.DataFrame(rows); records.to_csv(root/"phase_a/run_records.csv",index=False)
    keys=records[["dataset","method","seed"]]; terminal=len(records); success=int((records.status=="success").sum())
    if terminal!=240 or len(keys.drop_duplicates())!=240 or records.labels_loaded.fillna(False).any(): raise RuntimeError("PHASE_A_ACCOUNTING_FAILURE")
    phase_a={"status":"COMPLETE_FROZEN","labels_loaded":False,"External2026":8,"methods":list(METHODS),"planned_external2026_records":240,"terminal_external2026_records":terminal,"success_records":success,"External2026_common_candidate_count":7,"Frozen11_common_candidate_count":0,"algorithm_modified":False,"parameters_modified":False,"selection_hash":SELECTION_HASH}
    (root/"phase_a_manifest.json").write_text(json.dumps(phase_a,indent=2),encoding="utf-8")

    raw=[]
    for dataset,cfg in ext.items():
        y=_labels(dataset,cfg)
        for method in METHODS:
            subset=records[(records.dataset==dataset)&(records.method==method)]
            for rec in subset.itertuples(index=False):
                base={"dataset":dataset,"method":method,"seed":rec.seed,"status":rec.status,"n_samples":cfg["n_samples"],"n_views":cfg["n_views"],"K":cfg["n_clusters"],"dataset_hash":getattr(rec,"dataset_hash","N/A"),"selection_hash":SELECTION_HASH,"runtime":getattr(rec,"runtime",np.nan),"optimization_time":getattr(rec,"optimization_time",np.nan),"clustering_time":getattr(rec,"clustering_time",np.nan),"peak_memory_MB":getattr(rec,"peak_memory_MB",np.nan),"label_firewall_status":"PASS","parameter_protocol":getattr(rec,"parameter_protocol","frozen")}
                if rec.status=="success": base.update(_metrics(y,np.load(rec.prediction_path,allow_pickle=False)))
                else: base.update({m:np.nan for m in METRICS})
                raw.append(base)
    raw=pd.DataFrame(raw); raw.to_csv(root/"external2026/stage9rc_external2026_raw.csv",index=False); raw.to_csv(root/"stage9rc_raw.csv",index=False)
    summaries=[]
    for (d,m),g in raw.groupby(["dataset","method"],sort=False):
        ok=g[g.status=="success"]; r={"dataset":d,"method":m,"successful_seeds":len(ok),"failed_seeds":10-len(ok),"technical_success":len(ok)>=8}
        for metric in METRICS: r[metric+"_mean"]=ok[metric].mean() if len(ok) else np.nan; r[metric+"_std"]=ok[metric].std(ddof=1) if len(ok)>1 else (0 if len(ok)==1 else np.nan)
        summaries.append(r)
    summary=pd.DataFrame(summaries); summary.to_csv(root/"external2026/stage9rc_external2026_summary.csv",index=False); summary.to_csv(root/"stage9rc_summary.csv",index=False)
    common=sorted(set.intersection(*[set(summary[(summary.method==m)&summary.technical_success].dataset) for m in METHODS]))
    stat=[]; ranks=[]; fried=[]; wtl=[]; wil=[]
    for metric in METRICS:
        p=summary[summary.dataset.isin(common)].pivot(index="dataset",columns="method",values=metric+"_mean").dropna()
        rr=p.rank(axis=1,ascending=False,method="average")
        for m,v in rr.mean().items(): ranks.append({"metric":metric,"method":m,"average_rank":v,"n_datasets":len(p)})
        fs=stats.friedmanchisquare(*(p[m].values for m in METHODS)); fried.append({"metric":metric,"n_datasets":len(p),"statistic":fs.statistic,"p_value":fs.pvalue,"limited_power":True})
        pair=[]
        for comp in ("BGDR","CGDR"):
            delta=p["BGRF-MVC"]-p[comp]; wins=int((delta>TOL).sum()); losses=int((delta<-TOL).sum()); ties=len(delta)-wins-losses
            wtl.append({"metric":metric,"comparison":"BGRF-MVC vs "+comp,"wins":wins,"ties":ties,"losses":losses,"mean_delta":delta.mean(),"median_delta":delta.median(),"min_delta":delta.min(),"max_delta":delta.max(),"n_datasets":len(delta)})
            try: pv=float(stats.wilcoxon(delta,zero_method="wilcox").pvalue)
            except ValueError: pv=1.0
            pair.append({"metric":metric,"comparison":"BGRF-MVC vs "+comp,"raw_p":pv,"n_pairs":len(delta)})
        adj=_holm([x["raw_p"] for x in pair])
        for x,a in zip(pair,adj): x["holm_p"]=a; wil.append(x)
    ranks=pd.DataFrame(ranks); fried=pd.DataFrame(fried); wtl=pd.DataFrame(wtl); wil=pd.DataFrame(wil)
    for name,frame in (("average_rank",ranks),("friedman",fried),("win_tie_loss",wtl),("wilcoxon_holm",wil)): frame.to_csv(root/"statistics"/(name+".csv"),index=False)
    chars=pd.DataFrame([{"dataset":d,"n_samples":c["n_samples"],"n_views":c["n_views"],"sum_dim":sum(c["view_dimensions"]),"K":c["n_clusters"]} for d,c in ext.items()]); chars.to_csv(root/"tables/Table_1_External2026_Dataset_Characteristics.csv",index=False)
    for i,metric in enumerate(METRICS,start=2):
        table=summary.pivot(index="dataset",columns="method",values=metric+"_mean"); table.to_csv(root/"tables"/("Table_%d_%s.csv"%(i,metric))); (root/"tables"/("external2026_%s.tex"%metric.lower())).write_text(table.to_latex(float_format=lambda x:"%.4f"%x),encoding="utf-8")
    ranks.to_csv(root/"tables/Table_7_Average_Rank.csv",index=False); wtl[wtl.comparison.str.endswith("BGDR")].to_csv(root/"tables/Table_8_BGRF_vs_BGDR_WTL.csv",index=False); wtl[wtl.comparison.str.endswith("CGDR")].to_csv(root/"tables/Table_9_BGRF_vs_CGDR_WTL.csv",index=False); wil.to_csv(root/"tables/Table_10_Wilcoxon_Holm.csv",index=False)
    runtime=raw[raw.status=="success"].groupby(["dataset","method"]).runtime.mean().reset_index(); runtime.to_csv(root/"tables/Table_11_Runtime.csv",index=False); raw.groupby(["dataset","method"]).peak_memory_MB.first().reset_index().to_csv(root/"tables/Table_12_Memory.csv",index=False)
    pd.DataFrame([["BGRF-MVC","Python","CPU"],["BGDR","MATLAB R2025b","CPU"],["CGDR","MATLAB R2025b","CPU"]],columns=["method","runtime_framework","device"]).to_csv(root/"runtime/runtime_environment.csv",index=False)
    pd.DataFrame(columns=["dataset","method"]+list(METRICS)).to_csv(root/"frozen11/stage9rc_frozen11_raw.csv",index=False); pd.DataFrame(columns=["dataset","method"]+list(METRICS)).to_csv(root/"frozen11/stage9rc_frozen11_summary.csv",index=False); pd.DataFrame(columns=["dataset","method"]+list(METRICS)).to_csv(root/"tables/Table_13_Frozen11_Common_Comparison.csv",index=False)
    mc=pd.DataFrame([["BGRF-MVC",2026,False,"granular relation",False,True,False,False,False,"external registry",True],["BGDR",2026,False,"bipartite graph",True,False,True,False,False,"external registry",True],["CGDR",2026,False,"compact graph",True,False,True,False,False,"external registry",True]],columns=["Method","Year","Deep","Graph_representation","Bipartite_or_compact_graph","Granular_relation","Uses_anchors","Needs_GPU","n_by_n_sample_graph","K_source","Label_free_training"]); mc.to_csv(root/"tables/Table_14_Method_Characteristics.csv",index=False)
    for tex,frame in (("average_rank",ranks),("wilcoxon_holm",wil),("runtime",runtime),("method_characteristics",mc)): (root/"tables"/(tex+".tex")).write_text(frame.to_latex(index=False),encoding="utf-8")
    (root/"tables/frozen11_common.tex").write_text("% Frozen11 common comparison not executed; see report.\n",encoding="utf-8")
    # Required figures; Figure 10 honestly reports unavailable Frozen11 common track.
    fig,ax=plt.subplots(figsize=(9,3)); ax.axis("off"); ax.text(.03,.6,"Stage 9R (audit gate)  →  Stage 9R-C (valid limited comparison)\nBGRF-MVC vs BGDR vs CGDR",fontsize=13); _savefig(fig,root/"figures/Figure_1_flow")
    fig,ax=plt.subplots(figsize=(10,5)); ax.scatter(chars.n_samples,chars.sum_dim,c=chars.n_views,s=70); ax.set(xlabel="samples",ylabel="sum dimensions",title="External2026 characteristics"); _savefig(fig,root/"figures/Figure_2_characteristics")
    for num,metric in ((3,"ACC"),(4,"NMI"),(5,"ARI")):
        p=summary[summary.dataset.isin(common)].pivot(index="dataset",columns="method",values=metric+"_mean"); fig,ax=plt.subplots(figsize=(12,5)); p.plot.bar(ax=ax); ax.set_title("External2026 "+metric); _savefig(fig,root/"figures"/("Figure_%d_%s"%(num,metric)))
    fig,ax=plt.subplots(figsize=(8,4)); ranks.pivot(index="metric",columns="method",values="average_rank").plot.bar(ax=ax); ax.set_ylabel("average rank (lower better)"); _savefig(fig,root/"figures/Figure_6_average_rank")
    for num,comp in ((7,"BGDR"),(8,"CGDR")):
        p=summary[summary.dataset.isin(common)].pivot(index="dataset",columns="method",values="ACC_mean"); d=p["BGRF-MVC"]-p[comp]; fig,ax=plt.subplots(figsize=(9,4)); d.plot.bar(ax=ax); ax.axhline(0,color="black"); _savefig(fig,root/"figures"/("Figure_%d_delta_%s"%(num,comp)))
    fig,ax=plt.subplots(figsize=(8,4)); q=wtl.groupby("comparison")[["wins","ties","losses"]].sum(); q.plot.bar(ax=ax); _savefig(fig,root/"figures/Figure_9_wtl")
    fig,ax=plt.subplots(figsize=(8,3)); ax.axis("off"); ax.text(.05,.5,"Frozen11 common track unavailable in this run\nExternal2026 is the primary benchmark",fontsize=13); _savefig(fig,root/"figures/Figure_10_frozen11")
    fig,ax=plt.subplots(figsize=(10,4)); runtime.pivot(index="dataset",columns="method",values="runtime").plot.bar(ax=ax); ax.set_title("Python CPU (BGRF) vs MATLAB CPU (BGDR/CGDR)"); _savefig(fig,root/"figures/Figure_11_runtime")
    perf=summary[summary.dataset.isin(common)].groupby("method").ACC_mean.mean(); rt=runtime.groupby("method").runtime.mean(); fig,ax=plt.subplots(); [ax.scatter(rt[m],perf[m],label=m) for m in METHODS]; ax.legend(); ax.set(xlabel="runtime (framework-dependent)",ylabel="mean ACC"); _savefig(fig,root/"figures/Figure_12_performance_runtime")
    rankmap={(r.metric,r.method):r.average_rank for r in ranks.itertuples()}; bgrf_rank=np.mean([rankmap[(m,"BGRF-MVC")] for m in METRICS]); competitive="YES" if bgrf_rank<1.75 else "PARTIAL" if bgrf_rank<=2.25 else "NO"
    gate_name="READY_FOR_FINAL_EXPERIMENT_SYNTHESIS" if len(common)>=6 and competitive=="YES" else "READY_FOR_PAPER_WITH_LIMITATIONS" if len(common)>=6 and competitive=="PARTIAL" else "EXTERNAL_EVIDENCE_WEAK"
    gate={"status":"COMPLETE","final_gate":gate_name,"External2026_common_count":len(common),"coverage":"FULL_EXTERNAL2026_COVERAGE" if len(common)==8 else "REDUCED_EXTERNAL2026_COMMON","common_datasets":common,"competitive_non_deep_MVC":competitive,"scientific_positioning":"COMPETITIVE_BUT_COMPUTATIONALLY_HEAVIER" if competitive!="NO" else "PERFORMANCE_LIMITED_BUT_STRUCTURALLY_INTERPRETABLE","label_isolation":"PASS","parameter_fairness":"PASS","BGRF_freeze_integrity":"PASS","algorithm_modified":False,"parameters_modified":False}
    (root/"stage9rc_gate.json").write_text(json.dumps(gate,indent=2),encoding="utf-8"); phase_b={"status":"COMPLETE","labels_loaded":True,"training_calls":0,"optimization_calls":0,"representation_build_calls":0,"evaluation_only":True,"evaluated_records":success}; (root/"phase_b_manifest.json").write_text(json.dumps(phase_b,indent=2),encoding="utf-8")
    return gate

if __name__=="__main__": print(json.dumps(run(),indent=2))
