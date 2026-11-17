"""Create the ten required Stage 6 figures from frozen CSV outputs."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _save(fig, root, name):
    fig.tight_layout(); fig.savefig(str(root/(name+".png")),dpi=300); fig.savefig(str(root/(name+".pdf"))); plt.close(fig)


def plot_all(root_path="result/stage6_validation"):
    root=Path(root_path); out=root/"figures"; out.mkdir(exist_ok=True)
    raw=pd.read_csv(root/"stage6_raw.csv"); paired=pd.read_csv(root/"paired_performance.csv")
    rep=pd.read_csv(root/"diagnostics"/"representation_diagnostics.csv")
    dist=pd.read_csv(root/"diagnostics"/"pairwise_distance_preservation.csv")
    knn=pd.read_csv(root/"diagnostics"/"knn_neighborhood_preservation.csv")
    strata=pd.read_csv(root/"boundary"/"boundary_strata_evaluation.csv")
    view=pd.read_csv(root/"msrcv1_failure"/"msrcv1_view_structure.csv")
    loo=pd.read_csv(root/"msrcv1_failure"/"msrcv1_leave_one_view_out_summary.csv")
    eff=pd.read_csv(root/"efficiency"/"efficiency_summary.csv")
    methods=["D0_RawConcat_KM","D1_BoundaryOnly_Equal_KM","D2_RawGaussianHybrid_KM","D3_RawBoundaryHybrid_KM"]
    labels=["D0","D1","D2","D3"]
    # 1 paired deltas, not mean-only.
    fig,ax=plt.subplots(figsize=(9,4)); data=[]; names=[]
    for comp in ("D3-D0","D3-D1","D3-D2"):
        data.append(paired[(paired.primary)&(paired.comparison==comp)].delta.values); names.append(comp)
    ax.boxplot(data,labels=names,showmeans=True); ax.axhline(0,color="black",lw=.8); ax.set_ylabel("Paired Composite delta"); _save(fig,out,"01_performance_delta")
    # 2 dimension and effective rank.
    fig,axes=plt.subplots(1,2,figsize=(12,4)); x=np.arange(3); datasets=["3sources","MSRCV1","Handwritten_numerals"]
    for i,m in enumerate(methods):
        q=rep[rep.method==m].set_index("dataset"); axes[0].bar(x+(i-1.5)*.2,[q.loc[d,"representation_dim"] for d in datasets],.2,label=labels[i]); axes[1].bar(x+(i-1.5)*.2,[q.loc[d,"effective_rank"] for d in datasets],.2,label=labels[i])
    for ax,title in zip(axes,("Dimension","Effective rank")): ax.set_xticks(x); ax.set_xticklabels(datasets,rotation=15); ax.set_title(title); ax.set_yscale("log"); ax.legend()
    _save(fig,out,"02_dimension_effective_rank")
    # 3 pairwise preservation.
    fig,ax=plt.subplots(figsize=(10,4)); q=dist[dist.reference==methods[0]]; keys=q.dataset+":"+q.candidate.str[:2]; ax.bar(np.arange(len(q)),q.pearson,label="Pearson"); ax.plot(np.arange(len(q)),q.spearman,"o",label="Spearman"); ax.set_xticks(np.arange(len(q))); ax.set_xticklabels(keys,rotation=45,ha="right"); ax.legend(); ax.set_ylim(0,1.05); _save(fig,out,"03_pairwise_distance_preservation")
    # 4 kNN.
    fig,ax=plt.subplots(figsize=(10,4)); keys=knn.dataset+":"+knn.candidate.str[:2]; ax.bar(np.arange(len(knn))-.18,knn.neighbor_overlap,.36,label="overlap"); ax.bar(np.arange(len(knn))+.18,knn.jaccard_overlap,.36,label="Jaccard"); ax.set_xticks(np.arange(len(knn))); ax.set_xticklabels(keys,rotation=45,ha="right"); ax.legend(); _save(fig,out,"04_knn_preservation")
    # 5 boundary strata D2 vs D3.
    fig,axes=plt.subplots(1,3,figsize=(13,4)); order=["near_boundary","middle","interior_far"]
    agg=strata.groupby(["dataset","stratum","method"],as_index=False).ACC.mean()
    for ax,d in zip(axes,["3sources","MSRCV1","Handwritten_numerals"]):
        for m in methods[2:]:
            q=agg[(agg.dataset==d)&(agg.method==m)].set_index("stratum"); ax.plot(order,[q.loc[s,"ACC"] for s in order],marker="o",label=m[:2])
        ax.set_title(d); ax.set_xticklabels(order,rotation=25); ax.legend()
    _save(fig,out,"05_boundary_strata")
    # 6 MSRC view structure.
    fig,axes=plt.subplots(1,2,figsize=(10,4)); axes[0].bar(view.view_id,view.ball_count); axes[0].set_title("Ball count"); axes[1].bar(view.view_id,view.relation_effective_rank); axes[1].set_title("Relation effective rank"); [ax.set_xlabel("view") for ax in axes]; _save(fig,out,"06_msrcv1_view_structure")
    # 7 leave one view out.
    fig,ax=plt.subplots(figsize=(9,4)); x=np.arange(6)
    for i,m in enumerate((methods[0],methods[3])):
        q=loo[loo.method==m].set_index("omitted_view"); ax.bar(x+(i-.5)*.35,[q.loc[j,"Composite_mean"] for j in range(6)],.35,label=labels[0 if i==0 else 3])
    ax.set_xticks(x); ax.set_xlabel("Omitted view"); ax.set_ylabel("Composite"); ax.legend(); _save(fig,out,"07_msrcv1_leave_one_view_out")
    # 8 runtime.
    fig,ax=plt.subplots(figsize=(11,4)); q=eff.pivot(index="dataset",columns="method",values="total_runtime").loc[["3sources","MSRCV1","Handwritten_numerals"]]; q.plot.bar(ax=ax); ax.set_yscale("log"); ax.set_ylabel("seconds"); _save(fig,out,"08_runtime")
    # 9 peak memory.
    fig,ax=plt.subplots(figsize=(11,4)); q=eff.pivot(index="dataset",columns="method",values="peak_memory_MB").loc[["3sources","MSRCV1","Handwritten_numerals"]]; q.plot.bar(ax=ax); ax.set_yscale("log"); ax.set_ylabel("MB"); _save(fig,out,"09_peak_memory")
    # 10 D0/D3/DimCtrl.
    fig,ax=plt.subplots(figsize=(9,4)); agg=raw.groupby(["dataset","method"],as_index=False).Composite.mean(); x=np.arange(3)
    for i,m in enumerate((methods[0],methods[3],"D3_DimCtrl")):
        q=agg[agg.method==m].set_index("dataset"); ax.bar(x+(i-1)*.25,[q.loc[d,"Composite"] for d in datasets],.25,label=("D0","D3","DimCtrl")[i])
    ax.set_xticks(x); ax.set_xticklabels(datasets); ax.set_ylabel("Composite"); ax.legend(); _save(fig,out,"10_dimension_control")


if __name__=="__main__": plot_all()
