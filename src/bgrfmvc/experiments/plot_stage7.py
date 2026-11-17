"""Stage 7 descriptive engineering figures (no model fitting or labels)."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _save(fig, root, name):
    fig.tight_layout(); fig.savefig(str(root/(name+".png")),dpi=300)
    fig.savefig(str(root/(name+".pdf"))); plt.close(fig)


def plot(root_path):
    root=Path(root_path); figures=root/"figures"; figures.mkdir(exist_ok=True)
    bench=pd.read_csv(root/"benchmark"/"benchmark_raw.csv")
    prof=pd.read_csv(root/"profiling"/"reference_profile.csv")
    mem=pd.read_csv(root/"profiling"/"reference_memory.csv")
    eq=pd.read_csv(root/"equivalence"/"numerical_equivalence.csv")
    datasets=["3sources","MSRCV1","Handwritten_numerals"]
    # 1 reference build breakdown
    ref=prof[prof.implementation=="D3_reference"].set_index("dataset")
    fig,ax=plt.subplots(figsize=(8,4)); bottom=np.zeros(3)
    for col,label in (("standardization_time","standardize"),("granularization_time","granularize"),
                      ("boundary_relation_time","relation"),("hybrid_fusion_time","fusion")):
        vals=ref.loc[datasets,col].astype(float).values; ax.bar(datasets,vals,bottom=bottom,label=label); bottom+=vals
    ax.set_ylabel("seconds"); ax.set_title("Reference D3 representation build"); ax.legend(); _save(fig,figures,"01_reference_runtime_breakdown")
    def bars(mode,col,title,name):
        data=bench[bench["mode"]==mode].groupby(["dataset","implementation"])[col].mean().unstack()
        fig,ax=plt.subplots(figsize=(8,4)); data.loc[datasets].plot.bar(ax=ax); ax.set_ylabel("seconds" if "runtime" in col else col); ax.set_title(title); _save(fig,figures,name)
    bars("multi_seed","total_runtime","Reference vs D3-OPT: 10 seeds","02_reference_vs_opt_runtime")
    bars("cold","total_runtime","Cold-run runtime","03_cold_runtime")
    bars("multi_seed","total_runtime","Multi-seed amortized runtime","04_multiseed_runtime")
    ratios=pd.read_csv(root/"benchmark"/"efficiency_ratios.csv").set_index("dataset")
    fig,ax=plt.subplots(figsize=(8,4)); ratios.loc[datasets,["D3_OPT_vs_D0_cold_ratio","D3_OPT_vs_D0_multiseed_ratio"]].plot.bar(ax=ax); ax.axhline(6,color="k",ls="--",lw=1); ax.set_ylabel("runtime ratio"); _save(fig,figures,"05_runtime_ratio_vs_D0")
    fig,ax=plt.subplots(figsize=(8,4)); data=mem.groupby(["dataset","implementation"]).tracemalloc_peak_MB.mean().unstack(); data.loc[datasets].plot.bar(ax=ax); ax.set_yscale("log"); ax.set_ylabel("tracemalloc peak MB (log)"); _save(fig,figures,"06_peak_memory")
    data=bench[bench["mode"]=="multi_seed"].groupby(["dataset","implementation"])[["representation_build_time","kmeans_time"]].mean()
    fig,ax=plt.subplots(figsize=(8,4)); data.loc[(datasets,["D3_reference","D3_OPT"]),:].plot.bar(stacked=True,ax=ax); ax.set_ylabel("seconds"); _save(fig,figures,"07_build_vs_kmeans")
    fig,ax=plt.subplots(figsize=(8,4)); x=np.arange(3); width=.25
    r=prof[prof.implementation=="D3_reference"].set_index("dataset"); o=prof[prof.implementation=="D3_OPT"].set_index("dataset")
    reductions=[(r.loc[d,"boundary_relation_time"]-o.loc[d,"boundary_relation_time"]) for d in datasets]
    copies=[(r.loc[d,"hybrid_fusion_time"]-o.loc[d,"hybrid_fusion_time"]) for d in datasets]
    ax.bar(x-width,reductions,width,label="distance + vectorized relation"); ax.bar(x,copies,width,label="preallocated fusion");
    ax.bar(x+width,[r.loc[d,"representation_build_time"]-o.loc[d,"representation_build_time"] for d in datasets],width,label="total build reduction")
    ax.set_xticks(x); ax.set_xticklabels(datasets); ax.set_ylabel("seconds saved"); ax.legend(); _save(fig,figures,"08_optimization_modules")


def main():
    p=argparse.ArgumentParser(); p.add_argument("--root",default="result/stage7_freeze"); plot(p.parse_args().root)


if __name__=="__main__": main()
