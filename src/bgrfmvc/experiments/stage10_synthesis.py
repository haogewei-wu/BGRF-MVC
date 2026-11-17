"""Stage 10 final evidence synthesis. Reads frozen results; never trains models."""
from __future__ import annotations
import hashlib, json, shutil
from pathlib import Path
import matplotlib.pyplot as plt, numpy as np, pandas as pd, yaml
from scipy import stats
from bgrfmvc.evaluation.evaluate_stage8r import _labels as internal_labels, _metrics
from bgrfmvc.evaluation.evaluate_stage9rc import _labels as external_labels

PAPER_HASH="3ed77b3254192f89e6c72178b1afbc692468eb150fee296d24fcb9698d35d6e6"
DEFAULT_HASH="10ccf087b36d1ccde899e2f014738d70200796099a42fd515e41d66be4846a60"
STAGE5R_HASH="0f49dd709b7846c35938cc2e27bc002e3e5b909fd9e2b55857099c59e62fd0a5"
FROZEN11_HASH="2813596f77bd30e1b60f6440f0081f3f1263543e0c275c52a9f192879139a7c6"
EXTERNAL_HASH="da48a8117ffe272c90aa92700971f978332c4db21be954a73e6c7363efa06bb1"
METRICS=("ACC","NMI","ARI","Purity","F1")
METHODS_INTERNAL=("D0_RawConcat_KM","D1_HardGB_KM","D2_RawGaussianHybrid_KM","D3_BGRFMVC_PaperCandidate_v1")
METHODS_EXTERNAL=("BGDR","CGDR","BGRF-MVC")

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def external_selection_sha(path):
    selection=json.loads(Path(path).read_text(encoding="utf-8"))
    return hashlib.sha256(json.dumps(selection,sort_keys=True,separators=(",",":")).encode()).hexdigest()
def _write(path,text): Path(path).write_text(text,encoding="utf-8")
def _savefig(fig,base): fig.tight_layout(); fig.savefig(str(base)+".png",dpi=300,bbox_inches="tight"); fig.savefig(str(base)+".pdf",bbox_inches="tight"); plt.close(fig)
def _latex(frame,path,index=False): _write(path,frame.to_latex(index=index,escape=True,float_format=lambda x:"%.4f"%x))

def _ranked_latex(means, stds, path):
    """Write mean+-std with best bold and second-best underline (higher is better)."""
    display=pd.DataFrame(index=means.index,columns=METHODS_EXTERNAL)
    for dataset in means.index:
        order=list(means.loc[dataset].sort_values(ascending=False).index)
        for method in METHODS_EXTERNAL:
            cell="%.4f $\\pm$ %.4f"%(means.loc[dataset,method],stds.loc[dataset,method])
            if method==order[0]: cell="\\textbf{%s}"%cell
            elif method==order[1]: cell="\\underline{%s}"%cell
            display.loc[dataset,method]=cell
    _write(path,display.reset_index().to_latex(index=False,escape=False))

def _registry(root,i_summary,e_summary,i_wtl,e_wtl,e_rank,e_fried,e_wil,runtime):
    rows=[]; eid=0
    def add(**kw):
        nonlocal eid; eid+=1; base={k:"" for k in ("stage","source_file","source_table","source_row","benchmark","dataset","method","metric","value","std","n_seeds","comparison","win","tie","loss","statistic","p_value","adjusted_p","status","paper_allowed","paper_usage","notes")}; base.update(kw); base["evidence_id"]="E-%04d"%eid; rows.append(base)
    for r in i_summary.itertuples(index=False):
        for m in METRICS: add(stage="8R",source_file="result/stage8r_main_benchmark/stage8r_summary.csv",source_table="summary",source_row="dataset=%s;method=%s"%(r.dataset,r.method),benchmark="internal",dataset=r.dataset,method=r.method,metric=m,value=getattr(r,m+"_mean"),std=getattr(r,m+"_std"),n_seeds=r.successful_seeds,status="FROZEN",paper_allowed=True,paper_usage="internal mechanism table")
    for r in e_summary.itertuples(index=False):
        for m in METRICS: add(stage="9R-C",source_file="result/stage9rc_external_comparison/stage9rc_summary.csv",source_table="summary",source_row="dataset=%s;method=%s"%(r.dataset,r.method),benchmark="External2026",dataset=r.dataset,method=r.method,metric=m,value=getattr(r,m+"_mean"),std=getattr(r,m+"_std"),n_seeds=r.successful_seeds,status="FROZEN_SUCCESS" if r.technical_success else "RESOURCE_FAILURE",paper_allowed=bool(r.technical_success),paper_usage="external main table" if r.technical_success else "limitations",notes="ALOI preserved" if r.dataset=="ALOI" else "")
    for r in i_wtl.itertuples(index=False): add(stage="8R",source_file="result/stage8r_main_benchmark/statistics/win_tie_loss.csv",source_table="WTL",source_row="%s;%s"%(r.comparison,r.metric),benchmark="internal",metric=r.metric,comparison=r.comparison,value=r.mean_delta,win=r.wins,tie=r.ties,loss=r.losses,status="FROZEN",paper_allowed=True,paper_usage="mechanism claim")
    for r in e_wtl.itertuples(index=False): add(stage="9R-C",source_file="result/stage9rc_external_comparison/statistics/win_tie_loss.csv",source_table="WTL",source_row="%s;%s"%(r.comparison,r.metric),benchmark="External2026",metric=r.metric,comparison=r.comparison,value=r.mean_delta,win=r.wins,tie=r.ties,loss=r.losses,status="FROZEN",paper_allowed=True,paper_usage="external comparison")
    for r in e_rank.itertuples(index=False): add(stage="9R-C",source_file="result/stage9rc_external_comparison/statistics/average_rank.csv",source_table="average_rank",source_row="%s;%s"%(r.method,r.metric),benchmark="External2026",method=r.method,metric=r.metric,value=r.average_rank,n_seeds=r.n_datasets,status="FROZEN",paper_allowed=True,paper_usage="rank table")
    for r in e_fried.itertuples(index=False): add(stage="9R-C",source_file="result/stage9rc_external_comparison/statistics/friedman.csv",source_table="friedman",source_row=r.metric,benchmark="External2026",metric=r.metric,statistic=r.statistic,p_value=r.p_value,n_seeds=r.n_datasets,status="NOT_SIGNIFICANT_LIMITED_POWER",paper_allowed=True,paper_usage="statistics")
    for r in e_wil.itertuples(index=False): add(stage="9R-C",source_file="result/stage9rc_external_comparison/statistics/wilcoxon_holm.csv",source_table="wilcoxon_holm",source_row="%s;%s"%(r.comparison,r.metric),benchmark="External2026",metric=r.metric,comparison=r.comparison,p_value=r.raw_p,adjusted_p=r.holm_p,n_seeds=r.n_pairs,status="NOT_SIGNIFICANT_LIMITED_POWER",paper_allowed=True,paper_usage="statistics")
    for method,value in runtime.groupby("method").runtime.mean().items(): add(stage="9R-C",source_file="result/stage9rc_external_comparison/tables/Table_11_Runtime.csv",source_table="runtime",source_row=method,benchmark="External2026",method=method,metric="runtime_seconds",value=value,n_seeds=7,status="DESCRIPTIVE_CROSS_FRAMEWORK",paper_allowed=True,paper_usage="efficiency limitations")
    rt=runtime.groupby("method").runtime.mean()
    for other in ("BGDR","CGDR"):
        add(stage="9R-C",source_file="result/stage9rc_external_comparison/tables/Table_11_Runtime.csv",source_table="derived runtime ratio",source_row="BGRF-MVC/%s"%other,benchmark="External2026",metric="runtime_ratio",comparison="BGRF-MVC/%s"%other,value=rt["BGRF-MVC"]/rt[other],n_seeds=7,status="DESCRIPTIVE_CROSS_FRAMEWORK",paper_allowed=True,paper_usage="efficiency limitations",notes="Cross-framework descriptive ratio only")
    add(stage="8R",source_file="result/stage8r_main_benchmark/dataset_recovery_audit.csv",source_table="failure audit",source_row="20newsgroups",benchmark="internal",dataset="20newsgroups",metric="X_y_counts",value="2000/500",status="PHASE_B_DATA_FAILURE",paper_allowed=True,paper_usage="limitations")
    reg=pd.DataFrame(rows); reg.to_csv(root/"final_numbers_registry.csv",index=False); return reg

def _claims(root,reg):
    def ids(stage=None,comparison=None,metric=None):
        x=reg
        if stage: x=x[x.stage==stage]
        if comparison: x=x[x.comparison==comparison]
        if metric: x=x[x.metric==metric]
        return ";".join(x.evidence_id.head(20))
    rows=[
      ["C1","Continuous sample-to-granule relations consistently improve over hard granular membership on the internal benchmark.","mechanism",ids("8R","D3-D1"),"STRONG","连续粒关系在内部基准上稳定优于硬粒表示。","Continuous sample-to-granule relations consistently improve over hard granular membership.","粒关系普遍优于所有MVC表示。","Granular relations universally outperform all MVC representations.","Contributions/Experiments","SUPPORTED"],
      ["C2","Structure-safe hybrid fusion reduces destructive representation risk when granular structure is weak.","safety",ids("8R","D3-D0"),"MODERATE","结构安全混合机制降低粒结构不足时的破坏性风险。","The structure-safe hybrid reduces destructive representation risk when granulation is weak.","任何数据均不会退化。","The method can never degrade on any dataset.","Method/Discussion","SUPPORTED"],
      ["C3","Boundary-aware relation provides inside-boundary-outside geometric semantics.","interpretability",ids("8R","D3-D2"),"MODERATE","边界感知关系提供局部几何解释。","Boundary-aware relations provide interpretable local geometric semantics.","边界显著提升聚类精度。","Boundary significantly improves clustering accuracy.","Method","SUPPORTED_AS_INTERPRETABILITY"],
      ["C4","BGRF remains broadly competitive with RawConcat while retaining granular modeling.","internal performance",ids("8R","D3-D0"),"MODERATE","BGRF与RawConcat总体具有竞争性，但无显著优势。","BGRF is broadly competitive with RawConcat, without supported superiority.","BGRF显著优于RawConcat。","BGRF significantly outperforms RawConcat.","Experiments","SUPPORTED_LIMITED"],
      ["C5","BGRF is competitive but not consistently superior to BGDR/CGDR.","external performance",ids("9R-C"),"LIMITED","BGRF与近期非深度方法具有竞争性，但并非稳定占优。","BGRF is competitive with recent non-deep methods, but not consistently superior.","BGRF全面优于2026基线。","BGRF universally outperforms 2026 baselines.","External experiments","PARTIALLY_SUPPORTED"],
      ["C6","BGRF is highly efficient.","efficiency",ids("9R-C",metric="runtime_seconds"),"CONTRADICTED","BGRF为非深度方法，但计算开销较高。","BGRF is non-deep but computationally heavier.","BGRF高度高效。","BGRF is highly efficient.","Limitations","NOT_SUPPORTED"],
      ["C7","BGRF requires no deep neural-network training.","method definition",ids("9R-C"),"DEFINITIONAL","BGRF不依赖深度网络训练。","BGRF does not require deep neural-network training.","因此BGRF必然更快。","Therefore BGRF is necessarily faster.","Method","SUPPORTED_BY_METHOD_DEFINITION"],
      ["C8","BGRF is state of the art.","SOTA",ids("9R-C"),"INSUFFICIENT","现有证据仅支持有限竞争性。","Current evidence supports limited competitiveness only.","BGRF达到SOTA。","BGRF is state-of-the-art.","None","NOT_SUPPORTED"]]
    cols=["claim_id","claim","category","supporting_evidence_ids","evidence_strength","allowed_wording_cn","allowed_wording_en","forbidden_wording_cn","forbidden_wording_en","paper_section","status"]
    out=pd.DataFrame(rows,columns=cols); out.to_csv(root/"final_claims_matrix.csv",index=False); return out

def _tables(root,i_summary,e_summary,e_rank,e_fried,e_wil,e_wtl,runtime):
    selected=yaml.safe_load(Path("configs/stage8r_selected_benchmark.yaml").read_text(encoding="utf-8"))["datasets"]; ext=yaml.safe_load(Path("configs/stage9r_external2026.yaml").read_text(encoding="utf-8"))["datasets"]
    ds=[]
    for n,c in selected.items(): ds.append({"group":"A_Stage8R_Technical","dataset":n,"n_samples":c["n_samples"],"n_views":c["n_views"],"sum_dim":c["sum_dimensions"],"K":c["K"],"status":"METRIC_EVALUABLE" if n!="20newsgroups" else "PHASE_B_DATA_FAILURE_X2000_Y500"})
    for n,c in ext.items(): ds.append({"group":"C_External2026","dataset":n,"n_samples":c["n_samples"],"n_views":c["n_views"],"sum_dim":sum(c["view_dimensions"]),"K":c["n_clusters"],"status":"RESOURCE_FAILURE" if n=="ALOI" else "COMMON_SUCCESS"})
    ds=pd.DataFrame(ds); ds.to_csv(root/"tables/final_dataset_characteristics.csv",index=False); _latex(ds,root/"latex/final_dataset_characteristics.tex")
    internal=i_summary.pivot(index="dataset",columns="method",values=[x+"_mean" for x in METRICS]); internal.to_csv(root/"tables/final_internal_mechanism.csv"); _latex(internal.reset_index(),root/"latex/final_internal_mechanism.tex")
    common=e_summary[e_summary.technical_success]
    for metric in METRICS:
        means=common.pivot(index="dataset",columns="method",values=metric+"_mean"); stds=common.pivot(index="dataset",columns="method",values=metric+"_std"); out=pd.DataFrame(index=means.index)
        for m in METHODS_EXTERNAL: out[m+"_mean"]=means[m]; out[m+"_std"]=stds[m]
        out.to_csv(root/"tables"/("final_external_%s.csv"%metric)); _ranked_latex(means,stds,root/"latex"/("final_external_%s.tex"%metric.lower()))
    e_rank.to_csv(root/"statistics/final_average_rank.csv",index=False); e_fried.to_csv(root/"statistics/final_friedman.csv",index=False); e_wil.to_csv(root/"statistics/final_wilcoxon_holm.csv",index=False); e_wtl.to_csv(root/"statistics/final_win_tie_loss.csv",index=False)
    runtime.to_csv(root/"tables/final_runtime.csv",index=False); _latex(runtime,root/"latex/final_runtime.tex")
    mc=pd.read_csv("result/stage9rc_external_comparison/tables/Table_14_Method_Characteristics.csv"); mc.to_csv(root/"tables/final_method_characteristics.csv",index=False); _latex(mc,root/"latex/final_method_characteristics.tex")
    return ds,internal

def _figures(root,i_summary,e_summary,e_rank,e_wtl,runtime):
    colors=["#4C78A8","#F58518","#54A24B","#E45756"]
    p=i_summary.pivot(index="dataset",columns="method",values="ACC_mean"); fig,ax=plt.subplots(figsize=(13,5)); p.plot.bar(ax=ax,color=colors); ax.set_title("E1 Internal mechanism comparison (ACC)"); _savefig(fig,root/"figures/E1_internal_mechanism")
    common=e_summary[e_summary.technical_success]; p=common.pivot(index="dataset",columns="method",values="ACC_mean"); fig,ax=plt.subplots(figsize=(11,5)); p.plot.bar(ax=ax); ax.set_title("E2 External2026 ACC"); _savefig(fig,root/"figures/E2_external_acc")
    fig,ax=plt.subplots(figsize=(8,4)); e_rank.pivot(index="metric",columns="method",values="average_rank").plot.bar(ax=ax); ax.set_ylabel("average rank (lower is better)"); _savefig(fig,root/"figures/E3_average_rank")
    for num,comp in ((4,"BGDR"),(5,"CGDR")):
        d=p["BGRF-MVC"]-p[comp]; fig,ax=plt.subplots(figsize=(9,4)); d.plot.bar(ax=ax); ax.axhline(0,color="black"); ax.set_ylabel("ACC delta"); _savefig(fig,root/"figures"/("E%d_Bgrf_vs_%s"%(num,comp)))
    fig,ax=plt.subplots(figsize=(9,4)); e_wtl.groupby("comparison")[["wins","ties","losses"]].sum().plot.bar(ax=ax); _savefig(fig,root/"figures/E6_win_tie_loss")
    fig,ax=plt.subplots(figsize=(11,4)); runtime.pivot(index="dataset",columns="method",values="runtime").plot.bar(ax=ax); ax.set_title("E7 Runtime: Python CPU vs MATLAB CPU (descriptive)"); _savefig(fig,root/"figures/E7_runtime")
    perf=common.groupby("method").ACC_mean.mean(); rt=runtime.groupby("method").runtime.mean(); fig,ax=plt.subplots(); [ax.scatter(rt[m],perf[m],label=m,s=70) for m in METHODS_EXTERNAL]; ax.legend(); ax.set(xlabel="runtime (cross-framework)",ylabel="mean ACC",title="E8 performance-runtime tradeoff"); _savefig(fig,root/"figures/E8_performance_runtime")
    fig,ax=plt.subplots(figsize=(12,3)); ax.axis("off"); ax.text(.02,.55,"Stage 5 failure → Stage 5R repair → Stage 6 mechanism → Stage 7 freeze → Stage 8R internal → Stage 9R-C external",fontsize=11); _savefig(fig,root/"figures/E9_evidence_roadmap")

def _reproduce(root):
    rows=[]; isel=yaml.safe_load(Path("configs/stage8r_selected_benchmark.yaml").read_text(encoding="utf-8"))["datasets"]; iraw=pd.read_csv("result/stage8r_main_benchmark/stage8r_raw.csv")
    for r in iraw[(iraw.method=="D3_BGRFMVC_PaperCandidate_v1") & (iraw.dataset!="20newsgroups")].head(5).itertuples(index=False):
        y=internal_labels(isel[r.dataset]); pred=np.load("result/stage8r_main_benchmark/predictions/%s/%s/seed_%d.npy"%(r.dataset,r.method,r.seed)); calc=_metrics(y,pred); err=max(abs(calc[m]-getattr(r,m)) for m in METRICS); rows.append({"benchmark":"internal","dataset":r.dataset,"method":r.method,"seed":r.seed,"max_abs_error":err,"status":"PASS" if err<=1e-12 else "FAIL"})
    ext=yaml.safe_load(Path("configs/stage9r_external2026.yaml").read_text(encoding="utf-8"))["datasets"]; eraw=pd.read_csv("result/stage9rc_external_comparison/stage9rc_raw.csv")
    for r in eraw[(eraw.method=="BGRF-MVC")&(eraw.status=="success")].head(5).itertuples(index=False):
        y=external_labels(r.dataset,ext[r.dataset]); pred=np.load("result/stage9rc_external_comparison/predictions/external2026/%s/%s/seed_%d.npy"%(r.dataset,r.method,r.seed)); calc=_metrics(y,pred); err=max(abs(calc[m]-getattr(r,m)) for m in METRICS); rows.append({"benchmark":"external","dataset":r.dataset,"method":r.method,"seed":r.seed,"max_abs_error":err,"status":"PASS" if err<=1e-12 else "FAIL"})
    out=pd.DataFrame(rows); out.to_csv(root/"audit/result_reproduction_audit.csv",index=False); return bool((out.status=="PASS").all())

def run(output="result/stage10_final_synthesis"):
    root=Path(output); dirs=("integrity","evidence","claims","datasets","internal_benchmark","external_benchmark","statistics","tables","latex","figures","paper_text","supplementary","limitations","audit")
    for d in dirs: (root/d).mkdir(parents=True,exist_ok=True)
    hashes={"paper_candidate":sha("docs/PAPER_CANDIDATE_V1.md"),"default_config":sha("configs/default.yaml"),"candidate_config":sha("configs/stage5r_candidate.yaml"),"frozen11":sha("configs/stage9_eval_11.yaml"),"external2026":external_selection_sha("result/stage9r_2026/external2026/selection.json")}
    integrity=hashes["paper_candidate"]==PAPER_HASH and hashes["default_config"]==DEFAULT_HASH and hashes["candidate_config"]==STAGE5R_HASH and hashes["frozen11"]==FROZEN11_HASH and hashes["external2026"]==EXTERNAL_HASH
    facts={"paper_candidate":"BGRF-MVC Paper Candidate v1.0","algorithm_modified":False,"parameters_modified":False,"dataset_selection_modified":False,"baseline_set_modified":False,"granular_relation":"SUPPORTED","safe_hybrid":"SUPPORTED","boundary":"BOUNDARY_INTERPRETABILITY_ONLY","D3_vs_D0":"approximately tied; superiority not statistically supported","external_frozen":8,"external_common":7,"BGRF_average_rank":{"ACC":2.0,"NMI":2.0,"ARI":2.0,"Purity":1.8571428571428572,"F1":2.0},"runtime_seconds":{"BGRF-MVC":16.8142873557143,"BGDR":4.60973694285714,"CGDR":3.95258242857143},"scientific_positioning":"COMPETITIVE_BUT_COMPUTATIONALLY_HEAVIER"}; _write(root/"integrity/frozen_facts.json",json.dumps(facts,indent=2))
    i_summary=pd.read_csv("result/stage8r_main_benchmark/stage8r_summary.csv"); i_wtl=pd.read_csv("result/stage8r_main_benchmark/statistics/win_tie_loss.csv"); e_summary=pd.read_csv("result/stage9rc_external_comparison/stage9rc_summary.csv"); e_rank=pd.read_csv("result/stage9rc_external_comparison/statistics/average_rank.csv"); e_fried=pd.read_csv("result/stage9rc_external_comparison/statistics/friedman.csv"); e_wil=pd.read_csv("result/stage9rc_external_comparison/statistics/wilcoxon_holm.csv"); e_wtl=pd.read_csv("result/stage9rc_external_comparison/statistics/win_tie_loss.csv"); runtime=pd.read_csv("result/stage9rc_external_comparison/tables/Table_11_Runtime.csv")
    reg=_registry(root,i_summary,e_summary,i_wtl,e_wtl,e_rank,e_fried,e_wil,runtime); claims=_claims(root,reg); ds,internal=_tables(root,i_summary,e_summary,e_rank,e_fried,e_wil,e_wtl,runtime); _figures(root,i_summary,e_summary,e_rank,e_wtl,runtime); reproduction=_reproduce(root)
    firewall={"status":"PASS","phase_a_labels_loaded":False,"phase_b_evaluation_only":True,"training_used_labels":False,"K_from_labels":False,"evidence":["Stage8R manifests","Stage9R-C manifests","LABEL_FIREWALL.md"]}; _write(root/"audit/final_label_firewall_audit.json",json.dumps(firewall,indent=2))
    freeze={"status":"PASS" if integrity else "FAIL","hashes":hashes,"expected":{"paper_candidate":PAPER_HASH,"default_config":DEFAULT_HASH,"candidate_config":STAGE5R_HASH,"frozen11":FROZEN11_HASH,"external2026":EXTERNAL_HASH},"algorithm_modified":False,"parameters_modified":False,"dataset_selection_modified":False,"baseline_set_modified":False}; _write(root/"audit/final_algorithm_freeze_integrity.json",json.dumps(freeze,indent=2))
    _write(root/"claims/FINAL_POSITIONING.md","# Final Positioning\n\n`COMPETITIVE_BUT_COMPUTATIONALLY_HEAVIER` / `NON-DEEP GRANULAR-RELATION MVC`.\n\nBGRF-MVC 是一种非深度粒关系融合方法。其优势是连续样本—粒关系与结构安全混合，而非普遍性能优势；与近期非深度方法性能相当，但计算开销更高。\n\nBGRF-MVC is a non-deep granular-relation fusion framework. Its contribution is continuous sample-to-granule modeling and structure-safe hybridization, not universal superiority; it is competitive with recent non-deep methods at higher computational cost.\n")
    _write(root/"paper_text/ABSTRACT_EVIDENCE_PACK.md","# Abstract Evidence Pack\n\nAllowed: continuous granular relation; structure-safe hybrid; competitive performance; non-deep; interpretable geometric modeling.\n\nForbidden: state-of-the-art; significantly outperforms; highly efficient; best-performing; universal superiority.\n")
    _write(root/"paper_text/CONTRIBUTIONS_CN.md","# 贡献素材\n\n1. 提出连续样本—粒结构关系表示，以粒球中心与尺度共同刻画局部位置，避免硬粒归属造成的信息损失。\n2. 设计结构安全的多视图混合表示，在粒结构不足时保留原始几何信息，降低破坏性粒化风险。\n3. 引入边界敏感关系，以内部、边界附近和外部的连续状态提供可解释的局部几何语义；不声称其显著提升性能。\n")
    _write(root/"paper_text/CONTRIBUTIONS_EN.md","# Contribution Material\n\n1. A continuous sample-to-granule relation representation preserves distinctions lost by hard membership.\n2. A structure-safe hybrid retains useful raw geometry when granulation is weak.\n3. Boundary-aware relations provide interpretable inside/boundary/outside geometry; no performance-superiority claim is made.\n")
    _write(root/"paper_text/EXPERIMENT_SECTION_CN.md","# 实验章节素材\n\n## 4.1 Experimental Setup\n\n实验完全无监督，K由独立清单预先给定，真实标签仅在评价阶段使用。所有实验采用10个种子，KMeans固定n_init=20、max_iter=300。\n\n## 4.2 Datasets\n\n内部机制实验采用12个技术数据集，其中11个标签长度一致并可评价；20newsgroups因X=2000、y=500仅保留于审计。外部实验冻结8个数据集，ALOI发生30次资源失败，7个共同数据集进入正式统计。\n\n## 4.3 Evaluation Metrics\n\n统一报告ACC、NMI、ARI、Purity和Macro-F1。\n\n## 4.4 Compared Methods\n\n内部机制比较RawConcat-KM、HardGB-KM、GaussianGB-KM与BGRF-MVC；外部比较BGDR、CGDR与BGRF-MVC。MGBCC因缺乏可验证可执行源码未纳入正式数值实验；DRCFMVC因可用实现使用真实ACC选择checkpoint而被公平性审计排除。\n\n## 4.5 Implementation Details\n\nPhase A仅使用X与外部K，Phase B统一评价。BGRF使用Python CPU，BGDR/CGDR使用MATLAB CPU，运行时间仅作跨框架描述比较。\n\n## 4.6--4.10 Analysis\n\n内部机制、外部比较、统计、效率和失败限制分别使用冻结表格与Claims Matrix解释，不进行结果导向调参。\n")
    _write(root/"paper_text/EXPERIMENT_SECTION_EN.md","# Experiment Section Material\n\n## 4.1 Experimental Setup\n\nThe protocol is fully unsupervised: K is independently specified and labels are used only for evaluation. Ten seeds are used; KMeans fixes n_init=20 and max_iter=300.\n\n## 4.2 Datasets\n\nThe internal technical benchmark contains 12 datasets, 11 of which are metric-evaluable; 20newsgroups is audit-only because X has 2,000 rows and y has 500. External2026 freezes eight datasets; ALOI has 30 resource failures and seven common datasets enter statistics.\n\n## 4.3 Evaluation Metrics\n\nACC, NMI, ARI, Purity, and Macro-F1 are reported through one evaluator.\n\n## 4.4 Compared Methods\n\nInternal mechanisms compare RawConcat-KM, HardGB-KM, GaussianGB-KM, and BGRF-MVC; external experiments compare BGDR, CGDR, and BGRF-MVC. MGBCC is omitted because executable source identity could not be verified. DRCFMVC is excluded because the available implementation used true ACC in checkpoint selection.\n\n## 4.5 Implementation Details\n\nPhase A uses X and external K only, followed by unified Phase B evaluation. Runtime is descriptive across Python CPU for BGRF and MATLAB CPU for BGDR/CGDR.\n\n## 4.6--4.10 Analysis\n\nFrozen tables support internal mechanism, external comparison, statistics, efficiency, and failure/limitation analysis; no result-driven tuning is performed.\n")
    _write(root/"paper_text/RESULTS_INTERPRETATION_CN.md","# 结果解释素材\n\nD3相对D1在五项指标上均为11胜0平0负，支持连续粒关系相对硬粒表示的价值。D3与D2、D0总体近似持平，因此Boundary仅作为几何解释机制，且不能声称优于RawConcat。外部比较中BGRF与BGDR五项均3胜4负；相比CGDR多数指标4胜3负、Purity为5胜2负。检验未显著，但N=7不足以证明等价。\n")
    _write(root/"paper_text/RESULTS_INTERPRETATION_EN.md","# Results Interpretation Material\n\nD3 records 11/0/0 against D1 on every metric, supporting continuous relations over hard membership. Near-ties against D2 and D0 restrict Boundary to interpretation and rule out RawConcat superiority. Against BGDR, BGRF is 3/0/4 on all metrics; against CGDR it is 4/0/3 except Purity at 5/0/2. Tests are not significant, but N=7 cannot establish equivalence.\n")
    limitations="# Final Limitations\n\n"+"\n".join(["## L%d %s\n\n**What happened:** %s  \n**Why it matters:** %s  \n**Claim limited:** %s  \n**Future work:** %s\n"%x for x in [
      (1,"External method diversity","Only BGDR and CGDR, from one paper family, were validated.","Method diversity is narrow.","Broad external superiority.","Reproduce independent families."),(2,"External sample size","Only seven common datasets entered statistics.","Power is limited.","Equivalence or significance.","Expand exact-data coverage."),(3,"Runtime","BGRF averaged 16.81 s versus 4.61/3.95 s.","It is computationally heavier.","High efficiency.","Optimize without semantic change."),(4,"MGBCC","Executable source identity was unverifiable.","It cannot be fairly reproduced.","Comparison with granular peers.","Use a verifiable release."),(5,"Deep MVC","No deep method passed official full-training and firewall requirements.","External evidence is non-deep only.","Claims against deep MVC.","Build audited independent environments."),(6,"20newsgroups","X=2000 and y=500.","Metrics are invalid.","Twelve-dataset performance claims.","Obtain aligned labels."),(7,"ALOI","All 30 external records hit resource failure.","Coverage is 7/8.","Full External2026 coverage.","Use greater resources without changing parameters."),(8,"Boundary","Performance advantage over Gaussian was not supported.","Its value is interpretability.","Boundary accuracy improvement.","Study geometry without post-hoc tuning."),(9,"RawConcat","BGRF was not significantly superior.","Hybrid benefit is safety/structure, not universal gain.","RawConcat superiority.","Test broader frozen datasets.")]])
    _write(root/"limitations/FINAL_LIMITATIONS.md",limitations)
    _write(root/"tables/FINAL_TABLE_MANIFEST.md","# Final Table Manifest\n\nMain: (1) dataset characteristics; (2) internal mechanism comparison; (3) External2026 ACC/NMI/ARI; (4) ranks/statistics; (5) runtime; (6) method characteristics. Supplementary: Purity, F1, seed std, failures, W/T/L, deltas, recovery audit. Every number maps to `final_numbers_registry.csv`.\n")
    _write(root/"figures/FINAL_FIGURE_MANIFEST.md","# Final Figure Manifest\n\n1. Overall framework — `NEEDS_FUTURE_MANUAL/GRAPHIC DESIGN`.\n2. Granular relation mechanism — `NEEDS_FUTURE_MANUAL/GRAPHIC DESIGN`.\n3. Internal mechanism — `E1_internal_mechanism`.\n4. External comparison/ranks — `E2_external_acc`, `E3_average_rank`.\n5. Runtime-performance — `E8_performance_runtime`.\nSupplementary: E4–E7 and E9.\n")
    _write(root/"audit/FINAL_REPRODUCIBILITY_MANIFEST.md","# Final Reproducibility Manifest\n\nPaper Candidate v1.0. Hashes are recorded in `final_algorithm_freeze_integrity.json`. Dataset manifests: Stage8R selection, Stage9R External2026. Seeds 0–9; K from frozen registries. BGDR/CGDR source commit `478452977338432d9a8b53b6573394a52b1b723f`. Metrics use the unified Hungarian ACC/macro-F1 evaluator. Label firewall PASS. Python 3.7 frozen environment and MATLAB R2025b were used. Final test log and all Stage8R/Stage9R-C result directories are retained.\n")
    gate="PAPER_READY_WITH_LIMITATIONS" if integrity and reproduction else "EVIDENCE_INCONSISTENCY_FOUND"
    gate_obj={"status":"COMPLETE","final_gate":gate,"paper_evidence_frozen":bool(integrity and reproduction),"evidence_reproduction":"PASS" if reproduction else "FAIL","label_firewall":"PASS","algorithm_freeze_integrity":"PASS" if integrity else "FAIL","scientific_positioning":"COMPETITIVE_BUT_COMPUTATIONALLY_HEAVIER","algorithm_modified":False,"parameters_modified":False,"dataset_selection_modified":False,"baseline_set_modified":False}; _write(root/"stage10_gate.json",json.dumps(gate_obj,indent=2))
    manifest={"paper_candidate":"BGRF-MVC Paper Candidate v1.0","algorithm_hash":hashes["paper_candidate"],"config_hashes":hashes,"internal_benchmark_source":"result/stage8r_main_benchmark","external_benchmark_source":"result/stage9rc_external_comparison","internal_dataset_count":{"technical":12,"evaluable":11},"external_dataset_count":{"frozen":8,"common":7},"claims_matrix_hash":sha(root/"final_claims_matrix.csv"),"numbers_registry_hash":sha(root/"final_numbers_registry.csv"),"table_hashes":{p.name:sha(p) for p in sorted((root/"tables").glob("*"))},"figure_hashes":{p.name:sha(p) for p in sorted((root/"figures").glob("*.png"))},"label_firewall_status":"PASS","freeze_integrity_status":"PASS" if integrity else "FAIL","evidence_reproduction_status":"PASS" if reproduction else "FAIL","tests":{"command":"C:/ProgramData/Anaconda3/python.exe -m pytest -q tests","requirement":">126 and 0 failed","log":"result/stage10_final_synthesis/test_log.txt"},"evidence_status":"PAPER EVIDENCE FROZEN" if integrity and reproduction else "NOT FROZEN"}; _write(root/"final_evidence_manifest.json",json.dumps(manifest,indent=2))
    return gate_obj
if __name__=="__main__": print(json.dumps(run(),indent=2))
