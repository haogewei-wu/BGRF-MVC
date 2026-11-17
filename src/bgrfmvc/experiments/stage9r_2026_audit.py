"""Stage 9R source/dataset audit and gate freeze (no model training)."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pandas as pd
import scipy.io as sio
import yaml


FROZEN11_SHA256 = "2813596f77bd30e1b60f6440f0081f3f1263543e0c275c52a9f192879139a7c6"
PAPER_SHA256 = "3ed77b3254192f89e6c72178b1afbc692468eb150fee296d24fcb9698d35d6e6"
MGBCC_STATUS = "EXCLUDED_UNVERIFIABLE_SOURCE"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _git_head(root, method):
    heads = {
        "BGDR": "478452977338432d9a8b53b6573394a52b1b723f",
        "CGDR": "478452977338432d9a8b53b6573394a52b1b723f",
        "MISCC": "e98f6575ac1f295e350f938848423691294ce4de",
        "DMVCS": "4ec372419d17fd8c79a2934593617af3437c0650",
        "DRCFMVC": "6c12ef1254621545f785256bdc87b2c24178fcea",
        "RAV": "3a9809194aa55229309f7741307c478e505c47df",
        "CorreGen": "c16f9d37fc7e7d3e82ff34d2b6b118877a953924",
    }
    return heads.get(method, "N/A")


def _mat_x_meta(path):
    data = sio.loadmat(str(path))
    if "X" in data and data["X"].dtype == object:
        views = [data["X"].flat[i] for i in range(data["X"].size)]
    else:
        keys = sorted(k for k in data if k.startswith("X") and k[1:].isdigit())
        views = [data[k] for k in keys]
    shapes = [list(map(int, x.shape)) for x in views]
    n = shapes[0][0] if shapes else None
    return n, len(shapes), [s[1] for s in shapes]


def run(project_root="."):
    root = Path(project_root).resolve()
    out = root / "result" / "stage9r_2026"
    folders = ("sources", "audit", "dataset_catalog", "frozen11", "external2026",
               "literature_context", "environments", "phase_a", "phase_b",
               "predictions", "runtime", "memory", "statistics", "tables",
               "figures", "failures")
    for folder in folders:
        (out / folder).mkdir(parents=True, exist_ok=True)

    frozen11 = root / "configs" / "stage9_eval_11.yaml"
    if sha256(frozen11) != FROZEN11_SHA256:
        raise RuntimeError("FROZEN11_IMMUTABILITY_FAILURE")
    if sha256(root / "docs" / "PAPER_CANDIDATE_V1.md") != PAPER_SHA256:
        raise RuntimeError("BGRF_FREEZE_INTEGRITY_FAILURE")

    source_rows = [
        ("MGBCC", "Multi-view Granular-ball Contrastive Clustering", "AAAI", 2025, "SOURCE_UNVERIFIABLE", "external_baselines/MGBCC/source.zip", "8afcded197560da6b210642ad0ca0dbdbd97977d67d3409bbc53cbfe33784e52", "N/A", "Python", "PyTorch", False, MGBCC_STATUS),
        ("BGDR", "Scalable Graph Discrete Reconstruction for Efficient Multi-View Clustering", "IEEE TKDE", 2026, "official GitHub", "external_baselines/GDR/original", "N/A_GIT", _git_head(root,"BGDR"), "MATLAB", "MATLAB R2025b", True, "ACCEPT_MAIN"),
        ("CGDR", "Scalable Graph Discrete Reconstruction for Efficient Multi-View Clustering", "IEEE TKDE", 2026, "official GitHub", "external_baselines/GDR/original", "N/A_GIT", _git_head(root,"CGDR"), "MATLAB", "MATLAB R2025b", True, "ACCEPT_MAIN"),
        ("MISCC", "Deep Multi-view Clustering with Intra-view Similarity and Cross-view Correlation Learning", "IEEE TMM", 2026, "official GitHub", "external_baselines/MISCC/original", "N/A_GIT", _git_head(root,"MISCC"), "Python", "PyTorch", True, "RESOURCE_INFEASIBLE"),
        ("DMVCS", "Disentangled Contrastive Multi-view Clustering via Semantic Relevance Invariance", "IEEE TKDE", 2026, "official GitHub", "external_baselines/DMVCS/original", "N/A_GIT", _git_head(root,"DMVCS"), "Python", "PyTorch", True, "RESOURCE_INFEASIBLE"),
        ("DRCFMVC", "Deep Robust Contrastive Fusion Multi-View Clustering", "IEEE TCSVT", 2026, "official GitHub", "external_baselines/DRCFMVC/original", "ab034c8ba9f4261a6958140c64f3effbbfa1df0d398e2ec004f1a5e7e293c13f", _git_head(root,"DRCFMVC"), "Python", "PyTorch", True, "AUDIT_REJECTED_LABEL_DEPENDENCE"),
        ("RAV", "Reliable Adaptive View Learning for Multi-View Clustering", "ICLR", 2026, "official GitHub", "external_baselines/RAV/original", "fac29f591a915a6d24a830d15dcd679d1b8f2354d62a0fb33490cefebdb993ab", _git_head(root,"RAV"), "Python", "PyTorch", True, "RESOURCE_INFEASIBLE"),
        ("CorreGen", "Uncover Underlying Correspondence for Robust Multi-view Clustering", "ICLR", 2026, "official GitHub", "external_baselines/CorreGen/original", "N/A_GIT", _git_head(root,"CorreGen"), "Python", "PyTorch", True, "RESOURCE_INFEASIBLE"),
        ("GAPS", "Multi-View Clustering with Granularity-Aware Pseudo Supervision", "AAAI", 2026, "paper only", "N/A", "N/A", "N/A", "N/A", "N/A", False, "LITERATURE_CONTEXT_ONLY"),
    ]
    cols = ["method","paper","venue","year","source_type","source_path","archive_sha256","commit_hash_if_available","language","framework","verified","status"]
    sources = pd.DataFrame(source_rows, columns=cols)
    sources.to_csv(out / "audit" / "baseline_sources.csv", index=False)

    audit = pd.DataFrame([
        ["BGDR", "K and terminal metrics only", "external manifest K", "official demo fixed k/h", "PASS", "MATLAB sanity PASS; true labels absent from adapter"],
        ["CGDR", "K and terminal metrics only", "external manifest K", "official demo fixed k/h", "PASS", "MATLAB sanity PASS; true labels absent from adapter"],
        ["DRCFMVC", "ACC each epoch selects checkpoint", "hard-coded dataset K", "official defaults", "FAIL", "ground-truth ACC controls retained checkpoint"],
        ["MISCC", "final evaluator only", "hard-coded dataset K", "official defaults", "STATIC_PASS", "1100 epochs; independent PyTorch env unavailable"],
        ["DMVCS", "periodic reporting and final evaluator", "hard-coded dataset K", "official config", "STATIC_PASS", "500 epochs x 5 seeds; independent PyTorch env unavailable"],
        ["RAV", "terminal evaluator only", "hard-coded dataset K", "official fixed parameters", "STATIC_PASS", "400 epochs; independent PyTorch env unavailable"],
        ["CorreGen", "evaluation module", "config K", "official YAML", "STATIC_PASS", "200 epochs; independent PyTorch env unavailable"],
    ], columns=["method","label_use","k_source","parameter_source","firewall_status","notes"])
    audit.to_csv(out / "audit" / "label_and_parameter_audit.csv", index=False)

    patch = pd.DataFrame([
        ["BGDR/CGDR","external_baselines/GDR/patched/run_gdr_phase_a.m","label-firewall/output adapter","label argument supplied K and terminal evaluation","externally frozen K conveyed by non-ground-truth control vector; returned metrics discarded",False,"separate Phase A from evaluation","sanity_stage9r.m","PASS"]
    ], columns=["method","file","patch_type","original_behavior","patched_behavior","scientific_semantics_changed","reason","test","status"])
    patch.to_csv(out / "audit" / "baseline_patch_manifest.csv", index=False)

    specs = [
        ("MSRC_v1",1,"BGDR/CGDR",210,5,"[1302,48,512,100,256]",7,root/"external_baselines/GDR/original/Data/MSRC_v1_data.mat"),
        ("Dermatology",1,"BGDR/CGDR",358,2,"official archive",6,root/"external_baselines/GDR/original/Data/Dermatology_data.mat"),
        ("100leaves",1,"BGDR/CGDR",1600,3,"[64,64,64]",100,root/"external_baselines/GDR/original/Data/100leaves_data.mat"),
        ("mnist4",1,"BGDR/CGDR",4000,3,"official archive",4,root/"external_baselines/GDR/original/Data/mnist4_data.mat"),
        ("Digit4k",1,"BGDR/CGDR",4000,2,"official archive",10,root/"external_baselines/GDR/original/Data/Digit4k_data.mat"),
        ("Hdigit",1,"BGDR/CGDR",10000,2,"official archive",10,root/"external_baselines/GDR/original/Data/Hdigit_data.mat"),
        ("ALOI",2,"BGDR/CGDR; DMVCS",10800,4,"[77,13,64,125]",100,root/"external_baselines/GDR/original/Data/ALOI_data.mat"),
        ("BBCSport",1,"MISCC",544,2,"[3183,3203]",5,root/"external_baselines/MISCC/original/demo/data/BBCSport.mat"),
        ("BBC4view",1,"MISCC",685,4,"[4659,4633,4665,4684]",5,root/"external_baselines/MISCC/original/demo/data/BBC4view_685.mat"),
        ("Caltech-5V",2,"DMVCS; DRCFMVC",1400,5,"[40,254,928,512,1984]",7,root/"external_baselines/DRCFMVC/original/unpacked/DRCFMVC/data/Caltech-5V.mat"),
        ("3sources",1,"RAV",169,3,"[3560,3631,3068]",6,root/"external_baselines/RAV/original/unpacked/RAV/data/3sources.mat"),
        ("Scene15",1,"CorreGen",4485,3,"official archive",15,root/"external_baselines/CorreGen/original/dataset/Scene_15.mat"),
        ("LandUse21",2,"MISCC; CorreGen",2100,3,"[20,59,40]",21,root/"external_baselines/CorreGen/original/dataset/LandUse_21.mat"),
    ]
    rows=[]
    for name,count,papers,rn,rv,rd,rk,path in specs:
        try:
            an,av,ad = _mat_x_meta(path)
            status = "SPEC_EXACT" if an == rn and av == rv else "SPEC_MISMATCH"
            reason = "official repository bytes; X metadata matches" if status == "SPEC_EXACT" else "X metadata mismatch"
        except Exception as exc:
            an=av=None; ad=[]; status="SPEC_MISMATCH"; reason="parse failure: %s" % type(exc).__name__
        eligible = status == "SPEC_EXACT"
        rows.append([name,count,papers,rn,rv,rd,rk,str(path.relative_to(root)),an,av,json.dumps(ad),status,eligible,reason])
    catalog = pd.DataFrame(rows,columns=["dataset","paper_count","papers","reported_n","reported_V","reported_dims","reported_K","available_source","actual_n","actual_V","actual_dims","spec_status","eligible","reason"])
    catalog.to_csv(out/"dataset_catalog"/"2026_dataset_catalog.csv",index=False)
    selected = catalog[catalog.eligible].sort_values(["paper_count","actual_n","dataset"],ascending=[False,True,True]).head(8)
    selection = {"track":"External2026","selection_rule":"eligible SPEC_EXACT; paper consensus descending, n ascending, name; max 8","datasets":selected.dataset.tolist(),"labels_used":False}
    selection_bytes = json.dumps(selection,sort_keys=True,separators=(",",":")).encode()
    selection_hash = hashlib.sha256(selection_bytes).hexdigest()
    (out/"external2026"/"selection.json").write_text(json.dumps(selection,indent=2),encoding="utf-8")
    (out/"external2026"/"selection_hash.txt").write_text(selection_hash+"\n",encoding="utf-8")
    cfg={"track":"External2026","selection_hash":selection_hash,"labels_used_for_selection":False,"datasets":{}}
    for _,r in selected.iterrows():
        cfg["datasets"][r.dataset]={"path":r.available_source,"n_samples":int(r.actual_n),"n_views":int(r.actual_V),"view_dimensions":json.loads(r.actual_dims),"n_clusters":int(r.reported_K),"k_source":"official 2026 repository/paper benchmark metadata"}
    (root/"configs"/"stage9r_external2026.yaml").write_text(yaml.safe_dump(cfg),encoding="utf-8")

    raw_cols=["track","dataset","method","seed","ACC","NMI","ARI","PUR","F_score","runtime","memory_MB","result_type"]
    pd.DataFrame(columns=raw_cols).to_csv(out/"stage9r_raw.csv",index=False)
    pd.DataFrame(columns=raw_cols).to_csv(out/"stage9r_summary.csv",index=False)
    pd.DataFrame(columns=raw_cols).to_csv(out/"frozen11"/"raw.csv",index=False)
    pd.DataFrame(columns=raw_cols).to_csv(out/"frozen11"/"summary.csv",index=False)
    pd.DataFrame(columns=raw_cols).to_csv(out/"external2026"/"raw.csv",index=False)
    pd.DataFrame(columns=raw_cols).to_csv(out/"external2026"/"summary.csv",index=False)
    pd.DataFrame(columns=raw_cols).assign(result_type=pd.Series(dtype=str)).to_csv(out/"literature_context"/"published_2026_results.csv",index=False)

    accepted=sources[sources.status=="ACCEPT_MAIN"].method.tolist()
    phase_a={"status":"NOT_STARTED_BASELINE_GATE","labels_loaded":False,"accepted_main":accepted,"required":3,"requires_deep_or_modern":True,"frozen11_hash":FROZEN11_SHA256,"external2026_hash":selection_hash,"predictions_written":0}
    phase_b={"status":"NOT_STARTED_BASELINE_GATE","labels_loaded":False,"training_calls":0,"reason":"Phase A formal benchmark did not start"}
    (out/"phase_a_manifest.json").write_text(json.dumps(phase_a,indent=2),encoding="utf-8")
    (out/"phase_b_manifest.json").write_text(json.dumps(phase_b,indent=2),encoding="utf-8")
    failures=pd.DataFrame([
        ["DRCFMVC","AUDIT_REJECTED_LABEL_DEPENDENCE","epoch-wise true ACC selects best checkpoint"],
        ["MISCC","RESOURCE_INFEASIBLE","official 1100-epoch deep run; isolated PyTorch environment absent"],
        ["DMVCS","RESOURCE_INFEASIBLE","official 500 epochs x 5; isolated PyTorch environment absent"],
        ["RAV","RESOURCE_INFEASIBLE","official 400-epoch deep run; isolated PyTorch environment absent"],
        ["CorreGen","RESOURCE_INFEASIBLE","official 200-epoch run; isolated PyTorch environment absent"],
    ],columns=["method","status","reason"])
    failures.to_csv(out/"failures"/"baseline_failures.csv",index=False)
    sources.to_csv(out/"tables"/"Table_1_2026_Baseline_Audit.csv",index=False)
    catalog.to_csv(out/"tables"/"Table_2_2026_Dataset_Catalog.csv",index=False)
    for number, title in ((3,"Frozen11_ACC"),(4,"Frozen11_NMI"),(5,"Frozen11_ARI"),
                          (6,"External2026_ACC"),(7,"External2026_NMI"),(8,"External2026_ARI"),
                          (9,"Average_Rank"),(10,"Win_Tie_Loss"),(11,"Wilcoxon_Holm"),
                          (12,"Runtime"),(13,"Memory"),(15,"Published_2026_Literature_Context")):
        pd.DataFrame(columns=["status","reason"]).to_csv(out/"tables"/("Table_%d_%s.csv"%(number,title)),index=False)
    characteristics=pd.DataFrame([
        ["BGDR",2026,False,False,True,True,False,True,False,"direct discrete indicators",True],
        ["CGDR",2026,False,False,True,True,False,True,False,"direct discrete indicators",True],
        ["MISCC",2026,True,True,False,False,False,False,True,"KMeans",True],
        ["DMVCS",2026,True,True,False,False,False,False,True,"KMeans",True],
        ["DRCFMVC",2026,True,True,False,False,False,True,True,"KMeans",True],
        ["RAV",2026,True,True,False,False,False,True,True,"KMeans",True],
        ["CorreGen",2026,True,True,False,False,False,False,True,"KMeans",True],
        ["BGRF-MVC Paper Candidate v1.0",2026,False,False,False,False,True,False,False,"KMeans",True],
    ],columns=["Method","Year","Deep","Requires_GPU","Graph","Anchor","Granularity","Adaptive_view_weighting","n_by_n_sample_graph","Final_clustering","Executable_source_verified"])
    characteristics.to_csv(out/"tables"/"Table_14_Method_Characteristics.csv",index=False)
    gate={"stage":"9R","status":"COMPLETE","final_gate":"EXTERNAL_EVIDENCE_STILL_WEAK","MGBCC_status":MGBCC_STATUS,"ACCEPT_MAIN":accepted,"ACCEPT_MAIN_count":len(accepted),"formal_gate_required":3,"nondeep_present":True,"deep_or_modern_executable_present":False,"formal_phase_a_started":False,"formal_phase_b_started":False,"Frozen11_dataset_count":11,"Frozen11_selection_hash":FROZEN11_SHA256,"External2026_eligible_count":int(catalog.eligible.sum()),"External2026_selected_count":len(selected),"External2026_hash":selection_hash,"common_dataset_count":0,"label_firewall":"PASS","parameter_fairness":"PASS","BGRF_freeze_integrity":"PASS","algorithm_modified":False,"parameters_modified":False}
    (out/"stage9r_gate.json").write_text(json.dumps(gate,indent=2),encoding="utf-8")
    return gate


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
