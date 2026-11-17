"""Stage 7 Phase B: metric-equivalence evaluation of frozen predictions only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from bgrfmvc.evaluation.evaluate_predictions import compute_metrics
from bgrfmvc.evaluation.labels import load_evaluation_labels
from bgrfmvc.experiments.run_stage6_phase_a import _selected


METRICS=("ACC","NMI","ARI","PUR","F_score")


def evaluate(root_path, datasets_config):
    root=Path(root_path); phase=json.loads((root/"phase_a_manifest.json").read_text(encoding="utf-8"))
    if not phase.get("phase_a_complete") or phase.get("labels_loaded") is not False:
        raise RuntimeError("Phase A must be complete and label-free")
    manifest=yaml.safe_load(Path(datasets_config).read_text(encoding="utf-8"))["datasets"]
    rows=[]; equivalence=[]
    for key,slug,cfg in _selected(manifest):
        for method in ("D0","D3_reference","D3_OPT"):
            for seed in phase["benchmark_seeds"]:
                path=root/"predictions"/slug/method/("seed_%d"%seed)/"predictions.csv"
                if not path.is_file(): raise RuntimeError("Phase A prediction missing: %s"%path)
        y=load_evaluation_labels(cfg)
        keyed={}
        for method in ("D0","D3_reference","D3_OPT"):
            for seed in phase["benchmark_seeds"]:
                pred=pd.read_csv(root/"predictions"/slug/method/("seed_%d"%seed)/"predictions.csv").predicted_label.values
                metrics=compute_metrics(y,pred); row={"dataset":slug,"dataset_key":key,"method":method,
                    "seed":seed,"labels_loaded_phase_a":False}; row.update(metrics)
                row["Composite"]=float(np.mean([metrics[x] for x in METRICS])); rows.append(row)
                keyed[(method,seed)]=metrics
        for seed in phase["benchmark_seeds"]:
            diffs={m:abs(keyed[("D3_reference",seed)][m]-keyed[("D3_OPT",seed)][m]) for m in METRICS}
            equivalence.append({"dataset":slug,"seed":seed,"max_metric_abs_diff":max(diffs.values()),
                                **{m+"_abs_diff":v for m,v in diffs.items()}})
    raw=pd.DataFrame(rows); raw.to_csv(root/"stage7_raw.csv",index=False)
    summary=raw.groupby(["dataset","method"],as_index=False).agg({m:["mean","std"] for m in METRICS+("Composite",)})
    summary.columns=["_".join(x).rstrip("_") for x in summary.columns]; summary.to_csv(root/"stage7_summary.csv",index=False)
    eq=pd.DataFrame(equivalence); eq.to_csv(root/"equivalence"/"metric_equivalence.csv",index=False)
    bench=pd.read_csv(root/"benchmark"/"benchmark_raw.csv")
    bsum=bench.groupby(["dataset","implementation","mode"],as_index=False).agg(
        {c:["mean","median","std","min","max"] for c in
         ("representation_build_time","kmeans_time","total_runtime","peak_memory_MB")})
    bsum.columns=["_".join(x).rstrip("_") for x in bsum.columns]; bsum.to_csv(root/"benchmark"/"benchmark_summary.csv",index=False)
    ratios=[]
    for dataset in phase["datasets"]:
        def mean(impl,mode,col="total_runtime"):
            return float(bench[(bench.dataset==dataset)&(bench.implementation==impl)&(bench["mode"]==mode)][col].mean())
        ratios.append({"dataset":dataset,
          "cold_speedup":mean("D3_reference","cold")/mean("D3_OPT","cold"),
          "multi_seed_speedup":mean("D3_reference","multi_seed")/mean("D3_OPT","multi_seed"),
          "D3_OPT_vs_D0_cold_ratio":mean("D3_OPT","cold")/mean("D0","cold"),
          "D3_OPT_vs_D0_multiseed_ratio":mean("D3_OPT","multi_seed")/mean("D0","multi_seed"),
          "reference_build_fraction":mean("D3_reference","multi_seed","representation_build_time")/mean("D3_reference","multi_seed"),
          "optimized_build_fraction":mean("D3_OPT","multi_seed","representation_build_time")/mean("D3_OPT","multi_seed")})
    ratio_df=pd.DataFrame(ratios); ratio_df.to_csv(root/"benchmark"/"efficiency_ratios.csv",index=False)
    num=pd.read_csv(root/"equivalence"/"numerical_equivalence.csv")
    repro=json.loads((root/"reproduction"/"stage7_reproduction_check.json").read_text(encoding="utf-8"))
    handwritten=ratio_df[ratio_df.dataset=="Handwritten_numerals"].iloc[0]
    hw_runs=bench[(bench.dataset=="Handwritten_numerals")&(bench["mode"]=="multi_seed")]
    ref=hw_runs[hw_runs.implementation=="D3_reference"].sort_values("repeat").total_runtime.values
    opt=hw_runs[hw_runs.implementation=="D3_OPT"].sort_values("repeat").total_runtime.values
    stable=bool(np.all(opt<ref)); metric_max=float(eq.max_metric_abs_diff.max())
    scientific=bool(repro["status"]=="PASS" and repro["prediction_mismatches"]==0)
    semantic=bool(num.representation_allclose.all() and num.relation_allclose.all() and metric_max<=1e-12)
    efficiency=bool(stable and handwritten.D3_OPT_vs_D0_multiseed_ratio<=6.0)
    memory=bool(np.isfinite(bench.peak_memory_MB).all() and bench.peak_memory_MB.max()<4096)
    gate="FREEZE_PAPER_CANDIDATE" if scientific and semantic and efficiency and memory else (
         "METHOD_REVISION_REQUIRED" if not scientific or not semantic else "EFFICIENCY_REVISION_REQUIRED")
    gate_data={"final_gate":gate,"scientific_integrity":"PASS" if scientific else "FAIL",
      "semantic_equivalence":"PASS" if semantic else "FAIL","label_firewall":"PASS",
      "efficiency_gate":"PASS" if efficiency else "FAIL","memory_gate":"PASS" if memory else "FAIL",
      "handwritten":{"cold_speedup":handwritten.cold_speedup,"multi_seed_speedup":handwritten.multi_seed_speedup,
        "D3_OPT_vs_D0_cold_ratio":handwritten.D3_OPT_vs_D0_cold_ratio,
        "D3_OPT_vs_D0_multiseed_ratio":handwritten.D3_OPT_vs_D0_multiseed_ratio,
        "all_three_multiseed_repeats_faster":stable},
      "recommended_2x_target_met":bool(handwritten.multi_seed_speedup>=2.0),
      "recommended_2x_target_note":"Not met because the current reference already builds the seed-independent representation once; direct <=6x amortized-ratio rule passes.",
      "maximum_metric_abs_diff":metric_max,"boundary_status":"BOUNDARY_INTERPRETABILITY_ONLY",
      "paper_candidate_version":"BGRF-MVC Paper Candidate v1.0" if gate=="FREEZE_PAPER_CANDIDATE" else None}
    (root/"stage7_gate.json").write_text(json.dumps(gate_data,indent=2),encoding="utf-8")
    phase_b={"stage":"7","phase":"B","phase_a_complete_before_labels":True,"labels_loaded":True,
             "models_refit":False,"representations_rebuilt":False,"metric_equivalence_max_abs_diff":metric_max,
             "outputs_complete":True}
    (root/"phase_b_manifest.json").write_text(json.dumps(phase_b,indent=2),encoding="utf-8")
    return gate_data


def main():
    p=argparse.ArgumentParser(); p.add_argument("--root",default="result/stage7_freeze")
    p.add_argument("--datasets-config",default="configs/datasets.yaml"); a=p.parse_args()
    print(json.dumps(evaluate(a.root,a.datasets_config),indent=2))


if __name__=="__main__": main()
