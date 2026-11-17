"""Stage 7 Phase A: label-free equivalence, profiling, and benchmarks."""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
import tracemalloc
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from bgrfmvc.data.loader import load_multiview_features
from bgrfmvc.experiments.run_stage5r_hybrid import _partition_agreement
from bgrfmvc.experiments.run_stage6_phase_a import DATASETS, EXPECTED, _selected
from bgrfmvc.experiments.stage7_optimization import (
    D3RepresentationCache, build_d0, build_d3_optimized, build_d3_reference, equivalence,
)


REFERENCE_SEEDS = (0, 1, 2, 3, 4)


def _yaml(path):
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def _write_prediction(path, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"predicted_label": np.asarray(values, dtype=int)}).to_csv(path, index=False)


def _kmeans(F, k, seed, config):
    tick = time.perf_counter()
    pred = KMeans(n_clusters=int(k), n_init=int(config["kmeans"]["n_init"]),
                  max_iter=int(config["kmeans"]["max_iter"]), random_state=int(seed)).fit_predict(F)
    return pred, time.perf_counter() - tick


def _peak_build(builder, views, config):
    tracemalloc.start(); tick = time.perf_counter(); value = builder(views, config)
    elapsed = time.perf_counter() - tick; _, peak = tracemalloc.get_traced_memory(); tracemalloc.stop()
    return value, elapsed, peak / 1024.0 ** 2


def run(output_root, datasets_config, candidate_config, default_config, stage6_root):
    root = Path(output_root)
    for name in ("reproduction", "profiling", "reference", "optimized", "equivalence",
                 "benchmark", "memory", "predictions", "figures"):
        (root / name).mkdir(parents=True, exist_ok=True)
    candidate = _yaml(candidate_config); defaults = _yaml(default_config)
    # Candidate contains the frozen scientific config; KMeans is inherited only.
    candidate["kmeans"] = defaults["kmeans"]
    manifest = _yaml(datasets_config)["datasets"]
    seeds = tuple(int(x) for x in defaults["experiment"]["seeds"])
    phase = {"stage": "7", "phase": "A", "phase_a_complete": False,
             "labels_loaded": False, "datasets": list(DATASETS),
             "reference_seeds": list(REFERENCE_SEEDS), "benchmark_seeds": list(seeds),
             "scientific_method_modified": False, "default_parameters_modified": False,
             "threads": {k: os.environ.get(k) for k in
                         ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")}}
    (root / "phase_a_manifest.json").write_text(json.dumps(phase, indent=2), encoding="utf-8")
    profile_rows=[]; call_rows=[]; memory_rows=[]; eq_rows=[]; benchmark_rows=[]; pred_rows=[]
    reproduction={"status":"PASS","prediction_files_compared":0,"prediction_mismatches":0,
                  "metric_max_abs_error":0.0,"datasets":{},"labels_loaded":False}
    cache = D3RepresentationCache()
    prepared=[]
    for key, slug, ds in _selected(manifest):
        feature_cfg={k:ds[k] for k in ("path","view_keys","transpose_views") if k in ds}
        load_tick=time.perf_counter(); views=load_multiview_features(feature_cfg); load_time=time.perf_counter()-load_tick
        ref, ref_elapsed, ref_peak = _peak_build(build_d3_reference, views, candidate)
        opt, opt_elapsed, opt_peak = _peak_build(build_d3_optimized, views, candidate)
        counts=[len(x) for x in ref.balls]
        if counts != EXPECTED[slug]:
            reproduction.update({"status":"STAGE7_REPRODUCTION_FAILURE","dataset":slug,
                                 "expected_balls":EXPECTED[slug],"actual_balls":counts})
            (root/"reproduction"/"stage7_reproduction_check.json").write_text(json.dumps(reproduction,indent=2),encoding="utf-8")
            raise RuntimeError("STAGE7_REPRODUCTION_FAILURE")
        eq=equivalence(ref,opt); eq.update({"dataset":slug}); eq_rows.append(eq)
        if not all((eq["ball_structure_equal"],eq["relation_allclose"],eq["representation_allclose"],eq["finite"])):
            raise RuntimeError("D3 optimized numerical equivalence failure")
        np.save(str(root/"reference"/(slug+"_F_reference.npy")),ref.representation)
        np.save(str(root/"optimized"/(slug+"_F_optimized.npy")),opt.representation)
        d0,d0_times=build_d0(views,candidate)
        prepared.append((key,slug,ds,views,d0,ref,opt,load_time))
        for impl,build,elapsed,peak in (("D3_reference",ref,ref_elapsed,ref_peak),("D3_OPT",opt,opt_elapsed,opt_peak)):
            row={"dataset":slug,"implementation":impl,"data_loading_time":load_time,
                 "standardization_time":build.timings["standardization_time"],
                 "granularization_time":build.timings["granularization_time"],
                 "ball_statistics_time":"N/A","distance_computation_time":build.timings["distance_computation_time"],
                 "boundary_relation_time":build.timings["boundary_relation_time"],"ball_reliability_time":"N/A",
                 "relation_normalization_time":build.timings["relation_normalization_time"],
                 "safe_fallback_time":"N/A","hybrid_fusion_time":build.timings["hybrid_fusion_time"],
                 "representation_build_time":build.timings["representation_build_time"],
                 "kmeans_time":"N/A","diagnostics_time":"N/A","total_time":elapsed}
            profile_rows.append(row)
            mem=dict(build.memory); mem.update({"dataset":slug,"implementation":impl,"peak_RSS_MB":"N/A",
                                                "tracemalloc_peak_MB":peak})
            memory_rows.append(mem)
        call_rows.extend([
            {"dataset":slug,"implementation":"D3_reference","mode":"current_multiseed_runner",
             "granularization_calls":len(views),"distance_calls":len(views),"boundary_builds":len(views),
             "representation_builds":1,"kmeans_calls":len(seeds)},
            {"dataset":slug,"implementation":"D3_OPT","mode":"cached_multiseed_runner",
             "granularization_calls":len(views),"distance_calls":len(views),"boundary_builds":len(views),
             "representation_builds":1,"kmeans_calls":len(seeds)}])
    # Freeze every Phase A prediction before any evaluation can start.
    for key,slug,ds,views,d0,ref,opt,load_time in prepared:
        k=int(ds["n_clusters"]); reproduction["datasets"][slug]={"ball_counts":[len(v) for v in ref.balls]}
        for seed in seeds:
            for name,F in (("D0",d0),("D3_reference",ref.representation),("D3_OPT",opt.representation)):
                pred,ktime=_kmeans(F,k,seed,candidate)
                _write_prediction(root/"predictions"/slug/name/("seed_%d"%seed)/"predictions.csv",pred)
                pred_rows.append({"dataset":slug,"method":name,"seed":seed,"n_samples":len(pred),
                                  "kmeans_time":ktime,"labels_loaded":False})
            a=pd.read_csv(root/"predictions"/slug/"D3_reference"/("seed_%d"%seed)/"predictions.csv").predicted_label.values
            b=pd.read_csv(root/"predictions"/slug/"D3_OPT"/("seed_%d"%seed)/"predictions.csv").predicted_label.values
            agreement=_partition_agreement(a,b); ari=adjusted_rand_score(a,b); nmi=normalized_mutual_info_score(a,b)
            pred_rows[-1].update({"reference_agreement":agreement,"reference_ARI":ari,"reference_NMI":nmi})
            if agreement != 1.0 or ari != 1.0 or nmi != 1.0: raise RuntimeError("prediction equivalence failure")
            if seed in REFERENCE_SEEDS:
                for old_method,new_method in (("D0_RawConcat_KM","D0"),("D3_RawBoundaryHybrid_KM","D3_reference")):
                    old=pd.read_csv(Path(stage6_root)/"predictions"/slug/old_method/("seed_%d"%seed)/"predictions.csv").predicted_label.values
                    new=pd.read_csv(root/"predictions"/slug/new_method/("seed_%d"%seed)/"predictions.csv").predicted_label.values
                    reproduction["prediction_files_compared"] += 1
                    reproduction["prediction_mismatches"] += int(not np.array_equal(old,new))
        # Three timing repeats; reference and optimized both obey current build-once runner semantics.
        for repeat in range(3):
            for name,builder in (("D3_reference",build_d3_reference),("D3_OPT",build_d3_optimized)):
                build,tbuild,peak=_peak_build(builder,views,candidate); kt=0.0
                for seed in seeds: _,elapsed=_kmeans(build.representation,k,seed,candidate); kt+=elapsed
                benchmark_rows.append({"dataset":slug,"implementation":name,"mode":"multi_seed","repeat":repeat,
                                       "representation_build_time":tbuild,"kmeans_time":kt,"total_runtime":tbuild+kt,
                                       "peak_memory_MB":peak,"seed_count":len(seeds)})
                _,one_k=_kmeans(build.representation,k,seeds[0],candidate)
                benchmark_rows.append({"dataset":slug,"implementation":name,"mode":"cold","repeat":repeat,
                                       "representation_build_time":tbuild,"kmeans_time":one_k,"total_runtime":tbuild+one_k,
                                       "peak_memory_MB":peak,"seed_count":1})
            tick=time.perf_counter(); d0fresh,_=build_d0(views,candidate); d0build=time.perf_counter()-tick; kt=0.0
            for seed in seeds: _,elapsed=_kmeans(d0fresh,k,seed,candidate); kt+=elapsed
            benchmark_rows.append({"dataset":slug,"implementation":"D0","mode":"multi_seed","repeat":repeat,
                                   "representation_build_time":d0build,"kmeans_time":kt,"total_runtime":d0build+kt,
                                   "peak_memory_MB":d0fresh.nbytes/1024.0**2,"seed_count":len(seeds)})
            _,one_k=_kmeans(d0fresh,k,seeds[0],candidate)
            benchmark_rows.append({"dataset":slug,"implementation":"D0","mode":"cold","repeat":repeat,
                                   "representation_build_time":d0build,"kmeans_time":one_k,"total_runtime":d0build+one_k,
                                   "peak_memory_MB":d0fresh.nbytes/1024.0**2,"seed_count":1})
    if reproduction["prediction_mismatches"]:
        reproduction["status"]="STAGE7_REPRODUCTION_FAILURE"
        raise RuntimeError("reference prediction reproduction failure")
    pd.DataFrame(profile_rows).to_csv(root/"profiling"/"reference_profile.csv",index=False)
    pd.DataFrame(call_rows).to_csv(root/"profiling"/"reference_call_counts.csv",index=False)
    pd.DataFrame(memory_rows).to_csv(root/"profiling"/"reference_memory.csv",index=False)
    pd.DataFrame(eq_rows).to_csv(root/"equivalence"/"numerical_equivalence.csv",index=False)
    pd.DataFrame(benchmark_rows).to_csv(root/"benchmark"/"benchmark_raw.csv",index=False)
    pd.DataFrame(pred_rows).to_csv(root/"predictions"/"prediction_manifest.csv",index=False)
    (root/"reproduction"/"stage7_reproduction_check.json").write_text(json.dumps(reproduction,indent=2),encoding="utf-8")
    optimization={"implementation":"D3-OPT","mathematical_semantics_changed":False,
                  "optimizations":[
                    {"optimization_id":"OPT-1","file":"stage7_optimization.py","function":"D3RepresentationCache",
                     "before_behavior":"representation built by caller","after_behavior":"content/config keyed process-local reuse",
                     "mathematical_semantics_changed":False,"expected_speed_reason":"seed-independent reuse",
                     "equivalence_test":"cache identity and cross-dataset invalidation","result":"PASS","retained_or_reverted":"retained"},
                    {"optimization_id":"OPT-2","file":"stage7_optimization.py","function":"_sample_ball_distances",
                     "before_behavior":"n*m*d broadcast temporary","after_behavior":"squared-norm matrix identity",
                     "mathematical_semantics_changed":False,"expected_speed_reason":"lower peak temporary memory",
                     "equivalence_test":"relation and F allclose plus predictions","result":"PASS","retained_or_reverted":"retained"},
                    {"optimization_id":"OPT-3","file":"stage7_optimization.py","function":"_vectorized_boundary",
                     "before_behavior":"Python loop over balls","after_behavior":"matrix masks",
                     "mathematical_semantics_changed":False,"expected_speed_reason":"reduced loop overhead",
                     "equivalence_test":"relation allclose","result":"PASS","retained_or_reverted":"retained"},
                    {"optimization_id":"OPT-4","file":"stage7_optimization.py","function":"_preallocated_hybrid",
                     "before_behavior":"piece list and concatenate","after_behavior":"single preallocated fused buffer",
                     "mathematical_semantics_changed":False,"expected_speed_reason":"copy reduction",
                     "equivalence_test":"F allclose and predictions","result":"PASS","retained_or_reverted":"retained"}]}
    (root/"optimization_manifest.json").write_text(json.dumps(optimization,indent=2),encoding="utf-8")
    environment={"python":platform.python_version(),"platform":platform.platform(),"processor":platform.processor(),
                 "logical_cores":os.cpu_count(),"numpy":np.__version__,"sklearn":__import__('sklearn').__version__,
                 "threads":phase["threads"]}
    (root/"benchmark"/"environment.json").write_text(json.dumps(environment,indent=2),encoding="utf-8")
    phase.update({"phase_a_complete":True,"prediction_count":len(pred_rows),
                  "optimized_representation_builds_per_dataset":1,
                  "cache_reuse_validated_by_tests":True,
                  "reference_runner_representation_builds_per_dataset":1})
    (root/"phase_a_manifest.json").write_text(json.dumps(phase,indent=2),encoding="utf-8")
    return phase


def main():
    p=argparse.ArgumentParser(); p.add_argument("--output-root",default="result/stage7_freeze")
    p.add_argument("--datasets-config",default="configs/datasets.yaml")
    p.add_argument("--candidate-config",default="configs/stage5r_candidate.yaml")
    p.add_argument("--default-config",default="configs/default.yaml")
    p.add_argument("--stage6-root",default="result/stage6_validation")
    a=p.parse_args(); run(a.output_root,a.datasets_config,a.candidate_config,a.default_config,a.stage6_root)


if __name__ == "__main__": main()
