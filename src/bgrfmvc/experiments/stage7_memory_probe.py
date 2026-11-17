"""Isolated-process peak-RSS probe for one Stage 7 representation build."""

from __future__ import annotations

import argparse
import json
import os

import psutil
import yaml

from bgrfmvc.data.loader import load_multiview_features
from bgrfmvc.experiments.run_stage6_phase_a import _selected
from bgrfmvc.experiments.stage7_optimization import build_d0, build_d3_optimized, build_d3_reference


def main():
    p=argparse.ArgumentParser(); p.add_argument("--dataset",required=True)
    p.add_argument("--implementation",choices=("D0","D3_reference","D3_OPT"),required=True)
    a=p.parse_args(); manifest=yaml.safe_load(open("configs/datasets.yaml",encoding="utf-8"))["datasets"]
    config=yaml.safe_load(open("configs/stage5r_candidate.yaml",encoding="utf-8"))
    _,slug,ds=next(x for x in _selected(manifest) if x[1]==a.dataset)
    cfg={k:ds[k] for k in ("path","view_keys","transpose_views")}; views=load_multiview_features(cfg)
    if a.implementation=="D0": value,_=build_d0(views,config); rep=value
    else:
        value=(build_d3_reference if a.implementation=="D3_reference" else build_d3_optimized)(views,config)
        rep=value.representation
    info=psutil.Process(os.getpid()).memory_info()
    print(json.dumps({"dataset":slug,"implementation":a.implementation,
                      "peak_RSS_MB":float(info.peak_wset)/1024.0**2,
                      "current_RSS_MB":float(info.rss)/1024.0**2,
                      "representation_MB":rep.nbytes/1024.0**2}))


if __name__=="__main__": main()
