"""Export Frozen11 X-only arrays for the audited MATLAB adapters."""
from pathlib import Path
import json, numpy as np, scipy.io as sio, yaml
from bgrfmvc.data.loader import load_multiview_features

def run():
    cfg=yaml.safe_load(Path("configs/stage9_eval_11.yaml").read_text(encoding="utf-8"))["datasets"]
    out=Path("result/stage9rc_external_comparison/phase_a/frozen11_mat"); out.mkdir(parents=True,exist_ok=True); rows=[]
    for name,c in cfg.items():
        if name=="ALOI-100": continue
        source=yaml.safe_load(Path("configs/stage8r_selected_benchmark.yaml").read_text(encoding="utf-8"))["datasets"][name]
        views=load_multiview_features({k:source[k] for k in ("path","view_keys","transpose_views")})
        safe=name.replace("-","_"); cells=np.empty((1,len(views)),dtype=object)
        for i,x in enumerate(views): cells[0,i]=np.asarray(x,dtype=np.float64)
        sio.savemat(str(out/(safe+".mat")),{"X":cells},do_compression=True)
        rows.append({"dataset":name,"safe_name":safe,"K":int(c["K"]),"n_samples":int(c["n_samples"]),"n_views":int(c["n_views"]),"view_shapes":[list(x.shape) for x in views],"labels_loaded":False})
    (out/"manifest.json").write_text(json.dumps(rows,indent=2),encoding="utf-8"); return rows
if __name__=="__main__": print(json.dumps(run(),indent=2))
