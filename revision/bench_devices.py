"""Benchmark CPU vs GPU for every model on this dataset and recommend --gpu-models.

The dataset is small (616 x 9), so GPU kernels may be *slower* than one CPU
thread because of launch/transfer overhead.  Run once on the workstation:

    python bench_devices.py            # uses GPU 0
    python bench_devices.py --gpu 2

It prints the mean time of one fit per device and the recommended
``--gpu-models`` value for run_experiments.py.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--repeats", type=int, default=15)
    args = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    import numpy as np
    from rev.data import get_target_data
    from rev.features import MRMRPreprocessor
    from rev.models import build_model

    td = get_target_data("Y1")
    Z = MRMRPreprocessor(9).fit(td.X, td.y).transform(td.X)
    y = td.y.values
    params = {
        "XGBoost": {"n_estimators": 150, "max_depth": 5, "learning_rate": 0.1, "reg_alpha": 0.1, "reg_lambda": 1.0},
        "CatBoost": {"iterations": 150, "depth": 5, "learning_rate": 0.1, "l2_leaf_reg": 3.0},
        "LightGBM": {"n_estimators": 150, "max_depth": 5, "learning_rate": 0.1, "reg_alpha": 0.1, "reg_lambda": 1.0},
    }
    rec = []
    print(f"{'model':10s} {'cpu (s/fit)':>12s} {'gpu (s/fit)':>12s}  note")
    for m, p in params.items():
        res = {}
        for dev in ("cpu", "cuda"):
            try:
                kw = {"lgbm_gpu": True} if m == "LightGBM" else {}
                est = build_model(m, p, dev, threads=1, gpu_ram_part=0.2, **kw)
                est.fit(Z, y)  # warm-up
                t0 = time.perf_counter()
                for _ in range(args.repeats):
                    build_model(m, p, dev, threads=1, gpu_ram_part=0.2, **kw).fit(Z, y)
                res[dev] = (time.perf_counter() - t0) / args.repeats
            except Exception as e:  # GPU not available / not compiled in
                res[dev] = None
                res[dev + "_err"] = str(e).splitlines()[0][:80]
        cpu, gpu = res["cpu"], res["cuda"]
        note = res.get("cuda_err", "")
        if gpu is not None and cpu is not None and gpu < cpu:
            rec.append(m)
            note = note or "GPU faster"
        elif gpu is not None:
            note = note or "CPU faster (small data)"
        print(f"{m:10s} {cpu if cpu is None else f'{cpu:.4f}':>12} {gpu if gpu is None else f'{gpu:.4f}':>12}  {note}")
    print("\nRecommended:")
    if rec:
        extra = " --lgbm-gpu" if "LightGBM" in rec else ""
        print(f"  --gpu-models {','.join(rec)}{extra}")
    else:
        print("  GPU is not faster for any model on this data size. Use GPU slots anyway only for parallel "
              "capacity, e.g. --gpu-models XGBoost, or run CPU-only with --gpus none and more --cpu-workers.")


if __name__ == "__main__":
    main()
