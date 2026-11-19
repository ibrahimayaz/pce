"""Model definitions, search spaces and device handling.

Search spaces are identical to the original study (Table 4) except that
integer bounds are now decoded by rounding (the original code truncated with
``int()``, which allowed e.g. depth = 1 below the declared lower bound of 2).

Device policy (no NVLink => one job = one GPU):
* XGBoost  -> ``device="cuda"`` when the job runs on a GPU slot.
* CatBoost -> ``task_type="GPU"`` when the job runs on a GPU slot; the GPU RAM
  share is limited so that several workers can share one card.
* LightGBM -> CPU by default (the pip wheel has no CUDA build); GPU (OpenCL)
  only if explicitly enabled and available.
* AdaBoost, MLP (scikit-learn) -> CPU only.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List

GPU_CAPABLE = {"XGBoost", "CatBoost"}


@dataclass
class Param:
    name: str
    kind: str  # "int" | "float"
    low: float
    high: float


SPACES: Dict[str, List[Param]] = {
    "LightGBM": [
        Param("n_estimators", "int", 20, 150),
        Param("max_depth", "int", 2, 5),
        Param("learning_rate", "float", 0.001, 0.1),
        Param("reg_alpha", "float", 0.1, 10.0),
        Param("reg_lambda", "float", 0.1, 10.0),
    ],
    "XGBoost": [
        Param("n_estimators", "int", 20, 150),
        Param("max_depth", "int", 2, 5),
        Param("learning_rate", "float", 0.001, 0.1),
        Param("reg_alpha", "float", 0.1, 10.0),
        Param("reg_lambda", "float", 0.1, 10.0),
    ],
    "AdaBoost": [
        Param("n_estimators", "int", 20, 100),
        Param("learning_rate", "float", 0.01, 0.5),
    ],
    "CatBoost": [
        Param("iterations", "int", 50, 150),
        Param("depth", "int", 2, 5),
        Param("learning_rate", "float", 0.001, 0.1),
        Param("l2_leaf_reg", "float", 1.0, 10.0),
    ],
    "MLP": [
        Param("hidden_layer_size_1", "int", 16, 128),
        Param("hidden_layer_size_2", "int", 8, 64),
        Param("alpha", "float", 0.01, 1.0),
        Param("learning_rate_init", "float", 0.0001, 0.05),
    ],
}

# "default" tuner = library defaults, identical to the original default_params run
DEFAULT_PARAMS: Dict[str, Dict[str, Any]] = {
    "LightGBM": {},
    "XGBoost": {},
    "AdaBoost": {},
    "CatBoost": {},
    "MLP": {"hidden_layer_sizes": (32, 16)},
}


def canonical_params(model: str, raw: Dict[str, Any]) -> Dict[str, Any]:
    """Clip/round raw tuner output into valid hyper-parameters."""
    out: Dict[str, Any] = {}
    for p in SPACES[model]:
        v = raw[p.name]
        if p.kind == "int":
            v = int(min(max(int(round(float(v))), int(p.low)), int(p.high)))
        else:
            v = float(min(max(float(v), p.low), p.high))
            v = float(f"{v:.10g}")
        out[p.name] = v
    if model == "MLP":
        out = {
            "hidden_layer_sizes": (out.pop("hidden_layer_size_1"), out.pop("hidden_layer_size_2")),
            **out,
        }
    return out


def params_key(params: Dict[str, Any]) -> str:
    return "|".join(f"{k}={params[k]}" for k in sorted(params))


def build_model(model: str, params: Dict[str, Any], device: str = "cpu", threads: int = 1,
                seed: int = 42, gpu_ram_part: float = 0.2, lgbm_gpu: bool = False):
    """Instantiate an estimator. ``device`` is "cpu" or "cuda"."""
    params = dict(params)
    if model == "LightGBM":
        from lightgbm import LGBMRegressor
        extra = {"device_type": "gpu"} if (device == "cuda" and lgbm_gpu) else {}
        return LGBMRegressor(verbose=-1, random_state=seed, n_jobs=threads, **extra, **params)
    if model == "XGBoost":
        from xgboost import XGBRegressor
        return XGBRegressor(objective="reg:squarederror", random_state=seed, n_jobs=threads, verbosity=0,
                            tree_method="hist", device="cuda" if device == "cuda" else "cpu",
                            **params)
    if model == "CatBoost":
        from catboost import CatBoostRegressor
        extra: Dict[str, Any] = {"thread_count": threads}
        if device == "cuda":
            # CUDA_VISIBLE_DEVICES is set per worker, so the assigned card is always "0"
            extra.update(task_type="GPU", devices="0", gpu_ram_part=gpu_ram_part)
        return CatBoostRegressor(verbose=0, random_state=seed, allow_writing_files=False,
                                 **extra, **params)
    if model == "AdaBoost":
        from sklearn.ensemble import AdaBoostRegressor
        return AdaBoostRegressor(random_state=seed, **params)
    if model == "MLP":
        from sklearn.neural_network import MLPRegressor
        p = {"max_iter": 2000, "early_stopping": True, "random_state": seed}
        p.update(params)
        return MLPRegressor(**p)
    raise ValueError(model)


def to_cpu(model_name: str, est) -> None:
    """Make a fitted estimator portable to CPU-only machines (GUI)."""
    if model_name == "XGBoost":
        try:
            est.set_params(device="cpu")
        except Exception:
            pass


def set_thread_env(threads: int) -> None:
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[var] = str(threads)
