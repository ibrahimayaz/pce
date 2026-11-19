"""Job enumeration and execution.

A job = (block, target, data variant, model, tuner, outer fold) or, for the
sensitivity block, (target, model, pop, epoch, seed).  Each job writes exactly
one JSON result file atomically; a job whose result file exists is complete.
"""
from __future__ import annotations

import platform
import time
import traceback
import zlib
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from sklearn.metrics import (mean_absolute_error, mean_absolute_percentage_error,
                             mean_squared_error, r2_score)
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline

from .config import PROFILES, Profile, eo_evals, results_dir
from .data import get_target_data
from .features import MRMRPreprocessor
from .io_utils import EvalCache, atomic_joblib_dump, atomic_write_json, atomic_write_text
from .models import GPU_CAPABLE, build_model, to_cpu
from .splits import inner_splits, outer_splits
from .tuners import InnerObjective, tune

COST = {"MLP": 10.0, "CatBoost": 3.0, "AdaBoost": 2.0, "XGBoost": 2.0, "LightGBM": 1.0}


def run_root(profile: Profile) -> Path:
    return results_dir(profile) / f"budget_{profile.budget_hash()}"


def paths(profile: Profile) -> Dict[str, Path]:
    root = run_root(profile)
    d = {k: root / k for k in ("jobs", "cache", "locks", "errors", "final_models", "logs")}
    for p in d.values():
        p.mkdir(parents=True, exist_ok=True)
    d["root"] = root
    return d


def metrics(y, p) -> Dict[str, float]:
    y, p = np.asarray(y, float), np.asarray(p, float)
    mse = mean_squared_error(y, p)
    return {
        "R2": float(r2_score(y, p)) if len(y) > 1 else float("nan"),
        "RMSE": float(np.sqrt(mse)),
        "MAE": float(mean_absolute_error(y, p)),
        "MSE": float(mse),
        "MAPE": float(mean_absolute_percentage_error(y, p)),
    }


def enumerate_jobs(profile: Profile, gpu_models: List[str], use_gpu: bool) -> List[Dict[str, Any]]:
    jobs: List[Dict[str, Any]] = []
    for b in profile.blocks:
        for target in b.targets:
            td = get_target_data(target, b.data)
            folds = outer_splits(td, b.scheme, profile.rkf_folds, profile.rkf_repeats, profile.seed)
            if profile.max_outer_folds:
                folds = folds[: profile.max_outer_folds]
            for model in b.models:
                device = "cuda" if (use_gpu and model in gpu_models) else "cpu"
                for tuner in b.tuners:
                    if b.scheme == "sens":
                        for pop in b.sens_pop:
                            for epoch in b.sens_epoch:
                                for s in b.sens_seeds:
                                    jid = f"{b.name}__{target}__{b.data}__{model}__{tuner}__p{pop}e{epoch}s{s}"
                                    jobs.append(dict(id=jid, block=b.name, scheme=b.scheme, target=target,
                                                     data=b.data, model=model, tuner=tuner, fold="all",
                                                     pop=pop, epoch=epoch, seed=s, device=device,
                                                     priority=b.priority,
                                                     cost=COST[model] * eo_evals(pop, epoch)))
                        continue
                    for fold_name, _ in folds:
                        jid = f"{b.name}__{target}__{b.data}__{model}__{tuner}__{fold_name}"
                        seed = profile.seed + zlib.crc32(fold_name.encode()) % 10000
                        cost = COST[model] * (0.01 if tuner == "default" else profile.n_evals)
                        jobs.append(dict(id=jid, block=b.name, scheme=b.scheme, target=target, data=b.data,
                                         model=model, tuner=tuner, fold=fold_name, pop=profile.eo_pop,
                                         epoch=profile.eo_epoch, seed=seed, device=device,
                                         priority=b.priority, cost=cost))
    jobs.sort(key=lambda j: (j["priority"], -j["cost"], j["id"]))
    return jobs


def run_job(job: Dict[str, Any], profile: Profile, threads: int, gpu_ram_part: float,
            lgbm_gpu: bool, slot: str) -> Dict[str, Any]:
    P = paths(profile)
    t_start = time.time()
    td = get_target_data(job["target"], job["data"])
    folds = dict(outer_splits(td, job["scheme"], profile.rkf_folds, profile.rkf_repeats, profile.seed))
    tr, te = folds[job["fold"]]
    Xtr, ytr = td.X.iloc[tr], td.y.iloc[tr]

    inner = inner_splits(td, tr, job["scheme"], profile.inner_folds, job["seed"])
    # the identity baseline keeps all one-hot columns (mRMR would otherwise drop admixtures)
    k = None if job["data"] == "pceid" else profile.n_features
    cache = EvalCache(P["cache"] / f"{job['id']}.jsonl")
    try:
        obj = InnerObjective(job["model"], Xtr, ytr, inner, k, job["device"], threads,
                             profile.seed, gpu_ram_part, lgbm_gpu, cache)
        params, t_tune = tune(job["tuner"], obj, pop=job["pop"], epoch=job["epoch"],
                              n_evals=eo_evals(job["pop"], job["epoch"]), seed=job["seed"])
    finally:
        cache.close()

    hist = np.minimum.accumulate(np.asarray(obj.history, float)) if obj.history else np.array([])
    result: Dict[str, Any] = dict(job)
    result.update(
        profile=profile.name, budget=profile.budget_hash(), slot=slot, host=platform.node(),
        best_params=params, inner_best_mse=float(obj.best), n_evals=obj.n_calls,
        n_cache_hits=obj.n_cache_hits, tune_seconds=t_tune,
        best_so_far=[float(f"{v:.6g}") for v in hist],
        n_train=int(len(tr)), n_test=int(len(te)),
    )

    pipe = Pipeline([("prep", MRMRPreprocessor(k=k)),
                     ("model", build_model(job["model"], params, job["device"], threads, profile.seed,
                                           gpu_ram_part, lgbm_gpu))])

    if job["scheme"] in ("rkf", "lopo", "lofa"):
        pipe.fit(Xtr, ytr.values)
        pred = pipe.predict(td.X.iloc[te])
        result["selected_features"] = pipe.named_steps["prep"].selected_
        result["metrics"] = metrics(td.y.iloc[te].values, pred)
        result["test_index"] = te.tolist()
        result["test_mix_id"] = td.mix_id.iloc[te].tolist()
        result["y_true"] = td.y.iloc[te].tolist()
        result["y_pred"] = [float(v) for v in pred]
        result["measured"] = td.measured.iloc[te].astype(bool).tolist()
        result["X21"] = td.X["X21"].iloc[te].tolist() if "X21" in td.X else []
        result["fa_level"] = td.fa_level.iloc[te].tolist()
        result["pce"] = td.pce.iloc[te].tolist()
    elif job["scheme"] == "sens":
        # independent check of the tuned configuration on a different 5-fold split
        kf = KFold(n_splits=5, shuffle=True, random_state=1000)
        rm = []
        for a, b in kf.split(td.X):
            pipe_i = Pipeline([("prep", MRMRPreprocessor(k=k)),
                               ("model", build_model(job["model"], params, job["device"], threads,
                                                     profile.seed, gpu_ram_part, lgbm_gpu))])
            pipe_i.fit(td.X.iloc[a], td.y.iloc[a].values)
            rm.append(metrics(td.y.iloc[b].values, pipe_i.predict(td.X.iloc[b]))["RMSE"])
        result["check_cv_rmse_mean"] = float(np.mean(rm))
        result["check_cv_rmse_std"] = float(np.std(rm))
    elif job["scheme"] == "final":
        # final refit on all data on CPU so that the saved model is portable (GUI)
        final = Pipeline([("prep", MRMRPreprocessor(k=k)),
                          ("model", build_model(job["model"], params, "cpu", threads, profile.seed))])
        final.fit(td.X, td.y.values)
        to_cpu(job["model"], final.named_steps["model"])
        model_path = P["final_models"] / f"{job['target']}_{job['data']}_{job['model']}.joblib"
        atomic_joblib_dump(final, model_path)
        result["model_path"] = str(model_path.relative_to(P["root"]))
        result["selected_features"] = final.named_steps["prep"].selected_
        result["train_metrics"] = metrics(td.y.values, final.predict(td.X))
    result["total_seconds"] = time.time() - t_start
    atomic_write_json(P["jobs"] / f"{job['id']}.json", result)
    return result


def record_error(job: Dict[str, Any], profile: Profile) -> None:
    P = paths(profile)
    atomic_write_text(P["errors"] / f"{job['id']}.txt", traceback.format_exc())


def get_profile(name: str) -> Profile:
    return PROFILES[name]
