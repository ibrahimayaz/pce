"""Hyper-parameter tuners with an identical objective and evaluation budget.

Objective (Reviewer 1, #23): mean squared error averaged over the inner
cross-validation folds of the *outer-training* data.  Preprocessing
(zero-variance removal, standardisation, mRMR) is fitted inside every inner
training fold; because it does not depend on the hyper-parameters it is
computed once per inner fold and reused for all evaluations.

Tuners
------
* default : library defaults, no search
* random  : uniform random search, ``n_evals`` evaluations
* tpe     : Bayesian optimisation (Optuna TPE), ``n_evals`` evaluations
* eo      : Equilibrium Optimizer (mealpy ``OriginalEO``: V=1, a1=2, a2=1,
            GP=0.5), population ``pop`` x ``epoch`` iterations

All tuners are seeded.  Every evaluation is written to a persistent cache, so
after an interruption the tuner is replayed against the cache and continues
from the first evaluation that had not been completed.
"""
from __future__ import annotations

import math
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

from .features import MRMRPreprocessor
from .io_utils import EvalCache
from .models import SPACES, DEFAULT_PARAMS, build_model, canonical_params, params_key


class InnerObjective:
    def __init__(self, model: str, X: pd.DataFrame, y: pd.Series, inner_splits: List[Tuple[np.ndarray, np.ndarray]],
                 n_features: int, device: str, threads: int, seed: int, gpu_ram_part: float,
                 lgbm_gpu: bool, cache: Optional[EvalCache]):
        self.model = model
        self.device, self.threads, self.seed = device, threads, seed
        self.gpu_ram_part, self.lgbm_gpu = gpu_ram_part, lgbm_gpu
        self.cache = cache
        self.folds = []
        y = np.asarray(y, dtype=float)
        for tr, va in inner_splits:
            prep = MRMRPreprocessor(k=n_features).fit(X.iloc[tr], y[tr])
            self.folds.append((prep.transform(X.iloc[tr]), y[tr], prep.transform(X.iloc[va]), y[va]))
        self.n_calls = 0
        self.n_cache_hits = 0
        self.history: List[float] = []  # objective of every evaluation, in order
        self.best = math.inf
        self.best_params: Optional[Dict[str, Any]] = None

    def __call__(self, params: Dict[str, Any]) -> float:
        key = params_key(params)
        val = self.cache.get(key) if self.cache is not None else None
        if val is None:
            losses = []
            for Xtr, ytr, Xva, yva in self.folds:
                est = build_model(self.model, params, self.device, self.threads, self.seed,
                                  self.gpu_ram_part, self.lgbm_gpu)
                est.fit(Xtr, ytr)
                losses.append(mean_squared_error(yva, est.predict(Xva)))
            val = float(np.mean(losses))
            if not np.isfinite(val):
                val = 1e30
            if self.cache is not None:
                self.cache.put(key, val)
        else:
            self.n_cache_hits += 1
        self.n_calls += 1
        self.history.append(val)
        if val < self.best:
            self.best, self.best_params = val, dict(params)
        return val


def _sample_random(model: str, rng: np.random.Generator) -> Dict[str, Any]:
    raw = {}
    for p in SPACES[model]:
        if p.kind == "int":
            raw[p.name] = int(rng.integers(int(p.low), int(p.high) + 1))
        else:
            raw[p.name] = float(rng.uniform(p.low, p.high))
    return canonical_params(model, raw)


def run_default(obj: InnerObjective, **_) -> Dict[str, Any]:
    params = dict(DEFAULT_PARAMS[obj.model])
    # objective value of the defaults is still recorded for reference
    obj(params)
    return params


def run_random(obj: InnerObjective, n_evals: int, seed: int, **_) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    for _ in range(n_evals):
        obj(_sample_random(obj.model, rng))
    return obj.best_params


def run_tpe(obj: InnerObjective, n_evals: int, seed: int, **_) -> Dict[str, Any]:
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="minimize", sampler=sampler)

    def f(trial):
        raw = {}
        for p in SPACES[obj.model]:
            if p.kind == "int":
                raw[p.name] = trial.suggest_int(p.name, int(p.low), int(p.high))
            else:
                raw[p.name] = trial.suggest_float(p.name, p.low, p.high)
        return obj(canonical_params(obj.model, raw))

    study.optimize(f, n_trials=n_evals, show_progress_bar=False)
    return obj.best_params


def run_eo(obj: InnerObjective, pop: int, epoch: int, seed: int, **_) -> Dict[str, Any]:
    from mealpy import FloatVar, IntegerVar
    from mealpy.physics_based import EO

    space = SPACES[obj.model]
    bounds = [IntegerVar(lb=int(p.low), ub=int(p.high), name=p.name) if p.kind == "int"
              else FloatVar(lb=p.low, ub=p.high, name=p.name) for p in space]

    def decode(x) -> Dict[str, Any]:
        raw = {}
        for i, (p, b) in enumerate(zip(space, bounds)):
            v = b.decode(np.array([x[i]]))  # IntegerVar.decode rounds (fixes the int() truncation)
            raw[p.name] = np.asarray(v).ravel()[0]
        return canonical_params(obj.model, raw)

    problem = {"obj_func": lambda x: obj(decode(x)), "bounds": bounds, "minmax": "min",
               "log_to": None}
    opt = EO.OriginalEO(epoch=epoch, pop_size=pop)
    opt.solve(problem, seed=seed)
    return obj.best_params


TUNERS: Dict[str, Callable[..., Dict[str, Any]]] = {
    "default": run_default,
    "random": run_random,
    "tpe": run_tpe,
    "eo": run_eo,
}


def tune(tuner: str, obj: InnerObjective, pop: int, epoch: int, n_evals: int, seed: int):
    t0 = time.time()
    params = TUNERS[tuner](obj, pop=pop, epoch=epoch, n_evals=n_evals, seed=seed)
    elapsed = time.time() - t0
    return params, elapsed
