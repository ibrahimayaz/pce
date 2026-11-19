"""One-at-a-time hyper-parameter sensitivity around the EO optimum (Reviewer 1, #9).

For every model and target, each hyper-parameter is varied over an evenly
spaced grid across its full search range while all other hyper-parameters are
held at the EO optimum of the final model.  Performance is the RMSE of a
repeated (2 x 5) K-fold CV with in-fold preprocessing.

Crash-safe: every (target, model, parameter, value) result is cached, so the
script continues where it stopped.

    python 04_hyperparam_sensitivity.py --profile full --n-jobs 30

Outputs: results_<profile>/budget_<hash>/hyperparam_sensitivity/
  T_hp_sensitivity_<Y>.csv     RMSE per grid point
  T_hp_sensitivity_summary.csv RMSE range per hyper-parameter (sensitivity index)
  F_hp_sensitivity_<Y>.png/pdf one panel per model, normalised parameter axis
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402
from sklearn.model_selection import RepeatedKFold  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402

from rev.config import MODELS, PROFILES, TARGETS  # noqa: E402
from rev.data import VARIABLE_INFO, get_target_data  # noqa: E402
from rev.features import MRMRPreprocessor  # noqa: E402
from rev.io_utils import EvalCache  # noqa: E402
from rev.jobs import paths  # noqa: E402
from rev.models import SPACES, build_model, canonical_params, params_key  # noqa: E402

plt.rcParams.update({"font.family": "serif", "font.size": 9})


def flat(model: str, params: dict) -> dict:
    p = dict(params)
    if model == "MLP":
        h = p.pop("hidden_layer_sizes")
        p["hidden_layer_size_1"], p["hidden_layer_size_2"] = h
    return p


def cv_rmse(model: str, params: dict, target: str, k: int, seed: int) -> float:
    from rev.models import set_thread_env

    set_thread_env(1)
    td = get_target_data(target)
    errs = []
    for tr, te in RepeatedKFold(n_splits=5, n_repeats=2, random_state=seed).split(td.X):
        pipe = Pipeline([("prep", MRMRPreprocessor(k=k)), ("model", build_model(model, params, "cpu", 1, seed))])
        pipe.fit(td.X.iloc[tr], td.y.iloc[tr].values)
        e = pipe.predict(td.X.iloc[te]) - td.y.iloc[te].values
        errs.append(float(np.sqrt(np.mean(e ** 2))))
    return float(np.mean(errs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="full", choices=list(PROFILES))
    ap.add_argument("--points", type=int, default=9)
    ap.add_argument("--n-jobs", type=int, default=-1)
    ap.add_argument("--models", default=",".join(MODELS))
    args = ap.parse_args()
    prof = PROFILES[args.profile]
    P = paths(prof)
    out = P["root"] / "hyperparam_sensitivity"
    out.mkdir(exist_ok=True)
    models = [m for m in args.models.split(",") if m]
    summary = []
    for target in TARGETS:
        tasks = []
        for m in models:
            jf = P["jobs"] / f"final__{target}__full__{m}__eo__all.json"
            if not jf.exists():
                print(f"skip {target}/{m}: final job not finished")
                continue
            best = flat(m, json.loads(jf.read_text(encoding="utf-8"))["best_params"])
            for sp in SPACES[m]:
                grid = np.linspace(sp.low, sp.high, args.points)
                if sp.kind == "int":
                    grid = np.unique(np.round(grid).astype(int))
                for v in list(grid) + [best[sp.name]]:
                    raw = dict(best)
                    raw[sp.name] = v
                    params = canonical_params(m, raw)
                    tasks.append((m, sp.name, float(v), params, bool(v == best[sp.name])))
        cache = EvalCache(out / f"cache_{target}.jsonl")
        todo = [t for t in tasks if cache.get(f"{t[0]}|{params_key(t[3])}") is None]
        uniq = {f"{t[0]}|{params_key(t[3])}": t for t in todo}
        print(f"[{target}] {len(tasks)} grid points, {len(uniq)} to compute")
        batch = 4 * max(1, abs(args.n_jobs))
        keys = list(uniq)
        for i in range(0, len(keys), batch):
            chunk = keys[i:i + batch]
            vals = Parallel(n_jobs=args.n_jobs)(delayed(cv_rmse)(uniq[kk][0], uniq[kk][3], target, prof.n_features,
                                                                  prof.seed) for kk in chunk)
            for kk, v in zip(chunk, vals):
                cache.put(kk, v)
        rows = [{"Model": m, "Hyperparameter": hp, "Value": v, "Is_optimum": opt,
                 "CV_RMSE": cache.get(f"{m}|{params_key(p)}")} for m, hp, v, p, opt in tasks]
        cache.close()
        if not rows:
            continue
        df = pd.DataFrame(rows).drop_duplicates(["Model", "Hyperparameter", "Value"])
        df.to_csv(out / f"T_hp_sensitivity_{target}.csv", index=False)
        for (m, hp), g in df.groupby(["Model", "Hyperparameter"]):
            opt = g[g.Is_optimum].CV_RMSE.min()
            summary.append({"Target": target, "Model": m, "Hyperparameter": hp,
                            "RMSE_at_optimum": opt, "RMSE_min": g.CV_RMSE.min(), "RMSE_max": g.CV_RMSE.max(),
                            "RMSE_range": g.CV_RMSE.max() - g.CV_RMSE.min(),
                            "relative_range_%": 100 * (g.CV_RMSE.max() - g.CV_RMSE.min()) / opt})
        ms = [m for m in models if m in set(df.Model)]
        fig, axes = plt.subplots(1, len(ms), figsize=(3.1 * len(ms), 2.9), squeeze=False)
        u = VARIABLE_INFO[target]["unit"]
        for ax, m in zip(axes[0], ms):
            for sp in SPACES[m]:
                g = df[(df.Model == m) & (df.Hyperparameter == sp.name)].sort_values("Value")
                x = (g.Value - sp.low) / (sp.high - sp.low)
                ax.plot(x, g.CV_RMSE, "o-", ms=3, lw=1, label=sp.name)
                o = g[g.Is_optimum]
                ax.plot((o.Value - sp.low) / (sp.high - sp.low), o.CV_RMSE, "k*", ms=7)
            ax.set_title(m)
            ax.set_xlabel("Normalised value within search range")
            ax.set_ylabel(f"CV RMSE ({u})")
            ax.legend(fontsize=6)
            ax.grid(alpha=0.3)
        fig.suptitle(f"{target}: one-at-a-time hyper-parameter sensitivity (star = EO optimum)")
        fig.tight_layout()
        fig.savefig(out / f"F_hp_sensitivity_{target}.png", dpi=600, bbox_inches="tight")
        fig.savefig(out / f"F_hp_sensitivity_{target}.pdf", bbox_inches="tight")
        plt.close(fig)
    if summary:
        s = pd.DataFrame(summary).sort_values(["Target", "Model", "relative_range_%"], ascending=[True, True, False])
        s.to_csv(out / "T_hp_sensitivity_summary.csv", index=False)
        print(s.round(4).to_string(index=False))
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
