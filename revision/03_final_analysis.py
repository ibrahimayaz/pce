"""SHAP analysis, learning curves and GUI export for the final models.

Run after the ``final`` and ``rkf_main`` blocks are finished and
``02_aggregate.py`` has written ``best_models.json``:

    python 03_final_analysis.py --profile full
    python 03_final_analysis.py --profile full --model-Y1 CatBoost --model-Y2 LightGBM   # override

Outputs: results_<profile>/budget_<hash>/final_analysis/
  T_shap_<Y>.csv                 exact mean |SHAP| and rank of every model input       (R1 #30, #38)
  shap_settings.json             explainer, perturbation, background, n samples      (R1 #32)
  F_shap_bar_<Y>, F_shap_beeswarm_<Y>   separate 600-dpi panels                         (R1 #38)
  F_shap_dependence_<Y>_<X>, F_shap_dependence_grid_<Y>                                 (R1 #31, #33)
  T_learning_curve_<Y>.csv, F_learning_curve_<Y>  (random K-fold and grouped by PCE)    (R1 #20)
and the GUI artifacts in <project>/models_v2/.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import joblib  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.base import clone  # noqa: E402
from sklearn.model_selection import GroupKFold, KFold, learning_curve  # noqa: E402

from rev.config import PROFILES, PROJECT_ROOT, TARGETS  # noqa: E402
from rev.data import FEATURES, VARIABLE_INFO, get_target_data, label  # noqa: E402
from rev.io_utils import atomic_joblib_dump, atomic_write_json, read_json  # noqa: E402
from rev.jobs import paths  # noqa: E402

plt.rcParams.update({"font.family": "serif", "font.size": 9})


def save_fig(fig, path: Path):
    fig.savefig(path.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def shap_values_for(model_name: str, est, Z: np.ndarray, seed: int):
    import shap

    if model_name == "XGBoost":
        # shap 0.49 cannot parse the base_score format of XGBoost 3.x; XGBoost's built-in
        # TreeSHAP (pred_contribs) computes the same exact path-dependent Shapley values.
        import xgboost as xgb

        contrib = est.get_booster().predict(xgb.DMatrix(Z), pred_contribs=True)
        sv, base = contrib[:, :-1], float(contrib[0, -1])
        info = {"explainer": "XGBoost built-in TreeSHAP (pred_contribs=True), exact Tree SHAP",
                "feature_perturbation": "tree_path_dependent",
                "background": "none required: conditional expectations from the training-sample "
                              "cover stored in each tree",
                "explained_samples": int(Z.shape[0]), "base_value": base}
        return sv, info
    if model_name in ("LightGBM", "CatBoost"):
        expl = shap.TreeExplainer(est, feature_perturbation="tree_path_dependent")
        sv = expl.shap_values(Z)
        info = {"explainer": "TreeExplainer (exact, polynomial-time Tree SHAP)",
                "feature_perturbation": "tree_path_dependent",
                "background": "none required: conditional expectations from the training-sample "
                              "cover stored in each tree",
                "explained_samples": int(Z.shape[0]), "base_value": float(np.ravel(expl.expected_value)[0])}
    else:
        bg = shap.kmeans(Z, 50)
        expl = shap.KernelExplainer(est.predict, bg)
        sv = expl.shap_values(Z, nsamples=500, silent=True)
        info = {"explainer": "KernelExplainer", "background": "k-means summary of the training data (50 centroids)",
                "nsamples_per_explanation": 500, "explained_samples": int(Z.shape[0]),
                "base_value": float(np.ravel(expl.expected_value)[0])}
    return np.asarray(sv), info


def run_shap(target: str, model_name: str, pipe, td, out: Path, seed: int) -> dict:
    import shap

    prep, est = pipe.named_steps["prep"], pipe.named_steps["model"]
    feats = prep.selected_
    Z = prep.transform(td.X)
    Xd = td.X[feats].reset_index(drop=True)  # original units for display
    t0 = time.time()
    sv, info = shap_values_for(model_name, est, Z, seed)
    info["seconds"] = round(time.time() - t0, 1)
    pred = est.predict(Z)
    info["additivity_max_abs_error"] = float(np.max(np.abs(sv.sum(1) + info["base_value"] - pred)))
    info.update(model=model_name, target=target, features=feats,
                note="SHAP values are computed on the standardised model inputs; plots show the "
                     "corresponding values in original units (standardisation is monotonic).")

    mabs = np.abs(sv).mean(0)
    tab = pd.DataFrame({"feature": feats, "name": [VARIABLE_INFO[f]["name"] for f in feats],
                        "mean_abs_SHAP": mabs, "share_%": 100 * mabs / mabs.sum(),
                        "mean_SHAP": sv.mean(0), "sd_SHAP": sv.std(0),
                        "min_SHAP": sv.min(0), "max_SHAP": sv.max(0)})
    tab = tab.sort_values("mean_abs_SHAP", ascending=False).reset_index(drop=True)
    tab.insert(0, "rank", np.arange(1, len(tab) + 1))
    tab.to_csv(out / f"T_shap_{target}.csv", index=False)

    # Grouped SHAP: X19 and X20 are perfectly collinear (X19 + X20 = 700 g), so Tree SHAP may
    # attribute their common effect to either of them; summing SHAP values within a group gives
    # an attribution that does not depend on this arbitrary split.
    groups = {"Fly-ash replacement level (X19 + X20)": ["X19", "X20"],
              "PCE dosage (X21)": ["X21"],
              "PCE descriptors (X13-X18)": ["X13", "X14", "X15", "X16", "X17", "X18"]}
    grows = []
    for gname, members in groups.items():
        idx = [feats.index(f) for f in members if f in feats]
        if not idx:
            continue
        gsv = sv[:, idx].sum(1)
        grows.append({"group": gname, "members": ",".join(feats[i] for i in idx),
                      "mean_abs_SHAP": float(np.abs(gsv).mean())})
    gt = pd.DataFrame(grows).sort_values("mean_abs_SHAP", ascending=False)
    gt["share_%"] = 100 * gt.mean_abs_SHAP / gt.mean_abs_SHAP.sum()
    gt.insert(0, "rank", np.arange(1, len(gt) + 1))
    gt.to_csv(out / f"T_shap_grouped_{target}.csv", index=False)
    print(gt.round(4).to_string(index=False))

    names = [label(f, with_unit=False) for f in feats]
    u = VARIABLE_INFO[target]["unit"]
    shap.summary_plot(sv, Xd, feature_names=names, plot_type="bar", show=False, color="#4c72b0")
    plt.xlabel(f"mean |SHAP value| ({u})")
    plt.title(f"{target}: global importance ({model_name})")
    save_fig(plt.gcf(), out / f"F_shap_bar_{target}")
    shap.summary_plot(sv, Xd, feature_names=names, show=False)
    plt.xlabel(f"SHAP value ({u})")
    plt.title(f"{target}: SHAP beeswarm ({model_name})")
    save_fig(plt.gcf(), out / f"F_shap_beeswarm_{target}")

    expl = shap.Explanation(values=sv, base_values=np.full(len(sv), info["base_value"]),
                            data=Xd.values, feature_names=names)
    n = len(feats)
    cols = 3
    rows = int(np.ceil(n / cols))
    grid, gax = plt.subplots(rows, cols, figsize=(11, 3.1 * rows))
    gax = np.ravel(gax)
    for i, f in enumerate(feats):
        fig, ax = plt.subplots(figsize=(4.2, 3.2))
        shap.plots.scatter(expl[:, i], color=expl, ax=ax, show=False)
        ax.set_ylabel(f"SHAP value for {f} ({u})")
        save_fig(fig, out / f"F_shap_dependence_{target}_{f}")
        shap.plots.scatter(expl[:, i], color=expl, ax=gax[i], show=False)
        gax[i].set_ylabel(f"SHAP {f} ({u})")
    for ax in gax[n:]:
        ax.axis("off")
    grid.suptitle(f"{target}: SHAP dependence plots ({model_name}); colour = strongest interacting feature")
    grid.tight_layout()
    save_fig(grid, out / f"F_shap_dependence_grid_{target}")
    # ---- SHAP interaction values (R1 #33: does X13 absorb PCE-dosage effects?) ----
    inter = shap_interactions(model_name, est, Z)
    if inter is not None:
        mi = np.abs(inter).mean(0)
        pairs = []
        for i in range(len(feats)):
            for j in range(i + 1, len(feats)):
                pairs.append({"feature_i": feats[i], "feature_j": feats[j],
                              "mean_abs_interaction": float(2 * mi[i, j])})
        main_eff = pd.DataFrame({"feature": feats, "mean_abs_main_effect": np.diag(mi)})
        it = pd.DataFrame(pairs).sort_values("mean_abs_interaction", ascending=False)
        it.to_csv(out / f"T_shap_interactions_{target}.csv", index=False)
        main_eff.to_csv(out / f"T_shap_main_effects_{target}.csv", index=False)
        fig, ax = plt.subplots(figsize=(5.2, 4.4))
        M = mi.copy()
        im = ax.imshow(M, cmap="viridis")
        ax.set_xticks(range(len(feats)), feats, rotation=90)
        ax.set_yticks(range(len(feats)), feats)
        fig.colorbar(im, ax=ax, label=f"mean |SHAP interaction| ({u})")
        ax.set_title(f"{target}: SHAP interaction matrix ({model_name})")
        save_fig(fig, out / f"F_shap_interactions_{target}")
        info["interactions"] = "computed (diagonal = main effects)"
    else:
        info["interactions"] = "not available for this model type"
    # explicit X13 vs PCE dosage view
    if "X13" in feats and "X21" in feats:
        fig, ax = plt.subplots(figsize=(4.4, 3.3))
        shap.plots.scatter(expl[:, feats.index("X13")], color=expl[:, feats.index("X21")], ax=ax, show=False)
        ax.set_ylabel(f"SHAP value for X13 ({u})")
        ax.set_title(f"{target}: X13 effect coloured by PCE dosage X21")
        save_fig(fig, out / f"F_shap_X13_vs_X21_{target}")
    info["corr_X13_X21_in_data"] = float(np.corrcoef(td.X["X13"], td.X["X21"])[0, 1])
    atomic_write_json(out / f"shap_settings_{target}.json", info)
    return {"table": tab, "info": info}


def shap_interactions(model_name: str, est, Z: np.ndarray):
    """Exact Tree SHAP interaction values (n x p x p) or None."""
    try:
        if model_name == "XGBoost":
            import xgboost as xgb

            v = est.get_booster().predict(xgb.DMatrix(Z), pred_interactions=True)
            return v[:, :-1, :-1]
        if model_name == "CatBoost":
            from catboost import Pool

            v = est.get_feature_importance(Pool(Z), type="ShapInteractionValues")
            return np.asarray(v)[:, :-1, :-1]
        if model_name == "LightGBM":
            import shap

            return np.asarray(shap.TreeExplainer(est).shap_interaction_values(Z))
    except Exception as e:  # pragma: no cover
        print(f"SHAP interactions not available: {e}")
    return None


def run_learning_curve(target: str, pipe, td, out: Path, n_jobs: int, seed: int) -> pd.DataFrame:
    sizes = np.linspace(0.1, 1.0, 10)
    rows = []
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), sharey=True)
    schemes = [("Random 5-fold CV", KFold(5, shuffle=True, random_state=seed), None),
               ("Grouped by admixture (GroupKFold)", GroupKFold(n_splits=min(5, td.pce_group.nunique())),
                td.pce_group.values)]
    u = VARIABLE_INFO[target]["unit"]
    for ax, (name, cv, groups) in zip(axes, schemes):
        ns, tr, te = learning_curve(clone(pipe), td.X, td.y.values, cv=cv, groups=groups, train_sizes=sizes,
                                    scoring="neg_root_mean_squared_error", n_jobs=n_jobs, shuffle=True,
                                    random_state=seed)
        tr, te = -tr, -te
        for i, n in enumerate(ns):
            rows.append({"scheme": name, "n_train": int(n), "train_RMSE_mean": tr[i].mean(), "train_RMSE_sd": tr[i].std(),
                         "val_RMSE_mean": te[i].mean(), "val_RMSE_sd": te[i].std()})
        ax.plot(ns, tr.mean(1), "o-", label="Training")
        ax.fill_between(ns, tr.mean(1) - tr.std(1), tr.mean(1) + tr.std(1), alpha=0.15)
        ax.plot(ns, te.mean(1), "s--", label="Validation")
        ax.fill_between(ns, te.mean(1) - te.std(1), te.mean(1) + te.std(1), alpha=0.15)
        ax.set_title(name)
        ax.set_xlabel("Training set size (samples)")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel(f"RMSE ({u})")
    axes[0].legend()
    fig.suptitle(f"{target}: learning curves")
    fig.tight_layout()
    save_fig(fig, out / f"F_learning_curve_{target}")
    lc = pd.DataFrame(rows)
    lc.to_csv(out / f"T_learning_curve_{target}.csv", index=False)
    return lc


def export_gui(best: dict, pipes: dict, prof, P, A_dir: Path) -> Path:
    import sklearn

    dest = PROJECT_ROOT / "models_v2"
    dest.mkdir(exist_ok=True)
    meta = {"created": time.strftime("%Y-%m-%d %H:%M:%S"), "profile": prof.name, "budget": prof.budget_hash(),
            "sklearn_version": sklearn.__version__, "targets": {}, "inputs": {}}
    nested = {}
    for t in TARGETS:
        f = A_dir / f"T_nested_rkf_{t}.csv"
        if f.exists():
            d = pd.read_csv(f)
            r = d[d.Model == best[t]]
            if not r.empty:
                nested[t] = {"R2": r.iloc[0]["R2"], "RMSE": r.iloc[0]["RMSE"]}
    data_all = get_target_data("Y1")
    for t, pipe in pipes.items():
        atomic_joblib_dump(pipe, dest / f"pipeline_{t}.joblib")
        meta["targets"][t] = {"model": best[t], "name": VARIABLE_INFO[t]["name"], "unit": VARIABLE_INFO[t]["unit"],
                              "selected_features": pipe.named_steps["prep"].selected_,
                              "nested_cv": nested.get(t, {})}
    for f in FEATURES:
        s = data_all.X[f]
        meta["inputs"][f] = {"name": VARIABLE_INFO[f]["name"], "unit": VARIABLE_INFO[f]["unit"],
                             "min": float(s.min()), "max": float(s.max()),
                             "constant": bool(s.nunique() == 1), "default": float(s.iloc[0]),
                             "levels": sorted(map(float, s.unique())) if s.nunique() <= 25 else None}
    atomic_write_json(dest / "metadata.json", meta)
    return dest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="full", choices=list(PROFILES))
    ap.add_argument("--model-Y1", default="")
    ap.add_argument("--model-Y2", default="")
    ap.add_argument("--n-jobs", type=int, default=-1)
    ap.add_argument("--skip-gui", action="store_true")
    args = ap.parse_args()
    prof = PROFILES[args.profile]
    P = paths(prof)
    A_dir = P["root"] / "analysis"
    out = P["root"] / "final_analysis"
    out.mkdir(exist_ok=True)
    best = read_json(A_dir / "best_models.json") or {}
    if args.model_Y1:
        best["Y1"] = args.model_Y1
    if args.model_Y2:
        best["Y2"] = args.model_Y2
    missing = [t for t in TARGETS if t not in best]
    if missing:
        sys.exit(f"No best model for {missing}: run 02_aggregate.py first or pass --model-{missing[0]}")
    pipes = {}
    for t in TARGETS:
        mp = P["final_models"] / f"{t}_full_{best[t]}.joblib"
        if not mp.exists():
            sys.exit(f"Final model missing: {mp} (run the 'final' block)")
        pipe = joblib.load(mp)
        pipes[t] = pipe
        td = get_target_data(t)
        print(f"[{t}] SHAP for {best[t]} ...", flush=True)
        res = run_shap(t, best[t], pipe, td, out, prof.seed)
        print(res["table"].round(4).to_string(index=False))
        print(f"[{t}] learning curves ...", flush=True)
        run_learning_curve(t, pipe, td, out, args.n_jobs, prof.seed)
    if prof.name != "full":
        (out / "NOT_FOR_MANUSCRIPT.txt").write_text("Smoke-test output (tiny budget). Do not use in the manuscript.\n",
                                                    encoding="utf-8")
        if not args.skip_gui:
            print("Smoke profile: GUI export skipped (only the full profile may update models_v2/).")
            args.skip_gui = True
    if not args.skip_gui:
        dest = export_gui(best, pipes, prof, P, A_dir)
        print(f"GUI artifacts written to {dest}")
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
