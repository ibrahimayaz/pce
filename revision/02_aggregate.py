"""Collect finished jobs into manuscript tables, statistical tests and figures.

Can be run at any time (also while experiments are still running); tables are
built from whatever jobs are finished and the number of folds used is shown.

    python 02_aggregate.py --profile full

Outputs: results_<profile>/budget_<hash>/analysis/
  T4_hyperparameters.csv          complete Table 4: ranges + optimum for every model/target (R1 #21, R2 #5)
  T_nested_rkf_<Y>.csv            new Tables 5/6: nested repeated K-fold, EO-tuned          (R1 #14, #15, R2 #4)
  T_model_tests_<Y>.csv           best model vs others, corrected t-test + Wilcoxon (Holm)  (R2 #8)
  T_tuner_comparison_<Y>.csv      default / random / TPE / EO, same budget, paired tests    (R1 #24, R2 #8)
  T_lopo_<Y>.csv, T_lopo_pergroup_<Y>.csv   leave-one-PCE-out                               (R1 #17, #19, R2 #4)
  T_lofa_<Y>.csv                  leave-one-fly-ash-level-out
  T_nodup_Y1.csv                  near-duplicate admixtures removed                         (R1 #18)
  T_eo_sensitivity_<Y>.csv        EO population x iterations x seeds                        (R1 #9, #22)
  T_eo_settings.csv               all EO settings                                           (R1 #22, #23)
  F_*.png / .pdf                  convergence, actual-vs-predicted, LOPO per group, sensitivity
  best_models.json                best model per target (lowest nested-CV RMSE)
  summary.md                      everything in one readable file
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

from rev.config import MODELS, PROFILES, TARGETS  # noqa: E402
from rev.data import VARIABLE_INFO, get_target_data  # noqa: E402
from rev.io_utils import atomic_write_json, atomic_write_text  # noqa: E402
from rev.jobs import metrics, paths  # noqa: E402
from rev.models import SPACES  # noqa: E402
from rev.stats import corrected_resampled_ttest, holm, wilcoxon  # noqa: E402

METRICS = ["R2", "RMSE", "MAE", "MSE", "MAPE"]
TUNER_LABEL = {"default": "Default", "random": "Random search", "tpe": "Bayesian (TPE)", "eo": "EO"}
COLORS = {"LightGBM": "#1f77b4", "CatBoost": "#d62728", "XGBoost": "#2ca02c", "AdaBoost": "#9467bd",
          "MLP": "#ff7f0e", "eo": "#d62728", "random": "#7f7f7f", "tpe": "#1f77b4", "default": "#000000"}
plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.grid": True, "grid.alpha": 0.3})


def load_jobs(P) -> pd.DataFrame:
    rows = []
    for f in P["jobs"].glob("*.json"):
        try:
            r = json.loads(f.read_text(encoding="utf-8"))
        except ValueError:
            continue
        for k, v in (r.get("metrics") or {}).items():
            r[k] = v
        rows.append(r)
    return pd.DataFrame(rows)


def fmt(m, s, d=4):
    return f"{m:.{d}f} ± {s:.{d}f}"


def unit(target):
    return VARIABLE_INFO[target]["unit"]


def save_fig(fig, path: Path):
    fig.savefig(path.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def pooled(df: pd.DataFrame) -> dict:
    y = np.concatenate(df["y_true"].values)
    p = np.concatenate(df["y_pred"].values)
    return metrics(y, p)


def table_cv(df: pd.DataFrame, target: str) -> pd.DataFrame:
    rows = []
    for m in MODELS:
        d = df[df.model == m]
        if d.empty:
            continue
        row = {"Model": m, "n_outer_folds": len(d)}
        for k in METRICS:
            row[k] = fmt(d[k].mean(), d[k].std(ddof=0))
            row[k + "_mean"] = d[k].mean()
        rows.append(row)
    return pd.DataFrame(rows)


def ratio(d: pd.DataFrame) -> float:
    return float((d["n_test"] / d["n_train"]).mean())


def model_tests(df: pd.DataFrame, best: str) -> pd.DataFrame:
    piv = df.pivot_table(index="fold", columns="model", values="RMSE")
    r = ratio(df)
    rows = []
    for m in piv.columns:
        if m == best:
            continue
        ok = piv[[best, m]].dropna()
        t = corrected_resampled_ttest(ok[best], ok[m], r)
        rows.append({"Best": best, "Compared": m, "n_folds": len(ok), "mean_RMSE_diff (best-other)": t["mean_diff"],
                     "t_corrected": t["t"], "p_corrected": t["p"], "p_wilcoxon": wilcoxon(ok[best], ok[m])})
    out = pd.DataFrame(rows)
    if not out.empty:
        out["p_corrected_holm"] = holm(out["p_corrected"].tolist())
        out["p_wilcoxon_holm"] = holm(out["p_wilcoxon"].tolist())
    return out


def tuner_table(df: pd.DataFrame, target: str) -> pd.DataFrame:
    rows = []
    r = ratio(df) if not df.empty else 0.25
    for m in MODELS:
        dm = df[df.model == m]
        if dm.empty:
            continue
        piv = dm.pivot_table(index="fold", columns="tuner", values="RMSE")
        for tu in ["default", "random", "tpe", "eo"]:
            d = dm[dm.tuner == tu]
            if d.empty:
                continue
            row = {"Model": m, "Tuner": TUNER_LABEL[tu], "n_folds": len(d),
                   "R2": fmt(d.R2.mean(), d.R2.std(ddof=0)), "RMSE": fmt(d.RMSE.mean(), d.RMSE.std(ddof=0)),
                   "MAE": fmt(d.MAE.mean(), d.MAE.std(ddof=0)),
                   "inner_CV_RMSE": fmt(np.sqrt(d.inner_best_mse).mean(), np.sqrt(d.inner_best_mse).std(ddof=0)),
                   "evaluations": int(d.n_evals.median()), "tuning_time_s": round(d.tune_seconds.mean(), 1)}
            if tu != "eo" and "eo" in piv:
                ok = piv[["eo", tu]].dropna()
                t = corrected_resampled_ttest(ok["eo"], ok[tu], r)
                row.update({"EO_minus_this_RMSE": t["mean_diff"], "p_corrected (EO vs this)": t["p"],
                            "p_wilcoxon (EO vs this)": wilcoxon(ok["eo"], ok[tu])})
            rows.append(row)
    return pd.DataFrame(rows)


def grouped_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for m in MODELS:
        for tu in ["eo", "default"]:
            d = df[(df.model == m) & (df.tuner == tu)]
            if d.empty:
                continue
            pm = pooled(d)
            rows.append({"Model": m, "Tuner": TUNER_LABEL[tu], "held_out_groups": len(d),
                         "pooled_R2": round(pm["R2"], 4), "pooled_RMSE": round(pm["RMSE"], 4),
                         "pooled_MAE": round(pm["MAE"], 4), "pooled_MAPE": round(pm["MAPE"], 4),
                         "per_group_RMSE": fmt(d.RMSE.mean(), d.RMSE.std(ddof=0)),
                         "worst_group": d.loc[d.RMSE.idxmax(), "fold"], "worst_group_RMSE": round(d.RMSE.max(), 3)})
    return pd.DataFrame(rows)


def fig_actual_pred(df: pd.DataFrame, target: str, title: str, path: Path):
    models = [m for m in MODELS if m in set(df.model)]
    if not models:
        return
    fig, axes = plt.subplots(2, 3, figsize=(10, 6.8))
    axes = axes.ravel()
    for ax, m in zip(axes, models):
        d = df[df.model == m]
        y = np.concatenate(d.y_true.values)
        p = np.concatenate(d.y_pred.values)
        mm = metrics(y, p)
        ax.scatter(y, p, s=9, alpha=0.6, color=COLORS[m], edgecolors="none")
        lo, hi = min(y.min(), p.min()), max(y.max(), p.max())
        ax.plot([lo, hi], [lo, hi], "k--", lw=1)
        ax.set_title(m)
        ax.set_xlabel(f"Measured {target} ({unit(target)})")
        ax.set_ylabel(f"Predicted {target} ({unit(target)})")
        ax.text(0.04, 0.96, f"$R^2$ = {mm['R2']:.3f}\nRMSE = {mm['RMSE']:.3f}", transform=ax.transAxes,
                va="top", fontsize=8, bbox=dict(fc="white", ec="0.7"))
    for ax in axes[len(models):]:
        ax.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    save_fig(fig, path)


def fig_convergence(df: pd.DataFrame, target: str, path: Path):
    models = [m for m in MODELS if m in set(df.model)]
    if not models:
        return
    fig, axes = plt.subplots(1, len(models), figsize=(3.0 * len(models), 2.8), squeeze=False)
    for ax, m in zip(axes[0], models):
        for tu in ["random", "tpe", "eo"]:
            d = df[(df.model == m) & (df.tuner == tu)]
            if d.empty:
                continue
            L = min(len(h) for h in d.best_so_far)
            H = np.sqrt(np.vstack([np.asarray(h[:L]) for h in d.best_so_far]))
            mu, sd = H.mean(0), H.std(0)
            x = np.arange(1, L + 1)
            ax.plot(x, mu, color=COLORS[tu], label=TUNER_LABEL[tu], lw=1.2)
            ax.fill_between(x, mu - sd, mu + sd, color=COLORS[tu], alpha=0.15)
        ax.set_xscale("log")
        ax.set_title(m)
        ax.set_xlabel("Objective evaluations")
        ax.set_ylabel(f"Best inner-CV RMSE ({unit(target)})")
    axes[0][0].legend(fontsize=7)
    fig.suptitle(f"{target}: tuner convergence (mean ± SD over outer folds)")
    fig.tight_layout()
    save_fig(fig, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="full", choices=list(PROFILES))
    args = ap.parse_args()
    prof = PROFILES[args.profile]
    P = paths(prof)
    A = P["root"] / "analysis"
    A.mkdir(exist_ok=True)
    df = load_jobs(P)
    if df.empty:
        sys.exit("No finished jobs yet.")
    md = [f"# Revision results — profile `{prof.name}`, budget `{prof.budget_hash()}`", ""]
    if prof.name != "full":
        md += ["**SMOKE TEST — TINY BUDGET — NOT FOR THE MANUSCRIPT.**", ""]
        atomic_write_text(A / "NOT_FOR_MANUSCRIPT.txt", "Smoke-test output (tiny budget). Do not use in the manuscript.\n")
    md += [
          f"Finished jobs: {len(df)}", ""]
    best_models = {}

    # ---------- EO settings (R1 #22, #23) ----------
    eo_set = pd.DataFrame([
        ("Algorithm", "Equilibrium Optimizer, mealpy OriginalEO (Faramarzi et al., 2020)"),
        ("Population size (search agents)", prof.eo_pop), ("Iterations", prof.eo_epoch),
        ("Objective evaluations per run", f"{prof.eo_pop} + {prof.eo_epoch} x ({prof.eo_pop} + 1) = "
                                          f"{prof.eo_pop + prof.eo_epoch * (prof.eo_pop + 1)}"),
        ("a1 (exploration constant)", 2), ("a2 (exploitation constant)", 1), ("GP (generation probability)", 0.5),
        ("V (control volume)", 1), ("Equilibrium pool", "4 best agents + their average"),
        ("Stopping criterion", "fixed number of iterations"),
        ("Integer decoding", "rounding to nearest integer within bounds"),
        ("Objective", f"mean MSE over {prof.inner_folds}-fold inner CV of the outer-training data "
                      "(preprocessing + mRMR refitted in each inner fold)"),
        ("Random seeds", "one seed per outer fold (derived from base seed 42); sensitivity: 5 independent seeds"),
        ("Random search / TPE budget", "same number of objective evaluations as EO"),
    ], columns=["Setting", "Value"])
    eo_set.to_csv(A / "T_eo_settings.csv", index=False)
    md += ["## EO settings", eo_set.to_markdown(index=False), ""]

    # ---------- Table 4: complete hyper-parameters ----------
    fin = df[df.block == "final"]
    if not fin.empty:
        rows = []
        for m in MODELS:
            for p in SPACES[m]:
                row = {"Model": m, "Hyperparameter": p.name,
                       "Range": f"[{int(p.low)}, {int(p.high)}]" if p.kind == "int" else f"[{p.low}, {p.high}]"}
                for t in TARGETS:
                    r = fin[(fin.model == m) & (fin.target == t)]
                    if r.empty:
                        row[f"Optimum {t}"] = ""
                        continue
                    bp = r.iloc[0].best_params
                    if m == "MLP" and p.name.startswith("hidden_layer_size_"):
                        v = bp["hidden_layer_sizes"][int(p.name[-1]) - 1]
                    else:
                        v = bp[p.name]
                    row[f"Optimum {t}"] = v if isinstance(v, int) else round(v, 4)
                    row[f"Tuning time {t} (s)"] = round(r.iloc[0].tune_seconds, 1)
                rows.append(row)
        t4 = pd.DataFrame(rows)
        t4.to_csv(A / "T4_hyperparameters.csv", index=False)
        md += ["## Table 4 (complete)", t4.to_markdown(index=False), ""]

    for target in TARGETS:
        u = unit(target)
        md += [f"# Target {target} ({VARIABLE_INFO[target]['name']}, {u})", ""]
        # ---------- nested repeated K-fold ----------
        main_eo = df[(df.block == "rkf_main") & (df.target == target) & (df.tuner == "eo")]
        if not main_eo.empty:
            t = table_cv(main_eo, target)
            best = t.loc[t["RMSE_mean"].idxmin(), "Model"]
            best_models[target] = best
            t.drop(columns=[c for c in t.columns if c.endswith("_mean")]).to_csv(A / f"T_nested_rkf_{target}.csv", index=False)
            md += [f"## Nested {prof.rkf_repeats}x{prof.rkf_folds}-fold CV, EO-tuned (mean ± SD over outer folds)",
                   t.drop(columns=[c for c in t.columns if c.endswith("_mean")]).to_markdown(index=False),
                   f"Best model (lowest mean RMSE): **{best}**", ""]
            mt = model_tests(main_eo, best)
            mt.to_csv(A / f"T_model_tests_{target}.csv", index=False)
            md += ["### Paired tests: best vs other models (RMSE)", mt.round(4).to_markdown(index=False), ""]
            r0 = main_eo[main_eo.fold.str.startswith("r0")]
            fig_actual_pred(r0, target, f"{target}: nested CV out-of-fold predictions (repeat 1, EO-tuned)",
                            A / f"F_actual_vs_pred_rkf_{target}")
        # ---------- tuner comparison ----------
        tun = df[(df.scheme == "rkf") & (df.data == "full") & (df.target == target)]
        if not tun.empty:
            tt = tuner_table(tun, target)
            tt.to_csv(A / f"T_tuner_comparison_{target}.csv", index=False)
            md += ["## Tuner comparison (same evaluation budget, nested CV)", tt.round(4).to_markdown(index=False), ""]
            fig_convergence(tun, target, A / f"F_tuner_convergence_{target}")
        # ---------- LOPO / LOFA ----------
        for scheme, label in [("lopo", "Leave-one-PCE-group-out"), ("lofa", "Leave-one-fly-ash-level-out")]:
            d = df[(df.scheme == scheme) & (df.data == "full") & (df.target == target)]
            if d.empty:
                continue
            gt = grouped_table(d)
            gt.to_csv(A / f"T_{scheme}_{target}.csv", index=False)
            md += [f"## {label}", gt.to_markdown(index=False), ""]
            de = d[d.tuner == "eo"]
            if not de.empty:
                fig_actual_pred(de, target, f"{target}: {label} predictions (EO-tuned)",
                                A / f"F_actual_vs_pred_{scheme}_{target}")
            if scheme == "lopo" and not de.empty:
                pg = de.pivot_table(index="fold", columns="model", values="RMSE")
                pg.index = [i.replace("g_", "") for i in pg.index]
                pg.to_csv(A / f"T_lopo_pergroup_{target}.csv")
                fig, ax = plt.subplots(figsize=(max(6, 0.45 * len(pg)), 3.2))
                pg.plot.bar(ax=ax, width=0.8, color=[COLORS[c] for c in pg.columns])
                ax.set_ylabel(f"RMSE ({u})")
                ax.set_xlabel("Held-out admixture group")
                ax.set_title(f"{target}: error on each unseen admixture (EO-tuned)")
                fig.tight_layout()
                save_fig(fig, A / f"F_lopo_pergroup_{target}")
        # ---------- no-duplicate sensitivity ----------
        nd = df[(df.data == "nodup") & (df.target == target)]
        if not nd.empty:
            rows = []
            for (bl, tu, m), d in nd.groupby(["block", "tuner", "model"]):
                full_bl = "rkf_main" if bl == "nodup_rkf" else "lopo"
                f = df[(df.block == full_bl) & (df.tuner == tu) & (df.model == m) & (df.target == target)]
                rows.append({"Scheme": d.scheme.iloc[0], "Tuner": TUNER_LABEL[tu], "Model": m,
                             "RMSE (duplicates removed)": fmt(d.RMSE.mean(), d.RMSE.std(ddof=0)),
                             "R2 (duplicates removed)": fmt(d.R2.mean(), d.R2.std(ddof=0)),
                             "RMSE (full data)": fmt(f.RMSE.mean(), f.RMSE.std(ddof=0)) if not f.empty else ""})
            ndt = pd.DataFrame(rows)
            ndt.to_csv(A / f"T_nodup_{target}.csv", index=False)
            md += ["## Sensitivity: near-duplicate admixtures removed", ndt.to_markdown(index=False), ""]
        # ---------- baselines: descriptors vs identity vs no admixture information ----------
        rows = []
        for scheme in ["rkf", "lopo"]:
            for variant, vlabel in [("full", "Full model (PCE descriptors X13-X18 + X19-X21)"),
                                    ("pceid", "PCE identity (one-hot code) + X19-X21"),
                                    ("mixonly", "No admixture information (X19-X21 only)")]:
                for tu in ["eo", "default"]:
                    d = df[(df.scheme == scheme) & (df.data == variant) & (df.tuner == tu) & (df.target == target)
                           & df.model.isin(["LightGBM", "CatBoost", "XGBoost"])]
                    for m, dm in d.groupby("model"):
                        pm = pooled(dm)
                        rows.append({"Validation": "random K-fold (nested)" if scheme == "rkf" else "leave-one-PCE-out",
                                     "Input set": vlabel, "Tuner": TUNER_LABEL[tu], "Model": m, "n_folds": len(dm),
                                     "RMSE": fmt(dm.RMSE.mean(), dm.RMSE.std(ddof=0)),
                                     "R2": fmt(dm.R2.mean(), dm.R2.std(ddof=0)),
                                     "pooled_R2": round(pm["R2"], 4), "pooled_RMSE": round(pm["RMSE"], 4)})
        if rows and any(r["Input set"] != rows[0]["Input set"] for r in rows):
            bt = pd.DataFrame(rows)
            bt.to_csv(A / f"T_baselines_{target}.csv", index=False)
            md += ["## Input-set baselines (are PCE descriptors more than identifiers?)",
                   bt.to_markdown(index=False), ""]
        # ---------- original global-MICE procedure vs corrected data ----------
        lg = df[(df.data == "mice_global") & (df.target == target)]
        if not lg.empty:
            rows = []
            for (tu, m), d in lg.groupby(["tuner", "model"]):
                y = np.concatenate(d.y_true.values)
                p = np.concatenate(d.y_pred.values)
                meas = np.concatenate([np.asarray(v, bool) for v in d.measured.values])
                row = {"Model": m, "Tuner": TUNER_LABEL[tu],
                       "Original procedure: scored on all rows (incl. imputed targets)": "R2 {R2:.4f} / RMSE {RMSE:.4f}".format(**metrics(y, p)),
                       "Original procedure: scored on measured rows only": "R2 {R2:.4f} / RMSE {RMSE:.4f}".format(**metrics(y[meas], p[meas]))}
                if (~meas).sum() > 1:
                    row["Original procedure: imputed rows only"] = "R2 {R2:.4f} / RMSE {RMSE:.4f}".format(**metrics(y[~meas], p[~meas]))
                    row["imputed target range"] = f"[{y[~meas].min():.2f}, {y[~meas].max():.2f}]"
                f = df[(df.block == "rkf_main") & (df.model == m) & (df.tuner == tu) & (df.target == target)]
                if not f.empty:
                    row["Corrected procedure (measured only, nested)"] = "R2 {R2:.4f} / RMSE {RMSE:.4f}".format(**pooled(f))
                row["n_imputed_target_rows"] = int((~meas).sum())
                rows.append(row)
            lt = pd.DataFrame(rows)
            lt.to_csv(A / f"T_legacy_mice_{target}.csv", index=False)
            md += ["## Original global-MICE procedure vs corrected procedure (pooled out-of-fold)",
                   lt.to_markdown(index=False), ""]
        # ---------- residual analysis of the best model ----------
        if target in best_models:
            d = main_eo[main_eo.model == best_models[target]]
            td = get_target_data(target, "full")
            ti = np.concatenate([np.asarray(v, int) for v in d.test_index.values])
            res = pd.DataFrame({
                "y": np.concatenate(d.y_true.values), "pred": np.concatenate(d.y_pred.values),
                "X21": td.X["X21"].values[ti], "fa_level": td.fa_level.values[ti], "pce": td.pce.values[ti]})
            res["residual"] = res.pred - res.y
            out_rows = []
            for by, lab in [("X21", "PCE dosage (g)"), ("fa_level", "Fly-ash replacement (%)"), ("pce", "Admixture")]:
                for key, g in res.groupby(by):
                    out_rows.append({"factor": lab, "level": key, "n": len(g), "mean_measured": g.y.mean(),
                                     "bias (pred-meas)": g.residual.mean(), "MAE": g.residual.abs().mean(),
                                     "RMSE": float(np.sqrt((g.residual ** 2).mean())),
                                     "MAPE_%": 100 * (g.residual.abs() / g.y.abs()).mean()})
            rt = pd.DataFrame(out_rows)
            rt.to_csv(A / f"T_residuals_{target}.csv", index=False)
            md += [f"## Residual analysis of {best_models[target]} (nested CV, all repeats)",
                   rt.round(3).to_markdown(index=False), ""]
            fig, axes = plt.subplots(1, 3, figsize=(12, 3.4), gridspec_kw={"width_ratios": [7, 4, 9]})
            for ax, (by, lab) in zip(axes, [("X21", "PCE dosage X21 (g)"), ("fa_level", "Fly ash (%)"),
                                            ("pce", "Admixture")]):
                keys = sorted(res[by].unique(), key=lambda v: (str(type(v)), v))
                ax.boxplot([res.loc[res[by] == k, "residual"] for k in keys], showfliers=True,
                           flierprops=dict(markersize=2))
                ax.set_xticks(range(1, len(keys) + 1), [f"{k:g}" if isinstance(k, (int, float, np.number)) else k
                                                        for k in keys], rotation=90 if by == "pce" else 0, fontsize=7)
                ax.axhline(0, color="k", lw=0.8)
                ax.set_xlabel(lab)
            axes[0].set_ylabel(f"Residual, predicted - measured ({u})")
            fig.suptitle(f"{target}: residuals of {best_models[target]} (nested CV out-of-fold)")
            fig.tight_layout()
            save_fig(fig, A / f"F_residuals_{target}")
        # ---------- EO sensitivity ----------
        se = df[(df.block == "sens") & (df.target == target)]
        if not se.empty:
            se = se.assign(inner_rmse=np.sqrt(se.inner_best_mse))
            g = se.groupby(["model", "pop", "epoch"]).agg(
                seeds=("seed", "nunique"), inner_RMSE_mean=("inner_rmse", "mean"), inner_RMSE_sd=("inner_rmse", "std"),
                check_RMSE_mean=("check_cv_rmse_mean", "mean"), check_RMSE_sd=("check_cv_rmse_mean", "std"),
                evaluations=("n_evals", "median"), time_s=("tune_seconds", "mean")).reset_index()
            g.to_csv(A / f"T_eo_sensitivity_{target}.csv", index=False)
            md += ["## EO parameter sensitivity (independent seeds)", g.round(4).to_markdown(index=False), ""]
            ms = sorted(set(se.model))
            fig, axes = plt.subplots(1, len(ms), figsize=(3.2 * len(ms), 2.8), squeeze=False)
            for ax, m in zip(axes[0], ms):
                for (pop, ep), d in se[se.model == m].groupby(["pop", "epoch"]):
                    L = min(len(h) for h in d.best_so_far)
                    H = np.sqrt(np.vstack([np.asarray(h[:L]) for h in d.best_so_far]))
                    ax.plot(np.arange(1, L + 1), H.mean(0), lw=1, label=f"pop {pop}, iter {ep}")
                ax.set_xscale("log")
                ax.set_title(m)
                ax.set_xlabel("Objective evaluations")
                ax.set_ylabel(f"Best inner-CV RMSE ({u})")
            axes[0][0].legend(fontsize=6)
            fig.tight_layout()
            save_fig(fig, A / f"F_eo_sensitivity_{target}")

    atomic_write_json(A / "best_models.json", best_models)
    atomic_write_text(A / "summary.md", "\n".join(md))
    print("\n".join(md))
    print(f"\nSaved to {A}")


if __name__ == "__main__":
    main()
