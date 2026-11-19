"""Data audit (no training; runs in seconds).

Outputs (``results_audit/``):
  T_descriptive_stats.csv     descriptive statistics in original units        (R1 #25)
  T_missingness.csv           missing fraction per variable + interpretation  (R1 #26)
  T_missing_pattern.csv       joint missingness pattern                       (R1 #26)
  T_pce_descriptors.csv       descriptor set of every admixture, Y2 coverage  (R1 #19, R2 #1)
  T_duplicates.csv            exact / near-duplicate rows and admixture pairs (R1 #18)
  T_design.csv                factorial design check (PCE x fly ash x dosage)
  T_X21_X22_check.csv         X21 = PCE dosage, X22 = water evidence          (R2 #2)
  T_correlation_pearson.csv / _spearman.csv, F_correlation_matrix.png        (R1 #31)
  audit_summary.md            human-readable summary
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from rev.data import DUPLICATE_PAIRS, FEATURES, VARIABLE_INFO, load_corrected, load_raw  # noqa: E402

OUT = HERE / "results_audit"
OUT.mkdir(exist_ok=True)


def main() -> None:
    raw = load_raw()
    cor = load_corrected()
    lines = ["# Data audit summary", ""]

    # ---------------- descriptive statistics (original units, corrected X20) -------------
    rows = []
    for v in FEATURES + ["Y1", "Y2"]:
        s = cor[v]
        rows.append({
            "Variable": v, "Name": VARIABLE_INFO[v]["name"], "Unit": VARIABLE_INFO[v]["unit"],
            "Group": VARIABLE_INFO[v]["group"], "N": int(s.notna().sum()),
            "Missing (%)": round(100 * s.isna().mean(), 2), "Distinct": int(s.nunique()),
            "Min": s.min(), "Q1": s.quantile(.25), "Median": s.median(), "Mean": s.mean(),
            "Q3": s.quantile(.75), "Max": s.max(), "SD": s.std(), "CV (%)": 100 * s.std() / s.mean() if s.mean() else np.nan,
            "Constant": bool(s.nunique() <= 1),
        })
    desc = pd.DataFrame(rows)
    desc.to_csv(OUT / "T_descriptive_stats.csv", index=False)
    const = desc.loc[desc["Constant"], "Variable"].tolist()
    lines += [f"* Constant (zero-variance) variables: {', '.join(const)}",
              f"* Non-constant inputs: {', '.join(v for v in FEATURES if v not in const)}", ""]

    # ---------------- missingness -------------------------------------------------------
    miss = pd.DataFrame({
        "Variable": FEATURES + ["Y1", "Y2"],
        "Missing n (raw)": [int(raw[v].isna().sum()) for v in FEATURES + ["Y1", "Y2"]],
        "Missing % (raw)": [round(100 * raw[v].isna().mean(), 2) for v in FEATURES + ["Y1", "Y2"]],
    })
    fa = cor["fly_ash_pct"]
    miss["Interpretation"] = ""
    miss.loc[miss.Variable == "X20", "Interpretation"] = (
        f"all {int(raw.X20.isna().sum())} missing rows are UK0 mixtures (0% fly ash) -> structural zero, set to 0")
    pce_no_y2 = sorted(cor.loc[cor.Y2.isna(), "pce"].unique())
    pce_y2 = sorted(cor.loc[cor.Y2.notna(), "pce"].unique())
    miss.loc[miss.Variable == "Y2", "Interpretation"] = (
        f"not measured for {len(pce_no_y2)} of {cor.pce.nunique()} admixtures ({', '.join(pce_no_y2)}); "
        "missingness is block-wise by admixture, not random -> Y2 is modelled only on measured rows")
    miss.to_csv(OUT / "T_missingness.csv", index=False)
    pat = raw[FEATURES + ["Y1", "Y2"]].isna()
    pat = pat.loc[:, pat.any()].value_counts().rename("n_rows").reset_index()
    pat.to_csv(OUT / "T_missing_pattern.csv", index=False)
    assert (fa[raw.X20.isna()] == 0).all()
    lines += ["## Missingness",
              f"* X20 missing: {raw.X20.isna().sum()} rows, all with 0% fly ash (structural zero).",
              f"* Y2 missing: {raw.Y2.isna().sum()} / {len(raw)} rows; measured only for admixtures "
              f"{', '.join(pce_y2)}.", ""]

    # ---------------- admixture descriptor sets ----------------------------------------
    pdesc = cor.groupby("pce").agg(
        **{v: (v, "first") for v in ["X13", "X14", "X15", "X16", "X17", "X18"]},
        n_rows=("Y1", "size"), n_Y2=("Y2", lambda s: int(s.notna().sum())),
        Y1_mean=("Y1", "mean"), group=("pce_group", "first"))
    pdesc.to_csv(OUT / "T_pce_descriptors.csv")
    n_sets = pdesc[["X13", "X14", "X15", "X16", "X17", "X18"]].drop_duplicates().shape[0]
    lines += ["## Admixtures",
              f"* {cor.pce.nunique()} admixtures, {n_sets} distinct descriptor sets (X13-X18); "
              "X13-X18 are therefore admixture-level constants (each takes one value per admixture).", ""]

    # ---------------- design -------------------------------------------------------------
    design = cor.groupby(["fly_ash_pct", "X19", "X20"]).agg(
        n=("Y1", "size"), dosages=("X21", lambda s: ",".join(map(str, sorted(s.unique()))))).reset_index()
    design.to_csv(OUT / "T_design.csv", index=False)
    lines += ["## Design",
              f"* Full factorial: {cor.pce.nunique()} admixtures x {cor.fly_ash_pct.nunique()} fly-ash levels "
              f"x {cor.X21.nunique()} dosages = {len(cor)} rows.",
              f"* X19 + X20 = {sorted((cor.X19 + cor.X20).unique())} g in every row (binder fixed) -> "
              "X19 and X20 are perfectly collinear (one design factor: fly-ash replacement level).", ""]

    # ---------------- X21 / X22 meaning --------------------------------------------------
    dose_from_id = cor["mix_id"].str.extract(r"-(\d+[,.]?\d*)$")[0].str.replace(",", ".").astype(float)
    chk = pd.DataFrame({"mix_id": cor.mix_id, "dosage_in_id_%": dose_from_id, "X21": cor.X21,
                        "X21/700*100": cor.X21 / 700 * 100, "X22": cor.X22})
    ok = np.allclose(chk["dosage_in_id_%"], chk["X21/700*100"])
    chk.to_csv(OUT / "T_X21_X22_check.csv", index=False)
    lines += ["## Variable meaning (R2 #2)",
              f"* X21/700x100 equals the dosage (%) encoded in the mixture ID for every row: {ok} -> "
              "X21 is the PCE dosage (g), 0.5-2.0% by binder mass.",
              f"* X22 takes the single value {sorted(cor.X22.unique())} g -> mixing water, w/b = "
              f"{cor.X22.iloc[0] / 700:.2f}.", ""]

    # ---------------- duplicates ----------------------------------------------------------
    inputs = cor[FEATURES]
    dup_rows = []
    for a, b in DUPLICATE_PAIRS.items():
        A = cor[cor.pce == a].set_index(["X19", "X21"]).sort_index()
        B = cor[cor.pce == b].set_index(["X19", "X21"]).sort_index()
        same_y1 = int((A.Y1.values == B.Y1.values).sum())
        dmax = float(np.abs(A.Y1.values - B.Y1.values).max())
        ddesc = {v: (A[v].iloc[0], B[v].iloc[0]) for v in ["X13", "X14", "X15", "X16", "X17", "X18"]
                 if A[v].iloc[0] != B[v].iloc[0]}
        dup_rows.append({"admixture": a, "duplicate_of": b, "identical_Y1": f"{same_y1}/{len(A)}",
                         "max_abs_Y1_diff": dmax, "descriptor_differences": str(ddesc) or "none",
                         "Y2_measured_for": a if A.Y2.notna().any() else (b if B.Y2.notna().any() else "none")})
    dups = pd.DataFrame(dup_rows)
    dups.to_csv(OUT / "T_duplicates.csv", index=False)
    lines += ["## Duplicates (R1 #18)",
              f"* Exact duplicate rows (inputs+targets): {int(cor[FEATURES + ['Y1', 'Y2']].duplicated().sum())}",
              f"* Duplicate input vectors: {int(inputs.duplicated().sum())}",
              "* Near-duplicate admixture pairs (identical or almost identical descriptors and Y1 records):"]
    for r in dup_rows:
        lines.append(f"  * {r['admixture']} vs {r['duplicate_of']}: identical Y1 {r['identical_Y1']}, "
                     f"max |dY1| = {r['max_abs_Y1_diff']:.1f} s, descriptor differences {r['descriptor_differences']}")
    lines.append("")

    # ---------------- correlation matrix (non-constant variables) ---------------------
    var = [v for v in FEATURES if v not in const] + ["Y1", "Y2"]
    pear = cor[var].corr(method="pearson")
    spear = cor[var].corr(method="spearman")
    pear.to_csv(OUT / "T_correlation_pearson.csv")
    spear.to_csv(OUT / "T_correlation_spearman.csv")
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))
    for ax, m, t in zip(axes, [pear, spear], ["Pearson", "Spearman"]):
        im = ax.imshow(m.values, cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_xticks(range(len(var)), var, rotation=90)
        ax.set_yticks(range(len(var)), var)
        for i in range(len(var)):
            for j in range(len(var)):
                val = m.values[i, j]
                if np.isfinite(val):
                    ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=6.5,
                            color="white" if abs(val) > 0.6 else "black")
        ax.set_title(f"{t} correlation (Y2: pairwise, n = {int(cor.Y2.notna().sum())})")
    fig.colorbar(im, ax=axes, shrink=0.8)
    fig.savefig(OUT / "F_correlation_matrix.png", dpi=600, bbox_inches="tight")
    fig.savefig(OUT / "F_correlation_matrix.pdf", bbox_inches="tight")
    plt.close(fig)
    top = pear.loc[[v for v in var if v.startswith("X")], ["Y1", "Y2"]].round(3)
    lines += ["## Correlation with targets (Pearson)", top.to_markdown(), ""]
    (OUT / "audit_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nSaved to {OUT}")


if __name__ == "__main__":
    main()
