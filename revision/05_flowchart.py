"""Methodology flowchart of the revised workflow (Reviewer 1, #8).

    python 05_flowchart.py      ->  results_audit/F_methodology_flowchart.png / .pdf (600 dpi)
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

OUT = Path(__file__).resolve().parent / "results_audit"
OUT.mkdir(exist_ok=True)

C_DATA, C_PRE, C_LOOP, C_EVAL, C_OUT = "#dbe9f6", "#e8f4e0", "#fff2cc", "#fde2e1", "#ece4f5"


def box(ax, x, y, w, h, text, color, fs=7.0, bold_first=True):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.012,rounding_size=0.02",
                                fc=color, ec="0.25", lw=0.8))
    lines = text.split("\n")
    if bold_first:
        ax.text(x, y + h / 2 - 0.022, lines[0], ha="center", va="top", fontsize=fs + 0.6, weight="bold")
        ax.text(x, y + h / 2 - 0.05, "\n".join(lines[1:]), ha="center", va="top", fontsize=fs, linespacing=1.25)
    else:
        ax.text(x, y, text, ha="center", va="center", fontsize=fs)


def arrow(ax, x1, y1, x2, y2, text=None):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=9, lw=0.9, color="0.2"))
    if text:
        ax.text((x1 + x2) / 2 + 0.01, (y1 + y2) / 2, text, fontsize=6.3, va="center", style="italic")


def main():
    fig, ax = plt.subplots(figsize=(7.2, 9.6))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    box(ax, 0.5, 0.955, 0.86, 0.075, "1. Experimental dataset\n616 pastes = 22 PCEs × 4 fly-ash levels (0–45 %) × 7 PCE dosages "
        "(0.5–2.0 %); one cement, one fly ash,\nw/b = 0.35; targets: Marsh-funnel flow time Y1 (s, n = 616), "
        "mini-slump spread Y2 (cm, n = 252)", C_DATA)
    box(ax, 0.5, 0.845, 0.92, 0.09, "2. Data audit and corrections\nX20 '-' = no fly ash → 0 (no input imputation needed); "
        "no target imputation (Y2 only where measured)\nzero-variance inputs (X1-X12, X22) identified; near-duplicate "
        "admixture pairs grouped\ndescriptive statistics, missingness pattern, correlation matrix, mRMR scores", C_PRE)
    arrow(ax, 0.5, 0.917, 0.5, 0.891)

    # nested loop frame
    ax.add_patch(FancyBboxPatch((0.05, 0.315), 0.9, 0.455, boxstyle="round,pad=0.01", fc="none", ec="0.35",
                                lw=1.0, ls="--"))
    ax.text(0.07, 0.76, "Nested validation (repeated for every outer fold)", fontsize=7.5, weight="bold", va="top")
    arrow(ax, 0.5, 0.80, 0.5, 0.735)
    box(ax, 0.5, 0.70, 0.82, 0.07, "3. Outer split (test fold never used for any fitting)\n(a) repeated 3 × 5-fold CV   "
        "(b) leave-one-PCE-group-out   (c) leave-one-fly-ash-level-out", C_LOOP)
    arrow(ax, 0.5, 0.657, 0.5, 0.632, "outer-training part")
    box(ax, 0.275, 0.575, 0.38, 0.105, "4. Preprocessing (fit on training part)\nremove zero-variance inputs\n"
        "standardisation (z-score)\nmRMR ranking (F-test / |Pearson|), top 9", C_PRE)
    box(ax, 0.725, 0.575, 0.38, 0.105, "5. Hyper-parameter tuning\nEO (20 agents × 100 iterations), or\nrandom search / "
        "Bayesian TPE (same budget),\nor library defaults", C_LOOP)
    arrow(ax, 0.465, 0.575, 0.535, 0.575)
    box(ax, 0.725, 0.43, 0.38, 0.085, "Objective = mean inner-CV MSE\n3-fold inner CV on the outer-training part\n"
        "(preprocessing refitted in each inner fold)", "#fffbe6", fs=6.8)
    arrow(ax, 0.725, 0.522, 0.725, 0.473)
    arrow(ax, 0.275, 0.522, 0.275, 0.428)
    box(ax, 0.275, 0.39, 0.38, 0.075, "6. Fit tuned model\nLightGBM, CatBoost, XGBoost, AdaBoost, MLP\non the whole "
        "outer-training part", C_LOOP, fs=6.8)
    arrow(ax, 0.535, 0.43, 0.465, 0.40)
    arrow(ax, 0.5, 0.315, 0.5, 0.285, "predict held-out fold")

    box(ax, 0.5, 0.235, 0.86, 0.085, "7. Evaluation\nR², RMSE, MAE, MSE, MAPE (mean ± SD over outer folds, pooled out-of-fold)\n"
        "corrected resampled t-test and Wilcoxon (Holm); EO vs RS/TPE/default; EO and hyper-parameter sensitivity",
        C_EVAL)
    arrow(ax, 0.5, 0.192, 0.5, 0.165)
    box(ax, 0.5, 0.11, 0.86, 0.095, "8. Final models and interpretation\nEO tuning and refit on all data → complete "
        "Table 4; Tree SHAP (global importance, beeswarm,\ndependence and interaction plots); learning curves "
        "(random and grouped CV);\nweb GUI with range check", C_OUT)
    fig.savefig(OUT / "F_methodology_flowchart.png", dpi=600, bbox_inches="tight")
    fig.savefig(OUT / "F_methodology_flowchart.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {OUT / 'F_methodology_flowchart.png'}")


if __name__ == "__main__":
    main()
