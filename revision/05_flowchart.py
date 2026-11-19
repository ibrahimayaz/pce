"""Methodology flowchart of the revised workflow (Reviewer 1, #8); Fig. 2 of the manuscript.

Drawn at final print size: single-column width (3.45 in), all text >= 6.5 pt at 100 % scale.
Layout: a phase rail on the left (data / nested cross-validation / assessment), white step boxes with
numbered badges, and loop markers that make the nested structure explicit (inner CV inside tuning,
outer loop over folds).

    python 05_flowchart.py            ->  results_audit/F_methodology_flowchart.png / .pdf (600 dpi)
    python 05_flowchart.py out.png    ->  out.png / out.pdf
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle  # noqa: E402

OUT = Path(__file__).resolve().parent / "results_audit"

# ---------------------------------------------------------------- geometry (inches) and style
W = 3.45
FONT = ["Arial", "Helvetica", "DejaVu Sans"]
FS_T, FS_B, FS_S = 7.4, 6.9, 6.5          # title / body / small labels (pt)
PT = 1 / 72
TITLE = FS_T * 1.45 * PT
PAD_T, PAD_B = 0.065, 0.06
GAP = 0.15
RAIL_X, RAIL_W = 0.02, 0.20
BOX_X = RAIL_X + RAIL_W + 0.07
LOOP_W = 0.27                              # right-hand channel for the outer-loop arrow
BOX_W = W - BOX_X - LOOP_W
TEXT_X = 0.23                              # text indent inside a box (after the badge)
INK, MUTED, BORDER = "#1a1a1a", "#555555", "#8c8c8c"

PHASES = {  # name: (colour, rail label)
    "data": ("#2f5d8a", "DATA"),
    "cv": ("#2d7f6f", "NESTED CROSS-VALIDATION"),
    "out": ("#a3452f", "ASSESSMENT"),
}

STEPS = [
    ("data", "Experimental dataset", [
        "616 pastes: 22 PCEs × 4 fly-ash levels × 7 dosages",
        "one cement, one fly ash, w/b = 0.35",
        "Y1 Marsh-funnel time (n = 616)",
        "Y2 mini-slump spread (n = 252)"]),
    ("data", "Data audit and correction", [
        "X20 “–” (no fly ash) set to 0; no imputation",
        "constant inputs (X1–X12, X22) flagged",
        "near-duplicate PCEs grouped; mRMR, correlations"]),
    ("cv", "Outer split (test fold held out)", [
        "(a) repeated 3 × 5-fold CV",
        "(b) leave-one-PCE-group-out",
        "(c) leave-one-fly-ash-level-out"]),
    ("cv", "Preprocessing (training part only)", [
        "constant-input removal, z-score scaling",
        "mRMR ranking (F-test / |r|), top 9 features"]),
    ("cv", "Hyper-parameter tuning", [
        "EO (20 agents × 100 iterations) vs. random",
        "search, TPE (same budget) and defaults",
        "objective: mean MSE of inner 3-fold CV"]),
    ("cv", "Model fitting and prediction", [
        "LightGBM, CatBoost, XGBoost, AdaBoost, MLP",
        "refitted on the outer-training part"]),
    ("out", "Performance evaluation", [
        "R², RMSE, MAE, MSE, MAPE (mean ± SD)",
        "corrected resampled t-test, Wilcoxon (Holm)",
        "EO vs. RS / TPE / defaults; sensitivity"]),
    ("out", "Final models and interpretation", [
        "EO tuning and refit on all data (Table VI)",
        "Tree SHAP: importance, dependence, interaction",
        "learning curves; web GUI with range check"]),
]
ARROW_LABELS = {2: "outer-training part", 5: "out-of-fold predictions"}   # arrow leaving step index i

_BODY_H: dict[int, float] = {}


def _body_height(n: int) -> float:
    """Measured height (in) of an n-line body text block in the figure font."""
    if n not in _BODY_H:
        f = plt.figure(figsize=(W, 2))
        t = f.text(0, 1, "\n".join(["Xg"] * n), fontsize=FS_B, linespacing=1.25, va="top")
        _BODY_H[n] = t.get_window_extent(f.canvas.get_renderer()).height / f.dpi
        plt.close(f)
    return _BODY_H[n]


def _h(n: int) -> float:
    return PAD_T + TITLE + _body_height(n) + PAD_B


def draw(path: Path) -> None:
    plt.rcdefaults()
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": FONT, "mathtext.fontset": "dejavusans"})
    _BODY_H.clear()

    tops, y = [], 0.03
    for i, (ph, _, body) in enumerate(STEPS):
        if i and STEPS[i - 1][0] != ph:
            y += 0.06                      # a little extra air between phases
        tops.append(y)
        y += _h(len(body)) + GAP
    H = y - GAP + 0.03

    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.axis("off")
    renderer = fig.canvas.get_renderer()

    # ---- phase rail
    for ph, (col, label) in PHASES.items():
        idx = [i for i, s in enumerate(STEPS) if s[0] == ph]
        y0, y1 = tops[idx[0]], tops[idx[-1]] + _h(len(STEPS[idx[-1]][2]))
        ax.add_patch(FancyBboxPatch((RAIL_X, y0), RAIL_W, y1 - y0, boxstyle="round,pad=0,rounding_size=0.035",
                                    fc=col, ec="none"))
        ax.text(RAIL_X + RAIL_W / 2, (y0 + y1) / 2, label, rotation=90, ha="center", va="center", color="white",
                fontsize=FS_S, weight="bold")

    # ---- step boxes
    cx = BOX_X + BOX_W / 2
    for i, (ph, title, body) in enumerate(STEPS):
        col = PHASES[ph][0]
        top, h = tops[i], _h(len(body))
        ax.add_patch(FancyBboxPatch((BOX_X, top), BOX_W, h, boxstyle="round,pad=0,rounding_size=0.03",
                                    fc="white", ec=BORDER, lw=0.5))
        ax.add_patch(Rectangle((BOX_X, top + 0.02), 0.028, h - 0.04, fc=col, ec="none"))   # accent stripe
        by = top + PAD_T + TITLE * 0.42
        ax.add_patch(Circle((BOX_X + 0.12, by), 0.062, fc=col, ec="none"))
        ax.text(BOX_X + 0.12, by, str(i + 1), ha="center", va="center", color="white", fontsize=FS_S,
                weight="bold")
        ax.text(BOX_X + TEXT_X, by, title, ha="left", va="center", fontsize=FS_T, weight="bold", color=INK)
        t = ax.text(BOX_X + TEXT_X, top + PAD_T + TITLE, "\n".join(body), ha="left", va="top", fontsize=FS_B,
                    color="#333333", linespacing=1.25)
        bb = t.get_window_extent(renderer).transformed(ax.transData.inverted())
        if bb.x1 > BOX_X + BOX_W - 0.04:
            warnings.warn(f"text of step {i + 1} overflows its box")

    # ---- forward arrows
    for i in range(len(STEPS) - 1):
        y1, y2 = tops[i] + _h(len(STEPS[i][2])), tops[i + 1]
        ax.add_patch(FancyArrowPatch((cx, y1), (cx, y2), arrowstyle="-|>", mutation_scale=6.5, lw=0.7,
                                     color=MUTED, shrinkA=0, shrinkB=0))
        if i in ARROW_LABELS:
            ax.text(cx + 0.06, (y1 + y2) / 2, ARROW_LABELS[i], fontsize=FS_S, style="italic", color=MUTED,
                    ha="left", va="center")

    # ---- outer loop: from step 6 back to step 3 along the right channel
    cv = PHASES["cv"][0]
    xr, xl = BOX_X + BOX_W, BOX_X + BOX_W + 0.11
    ya = tops[5] + _h(len(STEPS[5][2])) / 2
    yb = tops[2] + _h(len(STEPS[2][2])) / 2
    ax.plot([xr, xl, xl], [ya, ya, yb], color=cv, lw=0.8, solid_capstyle="butt")
    ax.add_patch(FancyArrowPatch((xl, yb), (xr, yb), arrowstyle="-|>", mutation_scale=6.5, lw=0.8, color=cv,
                                 shrinkA=0, shrinkB=0))
    ax.text(xl + 0.05, (ya + yb) / 2, "next outer fold", rotation=270, ha="center", va="center", fontsize=FS_S,
            style="italic", color=cv)

    # ---- inner-loop marker on the tuning box, aligned with its title
    t5 = tops[4]
    ax.text(BOX_X + BOX_W - 0.07, t5 + PAD_T + TITLE * 0.42, "inner CV loop", ha="right", va="center",
            fontsize=FS_S - 0.3, style="italic", color=cv)
    ax.text(BOX_X + BOX_W - 0.66, t5 + PAD_T + TITLE * 0.42, "↻", ha="right", va="center",
            fontsize=FS_T + 2.5, color=cv, family="DejaVu Sans")

    fig.savefig(path, dpi=600, facecolor="white")
    fig.savefig(path.with_suffix(".pdf"), facecolor="white")
    plt.close(fig)


def main():
    if len(sys.argv) > 1:
        out = Path(sys.argv[1])
    else:
        OUT.mkdir(exist_ok=True)
        out = OUT / "F_methodology_flowchart.png"
    draw(out)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
