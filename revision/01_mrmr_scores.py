"""mRMR relevance / redundancy scores and ranks for X1-X22 (Reviewer 1, #10, #11).

For each target the complete mRMR trace (F-test relevance, mean |Pearson|
redundancy, FCQ score, selection step) is computed on the full data set and,
to show stability, within every training fold of the repeated K-fold used in
the nested CV.  Mutual-information relevance is reported as a complementary
non-linear measure.

Outputs (``results_audit/``):
  T_mrmr_<target>.csv, T_mrmr_stability_<target>.csv, T_mrmr_summary.md
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.feature_selection import mutual_info_regression  # noqa: E402

from rev.config import PROFILES  # noqa: E402
from rev.data import FEATURES, VARIABLE_INFO, get_target_data  # noqa: E402
from rev.features import mrmr_trace  # noqa: E402
from rev.splits import outer_splits  # noqa: E402

OUT = HERE / "results_audit"
OUT.mkdir(exist_ok=True)


def main(profile_name: str = "full") -> None:
    prof = PROFILES[profile_name]
    md = ["# mRMR scores (F-test relevance / mean |Pearson| redundancy / FCQ score)", ""]
    for target in ["Y1", "Y2"]:
        td = get_target_data(target)
        tr = mrmr_trace(td.X, td.y, k=None)
        std = td.X.std(ddof=0)
        mi = pd.Series(np.nan, index=FEATURES)
        nz = [c for c in FEATURES if std[c] > 1e-12]
        mi.loc[nz] = mutual_info_regression(td.X[nz].values, td.y.values, random_state=prof.seed)
        tr["name"] = tr.feature.map(lambda v: VARIABLE_INFO[v]["name"])
        tr["relevance_MI"] = tr.feature.map(mi)
        tr["rank_relevance_F"] = tr["relevance_F"].rank(ascending=False, method="min")
        tr["rank_MI"] = tr["relevance_MI"].rank(ascending=False, method="min")
        tr["mrmr_rank"] = tr["selection_step"]
        tr["selected_top9"] = tr["selection_step"] <= prof.n_features
        tr["note"] = np.where(tr["zero_variance"], "zero variance (constant) - not selectable", "")
        cols = ["feature", "name", "zero_variance", "relevance_F", "relevance_p", "rank_relevance_F",
                "relevance_MI", "rank_MI", "redundancy_at_selection", "mrmr_score_at_selection",
                "mrmr_rank", "selected_top9", "note"]
        tr = tr[cols].sort_values(["mrmr_rank", "feature"], na_position="last")
        tr.to_csv(OUT / f"T_mrmr_{target}.csv", index=False)

        # stability across outer training folds
        ranks = []
        for name, (a, _) in outer_splits(td, "rkf", prof.rkf_folds, prof.rkf_repeats, prof.seed):
            t = mrmr_trace(td.X.iloc[a], td.y.iloc[a], k=None)
            ranks.append(t.set_index("feature")["selection_step"].rename(name))
        R = pd.concat(ranks, axis=1)
        stab = pd.DataFrame({"mean_rank": R.mean(axis=1), "min_rank": R.min(axis=1),
                             "max_rank": R.max(axis=1),
                             f"selected_in_top{prof.n_features}_%": 100 * (R <= prof.n_features).mean(axis=1)})
        stab = stab.sort_values("mean_rank")
        stab.to_csv(OUT / f"T_mrmr_stability_{target}.csv")
        md += [f"## {target} (n = {len(td.y)})", tr.round(4).to_markdown(index=False), "",
               "Stability over outer training folds:", stab.round(2).to_markdown(), ""]
    (OUT / "T_mrmr_summary.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "full")
