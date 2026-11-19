"""Leakage-free preprocessing: zero-variance removal -> standardisation -> mRMR.

``MRMRPreprocessor`` is a scikit-learn transformer, so it is fitted only on the
training part of each fold.  The mRMR variant reproduces the one used by the
``mrmr_selection`` package in the original pipeline (``mrmr_regression``
defaults): relevance = F-statistic, redundancy = mean absolute Pearson
correlation with the already selected features, score = relevance /
redundancy (FCQ scheme, floor 0.001).  The full scoring trace is kept so that
relevance/redundancy scores and ranks can be reported (Reviewer 1, #10).
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import f_regression

FLOOR = 0.001


def mrmr_trace(X: pd.DataFrame, y, k: Optional[int] = None) -> pd.DataFrame:
    """Run mRMR (F-test / Pearson / mean) and return the full scoring trace.

    Columns: feature, variance_zero, relevance_F, relevance_p, selection_step,
    redundancy_at_selection, mrmr_score_at_selection, rank.
    Zero-variance features get NaN relevance and are never selected.
    """
    X = pd.DataFrame(X)
    y = np.asarray(y, dtype=float)
    std = X.std(axis=0, ddof=0)
    const = std <= 1e-12
    usable = [c for c in X.columns if not const[c]]
    F = pd.Series(np.nan, index=X.columns)
    P = pd.Series(np.nan, index=X.columns)
    if usable:
        f_vals, p_vals = f_regression(X[usable].values, y)
        F.loc[usable] = np.nan_to_num(f_vals, nan=0.0)
        P.loc[usable] = p_vals
    candidates = [c for c in usable if F[c] > 0]
    k = len(candidates) if k is None else min(k, len(candidates))
    corr = X[candidates].corr().abs().clip(lower=FLOOR) if candidates else pd.DataFrame()

    selected: List[str] = []
    rows = {}
    remaining = candidates.copy()
    for step in range(1, k + 1):
        if not selected:
            red = pd.Series(1.0, index=remaining)
        else:
            red = corr.loc[remaining, selected].mean(axis=1)
        score = F[remaining] / red
        best = score.idxmax()
        rows[best] = (step, float(red[best]) if selected else np.nan, float(score[best]))
        selected.append(best)
        remaining.remove(best)

    out = pd.DataFrame({
        "feature": X.columns,
        "zero_variance": const.values,
        "relevance_F": F.values,
        "relevance_p": P.values,
    })
    out["selection_step"] = out["feature"].map(lambda f: rows.get(f, (np.nan,))[0])
    out["redundancy_at_selection"] = out["feature"].map(lambda f: rows.get(f, (None, np.nan))[1])
    out["mrmr_score_at_selection"] = out["feature"].map(lambda f: rows.get(f, (None, None, np.nan))[2])
    out["selected"] = out["selection_step"].notna()
    return out


class MRMRPreprocessor(BaseEstimator, TransformerMixin):
    """Drop zero-variance columns, standardise, keep the top-k mRMR features."""

    def __init__(self, k: int = 9):
        self.k = k

    def fit(self, X, y):
        X = pd.DataFrame(X).astype(float)
        self.feature_names_in_ = list(X.columns)
        std = X.std(axis=0, ddof=0)
        self.kept_ = [c for c in X.columns if std[c] > 1e-12]
        self.mean_ = X[self.kept_].mean(axis=0)
        self.scale_ = X[self.kept_].std(axis=0, ddof=0)
        Z = (X[self.kept_] - self.mean_) / self.scale_
        trace = mrmr_trace(Z, y, k=self.k)
        sel = trace[trace["selected"]].sort_values("selection_step")
        self.selected_ = sel["feature"].tolist()
        self.trace_ = trace
        return self

    def transform(self, X):
        X = pd.DataFrame(X, columns=self.feature_names_in_).astype(float)
        Z = (X[self.selected_] - self.mean_[self.selected_]) / self.scale_[self.selected_]
        return Z.values

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.selected_, dtype=object)
