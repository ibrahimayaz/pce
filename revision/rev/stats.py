"""Paired statistical tests for cross-validated model comparisons."""
from __future__ import annotations

from typing import Dict, List

import numpy as np
from scipy import stats


def corrected_resampled_ttest(a, b, test_train_ratio: float) -> Dict[str, float]:
    """Nadeau & Bengio (2003) corrected resampled t-test for repeated K-fold.

    a, b : per-fold scores of two methods on the *same* folds.
    test_train_ratio : n_test / n_train of one fold.
    """
    d = np.asarray(a, float) - np.asarray(b, float)
    J = len(d)
    if J < 2:
        return {"t": np.nan, "p": np.nan, "mean_diff": float(np.mean(d)) if J else np.nan}
    var = np.var(d, ddof=1)
    if var == 0:
        return {"t": np.inf if d.mean() != 0 else 0.0, "p": 0.0 if d.mean() != 0 else 1.0,
                "mean_diff": float(d.mean())}
    t = d.mean() / np.sqrt((1.0 / J + test_train_ratio) * var)
    p = 2 * stats.t.sf(abs(t), df=J - 1)
    return {"t": float(t), "p": float(p), "mean_diff": float(d.mean())}


def wilcoxon(a, b) -> float:
    d = np.asarray(a, float) - np.asarray(b, float)
    if len(d) < 2 or np.allclose(d, 0):
        return float("nan")
    try:
        return float(stats.wilcoxon(a, b).pvalue)
    except ValueError:
        return float("nan")


def holm(pvals: List[float]) -> List[float]:
    p = np.asarray(pvals, float)
    out = np.full_like(p, np.nan)
    ok = np.isfinite(p)
    idx = np.argsort(p[ok])
    m = ok.sum()
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(idx):
        running = max(running, min(1.0, (m - rank) * p[ok][i]))
        adj[i] = running
    out[ok] = adj
    return out.tolist()
