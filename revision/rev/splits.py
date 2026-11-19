"""Outer and inner validation splits.

* rkf  : repeated shuffled K-fold (random mixtures held out)
* lopo : leave-one-PCE-group-out (a whole admixture, incl. its duplicate twin, held out)
* lofa : leave-one-fly-ash-level-out (a whole replacement level held out)

Inner splits used by the tuning objective follow the same logic on the
outer-training data: shuffled K-fold for rkf/final/sens, GroupKFold on PCE
groups for lopo, GroupKFold on fly-ash levels for lofa.
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np
from sklearn.model_selection import GroupKFold, KFold, LeaveOneGroupOut

from .data import TargetData

Split = Tuple[np.ndarray, np.ndarray]


def outer_splits(td: TargetData, scheme: str, folds: int, repeats: int, seed: int) -> List[Tuple[str, Split]]:
    n = len(td.y)
    idx = np.arange(n)
    out: List[Tuple[str, Split]] = []
    if scheme == "rkf":
        for r in range(repeats):
            kf = KFold(n_splits=folds, shuffle=True, random_state=seed + r)
            for f, (tr, te) in enumerate(kf.split(idx)):
                out.append((f"r{r}f{f}", (tr, te)))
    elif scheme == "lopo":
        g = td.pce_group.values
        for tr, te in LeaveOneGroupOut().split(idx, groups=g):
            name = str(g[te[0]]).replace("|", "+").replace(":", "_")
            out.append((f"g_{name}", (tr, te)))
    elif scheme == "lofa":
        g = td.fa_level.values
        for tr, te in LeaveOneGroupOut().split(idx, groups=g):
            out.append((f"fa{int(g[te[0]])}", (tr, te)))
    elif scheme in ("final", "sens"):
        out.append(("all", (idx, np.array([], dtype=int))))
    else:
        raise ValueError(scheme)
    return out


def inner_splits(td: TargetData, train_idx: np.ndarray, scheme: str, folds: int, seed: int) -> List[Split]:
    sub = np.arange(len(train_idx))
    if scheme in ("rkf", "final", "sens"):
        kf = KFold(n_splits=folds, shuffle=True, random_state=seed)
        return list(kf.split(sub))
    groups = (td.pce_group if scheme == "lopo" else td.fa_level).values[train_idx]
    k = min(folds, len(np.unique(groups)))
    return list(GroupKFold(n_splits=k).split(sub, groups=groups))
