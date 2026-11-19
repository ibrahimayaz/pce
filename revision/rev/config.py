"""Experiment configuration and run profiles.

Two profiles are provided:

* ``smoke`` – tiny budget, used only to check that every code path runs
  (minutes on a laptop).  Results are NOT meant for the manuscript.
* ``full``  – the budget used for the revised manuscript (run on the
  workstation).

Every experiment *block* defines a set of jobs.  The job identifier contains
the profile name and a hash of the budget settings, so results produced with
different budgets can never be mixed by accident.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List

REVISION_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = REVISION_DIR.parent
DATASET_PATH = PROJECT_ROOT / "data" / "dataset.csv"

TARGETS = ["Y1", "Y2"]
MODELS = ["LightGBM", "CatBoost", "XGBoost", "AdaBoost", "MLP"]
TUNERS = ["default", "random", "tpe", "eo"]


def eo_evals(pop: int, epoch: int) -> int:
    """Exact number of objective calls of mealpy OriginalEO: pop initial agents,
    then per epoch pop updated agents + 1 averaged equilibrium-pool agent.
    Random search and TPE receive exactly the same number of evaluations."""
    return pop + epoch * (pop + 1)


@dataclass
class Block:
    """One family of jobs.

    scheme : rkf (repeated random K-fold), lopo (leave-one-PCE-group-out),
             lofa (leave-one-fly-ash-level-out), final (tune + fit on all data),
             sens (EO parameter sensitivity on all data).
    data   : "full" or "nodup" (near-duplicate parent PCEs removed).
    """

    name: str
    scheme: str
    targets: List[str]
    models: List[str]
    tuners: List[str]
    data: str = "full"
    priority: int = 0
    # sensitivity grid (only for scheme == "sens")
    sens_pop: List[int] = field(default_factory=list)
    sens_epoch: List[int] = field(default_factory=list)
    sens_seeds: List[int] = field(default_factory=list)


@dataclass
class Profile:
    name: str
    # EO budget; RS and TPE receive the same number of objective evaluations
    eo_pop: int = 20
    eo_epoch: int = 100
    # outer CV
    rkf_folds: int = 5
    rkf_repeats: int = 3
    # inner CV used by the tuning objective
    inner_folds: int = 3
    # mRMR: number of features to keep (capped by the number of non-constant features)
    n_features: int = 9
    # limit the number of outer folds per block (smoke test only); 0 = no limit
    max_outer_folds: int = 0
    seed: int = 42
    blocks: List[Block] = field(default_factory=list)

    @property
    def n_evals(self) -> int:
        return eo_evals(self.eo_pop, self.eo_epoch)

    def budget_hash(self) -> str:
        d = asdict(self)
        d.pop("blocks")
        d.pop("name")
        return hashlib.sha1(json.dumps(d, sort_keys=True).encode()).hexdigest()[:8]


def _blocks(sens_pop, sens_epoch, sens_seeds) -> List[Block]:
    return [
        # final models: complete Table 4 (ranges + optimum for every model/target), GUI, SHAP
        Block("final", "final", TARGETS, MODELS, ["eo"], priority=0),
        # main nested-CV comparison: EO vs default vs Random Search vs Bayesian (TPE)
        Block("rkf_main", "rkf", TARGETS, MODELS, ["eo", "default"], priority=1),
        # leave-one-PCE-group-out: generalisation to an unseen admixture
        Block("lopo", "lopo", TARGETS, MODELS, ["eo", "default"], priority=2),
        Block("rkf_tuners", "rkf", TARGETS, MODELS, ["random", "tpe"], priority=3),
        # leave-one-fly-ash-level-out: extrapolation to an unseen replacement level
        Block("lofa", "lofa", TARGETS, MODELS, ["eo", "default"], priority=4),
        # sensitivity to the near-duplicate PCE pairs (Y2 is unaffected: parents have no Y2)
        Block("nodup_rkf", "rkf", ["Y1"], MODELS, ["eo", "default"], data="nodup", priority=5),
        Block("nodup_lopo", "lopo", ["Y1"], MODELS, ["eo"], data="nodup", priority=6),
        # baselines: are the PCE descriptors more than admixture identifiers? (R1 #19, #33)
        Block("base_mixonly", "rkf", TARGETS, ["LightGBM", "CatBoost", "XGBoost"], ["eo", "default"],
              data="mixonly", priority=5),
        Block("base_pceid", "rkf", TARGETS, ["LightGBM", "CatBoost", "XGBoost"], ["eo", "default"],
              data="pceid", priority=5),
        Block("base_mixonly_lopo", "lopo", TARGETS, ["LightGBM", "CatBoost", "XGBoost"], ["eo"],
              data="mixonly", priority=6),
        # original procedure (global MICE incl. targets) for comparison (R2 #1, R1 #27)
        Block("legacy_mice", "rkf", TARGETS, ["LightGBM", "CatBoost"], ["eo", "default"],
              data="mice_global", priority=6),
        # EO parameter sensitivity (population size x iterations x independent seeds)
        Block("sens", "sens", TARGETS, ["CatBoost", "LightGBM", "XGBoost"], ["eo"], priority=7,
              sens_pop=sens_pop, sens_epoch=sens_epoch, sens_seeds=sens_seeds),
    ]


PROFILES: Dict[str, Profile] = {
    "full": Profile(
        name="full", eo_pop=20, eo_epoch=100, rkf_folds=5, rkf_repeats=3, inner_folds=3,
        blocks=_blocks(sens_pop=[10, 20, 30], sens_epoch=[50, 100], sens_seeds=[1, 2, 3, 4, 5]),
    ),
    "smoke": Profile(
        name="smoke", eo_pop=5, eo_epoch=2, rkf_folds=3, rkf_repeats=1, inner_folds=2,
        max_outer_folds=2,
        blocks=_blocks(sens_pop=[5], sens_epoch=[1, 2], sens_seeds=[1]),
    ),
}


def results_dir(profile: Profile) -> Path:
    return REVISION_DIR / f"results_{profile.name}"
