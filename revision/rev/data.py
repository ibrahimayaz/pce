"""Dataset loading with the corrections identified during the revision.

Corrections relative to the original pipeline
--------------------------------------------
1. X20 (fly ash, g) is "-" only for the UK0 mixtures, i.e. mixtures *without*
   fly ash.  It is therefore set to 0 instead of being imputed.  After this
   correction no input variable contains missing values, so no imputation is
   needed at all.
2. Y2 (mini-slump, cm) is NOT imputed.  Y2 models use only the 252 rows with a
   measured mini-slump value.
3. Variable meaning: X21 is the PCE dosage (g per 700 g binder, i.e. 0.5-2.0 %
   by binder mass); X22 is the mixing water (g), constant at 245 g (w/b 0.35).
4. PCE groups: every admixture is identified from the mixture ID.  Three pairs
   of admixtures have (near-)identical descriptors and identical Y1 records
   (K/K3:1, S5/S5-3:1, F9/F9-3:1); each pair is placed in the same group so
   that grouped validation cannot leak information between them.

The S-series X13/X15 values are kept exactly as reported (confirmed by the
authors).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .config import DATASET_PATH

FEATURES: List[str] = [f"X{i}" for i in range(1, 23)]

VARIABLE_INFO: Dict[str, Dict[str, str]] = {
    "X1": {"name": "C3S", "unit": "%", "group": "cement"},
    "X2": {"name": "C2S", "unit": "%", "group": "cement"},
    "X3": {"name": "C3A", "unit": "%", "group": "cement"},
    "X4": {"name": "C4AF", "unit": "%", "group": "cement"},
    "X5": {"name": "Specific surface area of cement", "unit": "cm2/g", "group": "cement"},
    "X6": {"name": "Specific gravity of cement", "unit": "-", "group": "cement"},
    "X7": {"name": "SiO2 (fly ash)", "unit": "%", "group": "fly ash"},
    "X8": {"name": "Al2O3 (fly ash)", "unit": "%", "group": "fly ash"},
    "X9": {"name": "Fe2O3 (fly ash)", "unit": "%", "group": "fly ash"},
    "X10": {"name": "CaO (fly ash)", "unit": "%", "group": "fly ash"},
    "X11": {"name": "Specific surface area of fly ash", "unit": "cm2/g", "group": "fly ash"},
    "X12": {"name": "Specific gravity of fly ash", "unit": "-", "group": "fly ash"},
    "X13": {"name": "PCE density", "unit": "dataset scale", "group": "PCE"},
    "X14": {"name": "PCE solid content", "unit": "%", "group": "PCE"},
    "X15": {"name": "PCE pH", "unit": "-", "group": "PCE"},
    "X16": {"name": "PCE Mw", "unit": "g/mol", "group": "PCE"},
    "X17": {"name": "PCE PDI", "unit": "-", "group": "PCE"},
    "X18": {"name": "PCE viscosity", "unit": "dataset scale", "group": "PCE"},
    "X19": {"name": "Cement", "unit": "g", "group": "mixture"},
    "X20": {"name": "Fly ash", "unit": "g", "group": "mixture"},
    "X21": {"name": "PCE dosage", "unit": "g", "group": "mixture"},
    "X22": {"name": "Water", "unit": "g", "group": "mixture"},
    "Y1": {"name": "Marsh funnel flow time", "unit": "s", "group": "target"},
    "Y2": {"name": "Mini-slump spread", "unit": "cm", "group": "target"},
}

# admixture codes whose descriptors and Y1 records duplicate another admixture
DUPLICATE_PAIRS: Dict[str, str] = {"K3:1": "K", "S5-3:1": "S5", "F9-3:1": "F9"}


def label(var: str, with_unit: bool = True) -> str:
    info = VARIABLE_INFO.get(var, {"name": var, "unit": ""})
    if with_unit and info["unit"] not in ("", "-", "dataset scale"):
        return f"{var} {info['name']} ({info['unit']})"
    return f"{var} {info['name']}"


def pce_code(mix_id: str) -> str:
    s = re.sub(r"^UK\d+-?", "", str(mix_id))
    return re.sub(r"-[\d.,]+$", "", s)


def load_raw(path=DATASET_PATH) -> pd.DataFrame:
    """Original dataset with '-' converted to NaN (no other change)."""
    df = pd.read_csv(path)
    df = df.replace(r"^\s*-\s*$", np.nan, regex=True)
    for c in FEATURES + ["Y1", "Y2"]:
        df[c] = pd.to_numeric(df[c])
    df = df.rename(columns={"index": "mix_id"})
    return df


@dataclass
class TargetData:
    target: str
    X: pd.DataFrame          # all 22 inputs, original units
    y: pd.Series
    mix_id: pd.Series
    pce: pd.Series           # admixture code
    pce_group: pd.Series     # admixture group (duplicate pairs merged)
    fa_level: pd.Series      # fly-ash replacement level (%)
    measured: pd.Series      # True where the target value was actually measured


# Data variants
#   full        corrected data (main analysis)
#   nodup       parent admixture of each near-duplicate pair removed (R1 #18)
#   mixonly     baseline without any admixture information: X19, X20, X21 only (R1 #19, #33)
#   pceid       admixture identity (one-hot code) instead of the six descriptors (R1 #19)
#   mice_global original procedure, for comparison only: IterativeImputer fitted once on the
#               whole table including Y1 and Y2; Y2 imputed for all 616 rows (R2 #1, R1 #27)
DATA_VARIANTS = ["full", "nodup", "mixonly", "pceid", "mice_global"]
MIX_FEATURES = ["X19", "X20", "X21"]


def load_corrected(path=DATASET_PATH) -> pd.DataFrame:
    df = load_raw(path)
    df["fly_ash_pct"] = df["mix_id"].str.extract(r"^UK(\d+)")[0].astype(int)
    # Correction 1: '-' in X20 means no fly ash (all such rows are UK0)
    assert (df.loc[df["X20"].isna(), "fly_ash_pct"] == 0).all()
    df["X20"] = df["X20"].fillna(0.0)
    df["pce"] = df["mix_id"].map(pce_code)
    df["pce_group"] = df["pce"].map(lambda c: DUPLICATE_PAIRS.get(c, c))
    df["pce_group"] = df["pce_group"].map(
        lambda g: f"{g}|{[k for k, v in DUPLICATE_PAIRS.items() if v == g][0]}"
        if g in DUPLICATE_PAIRS.values() else g)
    return df


def get_target_data(target: str, data_variant: str = "full",
                    path=DATASET_PATH) -> TargetData:
    df = load_corrected(path)
    if data_variant not in DATA_VARIANTS:
        raise ValueError(data_variant)
    df["measured"] = df[target].notna()
    if data_variant == "mice_global":
        # Reproduction of the ORIGINAL procedure (src/data/imputer.py): one IterativeImputer
        # (default BayesianRidge, max_iter=10) fitted on the raw table incl. X20 '-' and both targets.
        from sklearn.experimental import enable_iterative_imputer  # noqa: F401
        from sklearn.impute import IterativeImputer

        raw = load_raw(path)
        cols = FEATURES + ["Y1", "Y2"]
        imp = IterativeImputer(random_state=42, max_iter=10).fit_transform(raw[cols])
        imp = pd.DataFrame(imp, columns=cols)
        for c in cols:
            df[c] = imp[c].values
        df = df.reset_index(drop=True)
        X = df[FEATURES].astype(float)
    else:
        if data_variant == "nodup":
            # drop the parent admixture of each duplicate pair (parents have no Y2)
            df = df[~df["pce"].isin(list(DUPLICATE_PAIRS.values()))]
        # Correction 2: no target imputation
        df = df[df[target].notna()].reset_index(drop=True)
        if data_variant == "mixonly":
            X = df[MIX_FEATURES].astype(float)
        elif data_variant == "pceid":
            onehot = pd.get_dummies(df["pce"], prefix="PCE").astype(float)
            X = pd.concat([df[MIX_FEATURES].astype(float), onehot], axis=1)
        else:
            X = df[FEATURES].astype(float)
    return TargetData(
        target=target,
        X=X,
        y=df[target].astype(float),
        mix_id=df["mix_id"],
        pce=df["pce"],
        pce_group=df["pce_group"],
        fa_level=df["fly_ash_pct"],
        measured=df["measured"].astype(bool),
    )
