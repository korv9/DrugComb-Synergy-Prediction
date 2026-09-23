"""Feature engineering.

Two kinds of features, kept strictly apart:

* **static features** depend only on the drugs' structures and the cell line's
  biology - never on synergy labels - so they are computed once for all rows.
  All pair features are symmetric (sum / |difference| / similarity), so the
  arbitrary order of drug_1 and drug_2 carries no information.
* **screen-history features** (smoothed mean ZIP per drug, cell line, pair,
  drug x cell, plus how often each was screened) *do* use labels. They are
  computed inside each training fold, out-of-fold for the training rows
  themselves, so the test fold never leaks into them.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from .drugs import DESCRIPTORS

FAMILIES = {
    "fps_": "Chemistry: fingerprint bits",
    "chem_": "Chemistry: descriptors & similarity",
    "rna_pc": "Biology: RNA expression PCs",
    "lineage": "Biology: tissue lineage",
    "te_": "Screen history: in-fold mean ZIP",
    "n_": "Screen history: in-fold counts",
}


def family(col: str) -> str:
    return next((v for k, v in FAMILIES.items() if col.startswith(k)), "Other")


def tanimoto(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    inter = np.minimum(a, b).sum(1).astype(np.float32)
    union = np.maximum(a, b).sum(1).astype(np.float32)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(union > 0, inter / union, np.nan)


def static_features(fact: pd.DataFrame, dim_drug: pd.DataFrame, fps: pd.DataFrame,
                    dim_cell: pd.DataFrame, rna: pd.DataFrame,
                    min_bit_freq: float = 0.01) -> pd.DataFrame:
    """Label-free features for every row of the fact table."""
    n = len(fact)
    out = {}

    # --- chemistry: fingerprints (bits set in >= 1% and <= 99% of drugs)
    fp = fps.set_index("drug_key")
    freq = fp.mean()
    keep = freq[(freq >= min_bit_freq) & (freq <= 1 - min_bit_freq)].index
    fp = fp[keep]
    pos = pd.Series(np.arange(len(fp)), index=fp.index)
    p1 = fact["drug_1"].map(pos).to_numpy()
    p2 = fact["drug_2"].map(pos).to_numpy()
    have = ~(np.isnan(p1) | np.isnan(p2))
    mat = fp.to_numpy(np.uint8)
    a = np.zeros((n, mat.shape[1]), np.uint8)
    b = np.zeros_like(a)
    a[have] = mat[p1[have].astype(int)]
    b[have] = mat[p2[have].astype(int)]
    fp_sum = (a + b).astype(np.float32)
    fp_sum[~have] = np.nan
    fp_cols = [f"fps_{c.removeprefix('fp_')}" for c in keep]

    # --- chemistry: descriptors (symmetric) + structural similarity
    sim = tanimoto(a, b)
    sim[~have] = np.nan
    out["chem_tanimoto"] = sim
    desc = dim_drug.set_index("drug_key")[DESCRIPTORS]
    d1 = desc.reindex(fact["drug_1"]).to_numpy(np.float32)
    d2 = desc.reindex(fact["drug_2"]).to_numpy(np.float32)
    for i, name in enumerate(DESCRIPTORS):
        out[f"chem_{name}_sum"] = d1[:, i] + d2[:, i]
        out[f"chem_{name}_absdiff"] = np.abs(d1[:, i] - d2[:, i])

    # --- biology: RNA PCs + lineage
    cell = dim_cell.set_index("cell_key")
    rna_idx = rna.set_index("model_id")
    rna_rows = rna_idx.reindex(fact["cell_key"].map(cell["model_id"]))
    for col in rna_idx.columns:
        out[col] = rna_rows[col].to_numpy(np.float32)
    out["lineage"] = pd.Categorical(fact["cell_key"].map(cell["lineage"]).fillna("Unknown"))

    feats = pd.DataFrame(out, index=fact.index)
    return pd.concat([feats, pd.DataFrame(fp_sum, columns=fp_cols, index=fact.index)], axis=1)


# ----------------------------------------------------------------- screen history
def _smoothed(keys: np.ndarray, y: np.ndarray, prior: float, m: float):
    g = pd.DataFrame({"k": keys, "y": y}).groupby("k")["y"].agg(["sum", "count"])
    return (g["sum"] + m * prior) / (g["count"] + m), g["count"]


def _history_keys(df: pd.DataFrame) -> dict[str, tuple[np.ndarray, ...]]:
    d1, d2, c = df["drug_1"].to_numpy(), df["drug_2"].to_numpy(), df["cell_key"].to_numpy()
    return {
        "drug": (d1, d2),
        "cell": (c,),
        "pair": (df["pair_key"].to_numpy(),),
        "drugcell": (d1 + "@" + c, d2 + "@" + c),
    }


def history_features(train: pd.DataFrame, y_train: np.ndarray, apply: pd.DataFrame,
                     m: float) -> pd.DataFrame:
    """Smoothed target means / counts learned on ``train`` and mapped onto ``apply``."""
    prior = float(np.mean(y_train))
    tk, ak = _history_keys(train), _history_keys(apply)
    out = {}
    for name, cols in tk.items():
        keys = np.concatenate(cols)
        ys = np.tile(y_train, len(cols))
        mean, count = _smoothed(keys, ys, prior, m)
        mapped = [pd.Series(k).map(mean).fillna(prior).to_numpy(np.float32) for k in ak[name]]
        counts = [np.log1p(pd.Series(k).map(count).fillna(0).to_numpy(np.float32))
                  for k in ak[name]]
        if len(mapped) == 1:
            out[f"te_{name}"] = mapped[0]
            out[f"n_{name}_log"] = counts[0]
        else:
            out[f"te_{name}_min"] = np.minimum(*mapped)
            out[f"te_{name}_max"] = np.maximum(*mapped)
            out[f"n_{name}_min_log"] = np.minimum(*counts)
            out[f"n_{name}_max_log"] = np.maximum(*counts)
    return pd.DataFrame(out, index=apply.index)


def history_features_oof(train: pd.DataFrame, y_train: np.ndarray, m: float,
                         n_splits: int = 5, seed: int = 0) -> pd.DataFrame:
    """Out-of-fold history features for the training rows themselves."""
    parts = []
    for fit_idx, oof_idx in KFold(n_splits, shuffle=True, random_state=seed).split(train):
        parts.append(history_features(train.iloc[fit_idx], y_train[fit_idx],
                                      train.iloc[oof_idx], m))
    return pd.concat(parts).loc[train.index]
