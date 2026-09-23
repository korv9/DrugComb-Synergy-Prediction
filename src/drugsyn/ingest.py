"""Stage raw DrugCombDB measurements into a typed, normalised staging table."""

from __future__ import annotations

import json
import logging
import re
import unicodedata

import numpy as np
import pandas as pd

from .config import Paths

log = logging.getLogger(__name__)

SCORES = ["zip", "bliss", "loewe", "hsa"]

# DrugCombDB header -> staging name (matched case-insensitively, ignoring spaces)
COLUMN_ALIASES = {
    "drug1": "drug_a_raw",
    "drug2": "drug_b_raw",
    "cellline": "cell_raw",
    "zip": "zip",
    "bliss": "bliss",
    "loewe": "loewe",
    "hsa": "hsa",
}

_DASHES = dict.fromkeys(map(ord, "‐‑‒–—−"), "-")
_ZERO_WIDTH = dict.fromkeys(map(ord, "​‌‍﻿"), None)


def norm_drug(name) -> str | None:
    """Lower-case, NFKC, unified dashes, collapsed whitespace."""
    if name is None or (isinstance(name, float) and np.isnan(name)):
        return None
    s = unicodedata.normalize("NFKC", str(name)).translate(_DASHES).translate(_ZERO_WIDTH)
    s = " ".join(s.strip().lower().split())
    return s or None


def norm_cell(name) -> str | None:
    """Cell-line key: lower-case alphanumerics only ("NCI-H460" -> "ncih460")."""
    if name is None or (isinstance(name, float) and np.isnan(name)):
        return None
    s = re.sub(r"[^a-z0-9]+", "", unicodedata.normalize("NFKC", str(name)).lower())
    return s or None


def _rename(df: pd.DataFrame) -> pd.DataFrame:
    mapping = {}
    for col in df.columns:
        key = re.sub(r"\s+", "", str(col)).lower()
        if key in COLUMN_ALIASES:
            mapping[col] = COLUMN_ALIASES[key]
    missing = {"drug_a_raw", "drug_b_raw", "cell_raw", "zip"} - set(mapping.values())
    if missing:
        raise ValueError(f"raw file lacks required columns {missing}; got {list(df.columns)}")
    return df.rename(columns=mapping)[list(mapping.values())]


def stage_measurements(raw: pd.DataFrame, zip_range=(-100, 100)) -> tuple[pd.DataFrame, list]:
    """Return (staged measurements, funnel steps)."""
    funnel = [{"step": "raw rows", "rows": len(raw)}]
    df = _rename(raw).copy()
    df.insert(0, "measurement_id", np.arange(len(df), dtype=np.int64))

    for col in SCORES:
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")
        else:
            df[col] = np.nan

    df["drug_a"] = df["drug_a_raw"].map(norm_drug)
    df["drug_b"] = df["drug_b_raw"].map(norm_drug)
    df["cell_key"] = df["cell_raw"].map(norm_cell)

    df = df.dropna(subset=["drug_a", "drug_b", "cell_key"])
    funnel.append({"step": "has drug & cell names", "rows": len(df)})

    df = df[np.isfinite(df["zip"])]
    funnel.append({"step": "numeric ZIP score", "rows": len(df)})

    lo, hi = zip_range
    df = df[df["zip"].between(lo, hi)]
    funnel.append({"step": f"ZIP within [{lo}, {hi}]", "rows": len(df)})

    for col in ("drug_a_raw", "drug_b_raw", "cell_raw", "drug_a", "drug_b", "cell_key"):
        df[col] = df[col].astype(str)
    return df.reset_index(drop=True), funnel


def run(cfg: dict, paths: Paths) -> None:
    paths.ensure()
    src = paths.raw_file("drugcombdb", cfg["sources"]["drugcombdb"]["files"]["scored"])
    log.info("reading %s", src)
    raw = pd.read_csv(src, low_memory=False)
    staged, funnel = stage_measurements(raw, tuple(cfg["clean"]["zip_plausible_range"]))
    staged.to_parquet(paths.measurements, index=False)
    (paths.interim / "funnel_ingest.json").write_text(json.dumps(funnel, indent=2))
    log.info("staged %d measurements -> %s", len(staged), paths.measurements)
