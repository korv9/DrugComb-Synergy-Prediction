"""Study labels and single-agent (monotherapy) responses from the dose-response matrices.

``drugcombs_response.csv`` holds every measured well of every combination block:
``BlockID, Row, Col, DrugRow, DrugCol, ConcRow, ConcCol, Response, …, source``.
``BlockID`` equals ``ID`` in ``drugcombs_scored.csv``.

For each block this stage extracts

* the source study (ONEIL, ALMANAC, …) - screens differ in protocol, dose
  ranges and readout, so the study is a strong nuisance variable;
* each drug's single-agent response *in that block* (the wells where the
  partner's concentration is 0): mean and maximum % inhibition over its doses.

Responses are reported as % viability by some studies and % inhibition by
others; the untreated well (both concentrations 0) tells which, per study.

Single-agent responses are *inputs*, not labels: synergy scores measure how
far the combination deviates from what the single agents predict. Using them
assumes the single agents were screened in the cell line, which is the usual
setting for combination screens (e.g. the AZ-DREAM challenge).
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .config import Paths

log = logging.getLogger(__name__)

MONO_COLS = ["mono_inh_mean", "mono_inh_max"]


def read_response(path) -> pd.DataFrame:
    cols = ["BlockID", "ConcRow", "ConcCol", "Response", "source"]
    df = pd.read_csv(path, usecols=cols, low_memory=False,
                     dtype={"source": "category"}, on_bad_lines="warn")
    for c in ("BlockID", "ConcRow", "ConcCol", "Response"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(subset=["BlockID", "Response"]).astype({"BlockID": "int64"})


def to_inhibition(resp: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame]:
    """Return (% inhibition per well, per-study readout table)."""
    ctrl = resp[(resp["ConcRow"] == 0) & (resp["ConcCol"] == 0)]
    baseline = ctrl.groupby("source")["Response"].median()
    viability = baseline > 50  # untreated wells ~100 % viable vs ~0 % inhibited
    readout = pd.DataFrame({"untreated_median": baseline,
                            "readout": np.where(viability, "viability", "inhibition")})
    is_via = resp["source"].map(viability).fillna(True).to_numpy()
    inh = np.where(is_via, 100 - resp["Response"].to_numpy(), resp["Response"].to_numpy())
    return pd.Series(inh, index=resp.index), readout.reset_index()


def block_features(resp: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """One row per block: study + single-agent summaries for the row and column drug."""
    inh, readout = to_inhibition(resp)
    resp = resp.assign(inh=inh.clip(-50, 150))

    study = resp.groupby("BlockID", observed=True)["source"].first().astype(str).rename("study")
    parts = [study]
    for side, alone in [("row", (resp["ConcCol"] == 0) & (resp["ConcRow"] > 0)),
                        ("col", (resp["ConcRow"] == 0) & (resp["ConcCol"] > 0))]:
        conc = "ConcRow" if side == "row" else "ConcCol"
        mono = resp.loc[alone, ["BlockID", conc, "inh"]]
        g = mono.groupby("BlockID")
        top = mono.sort_values(conc).groupby("BlockID")["inh"].last()
        parts += [g["inh"].mean().rename(f"mono_inh_mean_{side}"),
                  top.rename(f"mono_inh_max_{side}")]
    blocks = pd.concat(parts, axis=1)
    blocks.index.name = "block_id"
    return blocks.reset_index(), readout


def run(cfg: dict, paths: Paths) -> None:
    paths.ensure()
    src = paths.raw_file("drugcombdb", cfg["sources"]["drugcombdb"]["files"]["response"])
    if not src.exists():
        log.warning("%s missing - skipping study/monotherapy features", src)
        return
    log.info("reading %s", src)
    resp = read_response(src)
    log.info("%d wells in %d blocks", len(resp), resp["BlockID"].nunique())
    blocks, readout = block_features(resp)
    blocks.to_parquet(paths.processed / "block_monotherapy.parquet", index=False)
    readout.to_csv(paths.tables / "study_readout.csv", index=False)
    log.info("block features: %d blocks; studies: %s", len(blocks),
             blocks["study"].value_counts().to_dict())
