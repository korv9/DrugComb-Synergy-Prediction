"""Build the modelled fact table: one row per (drug pair, cell line).

Measurements are re-keyed on the resolved drug/cell entities *before*
aggregation, so synonyms of the same molecule are pooled and pairs such as
"5-FU + fluorouracil" are recognised as self-combinations.
"""

from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd

from .config import Paths
from .ingest import SCORES

log = logging.getLogger(__name__)

KEY = ["drug_1", "drug_2", "cell_key", "study"]
MONO = ["mono_inh_mean", "mono_inh_max"]


def build_fact(meas: pd.DataFrame, bridge: pd.DataFrame, dim_cell: pd.DataFrame, cutoff: float,
               blocks: pd.DataFrame | None = None):
    """Return (fact, replicate_pairs, funnel_steps).

    ``blocks`` (from the monotherapy stage) adds the source study and the
    single-agent responses of Drug1 (matrix row) and Drug2 (matrix column).
    """
    name2key = dict(zip(bridge["name_norm"], bridge["drug_key"]))
    df = meas.assign(key_a=meas["drug_a"].map(name2key), key_b=meas["drug_b"].map(name2key))
    if blocks is not None:
        df = df.merge(blocks, on="block_id", how="left")
    else:
        df["study"] = np.nan
    df["study"] = df["study"].astype(object).fillna("unknown").astype(str)
    for m in MONO:
        for side in ("row", "col"):
            if f"{m}_{side}" not in df:
                df[f"{m}_{side}"] = np.nan
    funnel = []

    df = df[df["key_a"] != df["key_b"]]
    funnel.append({"step": "two different molecules", "rows": len(df)})

    excluded = set(dim_cell.loc[dim_cell["match_method"] == "excluded_non_human", "cell_key"])
    df = df[~df["cell_key"].isin(excluded)]
    funnel.append({"step": "human cell lines only", "rows": len(df)})

    swap = df["key_a"] > df["key_b"]
    df = df.assign(
        drug_1=np.where(swap, df["key_b"], df["key_a"]),
        drug_2=np.where(swap, df["key_a"], df["key_b"]),
        **{f"{m}_1": np.where(swap, df[f"{m}_col"], df[f"{m}_row"]) for m in MONO},
        **{f"{m}_2": np.where(swap, df[f"{m}_row"], df[f"{m}_col"]) for m in MONO},
    )

    agg = {s: (s, "mean") for s in SCORES}
    agg.update({f"{m}_{i}": (f"{m}_{i}", "mean") for m in MONO for i in (1, 2)})
    fact = df.groupby(KEY, sort=False).agg(
        **agg,
        zip_std=("zip", "std"),
        n_replicates=("zip", "size"),
    ).reset_index()
    fact["pair_key"] = fact["drug_1"] + "|" + fact["drug_2"]
    fact["synergy_class"] = pd.cut(
        fact["zip"], [-np.inf, -cutoff, cutoff, np.inf],
        labels=["antagonistic", "additive", "synergistic"],
    ).astype(str)
    funnel.append({"step": "unique (pair, cell line, study) after replicate mean",
                   "rows": len(fact)})

    reps = (
        df.sort_values("measurement_id")
        .assign(rep=lambda d: d.groupby(KEY).cumcount())
        .query("rep < 2")
        .pivot_table(index=KEY, columns="rep", values="zip")
        .dropna()
        .rename(columns={0: "zip_rep1", 1: "zip_rep2"})
        .reset_index()
    )
    reps.columns.name = None
    return fact, reps, funnel


def quality_checks(fact: pd.DataFrame, dim_drug: pd.DataFrame,
                   dim_cell: pd.DataFrame) -> pd.DataFrame:
    """Declarative data-quality expectations; failures are reported, not raised."""
    checks = [
        ("fact grain is unique (drug_1, drug_2, cell_key, study)",
         not fact.duplicated(KEY).any(), f"{int(fact.duplicated(KEY).sum())} duplicates"),
        ("drug pairs are canonically ordered", bool((fact["drug_1"] < fact["drug_2"]).all()), ""),
        ("no self-combinations", bool((fact["drug_1"] != fact["drug_2"]).all()), ""),
        ("ZIP has no missing values", bool(fact["zip"].notna().all()),
         f"{int(fact['zip'].isna().sum())} missing"),
        ("every drug_key exists in dim_drug",
         bool(pd.concat([fact["drug_1"], fact["drug_2"]]).isin(dim_drug["drug_key"]).all()), ""),
        ("every cell_key exists in dim_cell",
         bool(fact["cell_key"].isin(dim_cell["cell_key"]).all()), ""),
        ("dim_drug key is unique", not dim_drug["drug_key"].duplicated().any(), ""),
        ("dim_cell key is unique", not dim_cell["cell_key"].duplicated().any(), ""),
    ]
    struct = dim_drug.set_index("drug_key")["has_structure"]
    both = fact["drug_1"].map(struct) & fact["drug_2"].map(struct)
    checks.append(("rows with structures for both drugs", True, f"{both.mean():.1%}"))
    rna = dim_cell.set_index("cell_key")["has_rna"] if "has_rna" in dim_cell else None
    if rna is not None:
        checks.append(("rows with DepMap RNA for the cell line", True,
                       f"{fact['cell_key'].map(rna).fillna(False).mean():.1%}"))
    return pd.DataFrame(checks, columns=["check", "passed", "detail"])


def entity_resolution_summary(dim_drug: pd.DataFrame, bridge: pd.DataFrame,
                              dim_cell: pd.DataFrame) -> pd.DataFrame:
    drug = bridge.groupby("resolution").size().rename("names").to_frame()
    drug["measurement_share"] = (
        dim_drug.groupby("resolution")["n_measurements"].sum() / dim_drug["n_measurements"].sum()
    )
    drug = drug.reset_index().rename(columns={"resolution": "method"})
    drug.insert(0, "entity", "drug")

    cell = dim_cell.groupby("match_method").agg(
        names=("cell_key", "size"), measurements=("n_measurements", "sum")).reset_index()
    cell["measurement_share"] = cell.pop("measurements") / dim_cell["n_measurements"].sum()
    cell = cell.rename(columns={"match_method": "method"})
    cell.insert(0, "entity", "cell_line")
    return pd.concat([drug, cell], ignore_index=True).fillna({"measurement_share": 0.0})


def run(cfg: dict, paths: Paths) -> None:
    paths.ensure()
    meas = pd.read_parquet(paths.measurements)
    bridge = pd.read_parquet(paths.processed / "bridge_drug_name.parquet")
    dim_drug = pd.read_parquet(paths.dim_drug)
    dim_cell = pd.read_parquet(paths.dim_cell)

    block_path = paths.processed / "block_monotherapy.parquet"
    blocks = pd.read_parquet(block_path) if block_path.exists() else None
    if blocks is None:
        log.warning("%s missing - no study labels or monotherapy features", block_path)
    fact, reps, funnel = build_fact(meas, bridge, dim_cell, cfg["clean"]["synergy_cutoff"],
                                    blocks)
    fact.to_parquet(paths.fact, index=False)
    reps.to_parquet(paths.replicates, index=False)

    ingest_funnel = json.loads((paths.interim / "funnel_ingest.json").read_text())
    pd.DataFrame(ingest_funnel + funnel).to_csv(paths.tables / "data_funnel.csv", index=False)
    qc = quality_checks(fact, dim_drug, dim_cell)
    qc.to_csv(paths.tables / "data_quality.csv", index=False)
    entity_resolution_summary(dim_drug, bridge, dim_cell).to_csv(
        paths.tables / "entity_resolution.csv", index=False)

    log.info("fact_combination: %d rows, %d pairs, %d replicate pairs",
             len(fact), fact["pair_key"].nunique(), len(reps))
    for _, row in qc[~qc["passed"]].iterrows():
        log.warning("QC FAILED: %s (%s)", row["check"], row["detail"])
