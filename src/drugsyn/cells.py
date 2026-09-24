"""Cell-line entity resolution against DepMap ``Model.csv`` + RNA expression PCs."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .config import REPO_ROOT, Paths
from .ingest import norm_cell

log = logging.getLogger(__name__)

ALIASES = REPO_ROOT / "configs" / "cell_line_aliases.csv"
EXCLUDE = "EXCLUDE"


def model_lookup(models: pd.DataFrame) -> pd.DataFrame:
    """Every normalised name a DepMap model is known by -> ModelID.

    Keys are ranked (stripped name > cell-line name > CCLE name prefix > CCLE
    name) so that an ambiguous key resolves to its most specific source.
    """
    rows = []
    for rank, col, fn in [
        (0, "StrippedCellLineName", norm_cell),
        (1, "CellLineName", norm_cell),
        (2, "CCLEName", lambda s: norm_cell(str(s).split("_")[0]) if isinstance(s, str) else None),
        (3, "CCLEName", norm_cell),
    ]:
        if col in models:
            rows.append(pd.DataFrame({
                "key": models[col].map(fn), "ModelID": models["ModelID"], "rank": rank,
            }))
    lut = pd.concat(rows).dropna(subset=["key"])
    return lut.sort_values(["key", "rank", "ModelID"]).drop_duplicates("key")


def match_cells(cells: pd.Series, models: pd.DataFrame, aliases: pd.DataFrame) -> pd.DataFrame:
    """cells: measurement counts indexed by normalised cell key."""
    lut = model_lookup(models).set_index("key")["ModelID"]
    alias = dict(zip(aliases["alias"].map(norm_cell), aliases["target"]))

    out = []
    for key, n in cells.items():
        target, method = key, "exact"
        if key in alias:
            target = alias[key]
            method = "alias"
        if target == EXCLUDE:
            out.append((key, n, None, "excluded_non_human"))
            continue
        model = lut.get(norm_cell(target))
        out.append((key, n, model, method if model is not None else "unmatched"))
    dim = pd.DataFrame(out, columns=["cell_key", "n_measurements", "model_id", "match_method"])

    meta_cols = [c for c in ("ModelID", "CellLineName", "OncotreeLineage",
                             "OncotreePrimaryDisease", "Sex", "PrimaryOrMetastasis")
                 if c in models]
    meta = models[meta_cols].rename(columns={
        "ModelID": "model_id", "CellLineName": "depmap_name", "OncotreeLineage": "lineage",
        "OncotreePrimaryDisease": "primary_disease", "Sex": "sex",
        "PrimaryOrMetastasis": "primary_or_metastasis",
    })
    dim = dim.merge(meta, on="model_id", how="left")
    dim["lineage"] = dim.get("lineage", pd.Series(index=dim.index, dtype=object)).fillna("Unknown")
    return dim


def load_expression(path, top_var_genes: int) -> pd.DataFrame:
    """Load DepMap log2(TPM+1) expression as a models x genes float32 frame."""
    header = pd.read_csv(path, nrows=0).columns
    meta = {"SequencingID", "ModelID", "IsDefaultEntryForMC", "ModelConditionID",
            "IsDefaultEntryForModel"}
    expr = pd.read_csv(path, engine="pyarrow")
    if "ModelID" in header:
        if "IsDefaultEntryForModel" in expr:
            default = expr["IsDefaultEntryForModel"].astype(str).str.lower()
            expr = expr[default.isin(["yes", "true"])]
        expr = expr.drop_duplicates("ModelID").set_index("ModelID")
    else:  # older releases: first (unnamed) column holds the ModelID
        expr = expr.set_index(expr.columns[0])
    genes = [c for c in expr.columns if c not in meta and not str(c).startswith("Unnamed")]
    expr = expr[genes].apply(pd.to_numeric, errors="coerce").astype(np.float32)
    expr = expr.loc[:, expr.notna().mean() > 0.95].fillna(expr.mean())
    top = expr.var().nlargest(min(top_var_genes, expr.shape[1])).index
    expr.index = expr.index.astype(str)
    return expr[top]


def rna_pcs(expr: pd.DataFrame, n_components: int,
            seed: int = 0) -> tuple[pd.DataFrame, np.ndarray]:
    """PCA on all DepMap models (unsupervised, external to the synergy labels).

    Fitting on every DepMap model rather than only the screened ones gives a
    more stable embedding and cannot leak synergy information.
    """
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    n = min(n_components, *expr.shape)
    x = StandardScaler().fit_transform(expr.values)
    pca = PCA(n_components=n, random_state=seed).fit(x)
    pcs = pd.DataFrame(pca.transform(x).astype(np.float32), index=expr.index,
                       columns=[f"rna_pc{i + 1}" for i in range(n)])
    pcs.index.name = "model_id"
    return pcs.reset_index(), pca.explained_variance_ratio_


def run(cfg: dict, paths: Paths) -> None:
    paths.ensure()
    files = cfg["sources"]["depmap"]["files"]
    meas = pd.read_parquet(paths.measurements, columns=["cell_key"])
    models = pd.read_csv(paths.raw_file("depmap", files["models"]), low_memory=False)
    aliases = pd.read_csv(ALIASES, dtype=str).fillna("")

    dim = match_cells(meas["cell_key"].value_counts(), models, aliases)
    raw_names = pd.read_parquet(paths.measurements, columns=["cell_key", "cell_raw"])
    display = raw_names.groupby("cell_key")["cell_raw"].agg(lambda s: s.mode().iat[0])
    dim.insert(1, "cell_name", dim["cell_key"].map(display))

    expr = load_expression(paths.raw_file("depmap", files["expression"]),
                           cfg["features"]["rna_top_var_genes"])
    pcs, evr = rna_pcs(expr, cfg["features"]["rna_pca_components"], cfg["model"]["seed"])
    dim["has_rna"] = dim["model_id"].isin(pcs["model_id"])

    # cell-line landscape for the figure: every DepMap model with lineage
    landscape = pcs[["model_id", "rna_pc1", "rna_pc2"]].merge(
        models[["ModelID", "OncotreeLineage"]].rename(
            columns={"ModelID": "model_id", "OncotreeLineage": "lineage"}),
        on="model_id", how="left")
    landscape["screened"] = landscape["model_id"].isin(dim["model_id"])

    dim.to_parquet(paths.dim_cell, index=False)
    pcs.to_parquet(paths.cell_rna_pcs, index=False)
    landscape.to_parquet(paths.processed / "cell_landscape.parquet", index=False)
    pd.DataFrame({"component": np.arange(1, len(evr) + 1), "explained_variance": evr}).to_csv(
        paths.tables / "rna_pca_explained_variance.csv", index=False)
    log.info("dim_cell: %d cell lines, %d matched to DepMap, %d with RNA",
             len(dim), int(dim["model_id"].notna().sum()), int(dim["has_rna"].sum()))
