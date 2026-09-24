"""Drug targets (STITCH, via DrugCombDB) and their state in each cell line (DepMap).

A drug only acts through its targets, so the most informative biology is not
the cell line's whole transcriptome but *what the cell line looks like at the
drug's targets*. For every (drug, DepMap model) this stage computes

* ``tgt_expr``  - mean expression z-score of the drug's targets,
* ``tgt_dep``   - strongest CRISPR dependency among the targets (Chronos gene
  effect; < -0.5 means the cell needs that gene),
* ``tgt_mut``   - whether any target carries a damaging mutation,

and, per drug, a low-dimensional **mechanism embedding** (truncated SVD of the
drug x target matrix) so that drugs hitting similar proteins sit close even
when their chemistry differs. None of these use synergy labels.
"""

from __future__ import annotations

import json
import logging
import re

import numpy as np
import pandas as pd
import pyarrow.csv as pacsv
import requests

from .config import Paths

log = logging.getLogger(__name__)

MYGENE = "https://mygene.info/v3/query"


# ----------------------------------------------------------------------------- STITCH
def read_links(path, min_score: int) -> pd.DataFrame:
    """High-confidence links backed by experiments or curated databases."""
    t = pacsv.read_csv(path, parse_options=pacsv.ParseOptions(delimiter="\t")).to_pandas()
    keep = (t["combined_score"] >= min_score) & ((t["experimental"] > 0) | (t["database"] > 0))
    t = t.loc[keep, ["chemical", "protein", "combined_score"]]
    t["cid"] = t["chemical"].str.extract(r"CID[ms]0*(\d+)")[0].astype("int64")
    t["ensp"] = t["protein"].str.replace("9606.", "", regex=False)
    return t[["cid", "ensp", "combined_score"]]


def ensp_to_symbol(ensps: list[str], cache_path) -> dict[str, str | None]:
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    todo = [e for e in dict.fromkeys(ensps) if e not in cache]
    for i in range(0, len(todo), 1000):
        chunk = todo[i:i + 1000]
        r = requests.post(MYGENE, data={"q": ",".join(chunk), "scopes": "ensembl.protein",
                                        "fields": "symbol", "species": "human"}, timeout=120)
        r.raise_for_status()
        for hit in r.json():
            if hit.get("symbol") and not hit.get("notfound"):
                cache.setdefault(hit["query"], hit["symbol"])
        for e in chunk:
            cache.setdefault(e, None)
    cache_path.write_text(json.dumps(cache))
    return {e: cache.get(e) for e in ensps}


def drug_cids(dim_drug: pd.DataFrame, bridge: pd.DataFrame, pubchem_cache) -> pd.DataFrame:
    """All PubChem CIDs we know for each drug (DrugCombDB's and name look-ups)."""
    rows = [dim_drug[["drug_key", "cid"]].dropna()]
    if pubchem_cache.exists():
        name2cid = json.loads(pubchem_cache.read_text()).get("name2cid", {})
        b = bridge.assign(cid=bridge["name_norm"].map(name2cid))
        rows.append(b[["drug_key", "cid"]].dropna())
    out = pd.concat(rows)
    out["cid"] = out["cid"].astype("int64")
    return out.drop_duplicates()


def same_connectivity(cids: list[int], pubchem) -> dict[int, list[int]]:
    """CIDs sharing each CID's atom connectivity (stereoisomers, parent/flat forms).

    STITCH keys chemicals by the flat or the stereo CID, which often differ
    from the CID a drug name resolves to (e.g. ruxolitinib).
    """
    from concurrent.futures import ThreadPoolExecutor

    cache = pubchem.cache.setdefault("cid2conn", {})
    todo = [c for c in dict.fromkeys(cids) if str(c) not in cache]

    def fetch(cid):
        r = pubchem._get(f"{pubchem.base}/compound/fastidentity/cid/{cid}/cids/JSON"
                          "?identity_type=same_connectivity")
        if r is not None and r.status_code in (200, 404):
            found = r.json().get("IdentifierList", {}).get("CID", []) if r.status_code == 200 \
                else []
            with pubchem._lock:
                cache[str(cid)] = found[:50]

    log.info("PubChem connectivity look-up: %d cached, %d to query", len(cids) - len(todo),
             len(todo))
    with ThreadPoolExecutor(pubchem.workers) as pool:
        list(pool.map(fetch, todo))
    pubchem.save()
    return {c: cache.get(str(c), []) for c in cids}


# ----------------------------------------------------------------------------- DepMap
def _gene_symbol(col: str) -> str:
    return re.sub(r"\s*\(\d+\)$", "", str(col))


def load_gene_matrix(path, genes: set[str]) -> pd.DataFrame:
    """Models x genes, restricted to ``genes``; ModelID taken from the first column."""
    header = pd.read_csv(path, nrows=0).columns
    idx = [i for i, c in enumerate(header) if i > 0 and _gene_symbol(c) in genes]
    dtypes = {header[i]: np.float32 for i in idx}
    t = pd.read_csv(path, usecols=[0, *idx], index_col=0, dtype=dtypes)
    t.index = t.index.astype(str)
    t.columns = [_gene_symbol(c) for c in t.columns]
    return t.T.groupby(level=0).mean().T.astype(np.float32)


def drug_cell_features(targets: pd.DataFrame, models: list[str], expr: pd.DataFrame,
                       dep: pd.DataFrame, mut: pd.DataFrame | None) -> pd.DataFrame:
    """Per (drug, model): target expression / dependency / mutation summaries."""
    z = (expr - expr.mean()) / expr.std().replace(0, np.nan)
    out = []
    for drug, genes in targets.groupby("drug_key")["gene"]:
        genes = list(dict.fromkeys(genes))
        rec = pd.DataFrame(index=pd.Index(models, name="model_id"))
        g = [x for x in genes if x in z.columns]
        rec["tgt_expr"] = z.reindex(models)[g].mean(axis=1) if g else np.nan
        g = [x for x in genes if x in dep.columns]
        rec["tgt_dep"] = dep.reindex(models)[g].min(axis=1) if g else np.nan
        if mut is not None:
            g = [x for x in genes if x in mut.columns]
            rec["tgt_mut"] = (mut.reindex(models)[g] > 0).any(axis=1).astype(np.float32) \
                if g else np.nan
        rec["drug_key"] = drug
        out.append(rec.reset_index())
    return pd.concat(out, ignore_index=True)


def mechanism_embedding(targets: pd.DataFrame, n_components: int, seed: int) -> pd.DataFrame:
    from sklearn.decomposition import TruncatedSVD

    m = targets.pivot_table(index="drug_key", columns="gene", values="combined_score",
                            aggfunc="max", fill_value=0) / 1000.0
    n = max(1, min(n_components, m.shape[0] - 1, m.shape[1] - 1))
    emb = TruncatedSVD(n_components=n, random_state=seed).fit_transform(m.to_numpy())
    out = pd.DataFrame(emb.astype(np.float32), index=m.index,
                       columns=[f"mech_{i + 1}" for i in range(n)])
    out["n_targets"] = (m > 0).sum(axis=1).astype(np.float32)
    return out.reset_index()


def run(cfg: dict, paths: Paths) -> None:
    paths.ensure()
    tcfg = cfg["features"]
    dcfg, mcfg = cfg["sources"]["drugcombdb"]["files"], cfg["sources"]["depmap"]["files"]

    links_path = paths.raw_file("drugcombdb", dcfg["protein_links"])
    if not links_path.exists() or not paths.raw_file("depmap", mcfg["crispr"]).exists():
        log.warning("drug-target or CRISPR file missing - skipping target features")
        return
    dim_drug = pd.read_parquet(paths.dim_drug)
    bridge = pd.read_parquet(paths.processed / "bridge_drug_name.parquet")
    cids = drug_cids(dim_drug, bridge, paths.pubchem_cache)

    links = read_links(links_path, tcfg["target_min_score"])
    pc_cfg = cfg["sources"]["pubchem"]
    if pc_cfg.get("enabled", True):
        from .drugs import PubChem

        missing = cids[~cids["drug_key"].isin(links.merge(cids, on="cid")["drug_key"])]
        conn = same_connectivity(missing["cid"].tolist(), PubChem(pc_cfg, paths.pubchem_cache))
        extra = missing.assign(cid=missing["cid"].map(conn)).explode("cid").dropna()
        cids = pd.concat([cids, extra.astype({"cid": "int64"})]).drop_duplicates()
    links = links.merge(cids, on="cid")
    sym = ensp_to_symbol(links["ensp"].unique().tolist(), paths.interim / "ensp_symbol.json")
    links["gene"] = links["ensp"].map(sym)
    targets = (links.dropna(subset=["gene"])
               .groupby(["drug_key", "gene"], as_index=False)["combined_score"].max())
    targets["rank"] = targets.groupby("drug_key")["combined_score"].rank(
        ascending=False, method="first")
    targets.to_parquet(paths.processed / "drug_targets.parquet", index=False)
    log.info("targets: %d links for %d / %d drugs, %d genes", len(targets),
             targets["drug_key"].nunique(), len(dim_drug), targets["gene"].nunique())

    mechanism_embedding(targets, tcfg["mechanism_components"], cfg["model"]["seed"]) \
        .to_parquet(paths.processed / "drug_mechanism.parquet", index=False)

    genes = set(targets.loc[targets["rank"] <= tcfg["target_top_k"], "gene"])
    dim_cell = pd.read_parquet(paths.dim_cell)
    models = sorted(dim_cell["model_id"].dropna().unique())
    expr = load_gene_matrix(paths.raw_file("depmap", mcfg["expression"]), genes)
    dep = load_gene_matrix(paths.raw_file("depmap", mcfg["crispr"]), genes)
    mut_path = paths.raw_file("depmap", mcfg["mutations"])
    mut = load_gene_matrix(mut_path, genes) if mut_path.exists() else None
    primary = targets[targets["rank"] <= tcfg["target_top_k"]]  # strongest links only
    feats = drug_cell_features(primary, models, expr, dep, mut)
    feats.to_parquet(paths.processed / "drug_cell_targets.parquet", index=False)
    log.info("drug x cell target features: %s", feats.shape)
