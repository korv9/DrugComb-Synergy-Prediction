"""End-to-end run on synthetic data (see tests/synthetic.py)."""

import duckdb
import pandas as pd
from synthetic import MALARIA


def test_drug_synonyms_collapse(pipeline):
    _, paths = pipeline
    dim = pd.read_parquet(paths.dim_drug)
    bridge = pd.read_parquet(paths.processed / "bridge_drug_name.parquet")
    key = dict(zip(bridge["name_norm"], bridge["drug_key"]))
    assert key["5-fu"] == key["fluorouracil"]
    assert key["zinc99999999"] == "NAME:zinc99999999"  # unresolved keeps a name key
    assert dim["drug_key"].is_unique
    assert dim.loc[dim["drug_key"] == key["cisplatin"], "has_structure"].item()


def test_cell_matching(pipeline):
    _, paths = pipeline
    cell = pd.read_parquet(paths.dim_cell).set_index("cell_key")
    assert cell.loc["7860", "match_method"] == "alias"
    assert cell.loc["7860", "model_id"] == "ACH-000649"
    assert cell.loc["ncih460", "match_method"] == "exact"
    assert cell.loc[MALARIA.lower(), "match_method"] == "excluded_non_human"


def test_fact_table_quality(pipeline):
    _, paths = pipeline
    fact = pd.read_parquet(paths.fact)
    qc = pd.read_csv(paths.tables / "data_quality.csv")
    assert qc["passed"].all(), qc[~qc["passed"]]
    assert MALARIA.lower() not in set(fact["cell_key"])
    assert (fact["n_replicates"] > 1).any()
    funnel = pd.read_csv(paths.tables / "data_funnel.csv")
    assert funnel["rows"].is_monotonic_decreasing


def test_warehouse_and_sql(pipeline):
    _, paths = pipeline
    con = duckdb.connect(str(paths.warehouse), read_only=True)
    n = con.execute("SELECT COUNT(*) FROM v_combination").fetchone()[0]
    assert n == len(pd.read_parquet(paths.fact))
    kpi = pd.read_csv(paths.tables / "sql_kpi_overview.csv")
    assert kpi["combinations"].item() == n


def test_model_outputs(pipeline):
    _, paths = pipeline
    m = pd.read_csv(paths.tables / "metrics.csv")
    assert set(m["scheme"]) == {"random", "cold_pair", "cold_drug", "cold_cell"}
    full = m[(m["model"] == "lgbm_full") & (m["scheme"] == "random")]["pearson"].item()
    assert full > 0.5  # the planted signal is recoverable


def test_figures_and_report(pipeline):
    _, paths = pipeline
    pngs = {p.stem for p in paths.figures.glob("*.png")}
    for name in ["de_01_data_funnel", "de_02_entity_resolution", "an_01_zip_distribution",
                 "an_02_replicate_agreement", "ml_01_performance_by_split"]:
        assert name in pngs
    assert "Modelling" in (paths.reports / "REPORT.md").read_text()


def test_study_and_monotherapy(pipeline):
    _, paths = pipeline
    fact = pd.read_parquet(paths.fact)
    assert {"ONEIL", "ALMANAC", "unknown"} <= set(fact["study"])
    known = fact["study"] != "unknown"
    assert fact.loc[known, "mono_inh_max_1"].notna().mean() > 0.95
    assert fact.loc[~known, "mono_inh_max_1"].isna().all()


def test_target_features(pipeline):
    _, paths = pipeline
    t = pd.read_parquet(paths.processed / "drug_targets.parquet")
    assert t["drug_key"].nunique() >= 10
    dc = pd.read_parquet(paths.processed / "drug_cell_targets.parquet")
    assert dc[["tgt_expr", "tgt_dep", "tgt_mut"]].notna().any().all()


def test_ablation_models(pipeline):
    _, paths = pipeline
    m = pd.read_csv(paths.tables / "metrics.csv")
    assert {"lgbm_study", "lgbm_mono", "lgbm_all", "lgbm_all_no_history"} <= set(m["model"])
    assert (paths.figures / "ml_04_ablation.png").exists()
