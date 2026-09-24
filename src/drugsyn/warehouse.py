"""Load the star schema into DuckDB and run the analyst SQL in ``sql/analysis``."""

from __future__ import annotations

import logging

import duckdb

from .config import REPO_ROOT, Paths

log = logging.getLogger(__name__)

SQL_DIR = REPO_ROOT / "sql"


def build(paths: Paths) -> duckdb.DuckDBPyConnection:
    paths.warehouse.unlink(missing_ok=True)
    con = duckdb.connect(str(paths.warehouse))
    tables = {
        "fact_combination": paths.fact,
        "dim_drug": paths.dim_drug,
        "dim_cell": paths.dim_cell,
        "replicate_pairs": paths.replicates,
        "bridge_drug_name": paths.processed / "bridge_drug_name.parquet",
    }
    for name, path in tables.items():
        con.execute(f"CREATE TABLE {name} AS SELECT * FROM read_parquet('{path.as_posix()}')")
    for sql in sorted((SQL_DIR / "models").glob("*.sql")):
        con.execute(sql.read_text())
    return con


def run_analysis(con: duckdb.DuckDBPyConnection, paths: Paths) -> dict:
    results = {}
    for sql in sorted((SQL_DIR / "analysis").glob("*.sql")):
        df = con.execute(sql.read_text()).df()
        df.to_csv(paths.tables / f"sql_{sql.stem}.csv", index=False)
        results[sql.stem] = df
        log.info("sql/%s -> %d rows", sql.name, len(df))
    return results


def run(cfg: dict, paths: Paths) -> None:
    paths.ensure()
    con = build(paths)
    run_analysis(con, paths)
    con.close()
    log.info("warehouse written to %s", paths.warehouse)
