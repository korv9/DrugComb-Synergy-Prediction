"""Configuration loading and canonical file locations."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "default.yaml"


def _deep_update(base: dict, override: dict) -> dict:
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def load_config(path: str | Path | None = None, overrides: dict | None = None) -> dict[str, Any]:
    with open(DEFAULT_CONFIG, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    if path is not None and Path(path).resolve() != DEFAULT_CONFIG:
        with open(path, encoding="utf-8") as fh:
            _deep_update(cfg, yaml.safe_load(fh) or {})
    if overrides:
        _deep_update(cfg, copy.deepcopy(overrides))
    return cfg


@dataclass(frozen=True)
class Paths:
    """Every file the pipeline reads or writes, derived from the config."""

    data: Path
    reports: Path

    @classmethod
    def from_config(cls, cfg: dict) -> Paths:
        def resolve(p: str) -> Path:
            p = Path(p)
            return p if p.is_absolute() else REPO_ROOT / p

        return cls(data=resolve(cfg["data_dir"]), reports=resolve(cfg["reports_dir"]))

    # layers
    @property
    def raw(self) -> Path:
        return self.data / "raw"

    @property
    def interim(self) -> Path:
        return self.data / "interim"

    @property
    def processed(self) -> Path:
        return self.data / "processed"

    @property
    def figures(self) -> Path:
        return self.reports / "figures"

    @property
    def tables(self) -> Path:
        return self.reports / "tables"

    # raw inputs
    def raw_file(self, source: str, name: str) -> Path:
        return self.raw / source / name

    @property
    def manifest(self) -> Path:
        return self.raw / "MANIFEST.json"

    # staged / modeled tables
    @property
    def measurements(self) -> Path:
        return self.interim / "stg_measurements.parquet"

    @property
    def pubchem_cache(self) -> Path:
        return self.interim / "pubchem_cache.json"

    @property
    def dim_drug(self) -> Path:
        return self.processed / "dim_drug.parquet"

    @property
    def dim_cell(self) -> Path:
        return self.processed / "dim_cell.parquet"

    @property
    def fact(self) -> Path:
        return self.processed / "fact_combination.parquet"

    @property
    def replicates(self) -> Path:
        return self.processed / "replicate_pairs.parquet"

    @property
    def cell_rna_pcs(self) -> Path:
        return self.processed / "cell_rna_pcs.parquet"

    @property
    def drug_fingerprints(self) -> Path:
        return self.processed / "drug_fingerprints.parquet"

    @property
    def warehouse(self) -> Path:
        return self.processed / "warehouse.duckdb"

    @property
    def predictions(self) -> Path:
        return self.processed / "oof_predictions.parquet"

    def ensure(self) -> None:
        for d in (self.raw, self.interim, self.processed, self.figures, self.tables):
            d.mkdir(parents=True, exist_ok=True)
