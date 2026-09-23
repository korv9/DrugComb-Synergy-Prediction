import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from synthetic import write_raw  # noqa: E402

from drugsyn.config import Paths, load_config  # noqa: E402


@pytest.fixture(scope="session")
def pipeline(tmp_path_factory):
    """Run the whole pipeline once on synthetic data; return (cfg, paths)."""
    from drugsyn.cli import STAGES

    root = tmp_path_factory.mktemp("run")
    cfg = load_config(overrides={
        "data_dir": str(root / "data"),
        "reports_dir": str(root / "reports"),
        "sources": {"pubchem": {"enabled": False}},
        "features": {"rna_top_var_genes": 200, "rna_pca_components": 8, "morgan_bits": 256},
        "model": {"n_folds": 3, "eval_folds": 2,
                  "lgbm": {"n_estimators": 40, "num_leaves": 15, "min_child_samples": 5}},
    })
    paths = Paths.from_config(cfg)
    write_raw(paths.raw)
    for stage, fn in STAGES.items():
        if stage != "download":
            fn(cfg, paths)
    return cfg, paths
