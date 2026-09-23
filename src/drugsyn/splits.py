"""Cross-validation schemes of increasing difficulty.

* ``random``    - rows shuffled; the same pair is seen in other cell lines.
* ``cold_pair`` - every test pair is unseen (in any cell line) during training.
* ``cold_drug`` - test rows contain at least one drug never seen in training.
* ``cold_cell`` - every test cell line is unseen during training.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

SCHEMES = ("random", "cold_pair", "cold_drug", "cold_cell")


def _group_folds(groups: pd.Series, n_folds: int, rng: np.random.Generator) -> np.ndarray:
    uniq = groups.unique()
    fold_of = dict(zip(rng.permutation(uniq), np.arange(len(uniq)) % n_folds))
    return groups.map(fold_of).to_numpy()


def make_folds(fact: pd.DataFrame, scheme: str, n_folds: int = 5, seed: int = 42):
    """Yield ``(fold, train_idx, test_idx)`` as positional indices."""
    rng = np.random.default_rng(seed)
    n = len(fact)
    if scheme == "random":
        fold = rng.permutation(np.arange(n) % n_folds)
    elif scheme == "cold_pair":
        fold = _group_folds(fact["pair_key"], n_folds, rng)
    elif scheme == "cold_cell":
        fold = _group_folds(fact["cell_key"], n_folds, rng)
    elif scheme == "cold_drug":
        drugs = pd.Series(pd.concat([fact["drug_1"], fact["drug_2"]]).unique())
        fold_of = dict(zip(rng.permutation(drugs), np.arange(len(drugs)) % n_folds))
        f1 = fact["drug_1"].map(fold_of).to_numpy()
        f2 = fact["drug_2"].map(fold_of).to_numpy()
        for k in range(n_folds):
            touches = (f1 == k) | (f2 == k)
            yield k, np.flatnonzero(~touches), np.flatnonzero(touches)
        return
    else:
        raise ValueError(f"unknown scheme {scheme!r}; choose from {SCHEMES}")
    for k in range(n_folds):
        yield k, np.flatnonzero(fold != k), np.flatnonzero(fold == k)
