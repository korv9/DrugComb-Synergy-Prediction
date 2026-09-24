"""Is the model any good, and is it overfitting? Diagnostics beyond the CV table.

* learning curves: train vs held-out performance as training data and trees grow
* y-scramble: the same pipeline on shuffled labels must score r ~ 0 (leakage check)
* calibration: mean observed ZIP per decile of predicted ZIP
* per-study performance: does one screen carry the headline number?
* enrichment: if you tested the model's top-ranked new combinations, how many
  would be synergistic compared with testing at random?
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

from .config import Paths
from .features import history_features, history_features_oof, select
from .model import _lgbm, load_modeling_data
from .splits import make_folds

log = logging.getLogger(__name__)

BEST = "lgbm_all"
BASELINE = "history_ridge"


def _xy(fact, static, tr, te, cfg, groups, y):
    m = cfg["features"]["target_encoding_smoothing"]
    seed = cfg["model"]["seed"]
    ftr, fte = fact.iloc[tr], fact.iloc[te]
    x_tr = pd.concat([static.iloc[tr], history_features_oof(ftr, y[tr], m, seed=seed)], axis=1)
    x_te = pd.concat([static.iloc[te], history_features(ftr, y[tr], fte, m)], axis=1)
    cols = select(x_tr.columns, groups)
    return x_tr[cols], x_te[cols]


def learning_curves(fact, static, cfg, scheme="cold_pair") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train/test r vs training-set size, and train/test RMSE per boosting round."""
    mcfg = cfg["model"]
    groups = mcfg["feature_sets"][BEST]
    y = fact["zip"].to_numpy(np.float64)
    _, tr_all, te = next(make_folds(fact, scheme, mcfg["n_folds"], mcfg["seed"]))
    rng = np.random.default_rng(mcfg["seed"])

    sizes, rounds = [], None
    for frac in (0.05, 0.15, 0.4, 1.0):
        tr = np.sort(rng.choice(tr_all, int(len(tr_all) * frac), replace=False))
        x_tr, x_te = _xy(fact, static, tr, te, cfg, groups, y)
        model = _lgbm(mcfg["lgbm"], mcfg["seed"])
        model.fit(x_tr, y[tr], eval_set=[(x_tr, y[tr]), (x_te, y[te])], eval_metric="l2")
        sizes.append({
            "train_fraction": frac, "n_train": len(tr),
            "train_r": pearsonr(y[tr], model.predict(x_tr))[0],
            "test_r": pearsonr(y[te], model.predict(x_te))[0],
        })
        log.info("learning curve %.2f: %s", frac, sizes[-1])
        if frac == 1.0:
            train_l2, test_l2 = (v["l2"] for v in model.evals_result_.values())
            rounds = pd.DataFrame({
                "round": np.arange(1, len(train_l2) + 1),
                "train_rmse": np.sqrt(train_l2),
                "test_rmse": np.sqrt(test_l2),
            })
    return pd.DataFrame(sizes), rounds


def y_scramble(fact, static, cfg, scheme="cold_pair") -> pd.DataFrame:
    """Refit on permuted labels; any real signal left would mean leakage."""
    mcfg = cfg["model"]
    y = fact["zip"].to_numpy(np.float64)
    y_perm = np.random.default_rng(0).permutation(y)
    _, tr, te = next(make_folds(fact, scheme, mcfg["n_folds"], mcfg["seed"]))
    x_tr, x_te = _xy(fact, static, tr, te, cfg, mcfg["feature_sets"][BEST], y_perm)
    p = _lgbm(mcfg["lgbm"], mcfg["seed"]).fit(x_tr, y_perm[tr]).predict(x_te)
    return pd.DataFrame([{"scheme": scheme, "labels": "shuffled",
                          "test_r": pearsonr(y_perm[te], p)[0]}])


def calibration(preds: pd.DataFrame, model: str = BEST) -> pd.DataFrame:
    out = []
    for scheme, g in preds.groupby("scheme", sort=False):
        dec = pd.qcut(g[f"pred_{model}"], 10, labels=False, duplicates="drop")
        t = g.groupby(dec).agg(pred=(f"pred_{model}", "mean"), obs=("y", "mean"),
                               obs_q25=("y", lambda v: v.quantile(0.25)),
                               obs_q75=("y", lambda v: v.quantile(0.75)), n=("y", "size"))
        out.append(t.reset_index(names="decile").assign(scheme=scheme))
    return pd.concat(out, ignore_index=True)


def by_study(preds: pd.DataFrame, fact: pd.DataFrame) -> pd.DataFrame:
    p = preds.merge(fact[["study"]], left_on="row", right_index=True)
    rows = []
    for (scheme, study), g in p.groupby(["scheme", "study"], sort=False):
        if len(g) < 200:
            continue
        for model in (BASELINE, BEST):
            rows.append({"scheme": scheme, "study": study, "model": model, "n": len(g),
                         "pearson": pearsonr(g["y"], g[f"pred_{model}"])[0],
                         "rmse": float(np.sqrt(np.mean((g["y"] - g[f"pred_{model}"]) ** 2)))})
    return pd.DataFrame(rows)


def enrichment(preds: pd.DataFrame, fact: pd.DataFrame, cutoff: float) -> pd.DataFrame:
    """Hit rate (ZIP > cutoff) among the top-ranked predictions, ranked within study.

    Ranking within each study mirrors how the model would be used - to pick
    what to test next in a given screen - and prevents a trivially "good"
    ranking that only separates screens with different baselines.
    """
    p = preds.merge(fact[["study"]], left_on="row", right_index=True)
    p = p[p["study"] != "unknown"]
    rows = []
    for scheme, g in p.groupby("scheme", sort=False):
        base = (g["y"] > cutoff).mean()
        for model in (BASELINE, BEST):
            pct = g.groupby("study")[f"pred_{model}"].rank(pct=True, ascending=False)
            for top in (0.01, 0.05, 0.10):
                hit = (g.loc[pct <= top, "y"] > cutoff).mean()
                rows.append({"scheme": scheme, "model": model, "top_fraction": top,
                             "hit_rate": hit, "base_rate": base, "enrichment": hit / base})
    return pd.DataFrame(rows)


def run(cfg: dict, paths: Paths) -> None:
    paths.ensure()
    fact, static = load_modeling_data(cfg, paths)
    preds = pd.read_parquet(paths.predictions)
    cutoff = cfg["clean"]["synergy_cutoff"]

    calibration(preds).to_csv(paths.tables / "eval_calibration.csv", index=False)
    by_study(preds, fact).to_csv(paths.tables / "eval_by_study.csv", index=False)
    enrichment(preds, fact, cutoff).to_csv(paths.tables / "eval_enrichment.csv", index=False)

    sizes, rounds = learning_curves(fact, static, cfg)
    sizes.to_csv(paths.tables / "eval_learning_curve.csv", index=False)
    rounds.to_csv(paths.tables / "eval_boosting_rounds.csv", index=False)
    y_scramble(fact, static, cfg).to_csv(paths.tables / "eval_y_scramble.csv", index=False)
    log.info("diagnostics written to %s", paths.tables)
