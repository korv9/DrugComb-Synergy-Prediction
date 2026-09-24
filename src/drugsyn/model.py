"""Evaluate baselines and gradient-boosted models under every CV scheme."""

from __future__ import annotations

import logging
import time

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import average_precision_score, mean_squared_error, r2_score

from .config import Paths
from .features import family, history_features, history_features_oof, select, static_features
from .splits import make_folds

log = logging.getLogger(__name__)

MODELS = {
    "mean": "Global mean",
    "history_ridge": "Screen history (ridge on in-fold means)",
    "lgbm_chem_bio": "LightGBM: chemistry + biology",
    "lgbm_full": "LightGBM: + screen history",
    "lgbm_study": "LightGBM: + study",
    "lgbm_mono": "LightGBM: + monotherapy",
    "lgbm_all": "LightGBM: + targets (all features)",
    "lgbm_all_no_history": "LightGBM: all features except screen history",
}


def metrics(y: np.ndarray, p: np.ndarray, cutoff: float) -> dict:
    const = np.allclose(p, p[0])
    positive = y > cutoff
    return {
        "pearson": np.nan if const else pearsonr(y, p)[0],
        "spearman": np.nan if const else spearmanr(y, p)[0],
        "rmse": float(np.sqrt(mean_squared_error(y, p))),
        "r2": r2_score(y, p),
        "auprc_synergy": average_precision_score(positive, p) if positive.any() else np.nan,
        "prevalence_synergy": positive.mean(),
        "n_test": len(y),
    }


def _lgbm(params: dict, seed: int):
    from lightgbm import LGBMRegressor

    return LGBMRegressor(random_state=seed, n_jobs=-1, **params)


def evaluate(fact: pd.DataFrame, static: pd.DataFrame, cfg: dict):
    mcfg, fcfg = cfg["model"], cfg["features"]
    seed, m = mcfg["seed"], fcfg["target_encoding_smoothing"]
    cutoff = cfg["clean"]["synergy_cutoff"]
    y = fact["zip"].to_numpy(np.float64)

    rows, preds, importances = [], [], []
    for scheme in mcfg["splits"]:
        for fold, tr, te in make_folds(fact, scheme, mcfg["n_folds"], seed):
            if fold >= mcfg["eval_folds"]:
                break
            t0 = time.time()
            ftr, fte = fact.iloc[tr], fact.iloc[te]
            h_tr = history_features_oof(ftr, y[tr], m, seed=seed)
            h_te = history_features(ftr, y[tr], fte, m)

            out = {"mean": np.full(len(te), y[tr].mean())}
            ridge = Ridge(alpha=1.0).fit(h_tr.to_numpy(), y[tr])
            out["history_ridge"] = ridge.predict(h_te.to_numpy())

            all_tr = pd.concat([static.iloc[tr], h_tr], axis=1)
            all_te = pd.concat([static.iloc[te], h_te], axis=1)
            for name, groups in mcfg["feature_sets"].items():
                cols = select(all_tr.columns, groups)
                if not cols:
                    continue
                x_tr, x_te = all_tr[cols], all_te[cols]
                model = _lgbm(mcfg["lgbm"], seed).fit(x_tr, y[tr])
                out[name] = model.predict(x_te)
                gain = model.booster_.feature_importance("gain")
                importances.append(pd.DataFrame({
                    "scheme": scheme, "fold": fold, "model": name,
                    "feature": x_tr.columns, "gain": gain / gain.sum(),
                }))

            for name, p in out.items():
                rows.append({"scheme": scheme, "fold": fold, "model": name,
                             **metrics(y[te], p, cutoff)})
            preds.append(pd.DataFrame({
                "scheme": scheme, "fold": fold, "row": te, "y": y[te],
                **{f"pred_{k}": v.astype(np.float32) for k, v in out.items()},
            }))
            summary = "  ".join(f"{r['model']}={r['pearson']:.3f}" for r in rows[-len(out):]
                                if r["model"].startswith("lgbm"))
            log.info("%-9s fold %d: n_train=%d n_test=%d  %s  (%.0fs)",
                     scheme, fold, len(tr), len(te), summary, time.time() - t0)

    return pd.DataFrame(rows), pd.concat(preds, ignore_index=True), pd.concat(importances)


def summarise(by_fold: pd.DataFrame) -> pd.DataFrame:
    metric_cols = ["pearson", "spearman", "rmse", "r2", "auprc_synergy", "prevalence_synergy"]
    g = by_fold.groupby(["scheme", "model"], sort=False)[metric_cols]
    summary = g.mean().join(g.std().add_suffix("_sd")).reset_index()
    summary["model_label"] = summary["model"].map(MODELS)
    return summary


def load_modeling_data(cfg: dict, paths: Paths) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fact table (optionally subsampled) and its label-free feature matrix."""
    fact = pd.read_parquet(paths.fact)
    max_rows = cfg["model"].get("max_rows")
    if max_rows and len(fact) > max_rows:
        fact = fact.sample(max_rows, random_state=cfg["model"]["seed"])
    fact = fact.reset_index(drop=True)

    def optional(name):
        p = paths.processed / name
        return pd.read_parquet(p) if p.exists() else None

    static = static_features(
        fact, pd.read_parquet(paths.dim_drug), pd.read_parquet(paths.drug_fingerprints),
        pd.read_parquet(paths.dim_cell), pd.read_parquet(paths.cell_rna_pcs),
        mechanism=optional("drug_mechanism.parquet"),
        drug_cell=optional("drug_cell_targets.parquet"),
    )
    log.info("static feature matrix: %s", static.shape)
    return fact, static


def run(cfg: dict, paths: Paths) -> None:
    paths.ensure()
    fact, static = load_modeling_data(cfg, paths)
    by_fold, preds, imp = evaluate(fact, static, cfg)
    preds = preds.merge(fact[["drug_1", "drug_2", "cell_key", "pair_key"]],
                        left_on="row", right_index=True)
    preds.to_parquet(paths.predictions, index=False)
    by_fold.to_csv(paths.tables / "metrics_by_fold.csv", index=False)
    summarise(by_fold).to_csv(paths.tables / "metrics.csv", index=False)

    imp["family"] = imp["feature"].map(family)
    fam = (imp.groupby(["scheme", "model", "fold", "family"])["gain"].sum()
           .groupby(["scheme", "model", "family"]).mean().reset_index())
    fam.to_csv(paths.tables / "feature_importance_family.csv", index=False)
    top = (imp.groupby(["scheme", "model", "feature", "family"])["gain"].mean()
           .reset_index().sort_values("gain", ascending=False)
           .groupby(["scheme", "model"]).head(25))
    top.to_csv(paths.tables / "feature_importance_top.csv", index=False)
    log.info("metrics written to %s", paths.tables / "metrics.csv")
