"""Portfolio figures (PNG + SVG) from the processed tables and model outputs."""

from __future__ import annotations

import logging

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from .config import Paths  # noqa: E402

log = logging.getLogger(__name__)

# Validated reference palette (light mode) - see the README's figure notes.
SURFACE = "#fcfcfb"
INK, INK_2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]  # blue, orange, aqua, yellow
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
NEUTRAL = "#c3c2b7"
SEQ = LinearSegmentedColormap.from_list("seq_blue", [SURFACE] + BLUE_RAMP)
DIV = LinearSegmentedColormap.from_list("div", ["#e34948", "#f0efec", "#2a78d6"])

SCHEME_LABELS = {
    "random": "Random rows",
    "cold_pair": "Unseen drug pair",
    "cold_drug": "Unseen drug",
    "cold_cell": "Unseen cell line",
}
MODEL_ORDER = ["mean", "history_ridge", "lgbm_chem_bio", "lgbm_full", "lgbm_all"]
MODEL_LABELS = {
    "mean": "Global mean",
    "history_ridge": "Screen-history baseline",
    "lgbm_chem_bio": "LightGBM · chemistry + biology",
    "lgbm_full": "LightGBM · + screen history",
    "lgbm_study": "LightGBM · + study",
    "lgbm_mono": "LightGBM · + monotherapy",
    "lgbm_all": "LightGBM · + targets (all)",
    "lgbm_all_no_history": "LightGBM · all but screen history",
}
LADDER = ["lgbm_chem_bio", "lgbm_full", "lgbm_study", "lgbm_mono", "lgbm_all"]
LADDER_LABELS = ["Chemistry\n+ biology", "+ screen\nhistory", "+ study", "+ mono-\ntherapy",
                 "+ targets"]


def _best_model(models) -> str:
    return next(m for m in ("lgbm_all", "lgbm_full", "lgbm_chem_bio") if m in set(models))


def _style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.family": "sans-serif",
        "font.sans-serif": ["Inter", "Helvetica Neue", "Arial", "DejaVu Sans"],
        "font.size": 10, "text.color": INK, "axes.labelcolor": INK_2,
        "axes.titlesize": 12, "axes.titleweight": "semibold", "axes.titlelocation": "left",
        "axes.titlecolor": INK, "axes.edgecolor": AXIS, "axes.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
        "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK_2,
        "ytick.labelcolor": INK_2, "xtick.major.size": 0, "ytick.major.size": 0,
        "legend.frameon": False, "legend.fontsize": 9,
    })


def _subtitle(ax, text: str) -> None:
    ax.text(0, 1.02, text, transform=ax.transAxes, color=INK_2, fontsize=9, va="bottom")


def _save(fig, paths: Paths, name: str) -> None:
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(paths.figures / f"{name}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    log.info("figure %s", name)


def _fmt(n: float) -> str:
    return f"{n:,.0f}".replace(",", " ")


# ------------------------------------------------------------------ data engineering
def fig_funnel(paths: Paths) -> None:
    f = pd.read_csv(paths.tables / "data_funnel.csv")
    fig, ax = plt.subplots(figsize=(8, 0.45 * len(f) + 1.4))
    y = np.arange(len(f))[::-1]
    ax.barh(y, f["rows"], color=SERIES[0], height=0.62)
    top = f["rows"].max()
    for yi, (_, r) in zip(y, f.iterrows()):
        pct = r["rows"] / f["rows"].iloc[0]
        ax.text(r["rows"] + top * 0.01, yi, f"{_fmt(r['rows'])}  ({pct:.0%})",
                va="center", color=INK_2, fontsize=9)
    ax.set_yticks(y, f["step"])
    ax.set_xlim(0, top * 1.22)
    ax.grid(axis="y", visible=False)
    ax.xaxis.set_major_formatter(lambda v, _: f"{v / 1e3:.0f}k")
    ax.set_title("From raw DrugCombDB rows to a modelling table", pad=22)
    _subtitle(ax, "Rows remaining after each cleaning step")
    _save(fig, paths, "de_01_data_funnel")


def fig_entity_resolution(paths: Paths) -> None:
    er = pd.read_csv(paths.tables / "entity_resolution.csv")
    group = {
        "drugcombdb": "Primary source", "exact": "Primary source",
        "pubchem": "Fallback (PubChem / alias)", "alias": "Fallback (PubChem / alias)",
        "unresolved": "Unresolved", "invalid_smiles": "Unresolved", "unmatched": "Unresolved",
        "excluded_non_human": "Excluded (non-human)",
    }
    er["group"] = er["method"].map(group).fillna("Unresolved")
    order = ["Primary source", "Fallback (PubChem / alias)", "Unresolved", "Excluded (non-human)"]
    colors = [SERIES[0], SERIES[2], NEUTRAL, SERIES[1]]
    tab = er.pivot_table(index="entity", columns="group", values="measurement_share",
                         aggfunc="sum").reindex(columns=order).fillna(0)
    tab = tab.reindex(["drug", "cell_line"])
    fig, ax = plt.subplots(figsize=(8, 2.8))
    left = np.zeros(len(tab))
    labels = ["Drugs → structure", "Cell lines → DepMap"]
    for g, c in zip(order, colors):
        v = tab[g].to_numpy()
        ax.barh(labels, v, left=left, color=c, height=0.55, label=g,
                edgecolor=SURFACE, linewidth=2)
        for i, (lv, vv) in enumerate(zip(left, v)):
            if vv > 0.06:
                ax.text(lv + vv / 2, i, f"{vv:.0%}", ha="center", va="center",
                        color="white" if c != NEUTRAL else INK, fontsize=9, weight="semibold")
        left += v
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.legend(ncol=4, loc="upper left", bbox_to_anchor=(0, -0.18))
    ax.set_title("Entity resolution coverage", pad=22)
    _subtitle(ax, "Share of measurements whose drug / cell line could be linked to external data")
    _save(fig, paths, "de_02_entity_resolution")


# ------------------------------------------------------------------ analysis
def fig_zip_distribution(paths: Paths, cutoff: float) -> None:
    fact = pd.read_parquet(paths.fact, columns=["zip"])
    lim = float(np.ceil(np.nanpercentile(np.abs(fact["zip"]), 99.5) / 10) * 10)
    lim = max(lim, cutoff + 10)
    z = fact["zip"].clip(-lim, lim)
    fig, ax = plt.subplots(figsize=(8, 3.8))
    bins = np.arange(-lim, lim + 1, 1)
    counts, edges = np.histogram(z, bins)
    centers = (edges[:-1] + edges[1:]) / 2
    colors = np.where(centers > cutoff, SERIES[0], np.where(centers < -cutoff, "#e34948", NEUTRAL))
    ax.bar(centers, counts, width=0.9, color=colors)
    for x in (-cutoff, cutoff):
        ax.axvline(x, color=INK_2, lw=0.8, ls=(0, (3, 3)))
    share_syn = (fact["zip"] > cutoff).mean()
    share_ant = (fact["zip"] < -cutoff).mean()
    ymax = counts.max()
    ax.text(cutoff + 2, ymax * 0.8, f"Synergistic\nZIP > {cutoff:g}\n{share_syn:.1%}",
            color=INK, fontsize=9)
    ax.text(-cutoff - 2, ymax * 0.8, f"Antagonistic\nZIP < {-cutoff:g}\n{share_ant:.1%}",
            color=INK, fontsize=9, ha="right")
    ax.set_xlabel(f"ZIP synergy score (clipped to ±{lim:g} for display)")
    ax.set_ylabel("Combinations")
    ax.yaxis.set_major_formatter(lambda v, _: f"{v / 1e3:.0f}k" if v >= 1e3 else f"{v:.0f}")
    ax.grid(axis="x", visible=False)
    ax.set_title("Distribution of ZIP synergy scores", pad=22)
    share_mid = 1 - share_syn - share_ant
    _subtitle(ax, f"Per (drug pair, cell line), n = {_fmt(len(fact))}; "
                  f"{share_mid:.0%} fall within ±{cutoff:g}")
    _save(fig, paths, "an_01_zip_distribution")


def fig_replicates(paths: Paths) -> float | None:
    reps = pd.read_parquet(paths.replicates)
    if len(reps) < 10:
        log.warning("too few replicate pairs for the noise-ceiling figure")
        return None
    r = float(np.corrcoef(reps["zip_rep1"], reps["zip_rep2"])[0, 1])
    fig, ax = plt.subplots(figsize=(5.2, 5))
    lim = (-50, 50)
    hb = ax.hexbin(reps["zip_rep1"].clip(*lim), reps["zip_rep2"].clip(*lim), gridsize=45,
                   cmap=SEQ, bins="log", mincnt=1, linewidths=0)
    ax.plot(lim, lim, color=INK_2, lw=0.8, ls=(0, (3, 3)))
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("ZIP, replicate 1")
    ax.set_ylabel("ZIP, replicate 2")
    ax.text(0.03, 0.95, f"Pearson r = {r:.2f}\n{_fmt(len(reps))} replicated combinations",
            transform=ax.transAxes, va="top", color=INK, fontsize=9)
    fig.colorbar(hb, ax=ax, shrink=0.7, label="Combinations (log)").outline.set_visible(False)
    ax.set_title("How reproducible is the label itself?", pad=22)
    _subtitle(ax, "Replicate agreement sets a ceiling for any model")
    _save(fig, paths, "an_02_replicate_agreement")
    return r


def fig_score_concordance(paths: Paths) -> None:
    fact = pd.read_parquet(paths.fact, columns=["zip", "bliss", "loewe", "hsa"]).dropna()
    if fact.empty:
        return
    c = fact.corr().to_numpy()
    names = ["ZIP", "Bliss", "Loewe", "HSA"]
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    im = ax.imshow(c, cmap=SEQ, vmin=0, vmax=1)
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f"{c[i, j]:.2f}", ha="center", va="center", fontsize=9,
                    color="white" if c[i, j] > 0.6 else INK)
    ax.set_xticks(range(4), names)
    ax.set_yticks(range(4), names)
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    fig.colorbar(im, ax=ax, shrink=0.75, label="Pearson r").outline.set_visible(False)
    ax.set_title("Agreement between synergy models", pad=22)
    _subtitle(ax, "Correlation of the four reference scores")
    _save(fig, paths, "an_03_score_concordance")


def fig_lineage(paths: Paths) -> None:
    t = pd.read_csv(paths.tables / "sql_synergy_by_lineage.csv")
    t = t[t["lineage"] != "Unknown"].sort_values("median_zip").tail(18)
    if t.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 0.34 * len(t) + 1.6))
    y = np.arange(len(t))
    ax.hlines(y, t["q25"], t["q75"], color=BLUE_RAMP[1], lw=5, capstyle="round")
    ax.scatter(t["median_zip"], y, color=SERIES[0], s=46, zorder=3,
               edgecolor=SURFACE, linewidth=2)
    ax.axvline(0, color=AXIS, lw=0.8)
    ax.set_yticks(y, [f"{lin}  ({_fmt(n)})" for lin, n in zip(t["lineage"], t["combinations"])])
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("ZIP synergy score (median, interquartile range)")
    ax.set_title("Synergy by tissue lineage", pad=22)
    _subtitle(ax, "Per DepMap lineage; combinations in brackets")
    _save(fig, paths, "an_04_synergy_by_lineage")


def fig_study(paths: Paths) -> None:
    t = pd.read_csv(paths.tables / "sql_study_overview.csv")
    if len(t) < 2 or "q25" not in t:
        return
    t = t.sort_values("median_zip")
    fig, ax = plt.subplots(figsize=(8, 0.5 * len(t) + 1.6))
    y = np.arange(len(t))
    ax.hlines(y, t["q25"], t["q75"], color=BLUE_RAMP[1], lw=6, capstyle="round")
    ax.scatter(t["median_zip"], y, color=SERIES[0], s=50, zorder=3,
               edgecolor=SURFACE, linewidth=2)
    for yi, (_, r) in zip(y, t.iterrows()):
        ax.text(r["q75"] + 0.6, yi, f"sd {r['sd_zip']:.1f}", va="center", fontsize=8.5,
                color=INK_2)
    ax.axvline(0, color=AXIS, lw=0.8)
    names = {"unknown": "no dose-response data"}
    ax.set_yticks(y, [f"{names.get(s, s)}  ({_fmt(n)})"
                      for s, n in zip(t["study"], t["combinations"])])
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("ZIP synergy score (median, interquartile range)")
    ax.set_title("Synergy by source study", pad=22)
    _subtitle(ax, "ZIP per study pooled in DrugCombDB; combinations in brackets")
    _save(fig, paths, "an_08_synergy_by_study")


def fig_coverage(paths: Paths) -> None:
    fact = pd.read_parquet(paths.fact, columns=["pair_key", "cell_key"])
    per_pair = fact.groupby("pair_key")["cell_key"].nunique()
    bins = [1, 2, 3, 6, 11, 21, 41, 81, np.inf]
    labels = ["1", "2", "3–5", "6–10", "11–20", "21–40", "41–80", "80+"]
    cats = pd.cut(per_pair, bins, right=False, labels=labels).value_counts().reindex(labels)
    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.bar(labels, cats.values, color=SERIES[0], width=0.7)
    for i, v in enumerate(cats.values):
        ax.text(i, v, f"{v / len(per_pair):.0%}", ha="center", va="bottom", color=INK_2,
                fontsize=9)
    ax.set_xlabel("Number of cell lines a drug pair was tested in")
    ax.set_ylabel("Drug pairs")
    ax.grid(axis="x", visible=False)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v / 1e3:.0f}k" if v >= 1e3 else f"{v:.0f}")
    ax.set_title("How many cell lines each drug pair was tested in", pad=22)
    _subtitle(ax, f"{_fmt(len(per_pair))} distinct drug pairs; "
                  f"{(per_pair == 1).mean():.0%} were tested in a single cell line")
    _save(fig, paths, "an_05_screen_coverage")


def fig_top_pairs(paths: Paths) -> None:
    t = pd.read_csv(paths.tables / "sql_top_synergistic_pairs.csv").head(15).iloc[::-1]
    if t.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 0.36 * len(t) + 1.6))
    y = np.arange(len(t))
    ax.barh(y, t["mean_zip"], color=SERIES[0], height=0.6)
    for yi, (_, r) in zip(y, t.iterrows()):
        ax.text(r["mean_zip"] + 0.3, yi, f"{r['mean_zip']:.1f}  · {r['cell_lines']} lines",
                va="center", color=INK_2, fontsize=8.5)
    names = [p if len(p) <= 48 else p[:46] + "…" for p in t["pair_name"]]
    ax.set_yticks(y, names, fontsize=8.5)
    ax.set_xlim(0, t["mean_zip"].max() * 1.3)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Mean ZIP across cell lines")
    ax.set_title("Consistently synergistic pairs", pad=22)
    _subtitle(ax, "Pairs tested in at least 10 cell lines, ranked by mean ZIP")
    _save(fig, paths, "an_06_top_pairs")


def fig_cell_landscape(paths: Paths) -> None:
    p = paths.processed / "cell_landscape.parquet"
    if not p.exists():
        return
    land = pd.read_parquet(p)
    fig, ax = plt.subplots(figsize=(6.4, 5))
    bg, fg = land[~land["screened"]], land[land["screened"]]
    ax.scatter(bg["rna_pc1"], bg["rna_pc2"], s=9, color=NEUTRAL, alpha=0.6, linewidth=0,
               label=f"Other DepMap models ({_fmt(len(bg))})")
    ax.scatter(fg["rna_pc1"], fg["rna_pc2"], s=22, color=SERIES[0], edgecolor=SURFACE,
               linewidth=1, label=f"Screened in DrugCombDB ({_fmt(len(fg))})")
    heme = land[land["lineage"].isin(["Lymphoid", "Myeloid"])]
    if len(heme) > 5:
        cx, cy = heme["rna_pc1"].median(), heme["rna_pc2"].median()
        ax.annotate("blood cancers", (cx, cy), xytext=(15, 15), textcoords="offset points",
                    color=INK_2, fontsize=9, arrowprops={"arrowstyle": "-", "color": MUTED})
    ax.set_xlabel("RNA PC 1")
    ax.set_ylabel("RNA PC 2")
    ax.legend(loc="lower left", bbox_to_anchor=(0, -0.3), ncol=2)
    ax.set_title("Where the screened cell lines sit in expression space", pad=22)
    _subtitle(ax, "PCA of DepMap RNA-seq (top-variance genes)")
    _save(fig, paths, "an_07_cell_landscape")


# ------------------------------------------------------------------ modelling
def fig_performance(paths: Paths, ceiling: float | None) -> None:
    m = pd.read_csv(paths.tables / "metrics.csv")
    schemes = [s for s in SCHEME_LABELS if s in set(m["scheme"])]
    models = [x for x in MODEL_ORDER if x in set(m["model"]) and x != "mean"]
    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    width = 0.8 / len(models)
    x = np.arange(len(schemes))
    for i, (model, color) in enumerate(zip(models, SERIES)):
        sub = m[m["model"] == model].set_index("scheme").reindex(schemes)
        xs = x - 0.4 + width * (i + 0.5)
        ax.bar(xs, sub["pearson"], width * 0.92, color=color, label=MODEL_LABELS[model])
        ax.errorbar(xs, sub["pearson"], yerr=sub["pearson_sd"], fmt="none", ecolor=INK_2,
                    elinewidth=0.8, capsize=0)
        tops = sub["pearson"] + sub["pearson_sd"].fillna(0)
        for xv, v, t in zip(xs, sub["pearson"], tops):
            if np.isfinite(v):
                ax.text(xv, max(t, 0) + 0.015, f"{v:.2f}", ha="center", fontsize=7.5,
                        color=INK_2)
    if ceiling is not None:
        ax.axhline(ceiling, color=INK, lw=0.9, ls=(0, (4, 3)))
        ax.text(len(schemes) - 0.55, ceiling, f"replicate agreement r = {ceiling:.2f}",
                ha="right",
                va="center", fontsize=8.5, color=INK, zorder=4,
                bbox={"facecolor": SURFACE, "edgecolor": "none", "pad": 2})
    ax.set_xticks(x, [SCHEME_LABELS[s] for s in schemes])
    ax.set_ylabel("Pearson r (predicted vs observed ZIP)")
    ax.set_ylim(min(0, m["pearson"].min() - 0.05), 1.08)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper left", bbox_to_anchor=(0, -0.1), ncol=len(models))
    ax.set_title("Model performance by split strategy", pad=22)
    _subtitle(ax, "Cross-validated Pearson r by split strategy (mean ± sd over folds)")
    _save(fig, paths, "ml_01_performance_by_split")


def fig_feature_families(paths: Paths) -> None:
    fam = pd.read_csv(paths.tables / "feature_importance_family.csv")
    fam = fam[fam["model"] == _best_model(fam["model"])]
    schemes = [s for s in ("random", "cold_drug") if s in set(fam["scheme"])]
    if not schemes:
        return
    order = (fam.groupby("family")["gain"].mean().sort_values().index.tolist())
    fig, axes = plt.subplots(1, len(schemes), figsize=(4.4 * len(schemes), 3.6), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, s in zip(axes, schemes):
        sub = fam[fam["scheme"] == s].set_index("family").reindex(order).fillna(0)
        ax.barh(order, sub["gain"], color=SERIES[0], height=0.6)
        for i, v in enumerate(sub["gain"]):
            ax.text(v + 0.01, i, f"{v:.0%}", va="center", fontsize=8.5, color=INK_2)
        ax.set_xlim(0, 1)
        ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
        ax.grid(axis="y", visible=False)
        ax.set_title(SCHEME_LABELS[s], fontsize=10.5)
    fig.suptitle("What the model relies on (share of LightGBM gain)", x=0.02, ha="left",
                 fontsize=12, weight="semibold", color=INK)
    _save(fig, paths, "ml_02_feature_families")


def fig_pred_vs_obs(paths: Paths, scheme: str = "cold_pair") -> None:
    if not paths.predictions.exists():
        return
    import pyarrow.parquet as pq

    cols = pq.read_schema(paths.predictions).names
    best = "pred_" + _best_model([c.removeprefix("pred_") for c in cols])
    p = pd.read_parquet(paths.predictions, columns=["scheme", "y", best])
    p = p[p["scheme"] == scheme].rename(columns={best: "pred"})
    if p.empty:
        return
    r = np.corrcoef(p["y"], p["pred"])[0, 1]
    lim = (-40, 40)
    fig, ax = plt.subplots(figsize=(5.2, 5))
    hb = ax.hexbin(p["pred"].clip(*lim), p["y"].clip(*lim), gridsize=45, cmap=SEQ,
                   bins="log", mincnt=1, linewidths=0)
    ax.plot(lim, lim, color=INK_2, lw=0.8, ls=(0, (3, 3)))
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("Predicted ZIP")
    ax.set_ylabel("Observed ZIP")
    ax.text(0.03, 0.95, f"Pearson r = {r:.2f}", transform=ax.transAxes, va="top", fontsize=9)
    fig.colorbar(hb, ax=ax, shrink=0.7, label="Combinations (log)").outline.set_visible(False)
    ax.set_title("Predictions for unseen drug pairs", pad=22)
    _subtitle(ax, f"{MODEL_LABELS[best.removeprefix('pred_')]}, out-of-fold")
    _save(fig, paths, "ml_03_pred_vs_obs")


def fig_ablation(paths: Paths) -> None:
    m = pd.read_csv(paths.tables / "metrics.csv")
    ladder = [x for x in LADDER if x in set(m["model"])]
    if len(ladder) < 3:
        return
    labels = [LADDER_LABELS[LADDER.index(x)] for x in ladder]
    schemes = [s for s in SCHEME_LABELS if s in set(m["scheme"])]
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    x = np.arange(len(ladder))
    for scheme, color in zip(schemes, SERIES):
        sub = m[m["scheme"] == scheme].set_index("model").reindex(ladder)
        ax.plot(x, sub["pearson"], color=color, lw=2, marker="o", ms=7,
                markeredgecolor=SURFACE, markeredgewidth=1.5, label=SCHEME_LABELS[scheme])
        ax.text(x[-1] + 0.12, sub["pearson"].iloc[-1], f"{sub['pearson'].iloc[-1]:.2f}",
                va="center", fontsize=9, color=INK)
    ax.set_xticks(x, labels)
    ax.set_xlim(-0.3, len(ladder) - 0.5)
    ax.set_ylabel("Pearson r")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper left", bbox_to_anchor=(0, -0.2), ncol=len(schemes))
    ax.set_title("What each data source adds", pad=22)
    _subtitle(ax, "LightGBM, cumulative feature groups, cross-validated Pearson r per split")
    _save(fig, paths, "ml_04_ablation")


def run(cfg: dict, paths: Paths) -> None:
    paths.ensure()
    _style()
    steps = [
        lambda: fig_funnel(paths),
        lambda: fig_entity_resolution(paths),
        lambda: fig_zip_distribution(paths, cfg["clean"]["synergy_cutoff"]),
        lambda: fig_score_concordance(paths),
        lambda: fig_lineage(paths),
        lambda: fig_coverage(paths),
        lambda: fig_top_pairs(paths),
        lambda: fig_study(paths),
        lambda: fig_cell_landscape(paths),
        lambda: fig_feature_families(paths),
        lambda: fig_pred_vs_obs(paths),
        lambda: fig_ablation(paths),
    ]
    ceiling = None
    try:
        ceiling = fig_replicates(paths)
    except FileNotFoundError as exc:
        log.warning("skipping replicate figure: %s", exc)
    steps.append(lambda: fig_performance(paths, ceiling))
    for step in steps:
        try:
            step()
        except FileNotFoundError as exc:
            log.warning("skipping figure, input missing: %s", exc)
