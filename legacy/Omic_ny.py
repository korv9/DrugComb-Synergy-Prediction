#%%
from __future__ import annotations
import re
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras import layers, callbacks

# =========================================================
# CONFIG
# =========================================================
# Sökvägar (anpassa vid behov)
DRUGCOMB_CSV = Path("C:/Users/46762/VSCODE/BIG_PHARMA/data/interim/scored_clean.csv")
MODEL_CSV    = Path("Model.csv")  # DepMap Model.csv
EXPR_CSV     = Path("OmicsExpressionTPMLogp1HumanProteinCodingGenes.csv")  # DepMap RNA (TPM log1p)

# Output
OUT_DIR = Path("C:/Users/46762/VSCODE/BIG_PHARMA/data/interim/out")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Autoencoder-parametrar
LATENT_DIM = 50
EPOCHS     = 40
BATCH_SIZE = 64
VAL_SPLIT  = 0.1
SEED       = 42

# =========================================================
# Hjälpfunktioner
# =========================================================
def norm_cellname(s: str | None) -> str | None:
    """Normalisera cellinjenamn (gemener + ta bort icke a-z0-9)."""
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return None
    s = str(s).strip().lower()
    s = re.sub(r"[^a-z0-9]+", "", s)
    return s or None

# Manuell mappning för knepiga alias -> DepMap/CCLEName-liknande etiketter
MANUAL_CELLNAME_MAP = {
    # Säkra/vanliga alias
    "7860": "786O_KIDNEY",
    "colo320dm": "COLO320",
    "colo858": "COLO858",
    "ctr": "SMS-CTR",
    "efm192b": "EFM19",
    "hl60tb": "HL60_HAEMATOPOIETIC_AND_LYMPHOID_TISSUE",
    "lncap": "LNCAPCLONEFGC",
    "mdamb435": "MDAMB435S_SKIN",
    "msto": "MSTO211H",
    "ovcar3": "NIHOVCAR3_OVARY",
    "pa1": "PA1_OVARY",
    "u251": "U251MG_BRAIN",
    "uwb1289brca1": "UWB1289BRCA1_OVARY",
    "colo320": "COLO320",

    # Ej DepMap/ej mänskliga
    "3d7": None, "dd2": None, "hb3": None,

    # Osäkra/ej hittade – lämna som None tills verifierat
    "kb31": None, "kbchr8511": None, "ed40515": None,
    "jhh136": None, "jhh520": None, "kbm7": None, "mak": None,
    "nciadrres": None, "sr": None, "sudipgxiii": None, "tmd8": None,
    "colo858_raw": None,
}

def load_drugcomb(path: Path) -> pd.DataFrame:
    """Läs DrugComb-data och normalisera cellinjenamn."""
    print(f"[DrugComb] Läser: {path}")
    df = pd.read_csv(path)
    if "cell_line" not in df.columns:
        raise ValueError("Saknar kolumnen 'cell_line' i DrugComb CSV.")
    df["cell_norm"] = df["cell_line"].astype(str).apply(norm_cellname)
    df["cell_norm"] = df["cell_norm"].apply(lambda x: MANUAL_CELLNAME_MAP.get(x, x) if x is not None else None)
    print(f"[DrugComb] Rader: {len(df)} | Unika celler (raw): {df['cell_line'].nunique()} | Unika (norm): {df['cell_norm'].nunique()}")
    return df

def load_model_map(model_csv: Path) -> pd.DataFrame:
    """Läs DepMap Model.csv och skapa 'best_key' för matchning."""
    print(f"[Model] Läser: {model_csv}")
    meta = pd.read_csv(model_csv, low_memory=False)

    # Kontrollera nödvändiga kolumner
    need_cols = {"ModelID", "CellLineName"}
    if not need_cols.issubset(set(meta.columns)):
        raise ValueError(f"Model.csv saknar någon av kolumnerna: {need_cols}")

    # Normalisera namn
    meta["CellLineName_norm"] = meta["CellLineName"].apply(norm_cellname)
    if "StrippedCellLineName" in meta.columns:
        meta["StrippedCellLineName_norm"] = meta["StrippedCellLineName"].apply(norm_cellname)
    else:
        meta["StrippedCellLineName_norm"] = np.nan

    # Välj bästa nyckel
    meta["best_key"] = np.where(meta["StrippedCellLineName_norm"].notna(),
                                meta["StrippedCellLineName_norm"],
                                meta["CellLineName_norm"])

    # Behåll relevanta kolumner och undvik dubletter
    keep = meta[["ModelID", "CellLineName", "best_key", "OncotreeLineage"]].dropna(subset=["best_key"])
    keep = keep.sort_values("ModelID").drop_duplicates("best_key", keep="first")
    keep["ModelID"] = keep["ModelID"].astype(str)
    print(f"[Model] Unika nycklar: {len(keep)}")
    return keep

def build_autoencoder(input_dim: int, latent_dim: int) -> tuple[keras.Model, keras.Model]:
    """Bygg RNA-autoencoder och returnera (autoencoder, encoder)."""
    inp = keras.Input(shape=(input_dim,), name="rna_input")
    x = layers.Dense(1024, activation="relu")(inp)
    x = layers.Dense(512, activation="relu")(x)
    x = layers.Dense(256, activation="relu")(x)
    z = layers.Dense(latent_dim, activation="linear", name="latent")(x)
    x = layers.Dense(256, activation="relu")(z)
    x = layers.Dense(512, activation="relu")(x)
    x = layers.Dense(1024, activation="relu")(x)
    out = layers.Dense(input_dim, activation="linear", name="recon")(x)

    autoenc = keras.Model(inp, out, name="rna_autoencoder")
    encoder = keras.Model(inp, z, name="rna_encoder")
    return autoenc, encoder

# =========================================================
# Huvudflöde
# =========================================================
def main():
    np.random.seed(SEED)
    keras.utils.set_random_seed(SEED)
    print("\n=== PIPELINE START ===\n")

    # 1) DrugComb
    drugcomb = load_drugcomb(DRUGCOMB_CSV)

    # 2) Model-karta
    model_map = load_model_map(MODEL_CSV)

    # 3) Matcha celler -> ModelID
    print("[Match] Mappar cell_norm till ModelID ...")
    cell_map = (
        drugcomb[["cell_norm"]].drop_duplicates()
        .merge(model_map[["best_key", "ModelID", "OncotreeLineage"]], left_on="cell_norm", right_on="best_key", how="left")
    )
    miss = cell_map["ModelID"].isna().sum()
    total = len(cell_map)
    print(f"[Match] Saknar ModelID för {miss}/{total} celler")

    # Mergar ModelID + Lineage in i DrugComb
    cell_map["ModelID"] = cell_map["ModelID"].astype(str)
    drugcomb = drugcomb.merge(cell_map[["cell_norm", "ModelID", "OncotreeLineage"]], on="cell_norm", how="left")

    need_models = drugcomb["ModelID"].dropna().astype(str).unique().tolist()
    print(f"[Match] Unika ModelIDs som behövs: {len(need_models)}")

    # 4) RNA-data (DepMap)
    print("\n[RNA] Läser uttrycksdata ...")
    rna = pd.read_csv(EXPR_CSV, low_memory=False)

    # Försök hitta ID-kolumn
    id_cols = [c for c in rna.columns if c.lower() in ("modelid", "depmap_id", "depmapid")]
    if id_cols:
        id_col = id_cols[0]
        rna = rna.set_index(id_col)
    else:
        # om filen redan har ModelID som index i första kolumnen, avkommentera nästa rad i stället:
        # rna = pd.read_csv(EXPR_CSV, index_col=0, low_memory=False)
        pass

    # Säkerställ strängindex
    rna.index = rna.index.astype(str)
    print(f"[RNA] Rå form (rader x kolumner): {rna.shape}")

    # Detektera orientering
    need_models_s = pd.Series(need_models, dtype=str)
    n_match_rows = need_models_s.isin(rna.index).sum()
    n_match_cols = need_models_s.isin(rna.columns.astype(str)).sum()
    print(f"[RNA] Models i rader: {n_match_rows} | i kolumner: {n_match_cols}")

    if n_match_rows == 0 and n_match_cols > 0:
        print("[RNA] Upptäckte modeller i kolumner -> transponerar")
        rna = rna.T
        rna.index = rna.index.astype(str)
    elif n_match_rows == 0 and n_match_cols == 0:
        print("[RNA] Exempel ModelIDs:", need_models[:5])
        print("[RNA] Exempel index:", rna.index.astype(str)[:5].tolist())
        print("[RNA] Exempel kolumner:", rna.columns.astype(str)[:5].tolist())
        raise ValueError("Ingen överlapp mellan ModelIDs och RNA-matrisens axlar. Kontrollera EXPR_CSV och ModelID-format.")

    # 5) Filtrera RNA till de modeller vi behöver
    keep_idx = rna.index.intersection(need_models_s)
    rna_sub = rna.loc[keep_idx].copy()
    print(f"[RNA] Subset efter model-filter: {rna_sub.shape}")

    # Ta bort icke-numeriska kolumner om sådana smugit in
    num_cols = rna_sub.columns[rna_sub.dtypes.apply(np.issubdtype, args=(np.number,))]
    if len(num_cols) < rna_sub.shape[1]:
        rna_sub = rna_sub[num_cols]
        print(f"[RNA] Tog bort icke-numeriska kolumner -> {rna_sub.shape}")

    # Variansfilter (ta bort konstanta gener)
    var = rna_sub.var(axis=0, numeric_only=True)
    rna_sub = rna_sub.loc[:, var > 0.0]
    print(f"[RNA] Efter variansfilter: {rna_sub.shape}")

    if rna_sub.shape[0] < 2 or rna_sub.shape[1] < 2:
        raise ValueError(f"För få sampel/feature efter filtrering: {rna_sub.shape}")

    # 6) Centering (features-wise), ej skalning (datan är log-transformerad)
    X = rna_sub.values.astype(np.float32)
    X = X - X.mean(axis=0, keepdims=True)

    # 7) Train/val-split
    X_train, X_val = train_test_split(X, test_size=VAL_SPLIT, random_state=SEED)
    print(f"[Train] Train: {X_train.shape} | Val: {X_val.shape}")

    # 8) Bygg och träna autoencoder
    print("\n[AE] Bygger och tränar autoencoder ...")
    autoenc, encoder = build_autoencoder(input_dim=X.shape[1], latent_dim=LATENT_DIM)
    autoenc.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3), loss="mse")

    cbs = [
        callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-5, verbose=1),
    ]

    hist = autoenc.fit(
        X_train, X_train,
        validation_data=(X_val, X_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        shuffle=True,
        callbacks=cbs,
        verbose=1,
    )

    best_val = float(np.min(hist.history.get("val_loss", [np.nan])))
    print(f"[AE] Klar! Bästa val-loss: {best_val:.6f}")

    # 9) Latenta embeddingar för alla modeller i rna_sub
    print("\n[AE] Extraherar latenta vektorer ...")
    Z = encoder.predict(X, batch_size=BATCH_SIZE)
    z_cols = [f"rna_latent{i+1}" for i in range(Z.shape[1])]
    rna_latent = pd.DataFrame(Z, index=rna_sub.index.astype(str), columns=z_cols)

     # 10) Tissues (one-hot)
    print("[Meta] Lägger till tissue (OncotreeLineage) ...")
    tissues = model_map.set_index("ModelID")["OncotreeLineage"].reindex(rna_sub.index.astype(str))
    tissues = tissues.fillna("unknown")
    tissue_ohe = pd.get_dummies(tissues, prefix="tissue")

# 11) Slå ihop cell-features (ModelID som kolumn – inte index)
    rna_latent.index = rna_sub.index.astype(str)   # säkerställ str
    rna_latent.index.name = "ModelID"              # döp indexet till ModelID
    cell_feats = rna_latent.join(tissue_ohe)       # join på index
    cell_feats = cell_feats.reset_index()          # gör ModelID till *kolumn*
    print(f"[Out] Cell features shape: {cell_feats.shape}")

# (valfritt men bra): säkerställ str-dtyp
    cell_feats["ModelID"] = cell_feats["ModelID"].astype(str)
    drugcomb["ModelID"] = drugcomb["ModelID"].astype(str)

# 12) Spara och merge
    out_cell_csv = OUT_DIR / "cell_rna_autoenc.csv"
    cell_feats.to_csv(out_cell_csv, index=False)
    print(f"[Out] Sparade cell-embeddingar: {out_cell_csv}")

    merged = drugcomb.merge(cell_feats, on="ModelID", how="left")
    out_merged_csv = OUT_DIR / "drugcomb_with_cell_autoenc.csv"
    merged.to_csv(out_merged_csv, index=False)
    print(f"[Out] Sparade sammanslagen fil: {out_merged_csv}")


#=========================================================
if __name__ == "__main__":
    main()
