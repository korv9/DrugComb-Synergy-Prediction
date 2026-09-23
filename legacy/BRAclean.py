# clean_scored.py
#%%
import pandas as pd, numpy as np
from pathlib import Path
from sklearn.preprocessing import LabelEncoder

# =========================
# CONFIG (enkla togglar)
# =========================
INPUT_PATH = (Path("C:\\Users\\46762\\VSCODE\\BIG_PHARMA\\data\\raw\\drugcombs_scored.csv"))
OUT_DIR = Path("C:\\Users\\46762\\VSCODE\\BIG_PHARMA\\data\\interim")
APPLY_WINSOR = True
WINSOR_Q = (0.005, 0.995)   # 0.5% / 99.5%

ADD_PAIR_ID = True
ADD_SYNERGY_SUMMARY = False      # focus on ZIP only; summary not needed
ADD_LABEL_ENCODINGS = True 
APPLY_CUTOFFS = True
TARGET_COL       = "synergy_zip"            # skapa synergy_class och synergy_binary
POS_CUTOFF = 10
NEG_CUTOFF = -10     # cell_id, drug_min_id, drug_max_id
EXPORT_SUMMARY_TABLES = True 
# Drop non-ZIP synergy scales from cleaned dataset (set True to keep them)
KEEP_OTHER_SCALES = False


#%%
# =========================
# Hjälpfunktioner
# =========================
def norm_str(x):
    if pd.isna(x): return np.nan
    s = str(x).strip().lower().replace("–","-")
    return " ".join(s.split())

def checkpoint(title, df=None, cols=None, head=5):
    print(f"\n=== {title} ===")
    if df is not None:
        print(f"Shape: {df.shape}")
        if cols:
            try:
                print(df[cols].head(head))
            except Exception:
                print(df.head(head))
        else:
            print(df.head(head))

def print_missing(df, cols=None):
    print("\n[Missing values per column]")
    if cols is None:
        print(df.isnull().sum().sort_values(ascending=False).head(20))
    else:
        print(df[cols].isnull().sum())

def winsorize_df(df, cols, lo_q, hi_q):
    for c in cols:
        lo, hi = df[c].quantile([lo_q, hi_q])
        df[c] = df[c].clip(lo, hi)
    return df

# --- Load ---
df = pd.read_csv(INPUT_PATH)
checkpoint("Loaded raw", df)
print_missing(df)

#%%
df.head(10)
# Basic dataframe info
print("Dataset shape:", df.shape)
print("\nDataframe info:")
df.info()

# Display numerical summaries for synergy scores
print("\nSynergy scores summary statistics:")
synergy_cols = ["ZIP", "Bliss", "Loewe", "HSA"]
print(df[synergy_cols].describe().round(2))



# Display top cell lines
print("\nMost frequent cell lines:")
print(df['Cell line'].value_counts().head(10))
#%%
# --- Rename ---
rename = {"Drug1":"drug1","Drug2":"drug2","Cell line":"cell_line",
          "ZIP":"synergy_zip","Bliss":"synergy_bliss",
          "Loewe":"synergy_loewe","HSA":"synergy_hsa"}
print("\n[Action] Rename columns ->", rename)
df = df.rename(columns=rename)
checkpoint("After rename (preview)", df, cols=list(rename.values()))
# Drop non-ZIP scales if requested
if not KEEP_OTHER_SCALES:
    drop_cols = [c for c in ["synergy_bliss","synergy_loewe","synergy_hsa"] if c in df.columns]
    if drop_cols:
        print("[Action] Dropping non-ZIP synergy columns:", drop_cols)
        df = df.drop(columns=drop_cols)

#%%
# --- String normalize drug/cell columns ---
for c in ["drug1","drug2","cell_line"]:
    print(f"[Action] Normalize strings in column: {c}")
    before_unique = df[c].nunique(dropna=True)
    df[c] = df[c].apply(norm_str)
    after_unique = df[c].nunique(dropna=True)
    print(f"   Unique before: {before_unique} -> after: {after_unique}")
checkpoint("After string normalization", df, cols=["drug1","drug2","cell_line"])


#%%
# --- Convert synergy cols to numeric ---
synergy_cols = [c for c in ["synergy_zip","synergy_bliss","synergy_loewe","synergy_hsa"] if c in df.columns]
print("\n[Action] To numeric for synergy columns:", synergy_cols)
for c in synergy_cols:
    df[c] = pd.to_numeric(df[c], errors="coerce")
print_missing(df, cols=synergy_cols)


#%%
# --- Remove rows with missing key strings ---
print("\n[Action] Drop rows with NA in ['drug1','drug2','cell_line']")
before = df.shape[0]
df = df.dropna(subset=["drug1","drug2","cell_line"])
print(f"   Dropped: {before - df.shape[0]} rows")
checkpoint("After dropping NA in keys", df, cols=["drug1","drug2","cell_line"])


#%%
# --- Remove self-pairs ---
print("\n[Action] Remove rows where drug1 == drug2")
before = df.shape[0]
df = df[df["drug1"] != df["drug2"]]
print(f"   Removed: {before - df.shape[0]} rows")


#%%
# --- Canonical pair (drug_min, drug_max) ---
print("\n[Action] Create canonical pair (drug_min, drug_max)")
pairs = df[["drug1","drug2"]].apply(lambda r: tuple(sorted([r["drug1"], r["drug2"]])), axis=1)
df["drug_min"] = [p[0] for p in pairs]
df["drug_max"] = [p[1] for p in pairs]
checkpoint("After canonical pairs", df, cols=["drug1","drug2","drug_min","drug_max"])


#%%
# --- Aggregate duplicates on (drug_min, drug_max, cell_line) ---
print("\n[Action] Aggregate duplicates with mean on synergy columns")
agg = {c: "mean" for c in synergy_cols}
df_clean = (
    df.groupby(["drug_min","drug_max","cell_line"], as_index=False)
      .agg(agg)
)
checkpoint("After aggregation", df_clean, cols=["drug_min","drug_max","cell_line"] + synergy_cols)


#%%
# --- Winsorization (optional) ---
if APPLY_WINSOR:
    print(f"\n[Action] Winsorize synergy cols at {int(WINSOR_Q[0]*100)}% / {int(WINSOR_Q[1]*100)}%")
    df_clean = winsorize_df(df_clean, synergy_cols, WINSOR_Q[0], WINSOR_Q[1])
    # Quick check on describe
    print("\n[Check] Describe after winsorization (synergy cols):")
    print(df_clean[synergy_cols].describe().round(3))


#%%
# --- Feature: pair_id ---
if ADD_PAIR_ID:
    print("\n[Action] Add pair_id (drug_min + '_' + drug_max)")
    df_clean["pair_id"] = df_clean["drug_min"] + "_" + df_clean["drug_max"]
    checkpoint("After pair_id", df_clean, cols=["pair_id","drug_min","drug_max"])


#%%
# --- Feature: synergy summaries ---
if ADD_SYNERGY_SUMMARY:
    print("\n[Action] Add synergy_mean and synergy_std")
    df_clean["synergy_mean"] = df_clean[synergy_cols].mean(axis=1)
    df_clean["synergy_std"]  = df_clean[synergy_cols].std(axis=1).fillna(0.0)
    checkpoint("After synergy summaries", df_clean, cols=["synergy_mean","synergy_std"])


#%%
# --- Label Encodings (for ML) ---
if ADD_LABEL_ENCODINGS:
    print("\n[Action] Label encode: cell_line, drug_min, drug_max")
    le_cell, le_drug = LabelEncoder(), LabelEncoder()

    df_clean["cell_id"] = le_cell.fit_transform(df_clean["cell_line"])
    # important: fit once on union of drugs to keep consistent mapping
    all_drugs = pd.Index(df_clean["drug_min"]).append(pd.Index(df_clean["drug_max"])).unique()
    le_drug.fit(all_drugs)
    df_clean["drug_min_id"] = le_drug.transform(df_clean["drug_min"])
    df_clean["drug_max_id"] = le_drug.transform(df_clean["drug_max"])

    checkpoint("After label encodings", df_clean,
               cols=["cell_line","cell_id","drug_min","drug_min_id","drug_max","drug_max_id"])

#%%
# --- Feature: drug and cell line frequency features ---
print("\n[Action] Add drug and cell line frequency features")

# Drug frequency features
drug_min_freq = df_clean['drug_min'].value_counts().to_dict()
drug_max_freq = df_clean['drug_max'].value_counts().to_dict()

df_clean['drug_min_freq'] = df_clean['drug_min'].map(drug_min_freq)
df_clean['drug_max_freq'] = df_clean['drug_max'].map(drug_max_freq)

# Cell line frequency feature
cell_freq = df_clean['cell_line'].value_counts().to_dict()
df_clean['cell_freq'] = df_clean['cell_line'].map(cell_freq)

# Optional: Add log-transformed frequencies (often helps with skewed distributions)
df_clean['drug_min_freq_log'] = np.log1p(df_clean['drug_min_freq'])
df_clean['drug_max_freq_log'] = np.log1p(df_clean['drug_max_freq'])
df_clean['cell_freq_log'] = np.log1p(df_clean['cell_freq'])

checkpoint("After frequency features", df_clean, 
           cols=['drug_min', 'drug_min_freq', 'drug_max', 'drug_max_freq', 'cell_line', 'cell_freq'])

#%%
# --- Quick imbalance checks (prints only) ---
print("\n[Quick check] Top cell_lines by count:")
print(df_clean["cell_line"].value_counts().head(10))
if ADD_PAIR_ID:
    print("\n[Quick check] Top pair_id by count:")
    print(df_clean["pair_id"].value_counts().head(10))

#%%
# --- Feature: synergy summaries ---
if ADD_SYNERGY_SUMMARY:
    print("\n[Action] Add synergy_mean and synergy_std")
    df_clean["synergy_mean"] = df_clean[synergy_cols].mean(axis=1)
    df_clean["synergy_std"]  = df_clean[synergy_cols].std(axis=1).fillna(0.0)
    checkpoint("After synergy summaries", df_clean, cols=["synergy_mean","synergy_std"])

#%%
# --- Synergy cutoffs (optional) ---
# --- Synergy cutoffs (optional) ---
if APPLY_CUTOFFS and TARGET_COL in df_clean:
    print(f"\n[Action] Apply synergy cutoffs on {TARGET_COL} at {NEG_CUTOFF}/{POS_CUTOFF}")

    def classify_synergy(x):
        if pd.isna(x): return np.nan
        if x > POS_CUTOFF:   return "synergistic"
        elif x < NEG_CUTOFF: return "antagonistic"
        else:                return "neutral"

    df_clean["synergy_class"]  = df_clean[TARGET_COL].map(classify_synergy)
    df_clean["synergy_binary"] = df_clean["synergy_class"].map(
        lambda z: np.nan if pd.isna(z) else (1 if z == "synergistic" else 0)
    )
    checkpoint("After synergy cutoffs", df_clean, cols=[TARGET_COL,"synergy_class","synergy_binary"])





#%%
# --- Save clean ---
out_file = OUT_DIR / "scored_clean.csv"
df_clean.to_csv(out_file, index=False)


#%%
# --- Final summary ---
print("\n=== DATA CLEANING SUMMARY ===")
print(f"Original shape: {df.shape}  (after initial NA/drop/self-pair filtering & before groupby)")
print(f"Final shape:    {df_clean.shape} -> {out_file}")
print(f"Columns renamed: {list(rename.keys())} -> {list(rename.values())}")
print(f"String columns normalized: {['drug1', 'drug2', 'cell_line']}")
print(f"Synergy columns numeric: {synergy_cols}")
print(f"Winsorization applied: {APPLY_WINSOR} at {WINSOR_Q}")
print(f"Added features: pair_id={ADD_PAIR_ID}, synergy_mean/std={ADD_SYNERGY_SUMMARY}, encodings={ADD_LABEL_ENCODINGS}")
if EXPORT_SUMMARY_TABLES:
    print("Exported: data/interim/cell_stats.csv, data/interim/pair_stats.csv")
print("================================")




