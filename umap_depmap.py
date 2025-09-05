#%%
import pandas as pd
from pathlib import Path
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

# Vill du köra UMAP, installera först:
# pip install umap-learn
try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False

# Paths (byt till dina riktiga paths)
scored_path = Path("C:/Users/46762/VSCODE/BIG_PHARMA/data/interim/scored_clean.csv")
merged_path = Path("C:/Users/46762/VSCODE/BIG_PHARMA/data/interim/out/drugcomb_with_cell_autoenc.csv")

# Load
scored = pd.read_csv(scored_path)
merged = pd.read_csv(merged_path)

# Funktion för PCA/UMAP plot med sampling
def plot_reduction(df: pd.DataFrame, title: str, max_samples=10000):
    num = df.select_dtypes(include='number').fillna(0)
    print(f"{title} - shape: {num.shape}")
    
    # Sample if necessary
    if num.shape[0] > max_samples:
        print(f"Sampling {max_samples} rows from {num.shape[0]} total")
        num = num.sample(n=max_samples, random_state=42)
    
    # PCA
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(num)
    plt.figure(figsize=(6,6))
    plt.scatter(coords[:,0], coords[:,1], s=2, alpha=0.4)
    plt.title(f"PCA - {title}")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.show()
    print(f"Explained variance ratio (PC1+PC2): {pca.explained_variance_ratio_.sum():.2%}")

    # UMAP om tillgänglig
    if UMAP_AVAILABLE:
        reducer = umap.UMAP(
            n_neighbors=15, 
            min_dist=0.1, 
            metric="cosine", 
            random_state=42,
            low_memory=True  # Lower memory usage
        )
        coords_umap = reducer.fit_transform(num)
        plt.figure(figsize=(6,6))
        plt.scatter(coords_umap[:,0], coords_umap[:,1], s=2, alpha=0.4)
        plt.title(f"UMAP - {title}")
        plt.show()

# Kör på båda
plot_reduction(scored, "scored_clean.csv", max_samples=8000)
plot_reduction(merged, "drugcomb_with_cell_autoenc.csv", max_samples=8000)
