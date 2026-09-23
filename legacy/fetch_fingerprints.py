#%%
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from rdkit import DataStructs
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

def smiles_to_mol(smile):
    """Konverterar SMILES-sträng till RDKit mol-objekt"""
    if pd.isna(smile):
        return None
    try:
        mol = Chem.MolFromSmiles(smile)
        return mol
    except:
        return None

def generate_morgan_fingerprint(mol, radius=2, nBits=2048):
    """
    Genererar Morgan fingerprint (ECFP4) för en molekyl
    
    Parameters:
    mol (rdkit.Chem.rdchem.Mol): RDKit molekyl-objekt
    radius (int): Radius för Morgan-algoritmen (2 = ECFP4)
    nBits (int): Längd på fingerprint-vektorn
    
    Returns:
    Morgan fingerprint som bit-vektor
    """
    if mol is None:
        return None
    
    try:
        # Morgan fingerprint (ECFP4) med radius 2 och 2048 bitar
        morgan_fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits)
        return morgan_fp
    except:
        return None

def fingerprint_to_numpy(fp, size=2048):
    """Konverterar RDKit fingerprint till numpy-array"""
    if fp is None:
        return np.zeros(size)
    
    arr = np.zeros(size)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr

def visualize_molecule(mol, filename, size=(400, 400)):
    """Skapar en bild av molekylen"""
    if mol is None:
        return False
    
    try:
        img = Draw.MolToImage(mol, size=size)
        img.save(filename)
        return True
    except Exception as e:
        print(f"Kunde inte skapa bild: {e}")
        return False

def create_morgan_fingerprints():
    """
    Huvudfunktion för att generera Morgan fingerprints från SMILES-strängar
    """
    # Definiera sökvägar
    smiles_path = "C:/Users/46762/VSCODE/BIG_PHARMA/data/interim/drug_smiles.csv"
    output_dir = "C:/Users/46762/VSCODE/BIG_PHARMA/data/interim/fingerprints"
    
    # Skapa output-katalog om den inte finns
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "molecule_images"), exist_ok=True)
    
    # Läs SMILES-data
    if not os.path.exists(smiles_path):
        print(f"Fel: SMILES-fil hittades inte - {smiles_path}")
        return
    
    df = pd.read_csv(smiles_path)
    print(f"Laddade {len(df)} läkemedel med SMILES")
    
    # Konvertera SMILES till molekyler
    print("Konverterar SMILES till molekyler...")
    df['mol'] = df['smile'].apply(smiles_to_mol)
    
    # Räkna giltiga molekyler
    valid_mols = df['mol'].apply(lambda x: x is not None).sum()
    print(f"Lyckades konvertera {valid_mols} av {len(df)} SMILES till molekyler")
    
    # Filtrera bort ogiltiga molekyler
    df = df[df['mol'].notnull()].reset_index(drop=True)
    
    # Generera Morgan fingerprints
    print("Genererar Morgan fingerprints (ECFP4)...")
    df['morgan_fp'] = df['mol'].apply(lambda x: generate_morgan_fingerprint(x))
    
    # Konvertera fingerprints till numpy-arrayer
    print("Konverterar fingerprints till numpy-arrayer...")
    fp_matrix = np.zeros((len(df), 2048))
    
    for i, fp in enumerate(tqdm(df['morgan_fp'])):
        if fp is not None:
            fp_matrix[i] = fingerprint_to_numpy(fp)
    
    # Skapa DataFrame med fingerprints
    fp_columns = [f"bit_{i}" for i in range(2048)]
    fp_df = pd.DataFrame(fp_matrix, columns=fp_columns)
    
    # Lägg till läkemedel-ID-kolumner
    fp_df.insert(0, 'drug', df['drug'])
    fp_df.insert(1, 'cid', df['cid'])
    
    # Spara till CSV
    fp_csv_path = os.path.join(output_dir, "morgan_fingerprints.csv")
    fp_df.to_csv(fp_csv_path, index=False)
    print(f"Sparade Morgan fingerprints till {fp_csv_path}")
    
    # Spara en komprimerad version med endast icke-noll-bitar (sparse format)
    sparse_fp_df = pd.DataFrame({
        'drug': df['drug'],
        'cid': df['cid']
    })
    
    # Räkna antalet unika bitar (icke-noll)
    nonzero_bits = (fp_matrix.sum(axis=0) > 0).sum()
    print(f"Antalet unika bitar som används i ditt dataset: {nonzero_bits} av 2048")
    
    # Välj de bitar som varierar mest för att skapa en komprimerad version
    bit_counts = fp_matrix.sum(axis=0)
    sorted_bits = np.argsort(-bit_counts)[:512]  # Ta de 512 mest förekommande bitarna
    compressed_fp = fp_matrix[:, sorted_bits]
    
    # Spara komprimerad version
    comp_columns = [f"top_bit_{i}" for i in range(512)]
    comp_df = pd.DataFrame(compressed_fp, columns=comp_columns)
    comp_df.insert(0, 'drug', df['drug'])
    comp_df.insert(1, 'cid', df['cid'])
    
    comp_csv_path = os.path.join(output_dir, "morgan_fingerprints_compressed.csv")
    comp_df.to_csv(comp_csv_path, index=False)
    print(f"Sparade komprimerade Morgan fingerprints till {comp_csv_path}")
    
    # Skapa en liten delmängd av molekylbilder för visualisering
    print("Genererar molekylvisualiseringar...")
    sample_size = min(20, len(df))
    sample_mols = df.sample(sample_size)
    
    for idx, row in sample_mols.iterrows():
        mol = row['mol']
        drug_name = row['drug']
        # Ersätt tecken som kan vara ogiltiga i filnamn
        safe_name = ''.join(c if c.isalnum() else '_' for c in drug_name)
        img_path = os.path.join(output_dir, "molecule_images", f"{safe_name}.png")
        visualize_molecule(mol, img_path)
    
    print(f"Sparade {sample_size} molekylbilder i {os.path.join(output_dir, 'molecule_images')}")
    
    # Generera en sammanfattande visualisering av bitfrekvenser
    print("Genererar visualisering av bitfrekvenser...")
    bit_counts = fp_matrix.sum(axis=0)
    
    plt.figure(figsize=(12, 6))
    plt.hist(bit_counts, bins=50)
    plt.title('Morgan Fingerprint Bitfrekvens')
    plt.xlabel('Antal läkemedel med bit aktiverad')
    plt.ylabel('Antal bitar')
    plt.savefig(os.path.join(output_dir, 'morgan_bit_frequency.png'), dpi=300)
    
    # Heatmap för att se mönster i de 20 mest förekommande bitarna
    top_n = 20
    top_bits = np.argsort(-bit_counts)[:top_n]
    top_bits_matrix = fp_matrix[:sample_size, top_bits]
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(top_bits_matrix, cmap='viridis', 
                xticklabels=[f"Bit {b}" for b in top_bits],
                yticklabels=sample_mols['drug'].values)
    plt.title(f'De {top_n} mest förekommande bitarna i Morgan fingerprints')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'morgan_top_bits.png'), dpi=300)
    
    print("Analys av fingerprints är klar!")
    
    # Spara statistik om fingerprints
    stats = {
        "total_drugs": len(df),
        "valid_molecules": valid_mols,
        "fingerprint_bits": 2048,
        "active_bits": int(nonzero_bits),
        "percent_bits_used": float(nonzero_bits/2048 * 100),
        "most_common_bit": int(np.argmax(bit_counts)),
        "most_common_bit_count": int(np.max(bit_counts)),
        "avg_bits_per_drug": float(np.mean(np.sum(fp_matrix, axis=1)))
    }
    
    stats_df = pd.DataFrame([stats])
    stats_df.to_csv(os.path.join(output_dir, 'morgan_fingerprint_stats.csv'), index=False)
    print("Statistik sparad i morgan_fingerprint_stats.csv")

if __name__ == "__main__":
    create_morgan_fingerprints()
# %%
