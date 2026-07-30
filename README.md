# DrugComb Synergy Prediction

An exploratory machine learning project investigating whether chemical and biological features can be used to predict how two drugs interact in a specific cancer cell line.

> **Note:** This was created as a school project and the repository reflects the exploratory nature of that work. The code is somewhat messy, several experiments overlap, and some scripts contain local file paths. It should therefore be viewed as a record of the learning and modeling process rather than a polished, production-ready pipeline.

## Project idea

Drug combinations can behave differently depending on the biological context. Two drugs may produce a stronger combined effect than expected, have little additional effect, or counteract each other. This project explores whether those interactions can be estimated from:

- observed drug-combination results from DrugCombDB
- molecular representations of the two drugs
- RNA expression data for the tested cell line
- cell-line and tissue metadata

The main target is the **ZIP synergy score**. A higher score indicates a more synergistic interaction, while a lower score indicates a more antagonistic interaction.

The broader purpose was to test whether combining information about both the **chemistry of the drugs** and the **biology of the cell line** could provide more useful predictions than relying only on drug names or historical synergy scores.

## Data and features

The project combines data from several sources:

- **DrugCombDB** for drug pairs, cell lines, and synergy scores
- **PubChem** for compound IDs and SMILES representations
- **Morgan fingerprints** for numerical representations of drug structures
- **DepMap** for cell-line metadata and RNA-expression data

The preprocessing includes:

- normalization of drug and cell-line names
- removal of self-pairs and incomplete observations
- canonical ordering of drug pairs
- aggregation of duplicate experiments
- optional winsorization of extreme synergy values
- mapping drug names to PubChem compounds
- generation and merging of molecular fingerprints
- matching cell lines to DepMap models
- dimensionality reduction of RNA-expression features with an autoencoder
- PCA reduction of molecular fingerprints

## Modeling and analysis

The repository contains experiments with both regression and classification:

- regression of continuous ZIP synergy scores
- classification into synergistic, neutral, and antagonistic combinations
- `HistGradientBoosting` and Random Forest baselines
- neural-network models combining drug fingerprints, RNA features, and frequency features
- PCA and UMAP for dimensionality reduction and visualization
- K-means clustering to explore patterns in the learned feature space
- grouped train/test splits and cross-validation to reduce leakage between related drug pairs

An important part of the work was avoiding target leakage. ZIP-derived labels are not included as input features when predicting synergy.

## Repository overview

| File | Purpose |
| --- | --- |
| `BRAclean.py` | Cleans and restructures the raw DrugCombDB data |
| `fetch_cid.py` | Maps normalized drug names to PubChem compound IDs |
| `fetch_smile.py` | Retrieves SMILES strings from PubChem |
| `fetch_fingerprints.py` | Generates molecular fingerprints from drug structures |
| `merge_fingerprint.py` | Merges drug fingerprints with synergy observations |
| `Omic_ny.py` | Matches cell lines to DepMap RNA data and creates latent RNA features with an autoencoder |
| `umap_depmap.py` | Runs PCA and UMAP on processed feature tables |
| `classification.ipynb` | Explores synergy classification with grouped validation |
| `main_eda (1).ipynb` | Main notebook for EDA, regression, classification, visualization, and model comparison |

## Simplified workflow

1. Clean and aggregate DrugCombDB observations.
2. Resolve drug names through PubChem.
3. Create molecular fingerprints for both drugs.
4. Match cell lines with DepMap metadata and RNA-expression profiles.
5. Reduce the high-dimensional chemical and biological features.
6. Merge all modalities into one modeling dataset.
7. Train and compare classical ML and neural-network models.
8. Explore the feature space with UMAP and clustering.

## Current limitations

- The required raw datasets are not included in the repository.
- Several scripts use hard-coded local Windows paths and must be updated before running.
- The project does not currently have a single reproducible entry point.
- Some notebook sections and experiments are duplicated.
- Dependencies are not pinned in a requirements file.
- Drug-name and cell-line matching partly relies on manual mappings.
- The models are experimental and are not intended for clinical use.

## Main takeaway

This project was primarily a learning exercise in working with multimodal biomedical data. Its main value is the full process: cleaning noisy public data, connecting several external sources, engineering chemical and biological features, thinking carefully about leakage, and comparing different approaches to drug-synergy prediction.

The next step would be to restructure the code into a reproducible pipeline, replace local paths with configuration, document the exact datasets and versions, and evaluate the models with stricter splits on previously unseen drug pairs and cell lines.

## Disclaimer

This repository is for educational and exploratory purposes only. It is not a medical tool, and its outputs should not be used to guide treatment or clinical decisions.
