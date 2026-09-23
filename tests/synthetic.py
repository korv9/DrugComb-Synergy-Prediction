"""Tiny *synthetic* raw files shaped like DrugCombDB / DepMap, for tests only.

Nothing here is real data: values are random with a planted signal so the
pipeline can be exercised end-to-end without network access.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# name -> SMILES; "5-fu" and "fluorouracil" are the same molecule on purpose,
# "fluorouracil sodium" differs only by a counter-ion (tests salt stripping).
DRUGS = {
    "fluorouracil": "O=c1[nH]cc(F)c(=O)[nH]1",
    "5-fu": "O=c1[nH]cc(F)c(=O)[nH]1",
    "aspirin": "CC(=O)Oc1ccccc1C(=O)O",
    "caffeine": "Cn1cnc2c1c(=O)n(C)c(=O)n2C",
    "ibuprofen": "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
    "paracetamol": "CC(=O)Nc1ccc(O)cc1",
    "metformin": "CN(C)C(=N)N=C(N)N",
    "temozolomide": "Cn1nnc2c(ncn2c1=O)C(N)=O",
    "gemcitabine": "Nc1ccn(C2OC(CO)C(O)C2(F)F)c(=O)n1",
    "cisplatin": "N.N.Cl[Pt]Cl",
    "vorinostat": "ONC(=O)CCCCCCC(=O)Nc1ccccc1",
    "bortezomib": "CC(C)CC(NC(=O)C(Cc1ccccc1)NC(=O)c1cnccn1)B(O)O",
    "erlotinib": "COCCOc1cc2ncnc(Nc3cccc(c3)C#C)c2cc1OCCOC",
    "imatinib": "Cc1ccc(NC(=O)c2ccc(CN3CCN(C)CC3)cc2)cc1Nc1nccc(n1)-c1cccnc1",
}
UNKNOWN_DRUG = "zinc99999999"  # not in drug_chemical_info -> unresolved (PubChem off)

CELLS = {  # DrugCombDB spelling -> (ModelID, DepMap name, lineage)
    "NCI-H460": ("ACH-000463", "NCI-H460", "Lung"),
    "A549": ("ACH-000681", "A549", "Lung"),
    "MCF7": ("ACH-000019", "MCF7", "Breast"),
    "T-47D": ("ACH-000147", "T47D", "Breast"),
    "HL-60": ("ACH-000002", "HL-60", "Myeloid"),
    "786-0": ("ACH-000649", "786-O", "Kidney"),  # needs the 7860 -> 786o alias
    "U251": ("ACH-000232", "U251MG", "CNS/Brain"),  # needs the u251 -> u251mg alias
    "SW-620": ("ACH-000421", "SW620", "Bowel"),
}
MALARIA = "3D7"


def write_raw(root: Path, seed: int = 0, n_rows: int = 3000) -> None:
    rng = np.random.default_rng(seed)
    names = list(DRUGS) + [UNKNOWN_DRUG]
    cells = list(CELLS) + [MALARIA]
    drug_eff = {d: rng.normal(0, 4) for d in names}
    drug_eff["5-fu"] = drug_eff["fluorouracil"]
    cell_eff = {c: rng.normal(0, 3) for c in cells}

    d1 = rng.choice(names, n_rows)
    d2 = rng.choice(names, n_rows)
    c = rng.choice(cells, n_rows)
    zip_ = np.array([drug_eff[a] + drug_eff[b] + cell_eff[k] for a, b, k in zip(d1, d2, c)])
    zip_ = zip_ + rng.normal(0, 3, n_rows)
    scored = pd.DataFrame({
        "ID": np.arange(1, n_rows + 1),
        "Drug1": [s.upper() if i % 7 == 0 else s for i, s in enumerate(d1)],  # case noise
        "Drug2": d2,
        "Cell line": c,
        "ZIP": zip_.round(3),
        "Bliss": (zip_ + rng.normal(0, 2, n_rows)).round(3),
        "Loewe": (zip_ + rng.normal(-3, 5, n_rows)).round(3),
        "HSA": (zip_ + rng.normal(1, 3, n_rows)).round(3),
    }).astype({"ZIP": object})
    scored.loc[5, "ZIP"] = "NA"          # non-numeric score
    scored.loc[6, "Drug2"] = None        # missing name
    scored.loc[7, "ZIP"] = 500.0         # implausible
    # replicates: repeat 200 rows with fresh noise
    rep = scored.iloc[10:210].copy()
    rep["ZIP"] = pd.to_numeric(rep["ZIP"]) + rng.normal(0, 3, len(rep))
    scored = pd.concat([scored, rep], ignore_index=True)

    (root / "drugcombdb").mkdir(parents=True, exist_ok=True)
    scored.to_csv(root / "drugcombdb" / "drugcombs_scored.csv", index=False)
    pd.DataFrame({
        "drugName": list(DRUGS),
        "cIds": [f"CIDs{1000 + i:08d}" for i in range(len(DRUGS))],
        "drugNameOfficial": list(DRUGS),
        "molecularWeight": 0.0,
        "smilesString": list(DRUGS.values()),
    }).to_csv(root / "drugcombdb" / "drug_chemical_info.csv", index=False)

    # DepMap: screened models + unscreened ones
    models = [(mid, name, lin) for mid, name, lin in CELLS.values()]
    lineages = ["Lung", "Breast", "Skin", "Bowel", "Myeloid", "Lymphoid"]
    models += [(f"ACH-9{i:05d}", f"EXTRA{i}", lineages[i % len(lineages)]) for i in range(40)]
    model_df = pd.DataFrame(models, columns=["ModelID", "CellLineName", "OncotreeLineage"])
    model_df["StrippedCellLineName"] = model_df["CellLineName"].str.replace(r"[^A-Za-z0-9]", "",
                                                                            regex=True)
    model_df["CCLEName"] = model_df["StrippedCellLineName"].str.upper() + "_" + \
        model_df["OncotreeLineage"].str.upper().str.replace(r"[^A-Z]", "", regex=True)
    model_df["OncotreePrimaryDisease"] = model_df["OncotreeLineage"] + " cancer"
    (root / "depmap").mkdir(parents=True, exist_ok=True)
    model_df.to_csv(root / "depmap" / "Model.csv", index=False)

    genes = [f"GENE{i} ({i})" for i in range(300)]
    lin_code = model_df["OncotreeLineage"].astype("category").cat.codes.to_numpy()
    expr = rng.normal(3, 1, (len(model_df), len(genes))) + lin_code[:, None] * \
        rng.normal(0, 0.5, len(genes))[None, :]
    # DepMap 24Q4 layout: ModelID in an unnamed first column
    expr_df = pd.DataFrame(expr.clip(0).round(4), columns=genes, index=model_df["ModelID"])
    expr_df.index.name = None
    expr_df.to_csv(root / "depmap" / "OmicsExpressionProteinCodingGenesTPMLogp1.csv")
