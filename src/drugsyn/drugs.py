"""Drug entity resolution: names -> structures -> one canonical drug per molecule.

Resolution order for every distinct (normalised) drug name:

1. DrugCombDB's own ``drug_chemical_info.csv`` (name, PubChem CID, SMILES)
2. PubChem PUG-REST fallback (name -> CID -> SMILES), cached on disk
3. unresolved (kept, but without chemistry features)

Structures are standardised with RDKit (largest fragment = salt stripping,
neutralisation) and keyed by InChIKey, so different names for the same
molecule ("5-FU", "fluorouracil") collapse into one ``drug_key``.
"""

from __future__ import annotations

import json
import logging
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import quote

import numpy as np
import pandas as pd
import requests

from .config import Paths
from .ingest import norm_drug

log = logging.getLogger(__name__)

DESCRIPTORS = [
    "mol_wt", "logp", "tpsa", "hbd", "hba", "rot_bonds",
    "rings", "aromatic_rings", "frac_csp3", "heavy_atoms", "qed",
]

_SMILES_KEYS = ("SMILES", "IsomericSMILES", "CanonicalSMILES", "ConnectivitySMILES")
_SALT_WORDS = re.compile(
    r"\b(hydrochloride|dihydrochloride|hcl|hydrobromide|maleate|mesylate|dimesylate|tosylate|"
    r"sulfate|phosphate|nitrate|acetate|tartrate|fumarate|succinate|citrate|lactate|"
    r"sodium|potassium|calcium|besylate|hydrate|monohydrate|dihydrate)\b"
)


# --------------------------------------------------------------------------- sources
def load_drugcombdb_info(path: Path) -> pd.DataFrame:
    """Parse ``drug_chemical_info.csv`` into (name_norm, cid, smiles)."""
    info = pd.read_csv(path, low_memory=False, encoding_errors="replace")
    lower = {c.lower(): c for c in info.columns}
    name_col = lower.get("drugname") or next(
        c for lc, c in lower.items() if "name" in lc and "official" not in lc
    )
    smiles_col = next((c for lc, c in lower.items() if "smiles" in lc), None)
    cid_col = next((c for lc, c in lower.items() if "cid" in lc), None)

    out = pd.DataFrame({"name_norm": info[name_col].map(norm_drug)})
    out["cid"] = (
        info[cid_col].astype(str).str.extract(r"(\d+)")[0].astype(float)
        if cid_col else np.nan
    )
    out.loc[out["cid"] <= 0, "cid"] = np.nan
    missing = {"", "none", "nan", "null", "na", "n/a"}

    def clean(col):
        if col is None:
            return None
        v = info[col].astype(str).str.strip()
        return v.where(~v.str.lower().isin(missing))

    out["smiles"] = clean(smiles_col)
    official_col = next((c for lc, c in lower.items() if "official" in lc), None)
    out["official_name"] = clean(official_col).map(norm_drug) if official_col else None
    out = out.dropna(subset=["name_norm"]).drop_duplicates("name_norm")
    return out


def name_variants(name: str) -> list[str]:
    """Query variants for PubChem name lookup, most specific first."""
    variants = [name]
    stripped = re.sub(r"\s*\([^)]*\)\s*$", "", name)          # trailing "(alias)"
    stripped = " ".join(_SALT_WORDS.sub("", stripped).split())  # salt / hydrate words
    for v in (stripped, stripped.replace(" ", "-"), stripped.replace("-", " ")):
        if v and v not in variants:
            variants.append(v)
    return variants


def display_name(name: str) -> str:
    """Human-readable name: drop salt/hydrate words and trailing "(alias)"."""
    variants = name_variants(name)
    return variants[1] if len(variants) > 1 else name


class PubChem:
    """Tiny PUG-REST client: rate-limited (PubChem allows 5 req/s), retried on
    "server busy", parallel for name lookups, and cached on disk so reruns
    never hit the API twice and an interrupted run resumes where it stopped."""

    def __init__(self, cfg: dict, cache_path: Path):
        self.base = cfg["base_url"].rstrip("/")
        self.min_interval = cfg.get("sleep_s", 0.25)
        self.timeout = cfg.get("timeout_s", 30)
        self.workers = cfg.get("workers", 3)
        self.cache_path = cache_path
        self.cache = {"name2cid": {}, "cid2smiles": {}, "cid2title": {}}
        if cache_path.exists():
            self.cache.update(json.loads(cache_path.read_text()))
        self.session = requests.Session()
        self.session.headers["User-Agent"] = "drugsyn/0.2 (drug-synergy research pipeline)"
        self._lock = threading.Lock()
        self._next_slot = 0.0

    def save(self) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            text = json.dumps(self.cache)
        self.cache_path.write_text(text)

    def _wait_for_slot(self) -> None:
        with self._lock:
            now = time.monotonic()
            slot = max(now, self._next_slot)
            self._next_slot = slot + self.min_interval
        time.sleep(max(0.0, slot - now))

    def _get(self, url: str, method: str = "GET", **kw):
        for attempt in range(6):
            self._wait_for_slot()
            try:
                r = self.session.request(method, url, timeout=self.timeout, **kw)
            except requests.RequestException as exc:
                log.warning("PubChem request failed: %s", exc)
                r = None
            if r is not None and r.status_code not in (429, 503, 504):
                return r
            time.sleep(2 ** attempt)
        return r

    def name_to_cid(self, name: str) -> int | None:
        if name in self.cache["name2cid"]:
            return self.cache["name2cid"][name]
        cid, definitive = None, True
        for q in name_variants(name):
            r = self._get(f"{self.base}/compound/name/{quote(q, safe='')}/cids/JSON")
            if r is not None and r.status_code == 200:
                cids = r.json().get("IdentifierList", {}).get("CID", [])
                if cids:
                    cid = int(cids[0])
                    break
            elif r is None or r.status_code != 404:
                definitive = False  # server busy / network error: not a real miss
        if cid is not None or definitive:  # never cache a transient failure
            with self._lock:
                self.cache["name2cid"][name] = cid
        return cid

    def names_to_cids(self, names: list[str]) -> dict[str, int | None]:
        for attempt in range(3):  # later passes retry names that hit transient errors
            todo = [n for n in names if n not in self.cache["name2cid"]]
            if not todo:
                break
            log.info("PubChem name lookup (pass %d): %d cached, %d to query",
                     attempt + 1, len(names) - len(todo), len(todo))
            with ThreadPoolExecutor(self.workers if attempt == 0 else 1) as pool:
                for i, _ in enumerate(pool.map(self.name_to_cid, todo), 1):
                    if i % 200 == 0 or i == len(todo):
                        self.save()
                        log.info("  %d / %d names queried", i, len(todo))
        return {n: self.cache["name2cid"].get(n) for n in names}

    def cids_to_smiles(self, cids: list[int], batch: int = 100) -> dict[int, str]:
        todo = [c for c in cids if str(c) not in self.cache["cid2smiles"]]
        for i in range(0, len(todo), batch):
            chunk = ",".join(map(str, todo[i:i + batch]))
            for props in ("SMILES,ConnectivitySMILES", "IsomericSMILES,CanonicalSMILES"):
                r = self._get(f"{self.base}/compound/cid/property/{props}/JSON",
                              "POST", data={"cid": chunk})
                if r is not None and r.status_code == 200:
                    for p in r.json().get("PropertyTable", {}).get("Properties", []):
                        smi = next((p[k] for k in _SMILES_KEYS if p.get(k)), None)
                        self.cache["cid2smiles"][str(p["CID"])] = smi
                    break
        return {c: self.cache["cid2smiles"].get(str(c)) for c in cids}

    def cids_to_titles(self, cids: list[int], batch: int = 100) -> dict[int, str | None]:
        todo = [c for c in dict.fromkeys(cids) if str(c) not in self.cache["cid2title"]]
        for i in range(0, len(todo), batch):
            chunk = ",".join(map(str, todo[i:i + batch]))
            r = self._get(f"{self.base}/compound/cid/property/Title/JSON", "POST",
                          data={"cid": chunk})
            if r is not None and r.status_code == 200:
                for p in r.json().get("PropertyTable", {}).get("Properties", []):
                    self.cache["cid2title"][str(p["CID"])] = p.get("Title")
        self.save()
        return {c: self.cache["cid2title"].get(str(c)) for c in cids}


# --------------------------------------------------------------------------- chemistry
def standardise(smiles: str | None):
    """Return (mol, canonical_smiles, inchikey) or (None, None, None)."""
    from rdkit import Chem, RDLogger
    from rdkit.Chem.MolStandardize import rdMolStandardize

    RDLogger.DisableLog("rdApp.*")
    if not isinstance(smiles, str) or not smiles:
        return None, None, None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, None, None
    try:
        mol = rdMolStandardize.LargestFragmentChooser().choose(mol)
        mol = rdMolStandardize.Uncharger().uncharge(mol)
    except Exception:  # standardisation is best effort
        pass
    return mol, Chem.MolToSmiles(mol), Chem.MolToInchiKey(mol) or None


def descriptors(mol) -> dict[str, float]:
    from rdkit.Chem import QED, Crippen, Descriptors, Lipinski, rdMolDescriptors

    return {
        "mol_wt": Descriptors.MolWt(mol),
        "logp": Crippen.MolLogP(mol),
        "tpsa": rdMolDescriptors.CalcTPSA(mol),
        "hbd": Lipinski.NumHDonors(mol),
        "hba": Lipinski.NumHAcceptors(mol),
        "rot_bonds": Lipinski.NumRotatableBonds(mol),
        "rings": rdMolDescriptors.CalcNumRings(mol),
        "aromatic_rings": rdMolDescriptors.CalcNumAromaticRings(mol),
        "frac_csp3": rdMolDescriptors.CalcFractionCSP3(mol),
        "heavy_atoms": mol.GetNumHeavyAtoms(),
        "qed": QED.qed(mol),
    }


def morgan_matrix(mols: list, radius: int, n_bits: int) -> np.ndarray:
    from rdkit.Chem import rdFingerprintGenerator

    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    return np.vstack([gen.GetFingerprintAsNumPy(m).astype(np.uint8) for m in mols])


# --------------------------------------------------------------------------- resolution
def resolve(names: pd.Series, info: pd.DataFrame | None, pubchem: PubChem | None) -> pd.DataFrame:
    """names: normalised drug names with measurement counts as values (index = name)."""
    res = pd.DataFrame({"name_norm": names.index, "n_measurements": names.values})
    res["cid"], res["smiles_raw"], res["resolution"] = np.nan, None, "unresolved"
    res["official_name"] = None

    if info is not None:
        res = res.drop(columns=["cid", "smiles_raw", "official_name"]).merge(
            info.rename(columns={"smiles": "smiles_raw"}), on="name_norm", how="left")
        res.loc[res["smiles_raw"].notna(), "resolution"] = "drugcombdb"

    if pubchem is not None:
        need_cid = res["smiles_raw"].isna() & res["cid"].isna()
        found = pubchem.names_to_cids(res.loc[need_cid, "name_norm"].tolist())
        res.loc[need_cid, "cid"] = res.loc[need_cid, "name_norm"].map(found).astype(float)
        need_smiles = res["smiles_raw"].isna() & res["cid"].notna()
        cids = res.loc[need_smiles, "cid"].astype(int).tolist()
        smiles = pubchem.cids_to_smiles(cids)
        res.loc[need_smiles, "smiles_raw"] = [smiles.get(c) for c in cids]
        res.loc[need_smiles & res["smiles_raw"].notna(), "resolution"] = "pubchem"
        pubchem.save()
    return res


_READABLE = r"[A-Za-z][A-Za-z0-9 \-]{2,30}"


def canonical_names(names: pd.Series, pubchem: PubChem) -> pd.Series:
    """Upgrade display names to PubChem's record title found *by name*.

    Looking the name up (rather than the structure) lands on the main record,
    e.g. "vepesid j" -> Etoposide, "lyovac-cosmegen" -> Dactinomycin. Titles
    that are not short readable names (IUPAC strings, IDs) are ignored.
    """
    cids = pubchem.names_to_cids(names.dropna().unique().tolist())
    titles = pubchem.cids_to_titles([c for c in cids.values() if c])
    by_name = {n: titles.get(c) for n, c in cids.items() if c}
    title = names.map(by_name)
    title = title.where(title.astype(str).str.fullmatch(_READABLE)).str.lower()
    return title.fillna(names)


def build_dimension(res: pd.DataFrame, radius: int, n_bits: int):
    """Standardise structures, merge synonyms, return (dim_drug, bridge, fingerprints)."""
    std = [standardise(s) for s in res["smiles_raw"]]
    res = res.assign(
        smiles=[s[1] for s in std],
        inchikey=[s[2] for s in std],
    )
    res.loc[res["smiles_raw"].notna() & res["inchikey"].isna(), "resolution"] = "invalid_smiles"
    res["drug_key"] = res["inchikey"].where(res["inchikey"].notna(), "NAME:" + res["name_norm"])
    mols = {k: s[0] for k, s in zip(res["drug_key"], std) if s[0] is not None}

    bridge = res[["name_norm", "drug_key", "resolution"]].copy()

    # display name: DrugCombDB official name of the most-measured synonym, else that
    # synonym (``run`` upgrades it to PubChem's canonical title when available)
    res = res.sort_values("n_measurements", ascending=False)
    res["display"] = res["official_name"].fillna(res["name_norm"]).map(display_name)
    dim = res.groupby("drug_key", sort=False).agg(
        drug_name=("display", "first"),
        synonyms=("name_norm", lambda s: " | ".join(sorted(s))),
        n_synonyms=("name_norm", "size"),
        n_measurements=("n_measurements", "sum"),
        cid=("cid", "first"),
        smiles=("smiles", "first"),
        inchikey=("inchikey", "first"),
        resolution=("resolution", "first"),
    ).reset_index()
    dim["has_structure"] = dim["drug_key"].isin(mols.keys())

    desc = pd.DataFrame(
        [descriptors(mols[k]) if k in mols else {} for k in dim["drug_key"]],
        columns=DESCRIPTORS,
    )
    dim = pd.concat([dim, desc], axis=1)

    keys = [k for k in dim["drug_key"] if k in mols]
    fp = pd.DataFrame(
        morgan_matrix([mols[k] for k in keys], radius, n_bits) if keys
        else np.zeros((0, n_bits), np.uint8),
        columns=[f"fp_{i}" for i in range(n_bits)],
    )
    fp.insert(0, "drug_key", keys)
    return dim, bridge, fp


def run(cfg: dict, paths: Paths) -> None:
    paths.ensure()
    meas = pd.read_parquet(paths.measurements, columns=["drug_a", "drug_b"])
    names = pd.concat([meas["drug_a"], meas["drug_b"]]).value_counts()

    info_path = paths.raw_file("drugcombdb", cfg["sources"]["drugcombdb"]["files"]["drug_info"])
    info = load_drugcombdb_info(info_path) if info_path.exists() else None
    if info is None:
        log.warning("%s not found - relying on PubChem only", info_path)
    pc_cfg = cfg["sources"]["pubchem"]
    pubchem = PubChem(pc_cfg, paths.pubchem_cache) if pc_cfg.get("enabled", True) else None

    res = resolve(names, info, pubchem)
    fcfg = cfg["features"]
    dim, bridge, fp = build_dimension(res, fcfg["morgan_radius"], fcfg["morgan_bits"])
    if pubchem is not None:
        dim["drug_name"] = canonical_names(dim["drug_name"], pubchem)
    dim.to_parquet(paths.dim_drug, index=False)
    bridge.to_parquet(paths.processed / "bridge_drug_name.parquet", index=False)
    fp.to_parquet(paths.drug_fingerprints, index=False)
    log.info("dim_drug: %d names -> %d drugs (%d with structure)",
             len(bridge), len(dim), int(dim["has_structure"].sum()))
