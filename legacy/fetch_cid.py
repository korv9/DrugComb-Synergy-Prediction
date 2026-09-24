# 1_fetch_cids.py
#%%
from __future__ import annotations
import re, json, time, unicodedata
from pathlib import Path
from typing import Optional, Dict, List, Tuple

import requests
import pandas as pd
from tqdm import tqdm

# ====== CONFIG ======
INPUT_CSV      = Path("C:/Users/46762/VSCODE/BIG_PHARMA/data/interim/scored_clean.csv")
DRUG_COLS      = ("drug_min", "drug_max")
OUT_CIDS_CSV   = Path("C:/Users/46762/VSCODE/BIG_PHARMA/data/interim/drug_cids.csv")
CACHE_JSON     = Path("drug_name2cid_cache.json")
TEST_MODE      = False 
TEST_MAX_DRUGS = 300
REQUESTS_TIMEOUT = 20
SLEEP_BETWEEN    = 0.12
VERBOSE          = True

# Alias/typos 
MANUAL_ALIASES = {
    "5-fu": "fluorouracil",
    "5 fu": "fluorouracil",
    "5-fluoro-2'-deoxyuridine": "floxuridine",
    "5-fluoro-2´-deoxyuridine": "floxuridine",
    "adm": "doxorubicin",         # adriamycin
    "cisplatino": "cisplatin",
    "mk 801": "dizocilpine",
    "mk-801": "dizocilpine",
}

#Normalisering/varianter 
ZERO_WIDTH = ["\u200b", "\u200c", "\u200d", "\ufeff"]
SALTS = r"(hydrochloride|hydrobromide|maleate|mesylate|tosylate|sulfate|phosphate|nitrate|acetate|tartrate|fumarate|succinate|bitartrate|oxalate|citrate|lactate|bicarbonate|carbonate|benzoate|pamoate|besylate)"
FORM_WORDS = r"(tablet|capsule|solution|injection|cream|ointment|suspension|powder|salt|hydrate|monohydrate|dihydrate)"
CAS_RE = re.compile(r"^\d{2,7}-\d{2}-\d$")

def _strip_zw(s: str) -> str:
    for z in ZERO_WIDTH: s = s.replace(z, "")
    return s
    
def _unify_dash(s: str) -> str:
    return s.replace("–","-").replace("—","-").replace("−","-")

def norm_name(raw) -> Optional[str]:
    if raw is None or (isinstance(raw, float) and pd.isna(raw)): return None
    s = str(raw).strip()
    s = unicodedata.normalize("NFKC", s)
    s = _strip_zw(_unify_dash(s))
    # stereoprefix
    s = re.sub(r"^\(\s*[+\-]\s*\)\s*-\s*", "", s, flags=re.I)
    s = re.sub(r"^\(\s*\+/-\s*\)\s*-\s*", "", s, flags=re.I)
    s = re.sub(r"^\(\s*[0-9a-z,\'\s]+\)\s*-\s*", "", s, flags=re.I)
    # alias i slutet
    s = re.sub(r"\s*\([^)]*\)\s*$", "", s)
    # salter/formulation
    s = re.sub(rf"\b({SALTS}|{FORM_WORDS})\b", "", s, flags=re.I)
    # mk 801 -> mk-801
    s = re.sub(r"\b(mk)\s*[- ]?\s*(\d+)\b", r"\1-\2", s, flags=re.I)
    s = " ".join(s.split())
    return s.lower() if s else None

def make_variants(name_raw: str) -> List[str]:
    vs: List[str] = []
    raw_l = _strip_zw(_unify_dash(str(name_raw))).strip().lower()

    # alias först
    alias = MANUAL_ALIASES.get(raw_l)
    if alias: vs.append(alias.lower())

    n = norm_name(name_raw)
    if n and n not in vs: vs.append(n)
    if raw_l and raw_l not in vs: vs.append(raw_l)

    # stereostrippad rå
    v = re.sub(r"^\(\s*[+\-]\s*\)\s*-\s*", "", raw_l, flags=re.I)
    v = re.sub(r"^\(\s*\+/-\s*\)\s*-\s*", "", v, flags=re.I)
    v = re.sub(r"^\(\s*[0-9a-z,\'\s]+\)\s*-\s*", "", v, flags=re.I)
    v = " ".join(v.split())
    if v and v not in vs: vs.append(v)

    # space<->dash
    for base in list(vs):
        for alt in (base.replace(" ","-"), base.replace("-", " ")):
            if alt not in vs: vs.append(alt)
    return vs

# ====== API ======
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "drugcomb-cid-fetcher/1.0 (+ml-pipeline)"})

def _get(url, **kw):
    try:
        return SESSION.get(url, timeout=REQUESTS_TIMEOUT, **kw)
    except Exception:
        return None

def pubchem_name_to_cid(q: str) -> Tuple[Optional[int], Optional[int]]:
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{requests.utils.quote(q)}/cids/JSON"
    r = _get(url)
    st = r.status_code if r else None
    if r and r.status_code == 200:
        try:
            cids = r.json().get("IdentifierList", {}).get("CID", [])
            return (cids[0] if cids else None, st)
        except Exception:
            return (None, st)
    return (None, st)

def pubchem_cas_to_cid(cas: str) -> Tuple[Optional[int], Optional[int]]:
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/xref/rn/{requests.utils.quote(cas)}/cids/JSON"
    r = _get(url)
    st = r.status_code if r else None
    if r and r.status_code == 200:
        try:
            cids = r.json().get("IdentifierList", {}).get("CID", [])
            return (cids[0] if cids else None, st)
        except Exception:
            return (None, st)
    return (None, st)

# ====== Cache ======
def load_cache(path: Path) -> Dict[str, Dict]:
    if path.exists(): return json.loads(path.read_text(encoding="utf-8"))
    return {}

def save_cache(path: Path, cache: Dict[str, Dict]) -> None:
    path.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")

# ====== MAIN ======
def main():
    df = pd.read_csv(INPUT_CSV)
    names = pd.unique(pd.concat([df[DRUG_COLS[0]], df[DRUG_COLS[1]]], axis=0).dropna())
    if TEST_MODE and len(names) > TEST_MAX_DRUGS:
        names = names[:TEST_MAX_DRUGS]
        print(f"[TEST] Kör på {len(names)} unika namn.")

    cache = load_cache(CACHE_JSON)
    rows = []
    hits, misses = 0, 0

    for raw in tqdm(names):
        raw_l = str(raw).lower()

        # CAS?
        if CAS_RE.match(raw_l):
            cid, st = pubchem_cas_to_cid(raw_l)
            if cid:
                rows.append({"drug": raw, "cid": cid, "source": "pubchem_cas", "query_used": raw_l})
                cache[raw_l] = {"cid": cid, "source": "pubchem_cas"}
                hits += 1
                if VERBOSE: print(f"[HIT][CAS] {raw} -> CID {cid}")
                continue

        # cache?
        if raw_l in cache and cache[raw_l].get("cid") is not None:
            cid = cache[raw_l]["cid"]
            rows.append({"drug": raw, "cid": cid, "source": cache[raw_l].get("source","cache"), "query_used": raw_l})
            hits += 1
            continue

        # multi-variants
        cid_found = None; src = None; used = None
        for q in make_variants(raw):
            if q in cache and cache[q].get("cid") is not None:
                cid_found = cache[q]["cid"]; src = cache[q].get("source","cache"); used = q
                break
            cid, st = pubchem_name_to_cid(q)
            if cid:
                cid_found = cid; src = "pubchem"; used = q
                cache[q] = {"cid": cid, "source": src}
                break
        if cid_found:
            rows.append({"drug": raw, "cid": cid_found, "source": src, "query_used": used})
            hits += 1
            if VERBOSE: print(f"[HIT][{src}] {raw} -> CID {cid_found} (via '{used}')")
        else:
            rows.append({"drug": raw, "cid": None, "source": "not_found", "query_used": None})
            cache[raw_l] = {"cid": None, "source": "not_found"}
            misses += 1
            if VERBOSE: print(f"[MISS] {raw}")

        save_cache(CACHE_JSON, cache)
        time.sleep(SLEEP_BETWEEN)

    pd.DataFrame(rows).to_csv(OUT_CIDS_CSV, index=False)
    print(f"[DONE] Wrote {OUT_CIDS_CSV} | hits={hits}, misses={misses}")

if __name__ == "__main__":
    main()
