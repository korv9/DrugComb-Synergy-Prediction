# Data sources

Nothing under `data/` is committed. `make data` (or `python -m drugsyn download`)
fetches every file and writes `data/raw/MANIFEST.json` with the URL, release,
size, SHA-256 and download time of each file. That manifest is what makes a run
reproducible. Commit a copy of it next to any results you publish.

If a host is unreachable (firewall, site down), download the files by hand and
put them in the paths below. The download step skips files that are already
there and still records their checksums.

| Source | File | Local path | Used for |
|---|---|---|---|
| DrugCombDB | `drugcombs_scored.csv` | `data/raw/drugcombdb/` | Drug pairs, cell lines, ZIP / Bliss / Loewe / HSA scores |
| DrugCombDB | `drug_chemical_info.csv` | `data/raw/drugcombdb/` | Drug name → PubChem CID and SMILES |
| DepMap | `Model.csv` | `data/raw/depmap/` | Cell-line identifiers, lineage, disease |
| DepMap | `OmicsExpressionProteinCodingGenesTPMLogp1.csv` | `data/raw/depmap/` | RNA-seq log2(TPM+1), models × genes |
| PubChem (API) | – | `data/interim/pubchem_cache.json` | Fallback name → CID → SMILES lookups |

## DrugCombDB

* Site: <http://drugcombdb.denglab.org/> (download page: `/download/`)
* Reference: Liu H. *et al.* "DrugCombDB: a comprehensive database of drug
  combinations toward the discovery of combinatorial therapy."
  *Nucleic Acids Research* 48(D1), 2020.
* `drugcombs_scored.csv` has one row per screened block:
  `ID, Drug1, Drug2, Cell line, ZIP, Bliss, Loewe, HSA`. The ingest step matches
  column names case- and whitespace-insensitively, so small header changes are
  tolerated.
* DrugCombDB combines several screens (e.g. NCI-ALMANAC, O'Neil et al.,
  CLOUD), so drug names come in mixed styles: brand names, CAS numbers, ZINC
  IDs, NSC numbers. It also includes *Plasmodium falciparum* strains (3D7, DD2,
  HB3) from anti-malaria screens. Those are excluded through
  `configs/cell_line_aliases.csv`.
* Check the site's terms of use before redistributing any derived table.

## DepMap

* Default source: the official Broad release **DepMap 24Q4 Public** on
  Figshare+ (<https://doi.org/10.25452/figshare.plus.27993248.v1>). The
  pipeline lists the article's files through the Figshare API and downloads
  them directly. Pinning a release keeps results reproducible. To use another
  Figshare release, change `sources.depmap.figshare_article_id`.
* Alternative: `provider: portal` reads the file index at
  `https://depmap.org/portal/api/download/files`. At the time of writing, the
  portal serves a browser verification page to scripted clients, so this mode
  only works from an environment that has passed that check.
* In 24Q4 the expression matrix has the ModelID in an unnamed first column.
  Newer releases prefix it with metadata columns (`SequencingID, ModelID,
  IsDefaultEntryForModel, …`). `cells.load_expression` handles both layouts.
* DepMap data is released under CC BY 4.0. Cite the release you used.

## PubChem

Used only for drugs that DrugCombDB's own chemical table does not cover. The
client batches CID → SMILES requests, rate-limits itself (`sleep_s`) and caches
every answer, so a re-run makes no new requests. PubChem renamed its SMILES
properties in 2025 (`IsomericSMILES` → `SMILES`, `CanonicalSMILES` →
`ConnectivitySMILES`), and the client accepts both.

## Network access

The pipeline needs outbound HTTP(S) to `drugcombdb.denglab.org`,
`api.figshare.com` and `ndownloader.figshare.com` (which redirect to S3), and
`pubchem.ncbi.nlm.nih.gov`.
