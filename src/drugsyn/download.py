"""Download raw sources and record provenance (URL, size, sha256, timestamp).

Files already present in ``data/raw/<source>/`` are never re-downloaded, so the
stage also works when the files were fetched manually (e.g. behind a firewall).
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

from .config import Paths

log = logging.getLogger(__name__)

CHUNK = 1 << 20
HEADERS = {"User-Agent": "drugsyn/0.2 (research pipeline; +https://github.com/korv9)"}


def sha256sum(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(CHUNK), b""):
            h.update(block)
    return h.hexdigest()


def download_file(url: str, dest: Path, timeout: int = 60) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with requests.get(url, stream=True, timeout=timeout, headers=HEADERS) as resp:
        resp.raise_for_status()
        with open(tmp, "wb") as fh:
            for block in resp.iter_content(CHUNK):
                fh.write(block)
    tmp.replace(dest)


def _record(source: str, url: str | None, dest: Path, release: str | None = None) -> dict:
    return {
        "source": source,
        "file": dest.name,
        "url": url,
        "release": release,
        "bytes": dest.stat().st_size,
        "sha256": sha256sum(dest),
        "recorded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def _fetch(source: str, url: str | None, dest: Path, release: str | None = None) -> dict:
    if dest.exists():
        log.info("[%s] %s already present, skipping download", source, dest.name)
    elif url is None:
        raise FileNotFoundError(f"{dest} is missing and no download URL is known")
    else:
        log.info("[%s] downloading %s", source, url)
        download_file(url, dest)
    return _record(source, url, dest, release)


def download_drugcombdb(cfg: dict, paths: Paths) -> list[dict]:
    src = cfg["sources"]["drugcombdb"]
    base = src["base_url"].rstrip("/") + "/"
    return [
        _fetch("drugcombdb", base + name, paths.raw_file("drugcombdb", name))
        for name in src["files"].values()
    ]


def resolve_depmap_release(index: pd.DataFrame, wanted: list[str], release: str | None):
    """Pick a release from the DepMap file index that contains every wanted file.

    Returns ``(release_name, {filename: url})``.
    """
    cols = {c.lower(): c for c in index.columns}
    rel_col, file_col, url_col = cols["release"], cols["filename"], cols["url"]
    date_col = cols.get("release_date")

    idx = index[index[file_col].isin(wanted)]
    complete = idx.groupby(rel_col)[file_col].nunique()
    candidates = complete[complete == len(wanted)].index.tolist()
    if release is not None:
        if release not in candidates:
            raise ValueError(f"DepMap release {release!r} does not contain all of {wanted}")
        chosen = release
    elif not candidates:
        raise ValueError(f"No DepMap release contains all of {wanted}")
    elif date_col is not None:
        dates = idx[idx[rel_col].isin(candidates)].groupby(rel_col)[date_col].max()
        chosen = pd.to_datetime(dates, errors="coerce").idxmax()
    else:  # the index is listed newest first
        chosen = next(r for r in idx[rel_col] if r in candidates)

    rows = idx[idx[rel_col] == chosen].drop_duplicates(file_col)
    return chosen, dict(zip(rows[file_col], rows[url_col]))


def figshare_files(article_id: int) -> tuple[str, dict[str, str]]:
    """Return (article title, {filename: download_url}) for a Figshare article."""
    api = f"https://api.figshare.com/v2/articles/{article_id}"
    meta = requests.get(api, timeout=60, headers=HEADERS)
    meta.raise_for_status()
    files = requests.get(f"{api}/files", params={"page_size": 1000}, timeout=60, headers=HEADERS)
    files.raise_for_status()
    return meta.json()["title"], {f["name"]: f["download_url"] for f in files.json()}


def download_depmap(cfg: dict, paths: Paths) -> list[dict]:
    """DepMap via its official Figshare release (default) or the portal file index."""
    src = cfg["sources"]["depmap"]
    wanted = list(src["files"].values())
    dests = {name: paths.raw_file("depmap", name) for name in wanted}
    if all(d.exists() for d in dests.values()):
        return [_fetch("depmap", None, d, src.get("release")) for d in dests.values()]

    if src.get("provider", "figshare") == "figshare":
        release, urls = figshare_files(src["figshare_article_id"])
        missing = set(wanted) - set(urls)
        if missing:
            raise ValueError(f"Figshare article {src['figshare_article_id']} lacks {missing}")
    else:  # the portal index sits behind bot verification for scripted clients
        resp = requests.get(src["files_index_url"], timeout=60, headers=HEADERS)
        resp.raise_for_status()
        index = pd.read_csv(io.StringIO(resp.text))
        release, urls = resolve_depmap_release(index, wanted, src.get("release"))
    log.info("[depmap] using release %s", release)
    return [_fetch("depmap", urls[name], dests[name], release) for name in wanted]


def write_manifest(records: list[dict], path: Path) -> None:
    existing = json.loads(path.read_text()) if path.exists() else []
    merged = {(r["source"], r["file"]): r for r in existing}
    merged.update({(r["source"], r["file"]): r for r in records})
    path.write_text(json.dumps(sorted(merged.values(), key=lambda r: (r["source"], r["file"])),
                               indent=2))


def run(cfg: dict, paths: Paths) -> None:
    paths.ensure()
    records = download_drugcombdb(cfg, paths) + download_depmap(cfg, paths)
    write_manifest(records, paths.manifest)
    log.info("manifest written to %s", paths.manifest)
