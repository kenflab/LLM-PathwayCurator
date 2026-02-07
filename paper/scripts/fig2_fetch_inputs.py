#!/usr/bin/env python3
# paper/scripts/fig2_fetch_inputs.py
from __future__ import annotations

import shutil
import urllib.request
from pathlib import Path

# =========================
# Fixed paths (v1)
# =========================
ROOT = Path(__file__).resolve().parents[1]  # paper/
SD = ROOT / "source_data" / "PANCAN_TP53_v1"
RAW = SD / "raw"

# =========================
# Public URLs (no auth)
# =========================
# Expression (EB++ adjusted; the one you pasted)
EXPR_URL = "https://tcga-pancan-atlas-hub.s3.us-east-1.amazonaws.com/download/EB%2B%2BAdjustPANCAN_IlluminaHiSeq_RNASeqV2.geneExp.xena.gz"

# MC3 mutation (Xena format; the one you pasted)
MC3_URL = (
    "https://tcga-pancan-atlas-hub.s3.us-east-1.amazonaws.com/download/mc3.v0.2.8.PUBLIC.xena.gz"
)

# Phenotype mapping (barcode -> cancer type)
XENA_PHENO_URL = "https://tcga-pancan-atlas-hub.s3.us-east-1.amazonaws.com/download/TCGA_phenotype_denseDataOnlyDownload.tsv.gz"


def _die(msg: str) -> None:
    """Abort execution with a message.

    Parameters
    ----------
    msg : str
        Human-readable error message.

    Raises
    ------
    SystemExit
        Always raised with ``msg``.
    """
    raise SystemExit(msg)


def _ensure_dir(p: Path) -> None:
    """Ensure a directory exists.

    Parameters
    ----------
    p : pathlib.Path
        Directory path to create.

    Notes
    -----
    Creates parent directories as needed and does not error if the directory
    already exists.
    """
    p.mkdir(parents=True, exist_ok=True)


def _download_url(url: str, out_path: Path) -> None:
    """Download a URL to a local file atomically.

    The download is written to a temporary file in the same directory and
    then moved into place to avoid partially-written outputs.

    Parameters
    ----------
    url : str
        Source URL (public, no auth).
    out_path : pathlib.Path
        Destination file path.

    Raises
    ------
    SystemExit
        If the download fails. Any temporary file is removed on failure.
    """
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    try:
        with urllib.request.urlopen(url) as r, tmp.open("wb") as w:
            shutil.copyfileobj(r, w)
        tmp.replace(out_path)
    except Exception as e:
        if tmp.exists():
            tmp.unlink()
        _die(f"[fetch_inputs] URL download failed: {url}\n{e}")


def main() -> None:
    """Fetch fixed public inputs for the Fig. 2 pipeline (v1).

    Downloads three UCSC Xena / PanCanAtlas-hosted files into
    ``paper/source_data/PANCAN_TP53_v1/raw`` if they are missing:

    - expression.xena.gz
    - mc3.v0.2.8.PUBLIC.xena.gz
    - TCGA_phenotype_dense.tsv.gz (downloaded from the Dense phenotype URL)

    Raises
    ------
    SystemExit
        If any expected output file is missing or empty after download.

    Notes
    -----
    Files are only downloaded when absent. Existing files are not
    re-downloaded.
    """
    _ensure_dir(RAW)

    expr_out = RAW / "expression.xena.gz"
    mc3_out = RAW / "mc3.v0.2.8.PUBLIC.xena.gz"
    pheno_out = RAW / "TCGA_phenotype_dense.tsv.gz"

    if not expr_out.exists():
        _download_url(EXPR_URL, expr_out)
    if not mc3_out.exists():
        _download_url(MC3_URL, mc3_out)
    if not pheno_out.exists():
        _download_url(XENA_PHENO_URL, pheno_out)

    for p in [expr_out, mc3_out, pheno_out]:
        if not p.exists() or p.stat().st_size == 0:
            _die(f"[fetch_inputs] expected non-empty file: {p}")

    print("[fetch_inputs] OK")
    print(f"  {expr_out}")
    print(f"  {mc3_out}")
    print(f"  {pheno_out}")


if __name__ == "__main__":
    main()
