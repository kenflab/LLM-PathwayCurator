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
    raise SystemExit(msg)


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _download_url(url: str, out_path: Path) -> None:
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
