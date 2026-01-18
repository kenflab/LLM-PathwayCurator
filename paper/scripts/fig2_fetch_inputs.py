#!/usr/bin/env python3
# paper/scripts/fig2_fetch_inputs.py
from __future__ import annotations

import gzip
import shutil
import urllib.request
from pathlib import Path

import synapseclient  # type: ignore

# =========================
# Fixed paths (v1)
# =========================
ROOT = Path(__file__).resolve().parents[1]  # paper/
SD = ROOT / "source_data" / "PANCAN_TP53_v1"
RAW = SD / "raw"

# =========================
# Inputs (edit if needed)
# =========================
SYN_EXPR = "syn4976369.3"  # PANCAN batch-normalized expression (Synapse entity)
SYN_MC3 = (
    "syn7824274"  # MC3 public MAF (Synapse entity; may be folder or file depending on permissions)
)

# Phenotype mapping (needed to assign TCGA barcode -> cancer type)
# If this URL ever changes, update it here.
XENA_PHENO_URL = "https://gdc.xenahubs.net/download/TCGA_phenotype_denseDataOnlyDownload.tsv.gz"


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


def _syn_login() -> synapseclient.Synapse:
    syn = synapseclient.Synapse(skip_checks=True)
    try:
        # Works if SYNAPSE_AUTH_TOKEN is set, or ~/.synapseConfig exists
        syn.login(silent=True)
    except Exception as e:
        _die(
            "[fetch_inputs] Synapse login failed.\n"
            "Set SYNAPSE_AUTH_TOKEN or configure ~/.synapseConfig.\n"
            f"{e}"
        )
    return syn


def _syn_get_to(syn: synapseclient.Synapse, syn_id: str, out_dir: Path) -> Path:
    """
    Download a Synapse entity to out_dir.
    If syn_id is a file, return the downloaded file path.
    If it is a folder, Synapse will download as a directory; we fail-fast and ask
    user to point to a file entity.
    """
    try:
        ent = syn.get(syn_id, downloadLocation=str(out_dir))
    except Exception as e:
        _die(f"[fetch_inputs] Synapse download failed for {syn_id}\n{e}")

    p = Path(ent.path).resolve()
    if p.is_dir():
        _die(
            f"[fetch_inputs] Synapse ID {syn_id} downloaded as a directory:\n{p}\n"
            "For v1, please set SYN_EXPR/SYN_MC3 to a FILE entity ID (not a folder)."
        )
    if not p.exists():
        _die(f"[fetch_inputs] Synapse returned non-existent path for {syn_id}: {p}")
    return p


def _maybe_gzip_copy(src: Path, dst: Path) -> None:
    """
    If src is already .gz, copy as-is.
    Else gzip it to dst (which should end with .gz).
    """
    _ensure_dir(dst.parent)
    if src.suffix == ".gz":
        shutil.copy2(src, dst)
        return

    if dst.suffix != ".gz":
        _die(f"[fetch_inputs] dst must end with .gz: {dst}")

    with src.open("rb") as r, gzip.open(dst, "wb") as w:
        shutil.copyfileobj(r, w)


def main() -> None:
    _ensure_dir(RAW)

    syn = _syn_login()

    # --- expression ---
    expr_src = _syn_get_to(syn, SYN_EXPR, RAW)
    expr_out = RAW / "expression.tsv.gz"
    _maybe_gzip_copy(expr_src, expr_out)

    # --- MC3 ---
    mc3_src = _syn_get_to(syn, SYN_MC3, RAW)
    mc3_out = RAW / "mc3.v0.2.8.PUBLIC.maf.gz"
    _maybe_gzip_copy(mc3_src, mc3_out)

    # --- phenotype ---
    pheno_out = RAW / "TCGA_phenotype_dense.tsv.gz"
    if not pheno_out.exists():
        _download_url(XENA_PHENO_URL, pheno_out)

    # minimal success signal
    for p in [expr_out, mc3_out, pheno_out]:
        if not p.exists() or p.stat().st_size == 0:
            _die(f"[fetch_inputs] expected non-empty file: {p}")

    print("[fetch_inputs] OK")
    print(f"  {expr_out}")
    print(f"  {mc3_out}")
    print(f"  {pheno_out}")


if __name__ == "__main__":
    main()
