#!/usr/bin/env python3
# paper/scripts/fig2_make_groups.py
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]  # paper/
SD = ROOT / "source_data" / "PANCAN_TP53_v1"
RAW = SD / "raw"
DER = SD / "derived"
OUT_GROUPS = DER / "groups"


def _die(msg: str) -> None:
    raise SystemExit(msg)


def _read_tsv_gz(path: Path) -> pd.DataFrame:
    if not path.exists():
        _die(f"[make_groups] missing file: {path}")
    try:
        return pd.read_csv(path, sep="\t", compression="gzip", dtype=str, low_memory=False).fillna(
            ""
        )
    except Exception as e:
        _die(f"[make_groups] failed to read tsv.gz: {path}\n{e}")


def _pick_col(df: pd.DataFrame, candidates: list[str]) -> str:
    cols = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols:
            return cols[cand.lower()]
    _die(f"[make_groups] none of columns found: {candidates}\navailable={list(df.columns)[:50]}...")


def _normalize_barcode(s: str) -> str:
    # We use first 16 chars (TCGA-XX-YYYY-ZZ...) for sample-level matching (common practice)
    s = str(s).strip()
    return s[:16]


def main() -> None:
    maf_path = RAW / "mc3.v0.2.8.PUBLIC.maf.gz"
    pheno_path = RAW / "TCGA_phenotype_dense.tsv.gz"

    maf = _read_tsv_gz(maf_path)
    pheno = _read_tsv_gz(pheno_path)

    # --- columns ---
    maf_gene = _pick_col(maf, ["Hugo_Symbol", "HUGO_SYMBOL", "gene"])
    maf_bar = _pick_col(maf, ["Tumor_Sample_Barcode", "tumor_sample_barcode"])
    maf_flt = _pick_col(maf, ["FILTER", "filter"])

    # phenotype: try common keys for cancer type / study
    # Xena phenotype often has "study" or "_study" for TCGA cancer (e.g., TCGA-LUAD)
    ph_sample = _pick_col(pheno, ["sample", "Sample", "_sample", "sampleID", "sample_id"])
    # robust cancer type field candidates:
    ph_cancer = None
    for cand in ["study", "_study", "cancer type", "_primary_disease", "project_id", "cohort"]:
        if cand.lower() in {c.lower() for c in pheno.columns}:
            ph_cancer = _pick_col(pheno, [cand])
            break
    if ph_cancer is None:
        _die(
            "[make_groups] could not find a cancer-type column in phenotype file.\n"
            "Inspect RAW/TCGA_phenotype_dense.tsv.gz and update candidates in fig2_make_groups.py."
        )

    ph = pheno[[ph_sample, ph_cancer]].copy()
    ph.columns = ["sample", "cancer_raw"]
    ph["sample"] = ph["sample"].astype(str).map(_normalize_barcode)

    # normalize cancer label -> like 'LUAD', 'HNSC' etc.
    def norm_cancer(x: str) -> str:
        x = str(x).strip()
        # examples: "TCGA-LUAD" -> "LUAD"
        if x.upper().startswith("TCGA-") and len(x) >= 9:
            return x.split("-", 1)[1].upper()
        return x.upper().replace(" ", "_")

    ph["cancer"] = ph["cancer_raw"].map(norm_cancer)
    ph = ph.drop_duplicates(subset=["sample"], keep="first")

    # --- TP53 mutations (PASS only) ---
    maf_tp53 = maf[
        (maf[maf_gene].astype(str).str.upper() == "TP53") & (maf[maf_flt] == "PASS")
    ].copy()
    if maf_tp53.empty:
        _die("[make_groups] no TP53 PASS mutations found in MC3 MAF (check FILTER column?)")

    maf_tp53["sample"] = maf_tp53[maf_bar].astype(str).map(_normalize_barcode)
    tp53_mut = set(maf_tp53["sample"].tolist())

    # --- map samples to cancer ---
    # Use only samples that appear in phenotype table
    # (expression matrix will later subset further)
    samples = ph["sample"].tolist()
    if not samples:
        _die("[make_groups] phenotype mapping produced 0 samples")

    df = ph[["sample", "cancer"]].copy()
    df["group"] = df["sample"].map(lambda s: "TP53_mut" if s in tp53_mut else "TP53_wt")

    OUT_GROUPS.mkdir(parents=True, exist_ok=True)

    # per cancer output
    cancers = sorted(df["cancer"].unique().tolist())
    if not cancers:
        _die("[make_groups] no cancers found after normalization")

    for cancer in cancers:
        g = df[df["cancer"] == cancer].copy()
        if g.empty:
            continue
        out_path = OUT_GROUPS / f"{cancer}.groups.tsv"
        g[["sample", "group"]].to_csv(out_path, sep="\t", index=False)

    # master
    (OUT_GROUPS / "PANCAN.groups.tsv").write_text(
        df[["sample", "cancer", "group"]].to_csv(sep="\t", index=False),
        encoding="utf-8",
    )

    print("[make_groups] OK")
    print(f"  wrote: {OUT_GROUPS}/PANCAN.groups.tsv")
    print(f"  cancers: {len(cancers)}")


if __name__ == "__main__":
    main()
