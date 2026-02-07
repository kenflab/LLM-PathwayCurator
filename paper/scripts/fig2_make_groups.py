#!/usr/bin/env python3
# paper/scripts/fig2_make_groups.py
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]  # paper/
SD = ROOT / "source_data" / "PANCAN_TP53_v1"
RAW = SD / "raw"
DER = SD / "derived"
OUT_GROUPS = DER / "groups"

# phenotype _primary_disease (normalized) -> TCGA code
DISEASE_TO_TCGA = {
    "adrenocortical cancer": "ACC",
    "bladder urothelial carcinoma": "BLCA",
    "brain lower grade glioma": "LGG",
    "breast invasive carcinoma": "BRCA",
    "cervical & endocervical cancer": "CESC",
    "cholangiocarcinoma": "CHOL",
    "colon adenocarcinoma": "COAD",
    "diffuse large b-cell lymphoma": "DLBC",
    "esophageal carcinoma": "ESCA",
    "glioblastoma multiforme": "GBM",
    "head & neck squamous cell carcinoma": "HNSC",
    "kidney chromophobe": "KICH",
    "kidney clear cell carcinoma": "KIRC",
    "kidney papillary cell carcinoma": "KIRP",
    "liver hepatocellular carcinoma": "LIHC",
    "lung adenocarcinoma": "LUAD",
    "lung squamous cell carcinoma": "LUSC",
    "mesothelioma": "MESO",
    "ovarian serous cystadenocarcinoma": "OV",
    "pancreatic adenocarcinoma": "PAAD",
    "pheochromocytoma & paraganglioma": "PCPG",
    "prostate adenocarcinoma": "PRAD",
    "rectum adenocarcinoma": "READ",
    "sarcoma": "SARC",
    "skin cutaneous melanoma": "SKCM",
    "stomach adenocarcinoma": "STAD",
    "testicular germ cell tumor": "TGCT",
    "thymoma": "THYM",
    "thyroid carcinoma": "THCA",
    "uterine carcinosarcoma": "UCS",
    "uterine corpus endometrioid carcinoma": "UCEC",
    "uveal melanoma": "UVM",
    "acute myeloid leukemia": "LAML",
}

# TP53 mut = protein-altering only
PROTEIN_ALTERING = {
    "Missense_Mutation",
    "Nonsense_Mutation",
    "Frame_Shift_Del",
    "Frame_Shift_Ins",
    "Splice_Site",
    "In_Frame_Del",
    "In_Frame_Ins",
}
PROTEIN_ALTERING_UP = {x.upper() for x in PROTEIN_ALTERING}


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


def _read_tsv_gz(path: Path) -> pd.DataFrame:
    """Read a gzipped TSV file into a DataFrame.

    Parameters
    ----------
    path : pathlib.Path
        Path to a ``.tsv.gz`` file.

    Returns
    -------
    pandas.DataFrame
        Loaded table with all columns as strings and missing values filled
        with empty strings.

    Raises
    ------
    SystemExit
        If the file does not exist or cannot be read.
    """
    if not path.exists():
        _die(f"[make_groups] missing file: {path}")
    try:
        return pd.read_csv(path, sep="\t", compression="gzip", dtype=str, low_memory=False).fillna(
            ""
        )
    except Exception as e:
        _die(f"[make_groups] failed to read tsv.gz: {path}\n{e}")


def _pick_col(df: pd.DataFrame, candidates: list[str]) -> str:
    """Pick the first matching column name from candidates.

    Matching is case-insensitive; the returned name preserves the exact
    spelling in the input DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        Input table.
    candidates : list of str
        Candidate column names to try in order.

    Returns
    -------
    str
        The selected column name.

    Raises
    ------
    SystemExit
        If none of the candidate columns exist.
    """
    cols = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols:
            return cols[cand.lower()]
    _die(
        f"[make_groups] none of columns found: {candidates}\navailable={list(df.columns)[:120]}..."
    )


def _normalize_barcode(s: str) -> str:
    """Normalize a TCGA barcode string for sample-level matching.

    The current implementation trims to the first 16 characters after
    stripping whitespace.

    Parameters
    ----------
    s : str
        Input barcode-like string.

    Returns
    -------
    str
        Normalized barcode prefix.
    """
    s = str(s).strip()
    return s[:16]


def _norm_disease(x: str) -> str:
    """Normalize disease strings for dictionary mapping.

    This lowercases the input, strips whitespace, and collapses internal
    whitespace to single spaces.

    Parameters
    ----------
    x : str
        Raw disease label.

    Returns
    -------
    str
        Normalized disease key.
    """
    x = str(x).strip().lower()
    x = re.sub(r"\s+", " ", x)
    return x


def _has_col(df: pd.DataFrame, name: str) -> bool:
    """Check whether a DataFrame has a column (case-insensitive).

    Parameters
    ----------
    df : pandas.DataFrame
        Input table.
    name : str
        Column name to check.

    Returns
    -------
    bool
        True if a column matches ``name`` ignoring case.
    """
    return name.lower() in {c.lower() for c in df.columns}


def main() -> None:
    """Build TP53 mutant vs wild-type group tables per TCGA cancer.

    This script:
    1) Reads phenotype and MC3 mutation tables.
    2) Keeps "primary-like" samples by ``sample_type_id`` in {1, 3}.
    3) Maps phenotype ``primary_disease`` to TCGA codes via
       ``DISEASE_TO_TCGA``.
    4) Extracts TP53 protein-altering mutations from MC3 (optionally
       requiring ``FILTER=PASS`` when available).
    5) Writes per-cancer ``{TCGA}.groups.tsv`` and a master
       ``PANCAN.groups.tsv``.

    Outputs
    -------
    derived/groups/{TCGA}.groups.tsv
        Two-column table: ``sample`` and ``group``.
    derived/groups/PANCAN.groups.tsv
        Three-column table: ``sample``, ``cancer``, ``group``.

    Raises
    ------
    SystemExit
        If required inputs are missing, mapping yields no samples, or no
        TP53 protein-altering calls are found after filtering.

    Notes
    -----
    - Group assignment is based on membership in the TP53-mutated sample
      set; otherwise samples are labeled TP53_wt.
    - The script prints warnings when a cancer has zero samples in either
      arm.
    """
    mc3_path = RAW / "mc3.v0.2.8.PUBLIC.xena.gz"
    pheno_path = RAW / "TCGA_phenotype_dense.tsv.gz"

    mc3 = _read_tsv_gz(mc3_path)
    pheno = _read_tsv_gz(pheno_path)

    # -------------------------
    # phenotype -> keep "primary-like" samples only (Fig2 v1)
    # -------------------------
    ph_sample_type_id = _pick_col(pheno, ["sample_type_id", "_sample_type_id"])

    def _norm_type_id(x: str) -> str:
        # Accept "01", "1", "03", "3", "1.0" -> "1" etc.
        s = str(x).strip()
        m = re.match(r"^(\d+)", s)
        if not m:
            return ""
        return str(int(m.group(1)))

    type_id_norm = pheno[ph_sample_type_id].map(_norm_type_id)
    keep_primary_like = type_id_norm.isin({"1", "3"})

    print(
        "[make_groups] sample_type_id raw unique (head):",
        pheno[ph_sample_type_id].astype(str).str.strip().unique()[:10].tolist(),
    )
    print(
        "[make_groups] sample_type_id norm counts:", type_id_norm.value_counts().head(10).to_dict()
    )

    pheno = pheno[keep_primary_like].copy()
    print(
        "[make_groups] filtered to primary-like sample_type_id in {1,3}:",
        int(keep_primary_like.sum()),
    )

    # -------------------------
    # phenotype -> (sample16, cancer)
    # -------------------------
    ph_sample = _pick_col(pheno, ["sampleID", "sample", "Sample", "_sample"])
    ph_disease = _pick_col(pheno, ["_primary_disease", "primary_disease", "disease"])

    ph = pheno[[ph_sample, ph_disease]].copy()
    ph.columns = ["sample", "primary_disease"]
    ph["sample"] = ph["sample"].map(_normalize_barcode)
    ph["primary_disease_norm"] = ph["primary_disease"].map(_norm_disease)
    ph["cancer"] = ph["primary_disease_norm"].map(DISEASE_TO_TCGA).fillna("")

    ph = ph[ph["cancer"].ne("")].copy()
    if ph.empty:
        uniq = sorted({_norm_disease(x) for x in pheno[ph_disease].astype(str).tolist()})
        _die(
            "[make_groups] no samples mapped to TCGA codes. Check DISEASE_TO_TCGA keys.\n"
            "Unique primary_disease (normalized) examples:\n" + "\n".join(uniq[:80])
        )

    ph = ph.drop_duplicates(subset=["sample"], keep="first")
    print("[make_groups] phenotype mapped samples:", len(ph))
    print("[make_groups] cancers:", sorted(ph["cancer"].unique().tolist()))

    # -------------------------
    # MC3 -> TP53 protein-altering set
    # -------------------------
    mc3_sample = _pick_col(mc3, ["sample", "Tumor_Sample_Barcode", "tumor_sample_barcode"])
    mc3_gene = _pick_col(mc3, ["gene", "Hugo_Symbol", "HUGO_SYMBOL", "symbol", "Gene"])
    mc3_effect = _pick_col(mc3, ["effect", "Variant_Classification", "variant_classification"])

    filter_col = None
    if _has_col(mc3, "FILTER"):
        filter_col = _pick_col(mc3, ["FILTER"])
    elif _has_col(mc3, "filter"):
        filter_col = _pick_col(mc3, ["filter"])

    gene_up = mc3[mc3_gene].astype(str).str.strip().str.upper()
    eff_up = mc3[mc3_effect].astype(str).str.strip().str.upper()

    tp53_mask = gene_up.eq("TP53")
    eff_ok = eff_up.isin(PROTEIN_ALTERING_UP)

    tp53_total = int(tp53_mask.sum())
    tp53_pa_total = int((tp53_mask & eff_ok).sum())
    print(f"[make_groups] MC3 TP53 rows: {tp53_total}")
    print(f"[make_groups] MC3 TP53 protein-altering rows (pre-FILTER): {tp53_pa_total}")

    if tp53_total == 0:
        top_genes = gene_up.value_counts().head(10).to_dict()
        _die(f"[make_groups] no TP53 rows found in MC3. Top genes: {top_genes}")

    if filter_col is not None:
        passed = mc3[filter_col].astype(str).str.strip().str.upper().eq("PASS")
        mc3_tp53 = mc3[tp53_mask & eff_ok & passed].copy()
        print("[make_groups] using FILTER=PASS")
    else:
        mc3_tp53 = mc3[tp53_mask & eff_ok].copy()
        print("[make_groups] FILTER column not found; not applying FILTER")

    if mc3_tp53.empty:
        eff_counts = eff_up[tp53_mask].value_counts().head(20).to_dict()
        _die(
            "[make_groups] no TP53 protein-altering mutations found after filtering.\n"
            f"TP53 effect top counts: {eff_counts}\n"
            f"Expected protein-altering set: {sorted(PROTEIN_ALTERING_UP)}"
        )

    mc3_tp53["sample16"] = mc3_tp53[mc3_sample].astype(str).map(_normalize_barcode)
    tp53_mut = set(mc3_tp53["sample16"].tolist())
    if not tp53_mut:
        _die("[make_groups] TP53_mut set is empty after barcode normalization")

    # -------------------------
    # groups table (mapped cancers only)
    # -------------------------
    df = ph[["sample", "cancer"]].copy()
    df["group"] = df["sample"].map(lambda x: "TP53_mut" if x in tp53_mut else "TP53_wt")

    OUT_GROUPS.mkdir(parents=True, exist_ok=True)

    cancers = sorted(df["cancer"].unique().tolist())
    for cancer in cancers:
        g = df[df["cancer"] == cancer].copy()
        out_path = OUT_GROUPS / f"{cancer}.groups.tsv"
        g[["sample", "group"]].to_csv(out_path, sep="\t", index=False)

        n_mut = int((g["group"] == "TP53_mut").sum())
        n_wt = int((g["group"] == "TP53_wt").sum())
        if n_mut == 0 or n_wt == 0:
            print(f"[make_groups] WARNING: {cancer} has n_mut={n_mut}, n_wt={n_wt}")

    master = OUT_GROUPS / "PANCAN.groups.tsv"
    master.write_text(
        df[["sample", "cancer", "group"]].to_csv(sep="\t", index=False),
        encoding="utf-8",
    )

    print("[make_groups] OK")
    print(f"  wrote: {master}")
    print(f"  cancers: {len(cancers)}")
    print(f"  samples (mapped): {len(df)}")
    print(f"  TP53_mut (mapped): {int((df['group'] == 'TP53_mut').sum())}")


if __name__ == "__main__":
    main()
