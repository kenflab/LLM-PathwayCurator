# LLM-PathwayCurator/src/llm_pathway_curator/noise_lists.py

"""
Noise module definitions (shared asset; conservative by default).

Rationale (paper-facing)
------------------------
Marker rankings and enrichment evidence often contain ubiquitous programs
(e.g., clonotypes, uninformative locus IDs) that can dominate prompts and
confuse LLM interpretation. This module centralizes *symbol-centric* noise
definitions that can be applied in prompt-facing layers while preserving
evidence identity in PathwayCurator.

Policy (PathwayCurator)
-----------------------
LLM-PathwayCurator evaluates enrichment interpretations as audited decisions.
Therefore, we do *not* pre-emptively remove broad biological programs
(cell cycle, interferon, ribosome/mitochondria, HLA, Ig constants) from
evidence by default, because they can be true biology and removing them can
inflate ABSTAIN via missing/unstable evidence.

Reproducibility
---------------
Edit conservatively: changes may affect benchmark comparability.
This file is dependency-free and safe to import.
"""

# -----------------------------------------------------------------------------
# Regex patterns for noise-like tokens (symbol-centric; human/mouse-aware)
# -----------------------------------------------------------------------------

#: Regex patterns used for optional masking.
#:
#: Design intent:
#: - Keep patterns *symbol-centric* and conservative.
#: - Avoid collapsing evidence identity in PathwayCurator defaults.
#: - Prefer enabling aggressive patterns only in prompt hygiene contexts
#:   (e.g., LLM-scCurator), not in enrichment-evidence QA.
#:
#: Note:
#: - Ensembl IDs (ENSG/ENSMUSG...) are often meaningless for LLM prompting,
#:   but they are critical evidence identifiers for PathwayCurator audits.
#:   Therefore, Ensembl-ID masking is intentionally disabled here.
NOISE_PATTERNS = {
    # --- Technical / Mapping Artifacts ---
    # "Ensembl_ID": r"^(ENSG|ENSMUSG)\d+", # Meaningless IDs for LLMs
    "LINC_Noise": r"^(LINC|linc)\d+$",  # Human LINC#### / linc####
    "Mouse_Predicted_Gm": r"^Gm\d+$",  # Mouse predicted genes
    "Mouse_Rik": r"^[0-9A-Za-z]+Rik$",  # Mouse Rik
    "LOC_Locus": r"^LOC\d+$",  # Unannotated locus identifiers
    # --- Lineage Noise (Variable Regions) ---
    "TCR_Clone": r"^TR[ABGD][VDJ]",  # T-cell Receptors (Human)
    "TCR_Clone_Mouse": r"^Tr[abgd][vdj]",  # T-cell Receptors (Mouse)
    "Ig_Clone": r"^IG[HKL][VDJ]",  # Immunoglobulins (Human)
    "Ig_Clone_Mouse": r"^Ig[hkl][vdj]",  # Immunoglobulins (Mouse)
    # IMPORTANT (PathwayCurator):
    # Do not treat Ig constant regions as removable "noise" for enrichment
    # evidence by default; they are core biology for B/plasma programs.
    # Enable only in marker-centric prompt hygiene if desired.
    # --- Ig Constant Regions (Human) ---
    # "Ig_Constant_Heavy": r"^IGH[-_]?((M|D)|G[1-4]|A[1-2]|E)$",  # IGHM/IGHD/IGHG1–4/IGHA1–2/IGHE
    # "Ig_Constant_Light_Kappa": r"^IGKC$",  # IGKC
    # "Ig_Constant_Light_Lambda": r"^IGLC.*$",  # IGLC1–7
    # --- Ig Constant Regions (Mouse) ---
    # "Ig_Constant_Heavy_Mouse": r"^Igh[-_]?((m|d)|g[1-4]|a|e)$",  # Ighm/Ighd/Ighg1–4/Igha/Ighe
    # "Ig_Constant_Light_Kappa_Mouse": r"^Igkc$",  # Igkc
    # "Ig_Constant_Light_Lambda_Mouse": r"^Iglc.*$",  # Iglc1–7
    # --- Biological State Noise ---
    # "Mito_Artifact": r"^[Mm][Tt]-",  # Mitochondrial (MT- or mt-)
    # "Ribo_Artifact": r"^[Rr][Pp][LSls]",  # Ribosomal (RPS/RPL or Rps/Rpl)
    # "HeatShock": r"^[Hh][Ss][Pp]",  # Heat shock (HSP or Hsp)
    # "JunFos_Stress": r"^(JUN|FOS|Jun|Fos)",  # Dissociation stress
    # "Hemo_Contam": r"^[Hh][Bb][ABab]",  # Hemoglobin
    # "Translation_Factor": r"^(EEF|EIF|TPT1|Eef|Eif|Tpt1)",  # Translation
    # --- Chromatin & Proliferation Artifacts ---
    # "Histone": r"^(HIST|Hist)",
    # --- Donor/Batch Confounders ---
    # 1. HLA Class I (Human): Ubiquitous & Interferon-sensitive.
    # "HLA_ClassI_Noise": r"^HLA-[ABCEFG]",  # Keeps Class II (HLA-D) for APC definition.
    # 2. MHC Class I (Mouse): H-2K, H-2D, H-2L.
    # "MHC_ClassI_Noise": r"^H2-[DKL]",  # Keeps Class II (H-2A, H-2E) for APC definition.
    # 3. Sex Chromosome (Gender Batch Effect Removal)
    # "SexChromosome": r"^(XIST|UTY|DDX3Y|Xist|Uty|Ddx3y)",  # Removes XIST (Female)
    # and Y-linked genes (Male)
}


# -----------------------------------------------------------------------------
# Cell-cycle gene sets (shared reference asset; not enabled by default here)
# -----------------------------------------------------------------------------

#: Full cell-cycle gene set (human uppercase) derived from published marker sets
#: (e.g., Tirosh et al., Science 2016). Kept as a shared reference asset.
_HUMAN_CC_GENES = {
    # G1/S
    "MCM5",
    "PCNA",
    "TYMS",
    "FEN1",
    "MCM2",
    "MCM4",
    "RRM1",
    "UNG",
    "GINS2",
    "MCM6",
    "CDCA7",
    "DTL",
    "PRIM1",
    "UHRF1",
    "MLF1IP",
    "HELLS",
    "RFC2",
    "RPA2",
    "NASP",
    "RAD51AP1",
    "GMNN",
    "WDR76",
    "SLBP",
    "CCNE2",
    "UBR7",
    "POLD3",
    "MSH2",
    "ATAD2",
    "RAD51",
    "RRM2",
    "CDC45",
    "CDC6",
    "EXO1",
    "TIPIN",
    "DSCC1",
    "BLM",
    "CASP8AP2",
    "USP1",
    "CLSPN",
    "POLA1",
    "CHAF1B",
    "BRIP1",
    "E2F8",
    # G2/M
    "HMGB2",
    "CDK1",
    "NUSAP1",
    "UBE2C",
    "BIRC5",
    "TPX2",
    "TOP2A",
    "NDC80",
    "CKS2",
    "NUF2",
    "CKS1B",
    "MKI67",
    "TMPO",
    "CENPF",
    "TACC3",
    "FAM64A",
    "SMC4",
    "CCNB2",
    "CKAP2L",
    "CKAP2",
    "AURKB",
    "BUB1",
    "KIF11",
    "ANP32E",
    "TUBB4B",
    "GTSE1",
    "KIF20B",
    "HJURP",
    "CDCA3",
    "HN1",
    "CDC20",
    "TTK",
    "CDC25C",
    "KIF2C",
    "RANGAP1",
    "NCAPD2",
    "DLGAP5",
    "CDCA2",
    "CDCA8",
    "ECT2",
    "KIF23",
    "HMMR",
    "AURKA",
    "PSRC1",
    "ANLN",
    "LBR",
    "CKAP5",
    "CENPE",
    "CTCF",
    "NEK2",
    "G2E3",
    "GAS2L3",
    "CBX5",
    "CENPA",
    # Melanoma Core Cycling (Additional)
    "TK1",
    "UBE2T",
    "MAD2L1",
    "ZWINT",
    "MCM7",
    "KIAA0101",
    "PTTG1",
    "CENPM",
    "KPNA2",
    "ASF1B",
    "KIF22",
    "FANCI",
    "TUBA1B",
    "CDKN3",
    "WDR34",
    "CCNB1",
    "PBK",
    "RPL39L",
    "SNRNP25",
    "TUBG1",
    "RNASEH2A",
    "DTYMK",
    "RFC3",
    "H2AFZ",
    "NUDT1",
    "RFC4",
    "RACGAP1",
    "KIFC1",
    "TUBB6",
    "ORC6",
    "CENPW",
    "CCNA2",
    "EZH2",
    "DEK",
    "DSN1",
    "DHFR",
    "TCF19",
    "HAT1",
    "VRK1",
    "SDF2L1",
    "PHF19",
    "SHCBP1",
    "SAE1",
    "CDCA5",
    "OIP5",
    "RANBP1",
    "LMNB1",
    "TROAP",
    "RFC5",
    "DNMT1",
    "MND1",
    "TIMELESS",
    "HMGB1",
    "ZWILCH",
    "ASPM",
    "POLA2",
    "FABP5",
    "TMEM194A",
}

# Sentinel proliferation markers we want to keep (not mask by default)
PROLIFERATION_SENTINELS = {
    "MKI67",
    "CDK1",
    "CCNB1",
    "CCNB2",
    "PCNA",
    "TOP2A",
    "BIRC5",
    "Mki67",
    "Cdk1",
    "Ccnb1",
    "Ccnb2",
    "Pcna",
    "Top2a",
    "Birc5",
}

#: Cell-cycle genes excluding proliferation sentinels.
#: Also includes a simple mouse-friendly capitalization variant.
_CELL_CYCLE_ALL = _HUMAN_CC_GENES.union({g.capitalize() for g in _HUMAN_CC_GENES})
CELL_CYCLE_GENES = _CELL_CYCLE_ALL.difference(PROLIFERATION_SENTINELS)

# -----------------------------------------------------------------------------
# Noise list profiles
# -----------------------------------------------------------------------------

#: Marker-centric suppression profile (opt-in).
#: Intended for prompt hygiene / LLM-scCurator-style usage.
NOISE_LISTS_CELLTYPE = {
    "CellCycle_State": CELL_CYCLE_GENES,
}

#: PathwayCurator default profile (conservative).
#:
#: Rationale:
#: - PathwayCurator audits enrichment evidence; broad programs can be true biology.
#: - Pre-emptive removal can bias evidence support and inflate ABSTAIN.
#:
#: Users who want marker-centric suppression should explicitly opt-in to
#: `NOISE_LISTS_CELLTYPE` (or a custom profile) in the prompt-facing layer.
NOISE_LISTS: dict[str, set[str]] = {}
