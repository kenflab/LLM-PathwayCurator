# LLM-PathwayCurator/src/llm_pathway_curator/distill.py
from __future__ import annotations

import math
from typing import Any

import pandas as pd

from .masking import apply_gene_masking
from .sample_card import SampleCard


def _split_genes(x: Any) -> list[str]:
    """Parse evidence_genes into a clean list[str]."""
    if x is None:
        return []
    if isinstance(x, (list, tuple, set)):
        return [str(g).strip() for g in x if str(g).strip()]
    try:
        if isinstance(x, float) and math.isnan(x):
            return []
    except Exception:
        pass
    s = str(x).strip()
    if not s or s.lower() in {"na", "nan", "none"}:
        return []
    parts = [p.strip() for p in s.replace(";", ",").split(",")]
    return [p for p in parts if p]


def distill_evidence(evidence: pd.DataFrame, card: SampleCard) -> pd.DataFrame:
    """
    Evidence hygiene (minimal v0):
    - Validate required columns
    - Normalize evidence_genes -> list[str]
    - Normalize direction
    - Add survival columns as stubs (real impl: LOO/jackknife + bootstrap/noise)
    """
    required = {
        "term_id",
        "term_name",
        "source",
        "stat",
        "qval",
        "direction",
        "evidence_genes",
    }
    missing = sorted(required - set(evidence.columns))
    if missing:
        raise ValueError(f"EvidenceTable missing columns: {missing}")

    out = evidence.copy()

    # basic string hygiene
    out["term_id"] = out["term_id"].astype(str).str.strip()
    out["term_name"] = out["term_name"].astype(str).str.strip()
    out["source"] = out["source"].astype(str).str.strip()

    # enforce numeric stat/qval (qval may remain NA)
    out["stat"] = pd.to_numeric(out["stat"], errors="coerce")
    if out["stat"].isna().any():
        i = int(out.index[out["stat"].isna()][0])
        raise ValueError(f"EvidenceTable has non-numeric stat at row index={i}")

    out["qval"] = pd.to_numeric(out["qval"], errors="coerce")

    # normalize direction vocabulary
    out["direction"] = (
        out["direction"]
        .astype(str)
        .str.strip()
        .str.lower()
        .replace({"activated": "up", "suppressed": "down", "nan": "na", "none": "na", "": "na"})
    )
    out.loc[~out["direction"].isin(["up", "down", "na"]), "direction"] = "na"

    # normalize evidence_genes to list[str]
    out["evidence_genes"] = out["evidence_genes"].map(_split_genes)

    # survival stubs (placeholders)
    out["term_survival"] = 1.0
    out["gene_survival"] = 1.0
    out["module_survival"] = 1.0

    out["n_evidence_genes"] = out["evidence_genes"].map(len)

    # early empty-field guard (cheap + saves pain later)
    bad = out["term_id"].eq("") | out["term_name"].eq("") | out["source"].eq("")
    if bad.any():
        i = int(out.index[bad][0])
        raise ValueError(f"EvidenceTable has empty required fields at row index={i}")

    masked = apply_gene_masking(out, genes_col="evidence_genes")
    out = masked.masked_distilled
    return out
