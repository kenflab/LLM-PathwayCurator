# LLM-PathwayCurator/src/llm_pathway_curator/distill.py
from __future__ import annotations

from typing import Any

import pandas as pd

from .masking import apply_gene_masking
from .sample_card import SampleCard


def _is_na_scalar(x: Any) -> bool:
    """
    Safe NA check that won't explode on list-like objects.

    pd.isna(list) returns an array, which cannot be used in boolean contexts.
    We therefore only treat pd.isna(x) as scalar NA when it returns a bool.
    """
    if x is None:
        return True

    # Do not call pd.isna on list-like containers
    if isinstance(x, (list, tuple, set, dict)):
        return False

    # Fast path for float NaN
    if isinstance(x, float):
        try:
            return bool(pd.isna(x))
        except Exception:
            return False

    # Generic scalar-ish check
    try:
        v = pd.isna(x)
        return bool(v) if isinstance(v, (bool,)) else False
    except Exception:
        return False


def _clean_required_str(x: Any) -> str:
    if _is_na_scalar(x):
        return ""
    s = str(x).strip()
    if s.lower() in {"na", "nan", "none"}:
        return ""
    return s


def _split_genes(x: Any) -> list[str]:
    """Parse evidence_genes into a clean list[str]."""
    # list-like first (most important to avoid pd.isna(list) ambiguity)
    if isinstance(x, (list, tuple, set)):
        genes = [str(g).strip() for g in x if str(g).strip()]
        return sorted(set(genes))

    if _is_na_scalar(x):
        return []

    # now it's string-ish / scalar-ish
    s = str(x).strip()
    if not s or s.lower() in {"na", "nan", "none"}:
        return []

    parts = [p.strip() for p in s.replace(";", ",").split(",")]
    genes = [p for p in parts if p]

    # stable de-dup (sorted) to keep deterministic outputs
    return sorted(set(genes))


def _norm_direction(x: Any) -> str:
    s = str(x).strip().lower()
    if s in {"up", "activated", "pos", "positive", "+", "1"}:
        return "up"
    if s in {"down", "suppressed", "neg", "negative", "-", "-1"}:
        return "down"
    return "na"


def distill_evidence(
    evidence: pd.DataFrame,
    card: SampleCard,
    *,
    seed: int | None = None,
) -> pd.DataFrame:
    """
    Evidence hygiene (v0 that matches paper semantics):
    - Validate required columns
    - Enforce canonical types / vocab
    - Normalize evidence_genes -> list[str]
    - Leave survival as NA (so audits can ABSTAIN for missing survival)
    - Apply masking (optional hygiene)

    Note:
      seed is accepted for forward-compatibility (future bootstrap/noise),
      but v0 remains deterministic.
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
        raise ValueError(f"distill_evidence: missing columns: {missing}")

    out = evidence.copy()

    # required string hygiene (NaN-safe)
    out["term_id"] = out["term_id"].map(_clean_required_str)
    out["term_name"] = out["term_name"].map(_clean_required_str)
    out["source"] = out["source"].map(_clean_required_str)

    bad_req = out["term_id"].eq("") | out["term_name"].eq("") | out["source"].eq("")
    if bad_req.any():
        i = int(out.index[bad_req][0])
        raise ValueError(f"distill_evidence: empty required fields at row index={i}")

    # numeric stat/qval (qval may remain NA)
    out["stat"] = pd.to_numeric(out["stat"], errors="coerce")
    if out["stat"].isna().any():
        i = int(out.index[out["stat"].isna()][0])
        raise ValueError(f"distill_evidence: non-numeric stat at row index={i}")
    out["qval"] = pd.to_numeric(out["qval"], errors="coerce")

    # canonical direction vocabulary
    out["direction"] = out["direction"].map(_norm_direction)

    # canonical evidence_genes
    out["evidence_genes"] = out["evidence_genes"].map(_split_genes)
    out["n_evidence_genes"] = out["evidence_genes"].map(len)

    # core method needs genes for factorization; fail early
    empty_genes = out["n_evidence_genes"].eq(0)
    if empty_genes.any():
        i = int(out.index[empty_genes][0])
        raise ValueError(f"distill_evidence: empty evidence_genes at row index={i}")

    # survival placeholders: NA means "not computed" (audits should ABSTAIN)
    out["term_survival"] = pd.NA
    out["gene_survival"] = pd.NA
    out["module_survival"] = pd.NA

    # Apply masking (seed is accepted but v0 masking is deterministic)
    masked = apply_gene_masking(out, genes_col="evidence_genes", seed=seed)
    out = masked.masked_distilled

    # Avoid using card prematurely to prevent accidental "context leakage" in hygiene
    _ = card

    return out
