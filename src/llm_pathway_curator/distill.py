# LLM-PathwayCurator/src/llm_pathway_curator/distill.py
from __future__ import annotations

from typing import Any

import pandas as pd

from .masking import apply_gene_masking
from .sample_card import SampleCard


def _is_na_scalar(x: Any) -> bool:
    if x is None:
        return True
    if isinstance(x, (list, tuple, set, dict)):
        return False
    if isinstance(x, float):
        try:
            return bool(pd.isna(x))
        except Exception:
            return False
    try:
        v = pd.isna(x)
        return bool(v) if isinstance(v, bool) else False
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
    if isinstance(x, (list, tuple, set)):
        genes = [str(g).strip() for g in x if str(g).strip()]
        return sorted(set(genes))
    if _is_na_scalar(x):
        return []
    s = str(x).strip()
    if not s or s.lower() in {"na", "nan", "none"}:
        return []
    parts = [p.strip() for p in s.replace(";", ",").split(",")]
    genes = [p for p in parts if p]
    return sorted(set(genes))


def _norm_direction(x: Any) -> str:
    s = str(x).strip().lower()
    if s in {"up", "activated", "pos", "positive", "+", "1"}:
        return "up"
    if s in {"down", "suppressed", "neg", "negative", "-", "-1"}:
        return "down"
    return "na"


def _demo_fill_survival(card: SampleCard) -> bool:
    try:
        extra = getattr(card, "extra", {}) or {}
        return bool(extra.get("demo_fill_survival", False))
    except Exception:
        return False


def distill_evidence(
    evidence: pd.DataFrame,
    card: SampleCard,
    *,
    seed: int | None = None,
) -> pd.DataFrame:
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

    out["term_id"] = out["term_id"].map(_clean_required_str)
    out["term_name"] = out["term_name"].map(_clean_required_str)
    out["source"] = out["source"].map(_clean_required_str)

    bad_req = out["term_id"].eq("") | out["term_name"].eq("") | out["source"].eq("")
    if bad_req.any():
        i = int(out.index[bad_req][0])
        raise ValueError(f"distill_evidence: empty required fields at row index={i}")

    out["stat"] = pd.to_numeric(out["stat"], errors="coerce")
    if out["stat"].isna().any():
        i = int(out.index[out["stat"].isna()][0])
        raise ValueError(f"distill_evidence: non-numeric stat at row index={i}")
    out["qval"] = pd.to_numeric(out["qval"], errors="coerce")

    out["direction"] = out["direction"].map(_norm_direction)

    out["evidence_genes"] = out["evidence_genes"].map(_split_genes)
    out["n_evidence_genes"] = out["evidence_genes"].map(len)

    empty_genes = out["n_evidence_genes"].eq(0)
    if empty_genes.any():
        i = int(out.index[empty_genes][0])
        raise ValueError(f"distill_evidence: empty evidence_genes at row index={i}")

    # Apply masking (seed accepted but v0 deterministic)
    masked = apply_gene_masking(out, genes_col="evidence_genes", seed=seed)
    out = masked.masked_distilled

    # ---- stable ids for joins/reports ----
    out = out.reset_index(drop=True).copy()
    out["term_row_id"] = range(len(out))
    out["term_uid"] = out["source"].astype(str) + ":" + out["term_id"].astype(str)

    # TSV-friendly genes
    out["evidence_genes_str"] = out["evidence_genes"].map(lambda xs: ",".join(xs))

    # survival placeholders (Float64 NA-capable)
    fill_demo = _demo_fill_survival(card)
    if fill_demo:
        out["term_survival"] = pd.Series([1.0] * len(out), dtype="Float64")
    else:
        out["term_survival"] = pd.Series([pd.NA] * len(out), dtype="Float64")
    out["gene_survival"] = pd.Series([pd.NA] * len(out), dtype="Float64")
    out["module_survival"] = pd.Series([pd.NA] * len(out), dtype="Float64")

    return out
