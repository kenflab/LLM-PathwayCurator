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
    try:
        v = pd.isna(x)
        return bool(v) if isinstance(v, bool) else False
    except Exception:
        return False


def _split_genes_loose(x: Any) -> list[str]:
    """
    Distill assumes EvidenceTable contract, but be tolerant:
    - list-like -> unique sorted
    - string -> split on , ; |
    """
    if _is_na_scalar(x):
        return []
    if isinstance(x, (list, tuple, set)):
        genes = [str(g).strip() for g in x if str(g).strip()]
        return sorted(set(genes))
    s = str(x).strip()
    if not s or s.lower() in {"na", "nan", "none"}:
        return []
    s = s.replace(";", ",").replace("|", ",")
    parts = [p.strip() for p in s.split(",") if p.strip()]
    return sorted(set(parts))


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
    """
    A) Evidence hygiene (v0):
    - expects EvidenceTable-like columns already normalized/validated by schema.py
    - applies masking to evidence_genes (optional)
    - creates stable IDs for joins and report
    - allocates survival columns (Float64) and keep_* gates (v0 all-pass)
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

    # Be tolerant about evidence_genes type (list vs string)
    out["evidence_genes"] = out["evidence_genes"].map(_split_genes_loose)
    out["n_evidence_genes"] = out["evidence_genes"].map(len)

    empty = out["n_evidence_genes"].eq(0)
    if empty.any():
        i = int(out.index[empty][0])
        raise ValueError(
            f"distill_evidence: empty evidence_genes at row index={i}. "
            "Fix upstream adapter/schema to supply overlap genes / leadingEdge."
        )

    # Apply masking (seed accepted; v0 deterministic unless masking uses RNG)
    masked = apply_gene_masking(out, genes_col="evidence_genes", seed=seed)
    out = masked.masked_distilled.copy()

    # stable ids for joins/reports
    out = out.reset_index(drop=True)
    out["term_row_id"] = range(len(out))
    out["term_uid"] = out["source"].astype(str) + ":" + out["term_id"].astype(str)

    # TSV-friendly genes
    out["evidence_genes_str"] = out["evidence_genes"].map(lambda xs: ",".join(xs))

    # survival placeholders (Float64 NA-capable). v0 may fill for demo.
    fill_demo = _demo_fill_survival(card)
    out["term_survival"] = pd.Series(
        [1.0] * len(out) if fill_demo else [pd.NA] * len(out),
        dtype="Float64",
    )
    out["gene_survival"] = pd.Series([pd.NA] * len(out), dtype="Float64")
    out["module_survival"] = pd.Series([pd.NA] * len(out), dtype="Float64")

    # gates for downstream (LLM input hygiene + audit)
    # v0: all pass; v1: set False when survival < tau or evidence too small
    out["keep_term"] = True
    out["keep_reason"] = "ok"

    return out
