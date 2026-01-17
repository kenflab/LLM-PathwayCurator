# LLM-PathwayCurator/src/llm_pathway_curator/distill.py
from __future__ import annotations

from typing import Any

import pandas as pd

from .masking import apply_gene_masking
from .sample_card import SampleCard

_NA_TOKENS = {"", "na", "nan", "none", "NA"}


def _is_na_scalar(x: Any) -> bool:
    """pd.isna is unsafe for list-like; only treat scalars here."""
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
    Tolerant parsing for evidence_genes:
      - list/tuple/set -> unique, sorted
      - string -> split on , ; | -> unique, sorted
      - NA/empty -> []
    Notes:
      - We intentionally sort to stabilize downstream artifacts/hashes.
      - If you need to preserve original order, do it upstream in adapters.
    """
    if _is_na_scalar(x):
        return []
    if isinstance(x, (list, tuple, set)):
        genes = [str(g).strip() for g in x if str(g).strip()]
        return sorted(set(genes))

    s = str(x).strip()
    if not s or s.lower() in {t.lower() for t in _NA_TOKENS}:
        return []
    s = s.replace(";", ",").replace("|", ",")
    parts = [p.strip() for p in s.split(",") if p.strip()]
    return sorted(set(parts))


def _ensure_float64_na_series(n: int) -> pd.Series:
    """Float64 NA-capable series of length n."""
    return pd.Series([pd.NA] * n, dtype="Float64")


def distill_evidence(
    evidence: pd.DataFrame,
    card: SampleCard,
    *,
    seed: int | None = None,
) -> pd.DataFrame:
    """
    A) Evidence hygiene (v1-minimal, deterministic)

    Responsibilities (and only these):
      1) Normalize EvidenceTable-like inputs into a stable, joinable table:
         - evidence_genes -> list[str] (tolerant to strings)
         - stable IDs: term_uid, term_row_id
         - TSV-friendly columns: evidence_genes_str
      2) Preserve survival columns if already provided; otherwise allocate NA placeholders.

    Non-responsibilities (explicitly NOT done here):
      - computing survival via LOO/jackknife/bootstrap
      - deciding PASS/ABSTAIN/FAIL
      - inventing/demo-filling survival values
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

    # Normalize evidence_genes (list[str]) and validate non-empty
    out["evidence_genes"] = out["evidence_genes"].map(_split_genes_loose)
    out["n_evidence_genes"] = out["evidence_genes"].map(len)

    empty = out["n_evidence_genes"].eq(0)
    if empty.any():
        i = int(out.index[empty][0])
        raise ValueError(
            f"distill_evidence: empty evidence_genes at row index={i}. "
            "Upstream adapter must supply overlap genes / leadingEdge."
        )

    # Optional masking (still deterministic if masking is deterministic for a given seed)
    masked = apply_gene_masking(out, genes_col="evidence_genes", seed=seed)
    out = masked.masked_distilled.copy()

    # Stable IDs for joins/reports
    out = out.reset_index(drop=True)
    out["term_row_id"] = range(len(out))
    out["term_uid"] = (
        out["source"].astype(str).str.strip() + ":" + out["term_id"].astype(str).str.strip()
    )

    # TSV-friendly genes
    out["evidence_genes_str"] = out["evidence_genes"].map(lambda xs: ",".join(xs))

    # Survival placeholders: preserve if present, else NA.
    n = len(out)

    if "term_survival" in out.columns:
        out["term_survival"] = pd.to_numeric(out["term_survival"], errors="coerce").astype(
            "Float64"
        )
    else:
        out["term_survival"] = _ensure_float64_na_series(n)

    if "gene_survival" in out.columns:
        out["gene_survival"] = pd.to_numeric(out["gene_survival"], errors="coerce").astype(
            "Float64"
        )
    else:
        out["gene_survival"] = _ensure_float64_na_series(n)

    if "module_survival" in out.columns:
        out["module_survival"] = pd.to_numeric(out["module_survival"], errors="coerce").astype(
            "Float64"
        )
    else:
        out["module_survival"] = _ensure_float64_na_series(n)

    # Gates for downstream (kept, but intentionally inert here)
    if "keep_term" not in out.columns:
        out["keep_term"] = True
    if "keep_reason" not in out.columns:
        out["keep_reason"] = "ok"

    return out
