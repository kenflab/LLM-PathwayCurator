# LLM-PathwayCurator/src/llm_pathway_curator/select.py
from __future__ import annotations

import hashlib
from typing import Any

import pandas as pd

from .claim_schema import Claim, EvidenceRef
from .sample_card import SampleCard

_ALLOWED_DIRECTIONS = {"up", "down", "activated", "suppressed", "na"}


def _make_id(s: str) -> str:
    return hashlib.md5(s.encode("utf-8")).hexdigest()[:10]


def _as_gene_list(x: Any) -> list[str]:
    # canonical: list[str]
    if x is None:
        return []
    if isinstance(x, (list, tuple, set)):
        return [str(g).strip() for g in x if str(g).strip()]
    # accept strings too (comma/semicolon)
    s = str(x).strip().replace(";", ",")
    if not s or s.lower() in {"na", "nan", "none"}:
        return []
    return [g.strip() for g in s.split(",") if g.strip()]


def _norm_direction(x: Any) -> str:
    s = str(x).strip().lower()
    if s in _ALLOWED_DIRECTIONS:
        return s
    if s in {"pos", "positive"}:
        return "up"
    if s in {"neg", "negative"}:
        return "down"
    return "na"


def select_claims(distilled: pd.DataFrame, card: SampleCard, *, k: int = 3) -> pd.DataFrame:
    """
    Deterministic placeholder for LLM-min selection.
    - Select top-k terms by stat (tie-break by term_id)
    - Bind evidence_ref to (module_id, genes, term_id)
    Output is a flat table for downstream audit/report.
    """
    required = {"term_id", "term_name", "stat", "direction", "evidence_genes"}
    missing = sorted(required - set(distilled.columns))
    if missing:
        raise ValueError(f"select_claims: missing columns in distilled: {missing}")

    df = distilled.sort_values(["stat", "term_id"], ascending=[False, True]).head(int(k)).copy()

    rows: list[dict[str, Any]] = []
    for _, r in df.iterrows():
        term_id = str(r["term_id"]).strip()
        term_name = str(r["term_name"]).strip()
        direction = _norm_direction(r.get("direction", "na"))
        genes = _as_gene_list(r.get("evidence_genes"))

        # Prefer real module_id from factorization; fallback only if absent
        module_id = (
            str(r["module_id"]).strip()
            if "module_id" in df.columns and pd.notna(r["module_id"])
            else f"M_fallback_{_make_id(term_id)}"
        )

        ctx_key = "|".join(
            [
                term_id,
                card.disease,
                card.tissue,
                card.perturbation,
                card.comparison,
            ]
        )

        claim = Claim(
            claim_id=f"c_{_make_id(ctx_key)}",
            entity=term_name,  # placeholder: later will become module-typed entity
            direction=direction,  # validated
            context_keys=["disease", "tissue", "perturbation", "comparison"],
            evidence_ref=EvidenceRef(
                module_id=module_id,
                gene_ids=genes[:10],
                term_ids=[term_id],
            ),
        )

        rows.append(
            {
                "claim_id": claim.claim_id,
                "entity": claim.entity,
                "direction": claim.direction,
                "context_keys": ",".join(claim.context_keys),
                "module_id": claim.evidence_ref.module_id,
                "gene_ids": ",".join(claim.evidence_ref.gene_ids),
                "term_ids": ",".join(claim.evidence_ref.term_ids),
                "claim_json": claim.model_dump_json(),  # future-proof for audit/report
            }
        )

    return pd.DataFrame(rows)
