from __future__ import annotations

import hashlib

import pandas as pd

from .claim_schema import Claim, EvidenceRef
from .sample_card import SampleCard


def _make_id(s: str) -> str:
    return hashlib.md5(s.encode("utf-8")).hexdigest()[:10]


def _split_csv(s: object) -> list[str]:
    xs = [x.strip() for x in str(s).split(",")] if s is not None else []
    return [x for x in xs if x and x.lower() not in {"na", "nan", "none"}]


def select_claims(distilled: pd.DataFrame, card: SampleCard) -> pd.DataFrame:
    """
    Placeholder for LLM-min selection.
    For now: deterministic top-k by stat with evidence_ref binding.
    """
    df = distilled.sort_values(["stat", "term_id"], ascending=[False, True]).head(3).copy()

    rows: list[dict] = []
    for _, r in df.iterrows():
        genes = _split_csv(r["evidence_genes"])
        module_id = f"m_{_make_id(str(r['term_id']))}"
        claim = Claim(
            claim_id=f"c_{_make_id(str(r['term_id']) + card.disease + card.tissue)}",
            entity=str(r["term_name"]),
            direction=str(r["direction"]),
            context_keys=["disease", "tissue", "perturbation", "comparison"],
            evidence_ref=EvidenceRef(
                module_id=module_id,
                gene_ids=genes[:10],
                term_ids=[str(r["term_id"])],
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
            }
        )

    return pd.DataFrame(rows)
