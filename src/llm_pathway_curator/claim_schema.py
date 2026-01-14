# LLM-PathwayCurator/src/llm_pathway_curator/claim_schema.py
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

Direction = Literal["up", "down", "activated", "suppressed", "na"]
Status = Literal["PASS", "ABSTAIN", "FAIL"]

ContextKey = Literal["disease", "tissue", "perturbation", "comparison"]


class EvidenceRef(BaseModel):
    module_id: str
    gene_ids: list[str] = Field(default_factory=list)
    term_ids: list[str] = Field(default_factory=list)

    @field_validator("gene_ids", "term_ids", mode="before")
    @classmethod
    def _ensure_list(cls, v):
        # allow None -> []
        return [] if v is None else v


class Claim(BaseModel):
    claim_id: str
    entity: str
    direction: Direction
    context_keys: list[ContextKey] = Field(default_factory=list)  # fixed vocabulary
    evidence_ref: EvidenceRef

    def polarity(self) -> Literal["pos", "neg", "na"]:
        """Collapse direction into a signed polarity for audits."""
        if self.direction in {"up", "activated"}:
            return "pos"
        if self.direction in {"down", "suppressed"}:
            return "neg"
        return "na"


# ---- optional but enables clean audited outputs ----

ReasonCode = Literal[
    "OK",
    "EVIDENCE_DRIFT",
    "LOW_STABILITY",
    "CONTEXT_NON_SPECIFIC",
    "COUNTERFACTUAL_VIOLATION",
    "CONTRADICTION",
    "INVALID_SCHEMA",
    "INSUFFICIENT_EVIDENCE",
]


class Decision(BaseModel):
    status: Status
    reason: ReasonCode = "OK"
    details: dict[str, str] = Field(default_factory=dict)


class AuditedClaim(BaseModel):
    claim: Claim
    decision: Decision
