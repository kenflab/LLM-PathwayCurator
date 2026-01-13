from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Direction = Literal["up", "down", "activated", "suppressed", "na"]
Status = Literal["PASS", "ABSTAIN", "FAIL"]


class EvidenceRef(BaseModel):
    module_id: str
    gene_ids: list[str] = Field(default_factory=list)
    term_ids: list[str] = Field(default_factory=list)


class Claim(BaseModel):
    claim_id: str
    entity: str
    direction: Direction
    context_keys: list[str] = Field(default_factory=list)  # e.g., ["disease", "tissue"]
    evidence_ref: EvidenceRef
