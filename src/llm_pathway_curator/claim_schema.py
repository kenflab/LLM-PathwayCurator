# LLM-PathwayCurator/src/llm_pathway_curator/claim_schema.py
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

# Canonical direction for auditable claims
Direction = Literal["up", "down", "na"]
Status = Literal["PASS", "ABSTAIN", "FAIL"]

ContextKey = Literal["disease", "tissue", "perturbation", "comparison"]


def _dedup_preserve_order(xs: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for x in xs:
        x = str(x).strip()
        if not x or x.lower() in {"na", "nan", "none"}:
            continue
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def _split_listlike(v: Any) -> list[str]:
    """
    Accept list/tuple/set OR comma/semicolon-separated strings.
    Normalize to list[str]. None/NA -> [].
    """
    if v is None:
        return []
    if isinstance(v, (list, tuple, set)):
        return [str(x).strip() for x in v if str(x).strip()]
    s = str(v).strip()
    if not s or s.lower() in {"na", "nan", "none"}:
        return []
    s = s.replace(";", ",")
    return [t.strip() for t in s.split(",") if t.strip()]


def _norm_direction(v: Any) -> str:
    s = str(v).strip().lower()
    if s in {"up", "upregulated", "activated", "pos", "positive", "+", "1"}:
        return "up"
    if s in {"down", "downregulated", "suppressed", "neg", "negative", "-", "-1"}:
        return "down"
    return "na"


class EvidenceRef(BaseModel):
    module_id: str
    gene_ids: list[str] = Field(default_factory=list)
    term_ids: list[str] = Field(default_factory=list)

    @field_validator("module_id", mode="before")
    @classmethod
    def _strip_module_id(cls, v: Any) -> str:
        if v is None:
            return ""
        return str(v).strip()

    @field_validator("gene_ids", "term_ids", mode="before")
    @classmethod
    def _ensure_list(cls, v: Any) -> list[str]:
        return _dedup_preserve_order(_split_listlike(v))


class Claim(BaseModel):
    claim_id: str
    entity: str
    direction: Direction
    context_keys: list[ContextKey] = Field(default_factory=list)
    evidence_ref: EvidenceRef

    @field_validator("claim_id", "entity", mode="before")
    @classmethod
    def _non_empty(cls, v: Any) -> str:
        s = "" if v is None else str(v).strip()
        if not s:
            raise ValueError("must be non-empty")
        return s

    @field_validator("direction", mode="before")
    @classmethod
    def _dir_canonical(cls, v: Any) -> str:
        return _norm_direction(v)


# ---- audited outputs ----
# IMPORTANT: keep reason codes aligned with audit_reasons.py (snake_case).
ReasonCode = Literal[
    "ok",
    # FAIL reasons
    "evidence_drift",
    "schema_violation",
    "contradiction",
    # ABSTAIN reasons
    "unstable",
    "missing_survival",
    "context_nonspecific",
    "under_supported",
    "hub_bridge",
    "inconclusive_stress",
]


class Decision(BaseModel):
    status: Status
    reason: ReasonCode = "ok"
    details: dict[str, Any] = Field(default_factory=dict)


class AuditedClaim(BaseModel):
    claim: Claim
    decision: Decision
