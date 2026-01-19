# LLM-PathwayCurator/src/llm_pathway_curator/claim_schema.py
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic.aliases import AliasChoices

from .audit_reasons import ALL_REASONS

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
    # empty allowed (term-only claims); still must include term_ids or gene_ids
    module_id: str = ""
    gene_ids: list[str] = Field(default_factory=list)
    term_ids: list[str] = Field(default_factory=list)

    gene_set_hash: str = ""  # e.g., sha256[:12] hex; empty allowed

    @field_validator("module_id", mode="before")
    @classmethod
    def _strip_module_id(cls, v: Any) -> str:
        if v is None:
            return ""
        s = str(v).strip()
        if s.lower() in {"na", "nan", "none"}:
            return ""
        return s

    @field_validator("gene_ids", "term_ids", mode="before")
    @classmethod
    def _ensure_list(cls, v: Any) -> list[str]:
        return _dedup_preserve_order(_split_listlike(v))

    @field_validator("gene_set_hash", mode="before")
    @classmethod
    def _strip_hash(cls, v: Any) -> str:
        if v is None:
            return ""
        s = str(v).strip().lower()
        if s in {"", "na", "nan", "none"}:
            return ""
        ok = all(ch in "0123456789abcdef" for ch in s) and len(s) >= 8
        if not ok:
            raise ValueError("gene_set_hash must be hex string (len>=8) or empty")
        return s

    @model_validator(mode="after")
    def _must_reference_something(self) -> EvidenceRef:
        if len(self.term_ids) == 0 and len(self.gene_ids) == 0:
            raise ValueError("EvidenceRef must include term_ids or gene_ids (non-empty)")
        return self


class Claim(BaseModel):
    claim_id: str
    entity: str
    direction: Direction
    context_keys: list[ContextKey] = Field(default_factory=list)

    # accept both keys from LLM / legacy payloads
    evidence_ref: EvidenceRef = Field(
        ...,
        validation_alias=AliasChoices("evidence_ref", "evidence_refs"),
        serialization_alias="evidence_ref",
    )

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


class Decision(BaseModel):
    status: Status
    reason: str = "ok"
    details: dict[str, Any] = Field(default_factory=dict)

    @field_validator("reason", mode="before")
    @classmethod
    def _reason_vocab(cls, v: Any) -> str:
        s = "ok" if v is None else str(v).strip()
        if not s or s.lower() in {"na", "nan", "none"}:
            s = "ok"
        allowed = {"ok"} | set(ALL_REASONS)
        if s not in allowed:
            raise ValueError(f"invalid reason={s} (must be 'ok' or one of ALL_REASONS)")
        return s


class AuditedClaim(BaseModel):
    claim: Claim
    decision: Decision
