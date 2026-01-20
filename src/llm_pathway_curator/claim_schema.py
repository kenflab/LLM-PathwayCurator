# LLM-PathwayCurator/src/llm_pathway_curator/claim_schema.py
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic.aliases import AliasChoices
from pydantic.config import ConfigDict

from .audit_reasons import ALL_REASONS

Direction = Literal["up", "down", "na"]
Status = Literal["PASS", "ABSTAIN", "FAIL"]

ContextKey = Literal["disease", "tissue", "perturbation", "comparison"]

_NA = {"na", "nan", "none", ""}


def _dedup_preserve_order(xs: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for x in xs:
        x = str(x).strip()
        if not x or x.lower() in _NA:
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
    if not s or s.lower() in _NA:
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


def _looks_like_12hex(s: str) -> bool:
    if not s:
        return False
    x = s.strip().lower()
    if len(x) != 12:
        return False
    return all(ch in "0123456789abcdef" for ch in x)


class EvidenceRef(BaseModel):
    """
    v1 contract (strict):
      - term_ids must contain EXACTLY ONE term_uid
      - gene_set_hash must be empty or 12-hex (sha256[:12])
      - gene_ids are optional but normalized to uppercase
    """

    model_config = ConfigDict(extra="forbid")

    module_id: str = ""
    gene_ids: list[str] = Field(default_factory=list)
    term_ids: list[str] = Field(default_factory=list)

    gene_set_hash: str = ""  # empty or 12-hex

    @field_validator("module_id", mode="before")
    @classmethod
    def _strip_module_id(cls, v: Any) -> str:
        if v is None:
            return ""
        s = str(v).strip()
        if s.lower() in _NA:
            return ""
        return s

    @field_validator("gene_ids", mode="before")
    @classmethod
    def _ensure_gene_list(cls, v: Any) -> list[str]:
        # parse + dedup
        xs = _dedup_preserve_order(_split_listlike(v))
        # normalize (align with audit.py/select.py)
        return [str(g).strip().upper() for g in xs if str(g).strip()]

    @field_validator("term_ids", mode="before")
    @classmethod
    def _ensure_term_list(cls, v: Any) -> list[str]:
        # parse + dedup (do NOT uppercase term_uids)
        return _dedup_preserve_order(_split_listlike(v))

    @field_validator("gene_set_hash", mode="before")
    @classmethod
    def _strip_hash(cls, v: Any) -> str:
        if v is None:
            return ""
        s = str(v).strip().lower()
        if s in _NA:
            return ""
        # v1: exactly 12-hex
        if not _looks_like_12hex(s):
            raise ValueError("gene_set_hash must be 12-hex (sha256[:12]) or empty")
        return s

    @model_validator(mode="after")
    def _enforce_v1_contract(self) -> EvidenceRef:
        # v1: exactly one term_uid
        if len(self.term_ids) != 1:
            raise ValueError("EvidenceRef.term_ids must contain exactly one term_uid (v1)")
        # gene_set_hash may be empty; if present already validated to 12-hex
        return self


class Claim(BaseModel):
    model_config = ConfigDict(extra="forbid")

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

    @field_validator("context_keys", mode="before")
    @classmethod
    def _ctx_keys_dedup(cls, v: Any) -> list[str]:
        # allow empty, but dedup/clean if provided list-like
        xs = _dedup_preserve_order(_split_listlike(v))
        return xs


class Decision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Status
    reason: str = "ok"
    details: dict[str, Any] = Field(default_factory=dict)

    @field_validator("reason", mode="before")
    @classmethod
    def _reason_vocab(cls, v: Any) -> str:
        s = "ok" if v is None else str(v).strip()
        if not s or s.lower() in _NA:
            s = "ok"
        allowed = {"ok"} | set(ALL_REASONS)
        if s not in allowed:
            raise ValueError(f"invalid reason={s} (must be 'ok' or one of ALL_REASONS)")
        return s


class AuditedClaim(BaseModel):
    model_config = ConfigDict(extra="forbid")

    claim: Claim
    decision: Decision
