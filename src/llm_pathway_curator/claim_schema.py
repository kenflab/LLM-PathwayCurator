# LLM-PathwayCurator/src/llm_pathway_curator/claim_schema.py
from __future__ import annotations

import hashlib
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


def _stable_claim_id(
    *, term_ids: list[str], direction: str, gene_set_hash: str, context_keys: list[str]
) -> str:
    """
    Tool-owned deterministic claim_id.
    Keep it minimal and stable:
      - depends on evidence identity + direction + which context KEYS are present
      - does NOT depend on free-text context VALUES (they can drift);
        values live in SampleCard anyway
    """
    payload = "|".join(
        [
            ",".join([str(t).strip() for t in term_ids if str(t).strip()]),
            str(direction or "na").strip().lower(),
            str(gene_set_hash or "").strip().lower(),
            ",".join([str(k).strip() for k in context_keys if str(k).strip()]),
        ]
    )
    return "c_" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


class EvidenceRef(BaseModel):
    """
    V2 contract (strict, simple):
      - term_ids: 1+ term_uid strings (module-level evidence allowed)
      - gene_set_hash: REQUIRED 12-hex (sha256[:12]) fingerprint of the *evidence gene set*
        (tool-owned; should match audit.py/select.py normalization rules)
      - gene_ids: optional, uppercase, compact (display/reference only)
      - module_id: optional (may be empty if unknown)
    """

    model_config = ConfigDict(extra="forbid")

    module_id: str = ""
    gene_ids: list[str] = Field(default_factory=list)
    term_ids: list[str] = Field(default_factory=list)
    gene_set_hash: str = ""  # REQUIRED 12-hex in v2

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
        xs = _dedup_preserve_order(_split_listlike(v))
        return [str(g).strip().upper() for g in xs if str(g).strip()]

    @field_validator("term_ids", mode="before")
    @classmethod
    def _ensure_term_list(cls, v: Any) -> list[str]:
        # do NOT uppercase term_uids
        return _dedup_preserve_order(_split_listlike(v))

    @field_validator("gene_set_hash", mode="before")
    @classmethod
    def _strip_hash(cls, v: Any) -> str:
        s = "" if v is None else str(v).strip().lower()
        if s in _NA:
            s = ""
        return s

    @model_validator(mode="after")
    def _enforce_v2_contract(self) -> EvidenceRef:
        if len(self.term_ids) < 1:
            raise ValueError("EvidenceRef.term_ids must contain at least one term_uid (v2)")
        if not _looks_like_12hex(self.gene_set_hash):
            raise ValueError(
                "EvidenceRef.gene_set_hash is required and must be 12-hex (sha256[:12]) (v2)"
            )
        return self


class Claim(BaseModel):
    """
    V2 Claim:
      - claim_id is tool-owned; may be omitted/empty on input and will be filled deterministically.
      - entity: term_id (not term_name)
      - direction: up|down|na
      - evidence_ref: schema-locked EvidenceRef
    """

    model_config = ConfigDict(extra="forbid")

    claim_id: str = ""  # tool-owned; auto-fill if empty
    entity: str
    direction: Direction
    context_keys: list[ContextKey] = Field(default_factory=list)

    evidence_ref: EvidenceRef = Field(
        ...,
        validation_alias=AliasChoices("evidence_ref", "evidence_refs"),
        serialization_alias="evidence_ref",
    )

    @field_validator("entity", mode="before")
    @classmethod
    def _entity_non_empty(cls, v: Any) -> str:
        s = "" if v is None else str(v).strip()
        if not s:
            raise ValueError("entity must be non-empty")
        return s

    @field_validator("direction", mode="before")
    @classmethod
    def _dir_canonical(cls, v: Any) -> str:
        return _norm_direction(v)

    @field_validator("context_keys", mode="before")
    @classmethod
    def _ctx_keys_dedup(cls, v: Any) -> list[str]:
        xs = _dedup_preserve_order(_split_listlike(v))
        # keep only allowed keys, preserve order
        allowed = {"disease", "tissue", "perturbation", "comparison"}
        xs2 = [x for x in xs if x in allowed]
        return xs2

    @field_validator("claim_id", mode="before")
    @classmethod
    def _claim_id_strip(cls, v: Any) -> str:
        # allow empty; will auto-fill
        if v is None:
            return ""
        s = str(v).strip()
        if s.lower() in _NA:
            return ""
        return s

    @model_validator(mode="after")
    def _fill_claim_id(self) -> Claim:
        if self.claim_id:
            return self
        cid = _stable_claim_id(
            term_ids=list(self.evidence_ref.term_ids or []),
            direction=str(self.direction or "na"),
            gene_set_hash=str(self.evidence_ref.gene_set_hash or ""),
            context_keys=list(self.context_keys or []),
        )
        # pydantic v2 safe copy
        try:
            return self.model_copy(update={"claim_id": cid})
        except Exception:
            d = self.model_dump()
            d["claim_id"] = cid
            return Claim.model_validate(d)


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
    """
    Keep as the single audited container (no v1/v2 split).
    """

    model_config = ConfigDict(extra="forbid")

    claim: Claim
    decision: Decision
