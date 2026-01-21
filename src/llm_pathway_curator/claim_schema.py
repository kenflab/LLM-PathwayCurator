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


def _sha256_12hex(payload: str) -> str:
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def _canonical_sorted_unique(xs: list[str]) -> list[str]:
    # Stable, order-invariant canonicalization for IDs/hashes.
    # - strip
    # - drop NA/empty
    # - dedup (case-sensitive; callers can pre-normalize)
    # - sort for determinism
    ys = [str(x).strip() for x in xs if str(x).strip() and str(x).strip().lower() not in _NA]
    return sorted(set(ys))


def _stable_claim_id(
    *,
    term_ids: list[str],
    direction: str,
    gene_set_hash: str,
    context_keys: list[str],
) -> str:
    """
    Tool-owned deterministic claim_id.

    Must be stable across:
      - term_id ordering differences
      - context_keys ordering differences
      - case/whitespace jitter in inputs (handled by upstream normalizers)

    Depends on:
      - evidence identity (term_ids + gene_set_hash)
      - direction
      - which context KEYS are present (NOT free-text context values)
    """
    term_ids_c = _canonical_sorted_unique([str(t) for t in term_ids])
    ctx_keys_c = _canonical_sorted_unique([str(k) for k in context_keys])
    d = str(direction or "na").strip().lower()
    h = str(gene_set_hash or "").strip().lower()

    payload = "|".join(
        [
            ",".join(term_ids_c),
            d,
            h,
            ",".join(ctx_keys_c),
        ]
    )
    return "c_" + _sha256_12hex(payload)


def _stable_gene_set_hash_from_gene_ids(gene_ids: list[str]) -> str:
    # Hash should be order-invariant and robust to case/whitespace jitter.
    # We intentionally do NOT attempt biological ID mapping here.
    canon = sorted(set([str(g).strip().upper() for g in gene_ids if str(g).strip()]))
    payload = ",".join(canon)
    return _sha256_12hex(payload)


def _stable_gene_set_hash_fallback(term_ids: list[str]) -> str:
    # Fallback when gene_ids are unavailable.
    # Still produces a deterministic fingerprint of "evidence identity",
    # but note it is term-driven, not gene-driven.
    canon = _canonical_sorted_unique([str(t) for t in term_ids])
    payload = ",".join(canon)
    return _sha256_12hex(payload)


class EvidenceRef(BaseModel):
    """
    EvidenceRef (strict, simple, tool-owned where possible):

    Required:
      - term_ids: 1+ term_uid strings (module-level evidence allowed)

    Optional (tool-owned; will be auto-filled if missing):
      - gene_set_hash: 12-hex (sha256[:12]) fingerprint of the *evidence gene set*
        If missing, it is deterministically derived from gene_ids when available,
        otherwise from term_ids as a fallback.

    Optional (display/reference only):
      - gene_ids: list[str]
      - module_id: str
    """

    model_config = ConfigDict(extra="forbid")

    module_id: str = ""
    gene_ids: list[str] = Field(default_factory=list)
    term_ids: list[str] = Field(default_factory=list)
    gene_set_hash: str = ""  # tool-owned; auto-fill if empty

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
        # Keep original casing for display as much as possible (only strip/dedup).
        # Hashing uses a canonicalized uppercase+sorted representation internally.
        return _dedup_preserve_order(_split_listlike(v))

    @field_validator("term_ids", mode="before")
    @classmethod
    def _ensure_term_list(cls, v: Any) -> list[str]:
        # Do NOT uppercase term_uids.
        return _dedup_preserve_order(_split_listlike(v))

    @field_validator("gene_set_hash", mode="before")
    @classmethod
    def _strip_hash(cls, v: Any) -> str:
        s = "" if v is None else str(v).strip().lower()
        if s in _NA:
            s = ""
        return s

    @model_validator(mode="after")
    def _enforce_contract_and_fill_hash(self) -> EvidenceRef:
        if len(self.term_ids) < 1:
            raise ValueError("EvidenceRef.term_ids must contain at least one term_uid")

        h = str(self.gene_set_hash or "").strip().lower()
        if not _looks_like_12hex(h):
            if self.gene_ids:
                h = _stable_gene_set_hash_from_gene_ids(self.gene_ids)
            else:
                h = _stable_gene_set_hash_fallback(self.term_ids)

        if not _looks_like_12hex(h):
            raise ValueError("EvidenceRef.gene_set_hash must be 12-hex (sha256[:12])")

        # IMPORTANT: return self (not a new object) to avoid pydantic warning
        self.gene_set_hash = h
        return self


class Claim(BaseModel):
    """
    Claim (typed, auditable):

      - claim_id is tool-owned; may be omitted/empty on input and will be filled deterministically.
      - entity: a stable identifier (prefer term_id; not free text)
      - direction: up|down|na
      - context_keys: which SampleCard keys this claim is conditioned on (values live elsewhere)
      - evidence_ref: schema-locked EvidenceRef (no free text evidence)
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
        allowed = {"disease", "tissue", "perturbation", "comparison"}
        xs2 = [x for x in xs if x in allowed]
        return xs2

    @field_validator("claim_id", mode="before")
    @classmethod
    def _claim_id_strip(cls, v: Any) -> str:
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

        # IMPORTANT: return self (not a new object) to avoid pydantic warning
        self.claim_id = cid
        return self


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
    Single audited container (stable contract).
    """

    model_config = ConfigDict(extra="forbid")

    claim: Claim
    decision: Decision
