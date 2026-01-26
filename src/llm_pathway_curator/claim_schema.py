# LLM-PathwayCurator/src/llm_pathway_curator/claim_schema.py
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from pydantic.aliases import AliasChoices

from . import _shared
from .audit_reasons import ALL_REASONS

Direction = Literal["up", "down", "na"]
Status = Literal["PASS", "ABSTAIN", "FAIL"]

# Tool-facing context keys:
# - Use neutral "condition" (not disease/cancer) for generality.
# - Accept legacy synonyms at input time and normalize to "condition".
ContextKey = Literal["condition", "tissue", "perturbation", "comparison"]

# Optional per-claim context relevance review (used by audit gate).
ContextStatus = Literal["PASS", "WARN", "FAIL"]
ContextMethod = Literal["llm", "proxy", "none"]

_NA = _shared.NA_TOKENS

# Keep context explanation minimal to avoid “narrative drift”.
_MAX_CONTEXT_REASON_CHARS = 160
_MAX_CONTEXT_NOTES_CHARS = 400


def _dedup_preserve_order(xs: list[str]) -> list[str]:
    # delegate to shared spec utility (preserves order, drops empties)
    return _shared.dedup_preserve_order(
        [str(x).strip() for x in xs if str(x).strip() and str(x).strip().lower() not in _NA]
    )


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
    return _shared.looks_like_12hex(s)


def _sha256_12hex(payload: str) -> str:
    return _shared.sha256_12hex(payload)


def _canonical_sorted_unique(xs: list[str]) -> list[str]:
    return _shared.canonical_sorted_unique([str(x) for x in xs])


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
    # align with tool-wide spec (trim-only; NO forced uppercasing)
    return _shared.hash_gene_set_12hex(list(gene_ids or []))


def _stable_gene_set_hash_fallback(term_ids: list[str]) -> str:
    # Fallback when gene_ids are unavailable.
    # Still produces a deterministic fingerprint of "evidence identity",
    # but note it is term-driven, not gene-driven.
    canon = _canonical_sorted_unique([str(t) for t in term_ids])
    payload = ",".join(canon)
    return _sha256_12hex(payload)


# Legacy input aliases for context keys (normalized to tool-facing keys).
_CTX_KEY_ALIASES: dict[str, str] = {
    # legacy / figure-specific terms
    "disease": "condition",
    "cancer": "condition",
    "tumor": "condition",
    # canonical keys
    "condition": "condition",
    "tissue": "tissue",
    "perturbation": "perturbation",
    "comparison": "comparison",
}
_ALLOWED_CTX_KEYS: set[str] = {"condition", "tissue", "perturbation", "comparison"}


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

    model_config = ConfigDict(extra="allow")

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

    Optional (for context gating / shuffle stress):
      - context_evaluated: whether a context relevance review was executed
      - context_method: llm|proxy|none
      - context_status: PASS|WARN|FAIL (meaning: relevant / weak / inconsistent)
      - context_reason/context_notes: short explanation (kept minimal; auditable gate uses status)

    IMPORTANT invariant (enforced here):
      - If context_evaluated is False:
          * context_method must be "none"
          * context_status/reason/notes must be None
      - If context_evaluated is True:
          * context_method must be "llm" or "proxy"
          * context_status must be provided
    """

    model_config = ConfigDict(extra="forbid")

    claim_id: str = ""  # tool-owned; auto-fill if empty
    entity: str
    direction: Direction
    context_keys: list[ContextKey] = Field(default_factory=list)

    # Optional context review metadata (used by audit gate).
    context_evaluated: bool = False
    context_method: ContextMethod = "none"
    context_status: ContextStatus | None = None
    context_reason: str | None = None
    context_notes: str | None = None

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
    def _ctx_keys_normalize_and_dedup(cls, v: Any) -> list[str]:
        xs = _dedup_preserve_order(_split_listlike(v))
        out: list[str] = []
        for x in xs:
            raw = str(x).strip()
            if not raw:
                continue
            k = raw.lower()
            k2 = _CTX_KEY_ALIASES.get(k, "")
            if k2 and k2 in _ALLOWED_CTX_KEYS:
                out.append(k2)
        return _dedup_preserve_order(out)

    @field_validator("claim_id", mode="before")
    @classmethod
    def _claim_id_strip(cls, v: Any) -> str:
        if v is None:
            return ""
        s = str(v).strip()
        if s.lower() in _NA:
            return ""
        return s

    @field_validator("context_reason", "context_notes", mode="before")
    @classmethod
    def _strip_optional_text(cls, v: Any) -> str | None:
        if v is None:
            return None
        s = str(v).strip()
        if not s or s.lower() in _NA:
            return None
        return s

    @model_validator(mode="after")
    def _normalize_context_review_fields(self) -> Claim:
        # Enforce a simple, machine-reliable contract for audit gating.
        if not self.context_evaluated:
            self.context_method = "none"
            self.context_status = None
            self.context_reason = None
            self.context_notes = None
            return self

        # context_evaluated=True:
        if self.context_method == "none":
            raise ValueError("context_method must be 'llm' or 'proxy' when context_evaluated=true")
        if self.context_status is None:
            raise ValueError("context_status must be provided when context_evaluated=true")

        # Keep explanation fields minimal (avoid narrative drift).
        if self.context_reason is not None and len(self.context_reason) > _MAX_CONTEXT_REASON_CHARS:
            raise ValueError(f"context_reason too long (max {_MAX_CONTEXT_REASON_CHARS} chars)")
        if self.context_notes is not None and len(self.context_notes) > _MAX_CONTEXT_NOTES_CHARS:
            raise ValueError(f"context_notes too long (max {_MAX_CONTEXT_NOTES_CHARS} chars)")

        return self

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
