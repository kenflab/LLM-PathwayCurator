# LLM-PathwayCurator/src/llm_pathway_curator/claim_schema.py

"""
Typed, auditable claim schema for LLM-PathwayCurator.

This module defines strict Pydantic models for:
- Evidence references (term IDs, optional gene IDs, module ID)
- Typed claims (entity, direction, context keys)
- Audit decisions (PASS/ABSTAIN/FAIL + reason codes)

Design:
- Claim and evidence identifiers are tool-owned and deterministic.
- Free-text evidence is disallowed; evidence must be referenced by IDs.
- Optional context review fields are supported for audit gating.

Notes
-----
- Status vocabulary is intentionally strict to keep denominators auditable.
- Gene ID casing is preserved for display; hashing follows tool-wide spec.
"""

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
    """
    De-duplicate tokens while preserving first occurrence order.

    This helper trims whitespace and drops empty/NA-like tokens before
    applying a stable de-duplication.

    Parameters
    ----------
    xs : list of str
        Input tokens.

    Returns
    -------
    list of str
        Cleaned tokens with first-seen order preserved.

    Notes
    -----
    NA detection delegates to `_shared.is_na_token()`.
    """
    cleaned: list[str] = []
    for x in xs or []:
        s = str(x).strip()
        if not s or _shared.is_na_token(s):
            continue
        cleaned.append(s)
    return _shared.dedup_preserve_order(cleaned)


def _split_listlike(v: Any) -> list[str]:
    """
    Parse a list-like field into a list of trimmed strings.

    Accepts:
    - list/tuple/set of scalars
    - a delimiter-separated string (comma or semicolon)

    Parameters
    ----------
    v : Any
        Input value (scalar or list-like).

    Returns
    -------
    list of str
        Trimmed tokens. Empty/NA-like input returns an empty list.

    Notes
    -----
    This is a tolerant splitter for schema inputs. It does not validate
    token formats (e.g., term UID structure).
    """
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
    """
    Normalize direction vocabulary to a canonical token.

    Parameters
    ----------
    v : Any
        Input direction value.

    Returns
    -------
    str
        One of {"up", "down", "na"}.

    Notes
    -----
    This function is intentionally lightweight and schema-local. If you
    want tool-wide vocabulary alignment, consider delegating to
    `_shared.normalize_direction()`.
    """
    s = str(v).strip().lower()
    if s in {"up", "upregulated", "activated", "pos", "positive", "+", "1"}:
        return "up"
    if s in {"down", "downregulated", "suppressed", "neg", "negative", "-", "-1"}:
        return "down"
    return "na"


def _looks_like_12hex(s: str) -> bool:
    """
    Check whether a string is a 12-hex lowercase digest.

    Parameters
    ----------
    s : str
        Input string.

    Returns
    -------
    bool
        True if the input matches the 12-hex pattern.
    """
    return _shared.looks_like_12hex(s)


def _sha256_12hex(payload: str) -> str:
    """
    Compute a deterministic short SHA-256 digest.

    Parameters
    ----------
    payload : str
        Input payload string.

    Returns
    -------
    str
        First 12 hex characters of SHA-256 (lowercase).

    Notes
    -----
    Delegates to `_shared.sha256_12hex()` to keep tool-wide stability.
    """
    return _shared.sha256_12hex(payload)


def _canonical_sorted_unique(xs: list[str]) -> list[str]:
    """
    Canonicalize tokens into sorted unique strings.

    Parameters
    ----------
    xs : list of str
        Input tokens.

    Returns
    -------
    list of str
        Sorted unique tokens after trimming and NA filtering.
    """
    return _shared.canonical_sorted_unique([str(x) for x in xs])


def _stable_claim_id(
    *,
    term_ids: list[str],
    direction: str,
    gene_set_hash: str,
    context_keys: list[str],
) -> str:
    """
    Build a deterministic, tool-owned claim identifier.

    The identifier is stable across:
    - term_id ordering differences
    - context key ordering differences
    - upstream whitespace/case jitter (expected to be normalized)

    Parameters
    ----------
    term_ids : list of str
        Term UID strings that define evidence identity.
    direction : str
        Canonical direction token (typically "up", "down", or "na").
    gene_set_hash : str
        12-hex fingerprint of the evidence gene set.
    context_keys : list of str
        Context key names the claim is conditioned on.

    Returns
    -------
    str
        Stable claim ID with prefix "c_".

    Notes
    -----
    Context *values* are not included by design. Only the presence of
    context keys affects claim identity.
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
    """
    Derive a stable gene-set hash from gene IDs.

    Parameters
    ----------
    gene_ids : list of str
        Gene identifiers for evidence.

    Returns
    -------
    str
        12-hex fingerprint of the gene set.

    Notes
    -----
    Delegates to `_shared.hash_gene_set_12hex()` (trim-only, no uppercasing).
    """
    return _shared.hash_gene_set_12hex(list(gene_ids or []))


def _stable_gene_set_hash_fallback(term_ids: list[str]) -> str:
    """
    Fallback hash when gene IDs are unavailable.

    Parameters
    ----------
    term_ids : list of str
        Term UID strings.

    Returns
    -------
    str
        12-hex fingerprint derived from canonicalized term IDs.

    Notes
    -----
    This is term-driven and may be less specific than a gene-driven hash.
    Use only when gene IDs cannot be provided.
    """
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
    Evidence reference container (strict, tool-friendly).

    Attributes
    ----------
    term_ids : list of str
        Required. One or more term UID strings that define evidence.
    gene_set_hash : str
        Optional input. If missing/invalid, it is deterministically filled:
        - from `gene_ids` when available, else
        - from `term_ids` as a fallback.
    gene_ids : list of str
        Optional. Evidence genes for display and hashing (tool spec).
    module_id : str
        Optional. Module identifier for module-level evidence.

    Notes
    -----
    - `gene_set_hash` must be a 12-hex digest (sha256[:12]).
    - Extra fields are allowed to support non-breaking provenance flags
      (e.g., `gene_set_hash_source`).
    - Term IDs are not uppercased.
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
        # Hashing is tool-wide spec: trim-only canonicalization (NO forced uppercasing).
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

        # Track how gene_set_hash was determined (auditable provenance).
        src = "input"

        h = str(self.gene_set_hash or "").strip().lower()
        if not _looks_like_12hex(h):
            if self.gene_ids:
                h = _stable_gene_set_hash_from_gene_ids(self.gene_ids)
                src = "from_gene_ids"
            else:
                h = _stable_gene_set_hash_fallback(self.term_ids)
                src = "from_term_ids"

        if not _looks_like_12hex(h):
            raise ValueError("EvidenceRef.gene_set_hash must be 12-hex (sha256[:12])")

        self.gene_set_hash = h

        # extra="allow" so this is safe and non-breaking.
        try:
            self.gene_set_hash_source = src
        except Exception:
            pass

        # Optional: flag raw term_ids (non-term_uid) for debugging, without rejecting.
        try:
            has_raw = any((":" not in str(t)) for t in (self.term_ids or []))
            self.term_ids_has_raw = bool(has_raw)
        except Exception:
            pass

        return self


class Claim(BaseModel):
    """
    Typed claim with auditable evidence linkage.

    Attributes
    ----------
    claim_id : str
        Tool-owned stable identifier. If empty, it is filled deterministically.
    entity : str
        Stable entity identifier (prefer IDs over free text).
    direction : {"up", "down", "na"}
        Canonical direction token.
    context_keys : list of {"condition", "tissue", "perturbation", "comparison"}
        Keys the claim is conditioned on. Values live in SampleCard.
    evidence_ref : EvidenceRef
        Evidence reference (IDs only; no free-text evidence).

    Optional context review fields
    ------------------------------
    context_evaluated : bool
        Whether context relevance review was executed.
    context_method : {"llm", "proxy", "none"}
        Method used for context review.
    context_status : {"PASS", "WARN", "FAIL"} or None
        Result of context review.
    context_reason : str or None
        Short reason (length-limited).
    context_notes : str or None
        Additional notes (length-limited).

    Notes
    -----
    Invariants enforced:
    - If `context_evaluated` is False:
      method="none" and status/reason/notes are cleared.
    - If `context_evaluated` is True:
      method must be "llm" or "proxy" and status must be provided.
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
    """
    Mechanical audit decision for a claim.

    Attributes
    ----------
    status : {"PASS", "ABSTAIN", "FAIL"}
        Final decision label.
    reason : str
        Reason code. Must be "ok" or one of `ALL_REASONS`.
    details : dict
        Optional structured metadata for debugging or reporting.

    Raises
    ------
    ValueError
        If `reason` is not in the allowed vocabulary.
    """

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
    Stable audited container.

    Attributes
    ----------
    claim : Claim
        Typed claim object.
    decision : Decision
        Mechanical decision and reason codes.

    Notes
    -----
    This object is intended as the unit of record for JSONL reports.
    """

    model_config = ConfigDict(extra="forbid")

    claim: Claim
    decision: Decision
