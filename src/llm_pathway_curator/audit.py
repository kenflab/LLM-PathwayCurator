# LLM-PathwayCurator/src/llm_pathway_curator/audit.py
from __future__ import annotations

import hashlib
import json
from typing import Any

import pandas as pd

from . import _shared
from .audit_reasons import (
    ABSTAIN_CONTEXT_MISSING,
    ABSTAIN_CONTEXT_NONSPECIFIC,
    ABSTAIN_HUB_BRIDGE,
    ABSTAIN_INCONCLUSIVE_STRESS,
    ABSTAIN_MISSING_EVIDENCE_GENES,
    ABSTAIN_MISSING_SURVIVAL,
    ABSTAIN_REASONS,
    ABSTAIN_UNDER_SUPPORTED,
    ABSTAIN_UNSTABLE,
    FAIL_CONTRADICTION,
    FAIL_EVIDENCE_DRIFT,
    FAIL_REASONS,
    FAIL_SCHEMA_VIOLATION,
)
from .claim_schema import Claim
from .sample_card import SampleCard

try:  # pragma: no cover
    from .audit_reasons import FAIL_CONTEXT  # type: ignore
except Exception:  # pragma: no cover
    # Safer than mapping to evidence drift: context failure is a distinct category.
    FAIL_CONTEXT = FAIL_SCHEMA_VIOLATION  # fallback (please add FAIL_CONTEXT in audit_reasons)

# Shared NA tokens / scalar NA check (single source of truth)
_NA_TOKENS_L = _shared.NA_TOKENS_L


def _is_na_scalar(x: Any) -> bool:
    """
    Wrapper for scalar NA detection.

    Parameters
    ----------
    x : Any
        Value to test.

    Returns
    -------
    bool
        True if `x` should be treated as NA-like scalar.
    """
    return _shared.is_na_scalar(x)


# -------------------------
# Gene canonicalization + hashing (compat-safe)
# -------------------------
def _clean_gene_id(g: Any) -> str:
    """
    Clean a gene identifier token conservatively.

    Parameters
    ----------
    g : Any
        Gene-like token (scalar).

    Returns
    -------
    str
        Cleaned token (trimmed and wrapper-stripped), preserving case.

    Notes
    -----
    This delegates to `_shared.clean_gene_token` and does not uppercase.
    """
    return _shared.clean_gene_token(g)


def _norm_gene_id(g: Any) -> str:
    """
    Normalize a gene identifier for audit comparisons.

    Parameters
    ----------
    g : Any
        Gene-like token.

    Returns
    -------
    str
        Canonical token used in audit comparisons.

    Notes
    -----
    Case is preserved (no uppercasing) to align with `_shared` policy.
    """
    return _clean_gene_id(g)


def _norm_gene_id_upper(g: Any) -> str:
    """
    Legacy uppercase normalization for gene identifiers.

    Parameters
    ----------
    g : Any
        Gene-like token.

    Returns
    -------
    str
        Uppercased canonical token.

    Notes
    -----
    This is a compatibility path only (not the default audit policy).
    """
    return _clean_gene_id(g).upper()


def _hash_gene_set_trim12(genes: list[str]) -> str:
    """
    Compute a 12-hex set-stable hash for genes (trim-only policy).

    Parameters
    ----------
    genes : list[str]
        Gene tokens (may be messy / duplicated).

    Returns
    -------
    str
        12-hex hash string.

    Notes
    -----
    - Tokens are normalized via `_norm_gene_id` (trim-only).
    - Hashing semantics are delegated to `_shared.hash_set_12hex`.
    """
    toks = [_norm_gene_id(g) for g in (genes or []) if str(g).strip()]
    return _shared.hash_set_12hex([t for t in toks if t])


def _hash_gene_set_upper12(genes: list[str]) -> str:
    """
    Compute a 12-hex set-stable hash for genes (legacy uppercasing policy).

    Parameters
    ----------
    genes : list[str]
        Gene tokens.

    Returns
    -------
    str
        12-hex hash string.

    Notes
    -----
    This is intended for backward-compat matching of historical outputs.
    """
    toks = [_norm_gene_id_upper(g) for g in (genes or []) if str(g).strip()]
    return _shared.hash_set_12hex([t for t in toks if t])


def _jaccard(a: set[str], b: set[str]) -> float:
    """
    Compute Jaccard similarity between two sets.

    Parameters
    ----------
    a : set[str]
        Left set.
    b : set[str]
        Right set.

    Returns
    -------
    float
        Jaccard index in [0, 1].

    Notes
    -----
    Empty-vs-empty returns 1.0 by convention.
    """
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return float(inter) / float(union) if union else 0.0


def _get_stress_jaccard_pass(card: SampleCard, default: float = 0.8) -> float:
    """
    Get the Jaccard threshold for passing stress robustness.

    Parameters
    ----------
    card : SampleCard
        Sample card holding knobs in `.extra`.
    default : float, optional
        Default threshold, by default 0.8.

    Returns
    -------
    float
        Clamped threshold in [0, 1].
    """
    ex = _get_extra(card)
    v = ex.get("stress_jaccard_pass", None)
    if v is None:
        return float(default)
    try:
        x = float(v)
    except Exception:
        return float(default)
    return min(1.0, max(0.0, x))


def _get_stress_jaccard_soft(card: SampleCard, default: float = 0.5) -> float:
    """
    Get the Jaccard threshold for the "soft" stress bucket.

    Parameters
    ----------
    card : SampleCard
        Sample card holding knobs in `.extra`.
    default : float, optional
        Default threshold, by default 0.5.

    Returns
    -------
    float
        Clamped threshold in [0, 1].

    Notes
    -----
    If J < soft: treated as strongly inconclusive.
    soft <= J < pass: treated as inconclusive.
    """
    ex = _get_extra(card)
    v = ex.get("stress_jaccard_soft", None)
    if v is None:
        return float(default)
    try:
        x = float(v)
    except Exception:
        return float(default)
    return min(1.0, max(0.0, x))


# -------- SampleCard knob access (single boundary, with robust fallback) --------
def _get_extra(card: SampleCard) -> dict[str, Any]:
    """
    Safely access `SampleCard.extra` as a dictionary.

    Parameters
    ----------
    card : SampleCard
        Sample card object.

    Returns
    -------
    dict[str, Any]
        Extra configuration dictionary (empty on failure).
    """
    try:
        ex = getattr(card, "extra", {}) or {}
        return ex if isinstance(ex, dict) else {}
    except Exception:
        return {}


def _as_bool(x: Any, default: bool) -> bool:
    """
    Parse a value into boolean with robust string handling.

    Parameters
    ----------
    x : Any
        Input value (bool-like).
    default : bool
        Default value when parsing fails or value is unknown.

    Returns
    -------
    bool
        Parsed boolean.

    Notes
    -----
    Avoids pitfalls like `bool("False") == True`.
    """
    if x is None:
        return bool(default)
    if isinstance(x, bool):
        return x
    s = str(x).strip().lower()
    if s in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if s in {"0", "false", "f", "no", "n", "off"}:
        return False
    return bool(default)


def _get_tau_default(card: SampleCard) -> float:
    """
    Get default audit tau from the sample card.

    Parameters
    ----------
    card : SampleCard
        Sample card.

    Returns
    -------
    float
        Tau value used as default (falls back to 0.8).
    """
    try:
        return float(card.audit_tau())
    except Exception:
        return 0.8


def _get_min_overlap_default(card: SampleCard) -> int:
    """
    Get default minimum gene overlap from the sample card.

    Parameters
    ----------
    card : SampleCard
        Sample card.

    Returns
    -------
    int
        Minimum overlap (falls back to 1).
    """
    try:
        return int(card.audit_min_gene_overlap())
    except Exception:
        return 1


def _get_min_union_genes(card: SampleCard, default: int = 3) -> int:
    """
    Get minimum union size required for evidence support.

    Parameters
    ----------
    card : SampleCard
        Sample card.
    default : int, optional
        Default minimum union size, by default 3.

    Returns
    -------
    int
        Minimum union size used in audit.
    """
    if hasattr(card, "min_union_genes") and callable(card.min_union_genes):
        try:
            return int(card.min_union_genes(default=default))  # type: ignore[attr-defined]
        except Exception:
            pass
    ex = _get_extra(card)
    try:
        v = ex.get("min_union_genes", None)
        return int(default) if v is None else int(v)
    except Exception:
        return int(default)


def _get_hub_term_degree(card: SampleCard, default: int = 200) -> int:
    """
    Get hub gene degree threshold for hub-bridge gating.

    Parameters
    ----------
    card : SampleCard
        Sample card.
    default : int, optional
        Default threshold, by default 200.

    Returns
    -------
    int
        Degree threshold; genes with degree > threshold are treated as hubs.
    """
    if hasattr(card, "hub_term_degree") and callable(card.hub_term_degree):
        try:
            return int(card.hub_term_degree(default=default))  # type: ignore[attr-defined]
        except Exception:
            pass
    ex = _get_extra(card)
    try:
        v = ex.get("hub_term_degree", None)
        return int(default) if v is None else int(v)
    except Exception:
        return int(default)


def _get_hub_frac_thr(card: SampleCard, default: float = 0.5) -> float:
    """
    Get hub fraction threshold for ABSTAIN_HUB_BRIDGE.

    Parameters
    ----------
    card : SampleCard
        Sample card.
    default : float, optional
        Default fraction threshold, by default 0.5.

    Returns
    -------
    float
        Fraction in [0, 1].
    """
    if hasattr(card, "hub_frac_thr") and callable(card.hub_frac_thr):
        try:
            return float(card.hub_frac_thr(default=default))  # type: ignore[attr-defined]
        except Exception:
            pass
    ex = _get_extra(card)
    try:
        v = ex.get("hub_frac_thr", None)
        return float(default) if v is None else float(v)
    except Exception:
        return float(default)


def _get_pass_notes(card: SampleCard, default: bool = True) -> bool:
    """
    Determine whether PASS notes should be populated.

    Parameters
    ----------
    card : SampleCard
        Sample card.
    default : bool, optional
        Default behavior, by default True.

    Returns
    -------
    bool
        True if PASS rows may receive a compact "ok" note.
    """
    if hasattr(card, "pass_notes") and callable(card.pass_notes):
        try:
            return bool(card.pass_notes(default=default))  # type: ignore[attr-defined]
        except Exception:
            pass
    ex = _get_extra(card)
    v = ex.get("pass_notes", None)
    if v is None:
        return bool(default)
    return _as_bool(v, default=bool(default))


def _get_context_proxy_swap_penalty(card: SampleCard, default: float = 0.6) -> float:
    """
    Get additive swap penalty for proxy context WARN probability.

    Parameters
    ----------
    card : SampleCard
        Sample card.
    default : float, optional
        Default penalty, by default 0.6.

    Returns
    -------
    float
        Penalty in [0, 1].

    Notes
    -----
    Applied only when context swap is active and proxy scoring is used.
    """
    ex = _get_extra(card)
    v = ex.get("context_proxy_swap_penalty", None)
    if v is None:
        return float(default)
    try:
        p = float(v)
    except Exception:
        return float(default)
    return min(1.0, max(0.0, p))


def _get_context_proxy_warn_p(card: SampleCard, default: float = 0.05) -> float:
    """
    Get base WARN probability for proxy context scoring.

    Parameters
    ----------
    card : SampleCard
        Sample card.
    default : float, optional
        Default probability, by default 0.05.

    Returns
    -------
    float
        Probability in [0, 1].
    """
    ex = _get_extra(card)
    v = ex.get("context_proxy_warn_p", None)
    if v is None:
        return float(default)
    try:
        p = float(v)
    except Exception:
        return float(default)
    return min(1.0, max(0.0, p))


def _get_context_proxy_key_fields(
    card: SampleCard, default: str = "ctx0,context_keys,term_uid,cancer"
) -> list[str]:
    """
    Get fields used to build deterministic proxy-context key.

    Parameters
    ----------
    card : SampleCard
        Sample card.
    default : str, optional
        Comma-separated field names, by default
        "ctx0,context_keys,term_uid,cancer".

    Returns
    -------
    list[str]
        Normalized list of field names.

    Notes
    -----
    - This controls which row/claim attributes contribute to the proxy key.
    - Unknown names are ignored by the proxy builder.
    - Allowed:
      context_keys,
      term_uid,
      module_id,
      gene_set_hash,
      comparison,
      cancer,
      disease,
      tissue,
      perturbation,
      condition,
      context_swap_from,
      context_swap_to,
      context_swap_active
    """
    ex = _get_extra(card)
    s = ex.get("context_proxy_key_fields", default)
    if _is_na_scalar(s):
        s = default
    items = [t.strip().lower() for t in str(s).split(",") if t.strip()]

    # Default safety: keep proxy key informative, but DO NOT make it swap-variant.
    # Swap strictness must be implemented via p_warn_eff penalty (monotone), not by changing u.
    if "cancer" not in items:
        items.append("cancer")

    # Do not force swap fields into the key. User can opt-in explicitly if desired.
    return items or [t.strip().lower() for t in str(default).split(",") if t.strip()]


def _get_context_gate_mode(card: SampleCard, default: str = "note") -> str:
    """
    Get context gate mode (off|note|hard) from card or extras.

    Parameters
    ----------
    card : SampleCard
        Sample card.
    default : str, optional
        Default gate mode, by default "note".

    Returns
    -------
    str
        Canonical gate mode: "off", "note", or "hard".

    Notes
    -----
    Normalization uses `_shared.normalize_gate_mode`.
    """
    try:
        if hasattr(card, "context_gate_mode") and callable(card.context_gate_mode):
            v = card.context_gate_mode(default=default)  # type: ignore[attr-defined]
        else:
            ex = _get_extra(card)
            v = ex.get("context_gate_mode", default)
    except Exception:
        v = default

    return _shared.normalize_gate_mode(v, default=default)


def _get_stress_gate_mode(card: SampleCard, default: str = "off") -> str:
    """
    Get stress gate mode (off|note|hard) from card or extras.

    Parameters
    ----------
    card : SampleCard
        Sample card.
    default : str, optional
        Default gate mode, by default "off".

    Returns
    -------
    str
        Canonical gate mode: "off", "note", or "hard".

    Notes
    -----
    Normalization uses `_shared.normalize_gate_mode`.
    """
    try:
        if hasattr(card, "stress_gate_mode") and callable(card.stress_gate_mode):
            v = card.stress_gate_mode(default=default)  # type: ignore[attr-defined]
        else:
            ex = _get_extra(card)
            v = ex.get("stress_gate_mode", default)
    except Exception:
        v = default

    return _shared.normalize_gate_mode(v, default=default)


def _get_audit_mode(card: SampleCard, default: str = "decision") -> str:
    """
    Get audit mode.

    Parameters
    ----------
    card : SampleCard
        Sample card.
    default : str, optional
        Default mode, by default "decision".

    Returns
    -------
    str
        "decision" or "diagnostic".

    Notes
    -----
    Diagnostic mode prefers annotation over status changes.
    """
    ex = _get_extra(card)
    s = str(ex.get("audit_mode", default)).strip().lower()
    if s in {"diagnostic", "debug", "inspect"}:
        return "diagnostic"
    return "decision"


def _get_context_swap_strict_mode(card: SampleCard, default: str = "warn_to_abstain") -> str:
    """
    Get strictness mode under context swap.

    Parameters
    ----------
    card : SampleCard
        Sample card.
    default : str, optional
        Default strict mode, by default "warn_to_abstain".

    Returns
    -------
    str
        "off" or "warn_to_abstain".

    Notes
    -----
    Under "warn_to_abstain", swap-active rows promote WARN->ABSTAIN and
    missing evaluation -> ABSTAIN.
    """
    ex = _get_extra(card)
    s = str(ex.get("context_swap_strict_mode", default)).strip().lower()
    if s in {"off", "none", "disable", "disabled"}:
        return "off"
    if s in {"warn_to_abstain", "hard", "strict"}:
        return "warn_to_abstain"
    d = str(default).strip().lower()
    return "warn_to_abstain" if d not in {"off"} else "off"


def _row_context_swap_active(row: pd.Series) -> bool:
    """
    Determine whether context swap is active for a given row.

    Parameters
    ----------
    row : pandas.Series
        Claim row.

    Returns
    -------
    bool
        True if swap is detected as active.

    Notes
    -----
    Priority:
    1) explicit boolean flag `context_swap_active`
    2) `variant` marker suggests swap (e.g., "context_swap")
    3) fallback on from/to mismatch (only if variant is unavailable)
    """
    # 1) Explicit flag wins
    v = row.get("context_swap_active", None)
    if not _is_na_scalar(v):
        # Robust: avoid bool("False")==True
        return _as_bool(v, default=False)

    # 2) Variant marker (recommended)
    variant = row.get("variant", None)
    if not _is_na_scalar(variant):
        s = str(variant).strip().lower()
        if s in {"context_swap", "swap", "contextswap"}:
            return True
        # if variant exists and is not swap, do NOT infer swap from from/to
        return False

    # 3) Fallback (only if we have no variant info at all)
    frm = row.get("context_swap_from", None)
    to = row.get("context_swap_to", None)
    if (not _is_na_scalar(frm)) and (not _is_na_scalar(to)):
        a = str(frm).strip()
        b = str(to).strip()
        if a and b and (a != b):
            return True
    return False


def _row_original_context_id(row: pd.Series) -> str:
    """
    Get a stable original-context identifier for proxy seeding.

    Parameters
    ----------
    row : pandas.Series
        Claim row.

    Returns
    -------
    str
        Original context id (may be empty string if unavailable).

    Notes
    -----
    Intended to be invariant under context swap so proxy `u` does not change.

    Preferred columns (if present):
      - context_ctx_id_original
      - ctx_id_original
      - context_id_original
      - context_original_id
    Fallback:
      - context_swap_from  (original context label)
      - context_keys       (less ideal but better than empty)
    """
    for k in [
        "context_ctx_id_original",
        "ctx_id_original",
        "context_id_original",
        "context_original_id",
    ]:
        v = row.get(k, None)
        if not _is_na_scalar(v):
            s = str(v).strip()
            if s:
                return s

    v = row.get("context_swap_from", None)
    if not _is_na_scalar(v):
        s = str(v).strip()
        if s:
            return s

    v = row.get("context_keys", None)
    if not _is_na_scalar(v):
        s = str(v).strip()
        if s:
            return s

    return ""


def _get_stability_gate_mode(card: SampleCard, default: str = "hard") -> str:
    """
    Get stability gate mode (off|note|hard).

    Parameters
    ----------
    card : SampleCard
        Sample card.
    default : str, optional
        Default mode, by default "hard".

    Returns
    -------
    str
        Canonical mode: "off", "note", or "hard".
    """
    try:
        if hasattr(card, "stability_gate_mode") and callable(card.stability_gate_mode):
            v = card.stability_gate_mode(default=default)  # type: ignore[attr-defined]
        else:
            ex = _get_extra(card)
            v = ex.get("stability_gate_mode", default)
    except Exception:
        v = default

    return _shared.normalize_gate_mode(v, default=default)


def _get_strict_evidence_check(card: SampleCard, default: bool = False) -> bool:
    """
    Get strict evidence linkage policy.

    Parameters
    ----------
    card : SampleCard
        Sample card.
    default : bool, optional
        Default behavior, by default False.

    Returns
    -------
    bool
        If True, missing evidence genes/hash becomes FAIL (schema violation).
        If False, missing evidence becomes ABSTAIN (decision-grade).
    """
    ex = _get_extra(card)
    return _as_bool(ex.get("strict_evidence_check", None), default=default)


def _get_evidence_dropout_p(card: SampleCard, default: float = 0.0) -> float:
    """
    Get internal evidence dropout probability.

    Parameters
    ----------
    card : SampleCard
        Sample card.
    default : float, optional
        Default probability, by default 0.0.

    Returns
    -------
    float
        Probability in [0, 1].
    """
    ex = _get_extra(card)
    v = ex.get("evidence_dropout_p", None)
    if v is None:
        return float(default)
    try:
        p = float(v)
    except Exception:
        return float(default)
    return min(1.0, max(0.0, p))


def _get_contradictory_p(card: SampleCard, default: float = 0.0) -> float:
    """
    Get internal contradiction-probe probability.

    Parameters
    ----------
    card : SampleCard
        Sample card.
    default : float, optional
        Default probability, by default 0.0.

    Returns
    -------
    float
        Probability in [0, 1].

    Notes
    -----
    This is a robustness probe tag, not a logical contradiction proof.
    """
    ex = _get_extra(card)
    v = ex.get("contradictory_p", None)
    if v is None:
        return float(default)
    try:
        p = float(v)
    except Exception:
        return float(default)
    return min(1.0, max(0.0, p))


def _append_note(old: str, msg: str) -> str:
    """
    Append a message to an audit note string.

    Parameters
    ----------
    old : str
        Existing note (may be NA-like).
    msg : str
        Message to append.

    Returns
    -------
    str
        Combined note string joined by " | ".
    """
    old = "" if _is_na_scalar(old) else str(old)
    return msg if not old else f"{old} | {msg}"


def _extract_claim_obj(cj: str) -> Claim | None:
    """
    Parse claim JSON into a `Claim` model.

    Parameters
    ----------
    cj : str
        Claim JSON string.

    Returns
    -------
    Claim or None
        Parsed `Claim` object, or None on validation failure.
    """
    try:
        return Claim.model_validate_json(cj)
    except Exception:
        return None


def _extract_evidence_from_claim_json(cj: str) -> tuple[list[str], list[str], Any, Any]:
    """
    Extract evidence reference fields from claim JSON.

    Parameters
    ----------
    cj : str
        Claim JSON string.

    Returns
    -------
    term_ids : list[str]
        Term identifiers (term_id or term_uid), as a list.
    gene_ids : list[str]
        Gene identifiers referenced by the claim, as a list.
    gene_set_hash : Any
        Gene set hash stored in the claim (may be None).
    module_id : Any
        Module identifier stored in the claim (may be None).

    Notes
    -----
    Evidence source is the Claim schema fields:
    `evidence_ref` or `evidence_refs` and their nested keys.
    """
    try:
        obj = json.loads(cj)
    except Exception:
        return ([], [], None, None)

    ev = None
    if isinstance(obj, dict):
        ev = obj.get("evidence_ref") or obj.get("evidence_refs")
        if ev is None and isinstance(obj.get("claim"), dict):
            ev = obj["claim"].get("evidence_ref") or obj["claim"].get("evidence_refs")

    if not isinstance(ev, dict):
        return ([], [], None, None)

    term_ids = _shared.parse_id_list(ev.get("term_ids") or ev.get("term_id") or "")
    gene_ids = _shared.parse_id_list(ev.get("gene_ids") or ev.get("gene_id") or "")
    gsh = ev.get("gene_set_hash")
    mid = ev.get("module_id")
    if isinstance(mid, (list, tuple, set)):
        mid = next(iter(mid), None)
    return (term_ids, gene_ids, gsh, mid)


def _extract_direction_from_claim_json(cj: str) -> str:
    """
    Extract normalized direction ("up"|"down"|"na") from claim JSON.

    Parameters
    ----------
    cj : str
        Claim JSON string.

    Returns
    -------
    str
        "up", "down", or "na".
    """
    try:
        obj = json.loads(cj)
    except Exception:
        return "na"

    d = None
    if isinstance(obj, dict):
        d = obj.get("direction")
        if d is None and isinstance(obj.get("claim"), dict):
            d = obj["claim"].get("direction")

    s = "" if d is None else str(d).strip().lower()
    if s in {"up", "down"}:
        return s
    return "na"


def _extract_context_review_from_claim_json(cj: str) -> tuple[bool, str, str]:
    """
    Extract context review fields from the claim schema.

    Parameters
    ----------
    cj : str
        Claim JSON string.

    Returns
    -------
    evaluated : bool
        Whether context was evaluated.
    status : str
        Context status string ("PASS", "WARN", "FAIL", or "").
    note : str
        Compact note suitable for `audit_notes`.

    Notes
    -----
    This is the primary source of truth when present.
    """
    c = _extract_claim_obj(cj)
    if c is None:
        return (False, "", "claim_json parse failed")

    evaluated = bool(getattr(c, "context_evaluated", False))
    method = str(getattr(c, "context_method", "none") or "none").strip().lower()
    status = str(getattr(c, "context_status", "") or "").strip().upper()
    reason = str(getattr(c, "context_reason", "") or "").strip()
    note = f"context_evaluated={evaluated} method={method}"
    if status:
        note += f" status={status}"
    if reason:
        note += f" reason={reason}"
    return (evaluated, status, note)


def _context_eval_from_row(row: pd.Series) -> tuple[bool, str, str]:
    """
    Extract context evaluation information from row-level columns (compat).

    Parameters
    ----------
    row : pandas.Series
        Claim row.

    Returns
    -------
    evaluated : bool
        Whether context evaluation is considered present.
    status : str
        "PASS", "WARN", "FAIL", or "".
    note : str
        Compact note explaining the source/decision.

    Notes
    -----
    Priority:
    1) `context_review_*` (LLM probe output)
    2) legacy `context_*`
    3) presence of `context_score` (proxy indicator)
    """
    if "context_review_evaluated" in row.index:
        v = row.get("context_review_evaluated")
        if not _is_na_scalar(v):
            try:
                ev = _as_bool(v, default=False)
            except Exception:
                ev = False

            st = ""
            if "context_review_status" in row.index:
                st = _shared.normalize_status_str(row.get("context_review_status"))
                if st == "ABSTAIN":
                    st = "WARN"
                if st and st not in {"PASS", "WARN", "FAIL"}:
                    st = ""

            rsn = ""
            if "context_review_reason" in row.index:
                rsn = (
                    ""
                    if _is_na_scalar(row.get("context_review_reason"))
                    else str(row.get("context_review_reason")).strip()
                )

            if (not ev) and st in {"PASS", "WARN", "FAIL"}:
                ev = True

            note = f"context_review_evaluated={ev}"
            if st:
                note += f" status={st}"
            if rsn:
                note += f" reason={rsn}"
            return (ev, st, note)

    if "context_evaluated" in row.index:
        v = row.get("context_evaluated")
        if _is_na_scalar(v):
            return (False, "", "context_evaluated=NA")
        try:
            ev = _as_bool(v, default=False)
        except Exception:
            return (False, "", "context_evaluated invalid")

        st = ""
        if "context_status" in row.index:
            st = _shared.normalize_status_str(row.get("context_status"))
            if st == "ABSTAIN":
                st = "WARN"
            if st and st not in {"PASS", "WARN", "FAIL"}:
                st = ""

        if (not ev) and st in {"PASS", "WARN", "FAIL"}:
            ev = True

        return (ev, st, f"context_evaluated={ev}" + (f" status={st}" if st else ""))

    if "context_status" in row.index:
        st = _shared.normalize_status_str(row.get("context_status"))
        if st in {"PASS", "WARN", "FAIL", "ABSTAIN"}:
            if st == "ABSTAIN":
                st = "WARN"
            return (True, st, f"context_status={st}")
        if not st:
            return (False, "", "context_status missing")
        return (False, "", f"context_status unknown={st}")

    if "context_score" in row.index:
        cs_raw = row.get("context_score")
        try:
            cs = None if _is_na_scalar(cs_raw) else float(cs_raw)
        except Exception:
            cs = None
        if cs is None:
            return (False, "", "context_score missing")
        return (True, "WARN", "context_score present (proxy)")

    return (False, "", "no context fields")


def _inject_stress_from_distilled(out: pd.DataFrame, distilled: pd.DataFrame) -> pd.DataFrame:
    """
    Backfill stress and contradiction columns on claims using distilled data.

    Parameters
    ----------
    out : pandas.DataFrame
        Claims table to be audited.
    distilled : pandas.DataFrame
        Distilled evidence table providing stress/contradiction annotations.

    Returns
    -------
    pandas.DataFrame
        Copy of `out` with injected columns when needed.

    Notes
    -----
    - If claims already include stress/contradiction columns, they are respected.
    - Mapping is primarily by `term_uid`, with a best-effort fallback from raw
      `term_id` to unique `term_uid`.
    - Stress and contradiction injections are treated as robustness probes and
      injected as ABSTAIN (inconclusive), not FAIL.
    """
    out2 = out.copy()

    claim_has_stress = any(
        c in out2.columns for c in ["stress_status", "stress_reason", "stress_notes", "stress_ok"]
    )
    claim_has_contra = any(
        c in out2.columns
        for c in [
            "contradiction_status",
            "contradiction_reason",
            "contradiction_notes",
            "contradiction_ok",
        ]
    )
    if claim_has_stress and claim_has_contra:
        return out2

    dist = distilled.copy()

    # Ensure dist.term_uid
    if "term_uid" not in dist.columns:
        if {"source", "term_id"}.issubset(set(dist.columns)):
            dist["term_uid"] = dist.apply(
                lambda r: _shared.make_term_uid(r.get("source"), r.get("term_id")),
                axis=1,
            )
        elif "term_id" in dist.columns:
            dist["term_uid"] = dist["term_id"].astype(str).str.strip()
        else:
            return out2

    dist["term_uid"] = dist["term_uid"].astype(str).str.strip()

    # Build raw term_id -> unique term_uid mapping (best-effort)
    known, raw_unique, _raw_amb = _build_term_uid_maps(dist[["term_uid"]].copy())

    # Build stress map
    stress_map_uid: dict[str, str] = {}
    stress_map_raw_unique: dict[str, str] = {}

    if (not claim_has_stress) and ("stress_tag" in dist.columns):
        # normalize tags per spec (tolerate legacy '+')
        tags_norm = []
        for x in dist["stress_tag"].tolist():
            if _is_na_scalar(x):
                tags_norm.append("")
            else:
                tags = _shared.split_tags(x)
                tags_norm.append(_shared.join_tags(tags) if tags else "")

        for tu, tag in zip(dist["term_uid"].tolist(), tags_norm, strict=False):
            if not tu or not tag:
                continue
            if tu not in stress_map_uid:
                stress_map_uid[tu] = tag

        # raw term_id -> tag only when raw term_id maps uniquely to a single term_uid
        for raw, tu in raw_unique.items():
            tag = stress_map_uid.get(tu, "")
            if tag and raw not in stress_map_raw_unique:
                stress_map_raw_unique[raw] = tag

    # Build contradiction map
    contra_uid: dict[str, bool] = {}
    contra_raw_unique: dict[str, bool] = {}

    if (not claim_has_contra) and ("contradiction_flip" in dist.columns):
        for tu, v in zip(
            dist["term_uid"].tolist(), dist["contradiction_flip"].tolist(), strict=False
        ):
            if not tu or tu in contra_uid:
                continue
            if _is_na_scalar(v):
                continue
            try:
                contra_uid[tu] = _as_bool(v, default=False)
            except Exception:
                continue

        for raw, tu in raw_unique.items():
            if raw in contra_raw_unique:
                continue
            if tu in contra_uid:
                contra_raw_unique[raw] = bool(contra_uid[tu])

    if (
        (not stress_map_uid)
        and (not stress_map_raw_unique)
        and (not contra_uid)
        and (not contra_raw_unique)
    ):
        return out2

    # Create columns only if absent (respect caller)
    if (not claim_has_stress) and (stress_map_uid or stress_map_raw_unique):
        out2["stress_status"] = ""
        out2["stress_reason"] = ""
        out2["stress_notes"] = ""
    if (not claim_has_contra) and (contra_uid or contra_raw_unique):
        out2["contradiction_status"] = ""
        out2["contradiction_reason"] = ""
        out2["contradiction_notes"] = ""

    for i, row in out2.iterrows():
        cj = row.get("claim_json", None)
        if cj is None or _is_na_scalar(cj) or (not str(cj).strip()):
            continue

        term_ids, _gene_ids, _gsh, _mid = _extract_evidence_from_claim_json(str(cj))
        if not term_ids:
            continue

        # Resolve each term_id to (1) exact term_uid, else (2) raw term_id if unique
        def _lookup_stress(t: str) -> str:
            tt = str(t).strip()
            if not tt:
                return ""
            if tt in stress_map_uid:
                return stress_map_uid.get(tt, "")
            raw = tt.split(":", 1)[-1].strip()
            return stress_map_raw_unique.get(raw, "")

        def _lookup_contra(t: str) -> bool:
            tt = str(t).strip()
            if not tt:
                return False
            if tt in contra_uid:
                return bool(contra_uid.get(tt, False))
            raw = tt.split(":", 1)[-1].strip()
            return bool(contra_raw_unique.get(raw, False))

        if (not claim_has_stress) and (stress_map_uid or stress_map_raw_unique):
            tags = [_lookup_stress(t) for t in term_ids]
            tags = [t for t in tags if t]
            if tags:
                # Robustness probe: treat as inconclusive stress (do not FAIL).
                out2.at[i, "stress_status"] = "ABSTAIN"
                out2.at[i, "stress_reason"] = "STRESS_TAG"
                # keep notes deterministic and compact
                uniq = _shared.dedup_preserve_order(tags)
                out2.at[i, "stress_notes"] = (
                    f"stress_tag={uniq[0]}" if len(uniq) == 1 else f"stress_tag={uniq}"
                )

        if (not claim_has_contra) and (contra_uid or contra_raw_unique):
            flips = [_lookup_contra(t) for t in term_ids]
            if any(bool(x) for x in flips):
                # IMPORTANT: contradiction_flip is a tag for direction-flip probe,
                # not a logical contradiction proof.
                # Mark as inconclusive (ABSTAIN),
                # and let audit layer map it to ABSTAIN_INCONCLUSIVE_STRESS.
                out2.at[i, "contradiction_status"] = "ABSTAIN"
                out2.at[i, "contradiction_reason"] = "CONTRADICTION_FLIP"
                out2.at[i, "contradiction_notes"] = "distilled.contradiction_flip=True (probe)"

    return out2


def _apply_external_contradiction(out: pd.DataFrame, i: int, row: pd.Series) -> None:
    """
    Apply external contradiction status to the audit result for a row.

    Parameters
    ----------
    out : pandas.DataFrame
        Audit output table (mutated in place).
    i : int
        Row index label used with `out.at`.
    row : pandas.Series
        Input claim row.

    Notes
    -----
    Policy:
    - Missing contradiction_status: do not abstain; only note.
    - PASS: ok
    - ABSTAIN/WARN: ABSTAIN with ABSTAIN_INCONCLUSIVE_STRESS
    - FAIL: FAIL with FAIL_CONTRADICTION
    - Unknown: FAIL with FAIL_SCHEMA_VIOLATION
    """
    if "contradiction_status" not in out.columns:
        return

    st = _shared.normalize_status_str(row.get("contradiction_status"))
    if not st:
        out.at[i, "audit_notes"] = _append_note(
            out.at[i, "audit_notes"], "contradiction_status missing"
        )
        return

    if st == "PASS":
        out.at[i, "contradiction_ok"] = True
        return

    if st in {"ABSTAIN", "WARN"}:
        out.at[i, "status"] = "ABSTAIN"
        out.at[i, "contradiction_ok"] = False
        # NOTE: we reuse ABSTAIN_INCONCLUSIVE_STRESS as a generic "inconclusive probe" code,
        # but we explicitly label it as contradiction in notes for interpretability.
        out.at[i, "abstain_reason"] = ABSTAIN_INCONCLUSIVE_STRESS
        out.at[i, "audit_notes"] = _append_note(
            out.at[i, "audit_notes"], f"contradiction_inconclusive={st}"
        )
        return

    if st == "FAIL":
        out.at[i, "status"] = "FAIL"
        out.at[i, "contradiction_ok"] = False
        out.at[i, "fail_reason"] = FAIL_CONTRADICTION
        rn = str(row.get("contradiction_reason", "")).strip()
        nt = str(row.get("contradiction_notes", "")).strip()
        note = "contradiction"
        if rn:
            note += f": {rn}"
        if nt:
            note += f" ({nt})"
        out.at[i, "audit_notes"] = _append_note(out.at[i, "audit_notes"], note)
        return

    out.at[i, "status"] = "FAIL"
    out.at[i, "contradiction_ok"] = False
    out.at[i, "fail_reason"] = FAIL_SCHEMA_VIOLATION
    out.at[i, "audit_notes"] = _append_note(
        out.at[i, "audit_notes"], f"contradiction_status unknown={st}"
    )


def _apply_external_stress(
    out: pd.DataFrame,
    i: int,
    row: pd.Series,
    *,
    gate_mode: str,
    input_stress_cols: list[str],
) -> None:
    """
    Apply stress results from input columns to the audit result for a row.

    Parameters
    ----------
    out : pandas.DataFrame
        Audit output table (mutated in place).
    i : int
        Row index label used with `out.at`.
    row : pandas.Series
        Input claim row.
    gate_mode : str
        Stress gate mode: "off", "note", or "hard".
    input_stress_cols : list[str]
        Names of stress-related columns available in the input.

    Notes
    -----
    Stress is treated as a robustness probe, not a validity proof.
    Therefore, stress failures should not become FAIL; at most ABSTAIN.
    """
    if not input_stress_cols:
        out.at[i, "stress_ok"] = pd.NA
        return

    gate_mode = (gate_mode or "").strip().lower()
    if gate_mode not in {"off", "note", "hard"}:
        gate_mode = "note"

    if ("stress_ok" in input_stress_cols) and ("stress_status" not in input_stress_cols):
        v = row.get("stress_ok")
        if _is_na_scalar(v):
            out.at[i, "stress_ok"] = pd.NA
            if gate_mode == "hard":
                out.at[i, "status"] = "ABSTAIN"
                out.at[i, "abstain_reason"] = ABSTAIN_INCONCLUSIVE_STRESS
                out.at[i, "audit_notes"] = _append_note(
                    out.at[i, "audit_notes"], "stress_ok missing (hard)"
                )
            else:
                out.at[i, "audit_notes"] = _append_note(
                    out.at[i, "audit_notes"], "stress_ok missing"
                )
            return

        ok = _as_bool(v, default=False)
        out.at[i, "stress_ok"] = ok

        if ok:
            out.at[i, "audit_notes"] = _append_note(out.at[i, "audit_notes"], "stress_ok=True")
            return

        if gate_mode == "hard":
            out.at[i, "status"] = "ABSTAIN"
            out.at[i, "abstain_reason"] = ABSTAIN_INCONCLUSIVE_STRESS
            out.at[i, "audit_notes"] = _append_note(
                out.at[i, "audit_notes"], "stress_ok=False -> ABSTAIN (hard)"
            )
        else:
            out.at[i, "audit_notes"] = _append_note(out.at[i, "audit_notes"], "stress_ok=False")
        return

    st = _shared.normalize_status_str(row.get("stress_status"))
    rs_raw = str(row.get("stress_reason", "")).strip()
    nt = str(row.get("stress_notes", "")).strip()

    if not st:
        out.at[i, "stress_ok"] = pd.NA
        if gate_mode == "hard":
            out.at[i, "status"] = "ABSTAIN"
            out.at[i, "abstain_reason"] = ABSTAIN_INCONCLUSIVE_STRESS
            out.at[i, "audit_notes"] = _append_note(
                out.at[i, "audit_notes"], "stress_status missing (hard)"
            )
        else:
            out.at[i, "audit_notes"] = _append_note(
                out.at[i, "audit_notes"], "stress_status missing"
            )
        return

    if st == "PASS":
        out.at[i, "stress_ok"] = True
        out.at[i, "audit_notes"] = _append_note(out.at[i, "audit_notes"], "stress=PASS")
        return

    if st in {"ABSTAIN", "WARN"}:
        out.at[i, "stress_ok"] = False
        rs = rs_raw if rs_raw in ABSTAIN_REASONS else ABSTAIN_INCONCLUSIVE_STRESS
        if gate_mode == "hard":
            out.at[i, "status"] = "ABSTAIN"
            out.at[i, "abstain_reason"] = rs
        note = f"stress={st}"
        if rs_raw:
            note += f": {rs_raw}"
        if nt:
            note += f" ({nt})"
        note += f" [{gate_mode}]"
        out.at[i, "audit_notes"] = _append_note(out.at[i, "audit_notes"], note)
        return

    if st == "FAIL":
        out.at[i, "stress_ok"] = False
        if gate_mode == "hard":
            out.at[i, "status"] = "ABSTAIN"
            out.at[i, "abstain_reason"] = ABSTAIN_INCONCLUSIVE_STRESS

        note = "stress=FAIL"
        if rs_raw:
            note += f": {rs_raw}"
        if nt:
            note += f" ({nt})"
        note += f" [{gate_mode}]"
        out.at[i, "audit_notes"] = _append_note(out.at[i, "audit_notes"], note)
        return

    out.at[i, "status"] = "FAIL"
    out.at[i, "stress_ok"] = False
    out.at[i, "fail_reason"] = FAIL_SCHEMA_VIOLATION
    out.at[i, "audit_notes"] = _append_note(out.at[i, "audit_notes"], f"stress_status unknown={st}")


def _enforce_reason_vocab(out: pd.DataFrame, i: int) -> None:
    """
    Enforce reason-code vocabulary for a single audited row.

    Parameters
    ----------
    out : pandas.DataFrame
        Audit output table (mutated in place).
    i : int
        Row index label used with `out.at`.

    Notes
    -----
    - Invalid fail_reason => replaced with FAIL_SCHEMA_VIOLATION.
    - Invalid abstain_reason => escalated to FAIL_SCHEMA_VIOLATION.
    - Ensures FAIL/ABSTAIN rows have non-empty `audit_notes`.
    """
    st = str(out.at[i, "status"]).strip().upper()

    if st == "FAIL":
        fr = (
            "" if _is_na_scalar(out.at[i, "fail_reason"]) else str(out.at[i, "fail_reason"]).strip()
        )
        if fr not in FAIL_REASONS:
            out.at[i, "fail_reason"] = FAIL_SCHEMA_VIOLATION
            out.at[i, "audit_notes"] = _append_note(
                out.at[i, "audit_notes"], f"invalid fail_reason={fr}"
            )

    if st == "ABSTAIN":
        ar = (
            ""
            if _is_na_scalar(out.at[i, "abstain_reason"])
            else str(out.at[i, "abstain_reason"]).strip()
        )
        if ar not in ABSTAIN_REASONS:
            out.at[i, "status"] = "FAIL"
            out.at[i, "fail_reason"] = FAIL_SCHEMA_VIOLATION
            out.at[i, "abstain_reason"] = ""
            out.at[i, "audit_notes"] = _append_note(
                out.at[i, "audit_notes"], f"invalid abstain_reason={ar}"
            )

    if st in {"FAIL", "ABSTAIN"}:
        note = (
            "" if _is_na_scalar(out.at[i, "audit_notes"]) else str(out.at[i, "audit_notes"]).strip()
        )
        if not note:
            out.at[i, "audit_notes"] = f"{st}: reason recorded"


def _apply_intra_run_contradiction(out: pd.DataFrame) -> None:
    """
    Detect and mark intra-run contradictions among PASS claims.

    Parameters
    ----------
    out : pandas.DataFrame
        Audit output table (mutated in place).

    Notes
    -----
    For PASS claims only:
    - If the same evidence_key appears with both direction "up" and "down",
      those claims are marked FAIL with FAIL_CONTRADICTION.
    """
    if "evidence_key" not in out.columns or "direction_norm" not in out.columns:
        return

    df = out.copy()
    df["status_u"] = df["status"].astype(str).str.strip().str.upper()
    df = df[df["status_u"] == "PASS"].copy()
    if df.empty:
        return

    grouped = df.groupby("evidence_key", dropna=False)
    for key, g in grouped:
        if _is_na_scalar(key) or str(key).strip() == "":
            continue
        dirs = set(g["direction_norm"].astype(str).tolist())
        if ("up" in dirs) and ("down" in dirs):
            idxs = g.index.tolist()
            note = f"intra-run contradiction: evidence_key={key} has both up and down"
            for i in idxs:
                out.at[i, "status"] = "FAIL"
                out.at[i, "fail_reason"] = FAIL_CONTRADICTION
                out.at[i, "contradiction_ok"] = False
                out.at[i, "audit_notes"] = _append_note(out.at[i, "audit_notes"], note)
                _enforce_reason_vocab(out, i)


def _deterministic_uniform_0_1(key: str, salt: str = "") -> float:
    """
    Generate a deterministic pseudo-uniform value in [0, 1).

    Parameters
    ----------
    key : str
        Deterministic key.
    salt : str, optional
        Extra salt for independent streams, by default "".

    Returns
    -------
    float
        Value in [0, 1).

    Notes
    -----
    Based on SHA-256 and a fixed modulus to ensure reproducibility.
    """
    payload = f"{salt}|{key}".encode()
    h = hashlib.sha256(payload).hexdigest()
    x = int(h[:12], 16)
    return (x % 1_000_000) / 1_000_000.0


def _proxy_context_status(
    *,
    card: SampleCard,
    row: pd.Series,
    term_ids: list[str],
    module_id: str,
    gene_set_hash: str,
) -> tuple[bool, str, str, float, float, str]:
    """
    Compute deterministic proxy context status for a row.

    Parameters
    ----------
    card : SampleCard
        Sample card holding proxy knobs.
    row : pandas.Series
        Claim row.
    term_ids : list[str]
        Term identifiers (resolved term_uids when possible).
    module_id : str
        Module identifier (may be empty).
    gene_set_hash : str
        Gene set hash (may be empty).

    Returns
    -------
    evaluated : bool
        Always True for proxy evaluation.
    status : str
        "PASS" or "WARN" from proxy scoring.
    note : str
        Compact note describing proxy parameters.
    u01 : float
        Deterministic uniform sample in [0, 1).
    p_warn_eff : float
        Effective WARN probability (includes swap penalty when active).
    key_base : str
        Base key used to seed `u01`.

    Notes
    -----
    Swap strictness is implemented by increasing `p_warn_eff` without
    changing the seeded `u01`.
    """
    p_warn = _get_context_proxy_warn_p(card)
    swap_penalty = _get_context_proxy_swap_penalty(card)

    def _get_row_str(k: str) -> str:
        v = row.get(k, "")
        if _is_na_scalar(v):
            return ""
        return str(v).strip()

    # --------
    # Build a BASE KEY that is invariant under swap.
    # User-controlled fields via SampleCard extra: context_proxy_key_fields
    # (but we still ensure swap dependence via context_swap_to + cancer by default).
    # --------
    fields = _get_context_proxy_key_fields(card)

    def _emit_field(name: str) -> str:
        n = (name or "").strip().lower()
        if not n:
            return ""

        if n in {"ctx0", "context_id_original", "context_original_id", "original_context_id"}:
            v = _row_original_context_id(row)
            return f"ctx0={v}" if v else ""

        if n == "term_uid":
            tu = ",".join(sorted({str(t).strip() for t in term_ids if str(t).strip()}))
            return f"term_uid={tu}" if tu else ""

        if n == "module_id":
            mid = str(module_id or "").strip()
            return f"module_id={mid}" if mid else ""

        if n == "gene_set_hash":
            gsh = str(gene_set_hash or "").strip()
            return f"gene_set_hash={gsh}" if gsh else ""

        if n == "context_keys":
            v = _get_row_str("context_keys")
            return f"context_keys={v}" if v else ""

        if n in {
            "comparison",
            "cancer",
            "disease",
            "tissue",
            "perturbation",
            "condition",
            "context_swap_from",
            "context_swap_to",
            "context_swap_active",
        }:
            v = _get_row_str(n)
            return f"{n}={v}" if v else ""

        # Unknown field name: ignore (contract-safe)
        return ""

    parts = [_emit_field(f) for f in fields]
    parts = [p for p in parts if p]

    key_base = "|".join(parts).strip()
    if not key_base:
        tu_fallback = ",".join(sorted({str(t).strip() for t in term_ids if str(t).strip()}))
        gsh_fallback = str(gene_set_hash or "").strip()
        key_base = gsh_fallback or tu_fallback or "proxy_context"

    u = _deterministic_uniform_0_1(key_base, salt="proxy_context_v3_base")

    try:
        swap_active = bool(_row_context_swap_active(row))
    except Exception:
        swap_active = False

    p_warn_eff = float(p_warn)
    if swap_active:
        p_warn_eff = min(1.0, float(p_warn) + float(swap_penalty))

    status = "WARN" if (u < float(p_warn_eff)) else "PASS"
    note = (
        f"proxy_context_v3: swap_active={int(swap_active)} "
        f"u={u:.3f} p_warn={float(p_warn):.3f} p_warn_eff={float(p_warn_eff):.3f} "
        f"key_fields={fields}"
    )
    return True, status, note, float(u), float(p_warn_eff), key_base


def _apply_internal_evidence_dropout(
    *,
    genes: list[str],
    evidence_key: str,
    p: float,
) -> tuple[list[str], str]:
    """
    Apply deterministic evidence dropout to a gene list (internal stress probe).

    Parameters
    ----------
    genes : list[str]
        Evidence gene list.
    evidence_key : str
        Evidence key used to seed deterministic sampling.
    p : float
        Dropout probability in [0, 1].

    Returns
    -------
    kept : list[str]
        Kept genes after dropout (at least one gene is kept if input non-empty).
    note : str
        Compact note including kept counts.
    """
    if p <= 0.0 or (not genes):
        return (genes, "dropout_p=0")

    kept: list[str] = []
    for g in genes:
        u = _deterministic_uniform_0_1(f"{evidence_key}|{g}", salt="dropout")
        if u >= p:
            kept.append(g)

    if not kept and genes:
        kept = [genes[0]]

    note = f"dropout_p={p:.3f} kept={len(kept)}/{len(genes)}"
    return (kept, note)


def _build_term_uid_maps(dist: pd.DataFrame) -> tuple[set[str], dict[str, str], set[str]]:
    """
    Build mappings to resolve raw term_ids into unique term_uids.

    Parameters
    ----------
    dist : pandas.DataFrame
        DataFrame containing a `term_uid` column.

    Returns
    -------
    known : set[str]
        Set of known term_uids.
    raw_unique : dict[str, str]
        Mapping raw term_id -> unique term_uid (only when unambiguous).
    raw_ambiguous : set[str]
        Raw term_ids that map to multiple term_uids.

    Notes
    -----
    Intended for best-effort linkage when claims provide raw term_ids.
    """
    term_uids = dist["term_uid"].astype(str).str.strip().tolist()
    known = {t for t in term_uids if t and t.lower() not in _NA_TOKENS_L}

    raw_to_uids: dict[str, set[str]] = {}
    for tu in known:
        raw = _term_uid_to_raw_term_id(tu)
        if not raw:
            continue
        raw_to_uids.setdefault(raw, set()).add(tu)

    raw_unique = {raw: next(iter(u)) for raw, u in raw_to_uids.items() if len(u) == 1}
    raw_amb = {raw for raw, u in raw_to_uids.items() if len(u) > 1}
    return known, raw_unique, raw_amb


def _term_uid_to_raw_term_id(tu: str) -> str:
    """
    Extract raw term_id from a term_uid string.

    Parameters
    ----------
    tu : str
        Term UID string, typically "<source>:<term_id>".

    Returns
    -------
    str
        Raw term_id portion (may be empty).

    Notes
    -----
    Splits from the right to tolerate sources that contain ':'.
    """
    s = "" if tu is None else str(tu).strip()
    if not s:
        return ""
    # Robust to sources that might contain ":" by splitting from the right.
    return s.rsplit(":", 1)[-1].strip()


def _resolve_term_ids_to_uids(
    term_ids: list[str],
    *,
    known: set[str],
    raw_unique: dict[str, str],
    raw_ambiguous: set[str],
) -> tuple[list[str], list[str], list[str]]:
    """
    Resolve term identifiers to term_uids with ambiguity tracking.

    Parameters
    ----------
    term_ids : list[str]
        Term identifiers from a claim (may include raw term_id or term_uid).
    known : set[str]
        Known term_uids.
    raw_unique : dict[str, str]
        Mapping raw term_id -> unique term_uid.
    raw_ambiguous : set[str]
        Raw term_ids that are ambiguous.

    Returns
    -------
    resolved : list[str]
        Resolved term_uids.
    unknown : list[str]
        Inputs that could not be resolved.
    ambiguous : list[str]
        Inputs that map to multiple term_uids.

    Notes
    -----
    Unknown and ambiguous terms are handled as audit failures upstream.
    """
    resolved: list[str] = []
    unknown: list[str] = []
    ambiguous: list[str] = []

    for t in term_ids:
        tt = str(t).strip()
        if not tt:
            continue
        if tt in known:
            resolved.append(tt)
            continue
        raw = tt.split(":", 1)[-1].strip()
        if raw in raw_unique:
            resolved.append(raw_unique[raw])
            continue
        if raw in raw_ambiguous:
            ambiguous.append(tt)
        else:
            unknown.append(tt)

    return resolved, unknown, ambiguous


def audit_claims(
    claims: pd.DataFrame,
    distilled: pd.DataFrame,
    card: SampleCard,
    *,
    tau: float | None = None,
) -> pd.DataFrame:
    """
    Mechanically audit claims against distilled evidence and sample context.

    Parameters
    ----------
    claims : pandas.DataFrame
        Claims table. Must include `claim_json` with Claim schema JSON.
    distilled : pandas.DataFrame
        Distilled evidence table. Must provide term linkage via `term_uid` or
        (`source`, `term_id`). Evidence genes are read from `evidence_genes` or
        `evidence_genes_str`. Stability uses `term_survival` when available.
    card : SampleCard
        Sample card providing audit knobs and gate modes.
    tau : float or None, optional
        Override stability tau. If None, uses `card.audit_tau()`.

    Returns
    -------
    pandas.DataFrame
        Audited claims with status, reasons, and audit notes.

    Raises
    ------
    ValueError
        If `distilled` cannot provide term linkage (missing required columns).

    Notes
    -----
    Status priority is: FAIL > ABSTAIN > PASS.

    Major checks:
    - Linkage: term_id -> term_uid resolution; reject unknown/ambiguous terms.
    - Evidence identity: gene_set_hash match against computed union evidence genes.
    - Stability: term-level survival aggregation (min across referenced terms).
    - Under-support: minimum union evidence genes.
    - Hub-bridge: abstain when evidence is dominated by hub genes.
    - Context gate: uses claim schema context review, with optional proxy fallback.
    - Stress probes: optional internal dropout and contradiction probes and/or
      external stress columns; treated as ABSTAIN (inconclusive), not FAIL.
    """
    out = claims.copy()

    # Backfill stress/contradiction from distilled (if claims don't already have them).
    out = _inject_stress_from_distilled(out, distilled)

    # IMPORTANT:
    # Use *out.columns* (post-injection) so injected columns are actually connected to gating.
    cols_now = set(out.columns)

    input_stress_cols = [
        c for c in ["stress_status", "stress_ok", "stress_reason", "stress_notes"] if c in cols_now
    ]
    input_contra_cols = [
        c
        for c in ["contradiction_status", "contradiction_reason", "contradiction_notes"]
        if c in cols_now
    ]

    out["status"] = "PASS"
    out["link_ok"] = True
    out["stability_ok"] = True
    out["contradiction_ok"] = pd.NA
    out["stress_ok"] = pd.NA
    out["stress_evaluated"] = False
    out["abstain_reason"] = ""
    out["fail_reason"] = ""
    out["audit_notes"] = ""

    if "tau_used" not in out.columns:
        out["tau_used"] = pd.NA
    if "term_survival_agg" not in out.columns:
        out["term_survival_agg"] = pd.NA
    if "stability_scope" not in out.columns:
        out["stability_scope"] = ""
    if "module_id_effective" not in out.columns:
        out["module_id_effective"] = ""
    if "gene_set_hash_effective" not in out.columns:
        out["gene_set_hash_effective"] = ""
    if "term_ids_set_hash" not in out.columns:
        out["term_ids_set_hash"] = ""
    if "evidence_key" not in out.columns:
        out["evidence_key"] = ""
    if "direction_norm" not in out.columns:
        out["direction_norm"] = "na"

    if "gene_set_hash_computed" not in out.columns:
        out["gene_set_hash_computed"] = ""
    if "gene_set_hash_match_mode" not in out.columns:
        out["gene_set_hash_match_mode"] = ""

    if "context_evaluated" not in out.columns:
        out["context_evaluated"] = False
    if "context_method" not in out.columns:
        out["context_method"] = "none"
    if "context_status" not in out.columns:
        out["context_status"] = ""
    if "context_reason" not in out.columns:
        out["context_reason"] = ""
    if "context_notes" not in out.columns:
        out["context_notes"] = ""

    if "context_score_proxy_u01" not in out.columns:
        out["context_score_proxy_u01"] = pd.NA
    if "context_score_proxy_p_warn" not in out.columns:
        out["context_score_proxy_p_warn"] = pd.NA
    if "context_score_proxy_p_warn_eff" not in out.columns:
        out["context_score_proxy_p_warn_eff"] = pd.NA
    if "context_score_proxy_key_base" not in out.columns:
        out["context_score_proxy_key_base"] = ""

    if "module_reason" not in out.columns:
        out["module_reason"] = ""

    dist = distilled.copy()
    if "term_uid" not in dist.columns:
        if {"source", "term_id"}.issubset(set(dist.columns)):
            dist["term_uid"] = dist.apply(
                lambda r: _shared.make_term_uid(r.get("source"), r.get("term_id")),
                axis=1,
            )
        elif "term_id" in dist.columns:
            dist["term_uid"] = dist["term_id"].astype(str).str.strip()
        else:
            raise ValueError("audit_claims: distilled must have term_uid OR (source, term_id)")

    known_terms, raw_unique, raw_amb = _build_term_uid_maps(dist)

    has_survival = "term_survival" in distilled.columns
    if has_survival:
        tmp = distilled.copy()
        tmp["term_key"] = tmp["term_uid"].astype(str).str.strip()
        tmp["term_survival"] = pd.to_numeric(tmp["term_survival"], errors="coerce")
        term_surv = tmp.groupby("term_key")["term_survival"].min()
    else:
        term_surv = None

    # NOTE (spec):
    # Stability aggregation is defined strictly at the TERM level:
    #   term_survival_agg = min(term_survival over evidence_ref.term_ids)
    #
    # We intentionally DO NOT use module-level survival here because:
    #   - module_survival is often derived from the same term_survival minima,
    #     making "module" vs "term" paths redundant and confusing.
    #   - a single, deterministic definition is easier to audit and to describe in Methods.
    #
    # module_id remains useful for redundancy control / grouping elsewhere, but not for stability.

    tau_default = _get_tau_default(card)
    if tau is not None:
        try:
            tau_default = float(tau)
        except Exception:
            pass

    min_overlap_default = _get_min_overlap_default(card)

    term_to_gene_set: dict[str, set[str]] = {}

    if "evidence_genes" in dist.columns:
        for tk, xs in zip(
            dist["term_uid"].astype(str).str.strip(),
            dist["evidence_genes"],
            strict=True,
        ):
            if not tk or tk.lower() in _NA_TOKENS_L:
                continue

            genes = [_norm_gene_id(g) for g in _shared.parse_genes(xs) if str(g).strip()]
            gs = {g for g in genes if g}
            if not gs:
                continue
            term_to_gene_set.setdefault(tk, set()).update(gs)

    elif "evidence_genes_str" in dist.columns:
        for tk, s in zip(
            dist["term_uid"].astype(str).str.strip(),
            dist["evidence_genes_str"],
            strict=True,
        ):
            if not tk or tk.lower() in _NA_TOKENS_L:
                continue

            genes = [_norm_gene_id(g) for g in _shared.parse_genes(s) if str(g).strip()]
            gs = {g for g in genes if g}
            if not gs:
                continue
            term_to_gene_set.setdefault(tk, set()).update(gs)

    gene_to_term_degree: dict[str, int] = {}
    for _tk, gs in term_to_gene_set.items():
        for g in gs:
            gene_to_term_degree[g] = gene_to_term_degree.get(g, 0) + 1

    hub_thr = _get_hub_term_degree(card, default=200)
    hub_genes = {g for g, deg in gene_to_term_degree.items() if deg > int(hub_thr)}

    audit_mode = _get_audit_mode(card, default="decision")

    # Keep notes lean in decision mode (PASS should look clean).
    pass_note_enabled = _get_pass_notes(card, default=(audit_mode != "decision"))

    # Gate defaults depend on audit_mode:
    # - decision: gates should actually change status (decision-grade)
    # - diagnostic: gates may annotate without changing status (debug-grade)
    context_gate_default = "hard" if audit_mode == "decision" else "note"
    stress_gate_default = "hard" if audit_mode == "decision" else "off"

    pass_note_enabled = _get_pass_notes(card, default=bool(pass_note_enabled))
    context_gate_mode = _get_context_gate_mode(card, default=context_gate_default)
    stability_gate_mode = _get_stability_gate_mode(card, default="hard")
    stress_gate_mode = _get_stress_gate_mode(card, default=stress_gate_default)
    strict_evidence = _get_strict_evidence_check(card, default=False)

    min_union = _get_min_union_genes(card, default=3)
    hub_frac_thr = _get_hub_frac_thr(card, default=0.5)

    evidence_dropout_p = _get_evidence_dropout_p(card, default=0.0)
    contradictory_p = _get_contradictory_p(card, default=0.0)

    # -------------------------
    # CRITICAL FIX:
    # Only enforce stress gating if a stress probe is actually enabled/present.
    # Otherwise, "hard" would abstain everything due to stress_missing(hard_mode).
    # -------------------------
    has_stress_probe = (
        bool(input_stress_cols)
        or (float(evidence_dropout_p) > 0.0)
        or (float(contradictory_p) > 0.0)
    )
    if not has_stress_probe:
        stress_gate_mode = "off"

    for i, row in out.iterrows():
        cj = row.get("claim_json", None)
        if cj is None or _is_na_scalar(cj) or (not str(cj).strip()):
            out.at[i, "status"] = "FAIL"
            out.at[i, "link_ok"] = False
            out.at[i, "fail_reason"] = FAIL_SCHEMA_VIOLATION
            out.at[i, "audit_notes"] = "missing claim_json"
            _enforce_reason_vocab(out, i)
            continue

        cj_str = str(cj)
        cobj = _extract_claim_obj(cj_str)
        if cobj is None:
            out.at[i, "status"] = "FAIL"
            out.at[i, "link_ok"] = False
            out.at[i, "fail_reason"] = FAIL_SCHEMA_VIOLATION
            out.at[i, "audit_notes"] = "claim_json schema violation"
            _enforce_reason_vocab(out, i)
            continue

        try:
            out.at[i, "claim_json"] = json.dumps(
                cobj.model_dump(),
                ensure_ascii=False,
                separators=(",", ":"),
            )
            cj_str = str(out.at[i, "claim_json"])
        except Exception:
            pass

        d = str(row.get("direction", "")).strip().lower()
        if d not in {"up", "down", "na"}:
            d = _extract_direction_from_claim_json(cj_str)
        out.at[i, "direction_norm"] = d if d in {"up", "down"} else "na"

        tau_row = tau_default
        if "tau" in out.columns and not _is_na_scalar(row.get("tau")):
            try:
                tau_row = float(row.get("tau"))
            except Exception:
                tau_row = tau_default
        out.at[i, "tau_used"] = float(tau_row)

        min_overlap = min_overlap_default
        if "min_overlap" in out.columns and not _is_na_scalar(row.get("min_overlap")):
            try:
                min_overlap = int(row.get("min_overlap"))
            except Exception:
                min_overlap = min_overlap_default

        term_ids_raw, gene_ids, gsh, module_id = _extract_evidence_from_claim_json(cj_str)
        module_id = "" if module_id is None else str(module_id).strip()
        out.at[i, "module_id_effective"] = module_id

        gene_ids_trim = [_norm_gene_id(g) for g in gene_ids if str(g).strip()]
        gene_ids_upper = [_norm_gene_id_upper(g) for g in gene_ids if str(g).strip()]

        if not term_ids_raw:
            out.at[i, "status"] = "FAIL"
            out.at[i, "link_ok"] = False
            out.at[i, "fail_reason"] = FAIL_EVIDENCE_DRIFT
            out.at[i, "audit_notes"] = "empty term_ids"
            _enforce_reason_vocab(out, i)
            continue

        term_ids, unknown, ambiguous = _resolve_term_ids_to_uids(
            term_ids_raw,
            known=known_terms,
            raw_unique=raw_unique,
            raw_ambiguous=raw_amb,
        )
        if ambiguous:
            out.at[i, "status"] = "FAIL"
            out.at[i, "link_ok"] = False
            out.at[i, "fail_reason"] = FAIL_SCHEMA_VIOLATION
            out.at[i, "audit_notes"] = f"ambiguous term_ids={ambiguous} (use term_uid)"
            _enforce_reason_vocab(out, i)
            continue
        if unknown:
            out.at[i, "status"] = "FAIL"
            out.at[i, "link_ok"] = False
            out.at[i, "fail_reason"] = FAIL_EVIDENCE_DRIFT
            out.at[i, "audit_notes"] = f"unknown term_ids={unknown}"
            _enforce_reason_vocab(out, i)
            continue

        ev_union: list[str] = []
        ev_seen_trim: set[str] = set()
        for t in term_ids:
            for g in sorted(term_to_gene_set.get(t, set())):
                if g not in ev_seen_trim:
                    ev_seen_trim.add(g)
                    ev_union.append(g)

        ev_seen_upper: set[str] = {_norm_gene_id_upper(g) for g in ev_seen_trim if str(g).strip()}

        computed_gsh_trim = _hash_gene_set_trim12(ev_union) if ev_union else ""
        computed_gsh_upper = _hash_gene_set_upper12(ev_union) if ev_union else ""
        out.at[i, "term_ids_set_hash"] = _shared.hash_set_12hex(term_ids)

        gsh_norm = "" if _is_na_scalar(gsh) else str(gsh).strip().lower()
        if not _shared.looks_like_12hex(gsh_norm):
            if strict_evidence:
                out.at[i, "status"] = "FAIL"
                out.at[i, "link_ok"] = False
                out.at[i, "fail_reason"] = FAIL_SCHEMA_VIOLATION
                out.at[i, "audit_notes"] = "gene_set_hash missing/invalid (strict)"
            else:
                out.at[i, "status"] = "ABSTAIN"
                out.at[i, "link_ok"] = False
                out.at[i, "abstain_reason"] = ABSTAIN_MISSING_EVIDENCE_GENES
                out.at[i, "audit_notes"] = "gene_set_hash missing/invalid"
            _enforce_reason_vocab(out, i)
            continue

        if not ev_union:
            if strict_evidence:
                out.at[i, "status"] = "FAIL"
                out.at[i, "link_ok"] = False
                out.at[i, "fail_reason"] = FAIL_SCHEMA_VIOLATION
                out.at[i, "audit_notes"] = (
                    "evidence_genes unavailable for referenced term_ids (strict)"
                )
            else:
                out.at[i, "status"] = "ABSTAIN"
                out.at[i, "link_ok"] = False
                out.at[i, "abstain_reason"] = ABSTAIN_MISSING_EVIDENCE_GENES
                out.at[i, "audit_notes"] = "evidence_genes unavailable for referenced term_ids"
            _enforce_reason_vocab(out, i)
            continue

        match_mode = ""
        if computed_gsh_trim.lower() == gsh_norm:
            match_mode = "trim"
            out.at[i, "gene_set_hash_computed"] = computed_gsh_trim.lower()
        elif computed_gsh_upper.lower() == gsh_norm:
            match_mode = "upper_compat"
            out.at[i, "gene_set_hash_computed"] = computed_gsh_upper.lower()
        else:
            out.at[i, "status"] = "FAIL"
            out.at[i, "link_ok"] = False
            out.at[i, "fail_reason"] = FAIL_EVIDENCE_DRIFT
            out.at[i, "audit_notes"] = (
                "gene_set_hash mismatch: "
                f"trim={computed_gsh_trim.lower()} "
                f"upper={computed_gsh_upper.lower()} "
                f"!= claim={gsh_norm}"
            )
            _enforce_reason_vocab(out, i)
            continue

        out.at[i, "gene_set_hash_match_mode"] = match_mode
        out.at[i, "gene_set_hash_effective"] = gsh_norm

        if gene_ids_trim:
            hits_trim = sum(1 for g in gene_ids_trim if g in ev_seen_trim)
            hits_upper = sum(1 for g in gene_ids_upper if g in ev_seen_upper)
            n_hit = max(hits_trim, hits_upper)

            if n_hit < int(min_overlap):
                out.at[i, "status"] = "FAIL"
                out.at[i, "link_ok"] = False
                out.at[i, "fail_reason"] = FAIL_EVIDENCE_DRIFT
                out.at[i, "audit_notes"] = (
                    f"gene_ids drift: hits={n_hit} < min_overlap={min_overlap} "
                    f"(hits_trim={hits_trim}, hits_upper={hits_upper})"
                )
                _enforce_reason_vocab(out, i)
                continue

        evk = (
            str(out.at[i, "gene_set_hash_effective"]).strip()
            or str(out.at[i, "term_ids_set_hash"]).strip()
        )
        out.at[i, "evidence_key"] = evk

        if (not has_survival) or (term_surv is None):
            out.at[i, "status"] = "ABSTAIN"
            out.at[i, "stability_ok"] = False
            out.at[i, "abstain_reason"] = ABSTAIN_MISSING_SURVIVAL
            out.at[i, "audit_notes"] = _append_note(
                out.at[i, "audit_notes"], "term_survival column missing"
            )
            _enforce_reason_vocab(out, i)
            continue

        # Stability aggregation (TERM-only; spec-level contract)
        vals: list[float] = []
        for t in term_ids:
            v = term_surv.get(t, float("nan"))
            try:
                vals.append(float(v))
            except Exception:
                vals.append(float("nan"))

        agg = float(pd.Series(vals).min(skipna=True)) if vals else float("nan")

        out.at[i, "term_survival_agg"] = agg
        out.at[i, "stability_scope"] = "term"
        out.at[i, "module_reason"] = "term_survival_used"

        if pd.isna(agg):
            out.at[i, "status"] = "ABSTAIN"
            out.at[i, "stability_ok"] = False
            out.at[i, "abstain_reason"] = ABSTAIN_MISSING_SURVIVAL
            out.at[i, "audit_notes"] = _append_note(
                out.at[i, "audit_notes"], "term_survival missing for referenced terms/module"
            )
            _enforce_reason_vocab(out, i)
            continue

        if agg < float(tau_row):
            msg = f"survival[term]={agg:.3f} < tau={float(tau_row):.2f}"

            if stability_gate_mode == "off":
                out.at[i, "stability_ok"] = True
                out.at[i, "audit_notes"] = _append_note(
                    out.at[i, "audit_notes"], f"unstable(off): {msg}"
                )
                _enforce_reason_vocab(out, i)
            elif stability_gate_mode == "note":
                out.at[i, "stability_ok"] = True
                out.at[i, "audit_notes"] = _append_note(
                    out.at[i, "audit_notes"], f"unstable(note): {msg}"
                )
                _enforce_reason_vocab(out, i)
            else:
                out.at[i, "status"] = "ABSTAIN"
                out.at[i, "stability_ok"] = False
                out.at[i, "abstain_reason"] = ABSTAIN_UNSTABLE
                out.at[i, "audit_notes"] = _append_note(out.at[i, "audit_notes"], msg)
                _enforce_reason_vocab(out, i)
                continue

        if input_contra_cols:
            row_has_contra_value = False
            for c in input_contra_cols:
                v = row.get(c)
                if not _is_na_scalar(v) and str(v).strip():
                    row_has_contra_value = True
                    break

            if row_has_contra_value:
                _apply_external_contradiction(out, i, row)
                if str(out.at[i, "status"]).strip().upper() in {"FAIL", "ABSTAIN"}:
                    _enforce_reason_vocab(out, i)
                    continue

        if len(ev_union) < int(min_union):
            out.at[i, "status"] = "ABSTAIN"
            out.at[i, "abstain_reason"] = ABSTAIN_UNDER_SUPPORTED
            msg = (
                f"under_supported: |union_evidence_genes|={len(ev_union)} "
                f"< min_union_genes={int(min_union)}"
            )
            out.at[i, "audit_notes"] = _append_note(out.at[i, "audit_notes"], msg)
            _enforce_reason_vocab(out, i)
            continue

        if ev_union and hub_genes:
            hub_hits = [g for g in ev_union if g in hub_genes]
            frac = (len(hub_hits) / len(ev_union)) if ev_union else 0.0
            if frac >= float(hub_frac_thr):
                out.at[i, "status"] = "ABSTAIN"
                out.at[i, "abstain_reason"] = ABSTAIN_HUB_BRIDGE
                out.at[i, "audit_notes"] = _append_note(
                    out.at[i, "audit_notes"],
                    f"hub_bridge: hub_frac={frac:.2f} (n_hub={len(hub_hits)}/{len(ev_union)})",
                )
                _enforce_reason_vocab(out, i)
                continue

        # --- Context integration priority (LLM-first, but keep swap-sensitive proxy) ---
        c_eval, c_status, c_note = _extract_context_review_from_claim_json(cj_str)

        context_review_mode = str(row.get("context_review_mode", "")).strip().lower()

        cs_val = None
        if "context_score" in row.index:
            cs_raw = row.get("context_score")
            try:
                cs_val = None if _is_na_scalar(cs_raw) else float(cs_raw)
            except Exception:
                cs_val = None

        swap_active = False
        try:
            swap_active = bool(_row_context_swap_active(row))
        except Exception:
            swap_active = False

        # (A) If claim_json has no context, and proxy is allowed, run proxy BEFORE row_fallback
        #     ONLY when the only available signal is row-level proxy-ish stuff
        #     (context_score / proxy mode).
        #     This restores swap effect (swap_penalty) and avoids row_fallback masking it.
        if (not c_eval) and (not str(c_status or "").strip()):
            if (context_review_mode == "proxy") or (cs_val is not None):
                # If row has true LLM outputs (context_review_*), we must NOT override them.
                has_llm_cols = (
                    ("context_review_evaluated" in row.index)
                    or ("context_review_status" in row.index)
                    or ("context_review_reason" in row.index)
                )
                if not has_llm_cols:
                    c_eval, c_status, c_note, u01, p_warn_eff, key_base = _proxy_context_status(
                        card=card,
                        row=row,
                        term_ids=term_ids,
                        module_id=str(module_id or ""),
                        gene_set_hash=str(gsh_norm or ""),
                    )
                    out.at[i, "context_score_proxy_u01"] = float(u01)
                    out.at[i, "context_score_proxy_p_warn"] = float(_get_context_proxy_warn_p(card))
                    out.at[i, "context_score_proxy_p_warn_eff"] = float(p_warn_eff)
                    out.at[i, "context_score_proxy_key_base"] = str(key_base)

        # (B) If still missing, use row fallback
        #     (LLM outputs > legacy context_* > presence of context_score)
        if (not c_eval) and (not str(c_status or "").strip()):
            ev2, st2, note2 = _context_eval_from_row(row)
            if ev2 or st2:
                c_eval, c_status, c_note = ev2, st2, f"row_fallback: {note2}"

        st_norm = str(c_status or "").strip().upper()
        if (not c_eval) and st_norm in {"PASS", "WARN", "FAIL"}:
            c_eval = True

        try:
            out.at[i, "context_evaluated"] = bool(getattr(cobj, "context_evaluated", False))
            out.at[i, "context_method"] = str(getattr(cobj, "context_method", "none") or "none")
            out.at[i, "context_status"] = str(getattr(cobj, "context_status", "") or "")
            out.at[i, "context_reason"] = str(getattr(cobj, "context_reason", "") or "")
            out.at[i, "context_notes"] = str(getattr(cobj, "context_notes", "") or "")
        except Exception:
            pass

        if str(out.at[i, "context_status"]).strip() == "":
            if c_eval or st_norm:
                out.at[i, "context_evaluated"] = bool(c_eval)
                # Robust method labeling (do not depend on free-text notes).
                if "proxy_context_v" in str(c_note):
                    out.at[i, "context_method"] = "proxy"
                elif "row_fallback:" in str(c_note):
                    out.at[i, "context_method"] = "row"
                else:
                    # If the row-level LLM probe columns exist, treat as llm-originated.
                    has_llm_cols = (
                        ("context_review_evaluated" in row.index)
                        or ("context_review_status" in row.index)
                        or ("context_review_reason" in row.index)
                    )
                    out.at[i, "context_method"] = "llm" if has_llm_cols else "row"

                out.at[i, "context_status"] = st_norm
                out.at[i, "context_reason"] = (
                    "proxy_context_score" if "proxy_context" in str(c_note) else ""
                )
                out.at[i, "context_notes"] = str(c_note)

        st_out = str(out.at[i, "context_status"]).strip().upper()
        if st_out in {"PASS", "WARN", "FAIL"} and (not bool(out.at[i, "context_evaluated"])):
            out.at[i, "context_evaluated"] = True

        # --- Swap ablation strictness (paper contract) ---
        # If context_swap_active, we require strict context consistency:
        #   - missing evaluation => ABSTAIN_CONTEXT_MISSING
        #   - WARN => ABSTAIN_CONTEXT_NONSPECIFIC
        swap_strict_mode = _get_context_swap_strict_mode(card, default="warn_to_abstain")
        swap_active = _row_context_swap_active(row)

        if swap_active and (swap_strict_mode == "warn_to_abstain") and (context_gate_mode != "off"):
            # If missing evaluation, treat as hard-missing under swap.
            if not bool(c_eval):
                out.at[i, "audit_notes"] = _append_note(
                    out.at[i, "audit_notes"], f"swap_strict: context_missing -> ABSTAIN ({c_note})"
                )
                out.at[i, "status"] = "ABSTAIN"
                out.at[i, "abstain_reason"] = ABSTAIN_CONTEXT_MISSING
                _enforce_reason_vocab(out, i)
                continue

            # If WARN, treat as hard under swap regardless of note/hard mode.
            if st_norm == "WARN":
                out.at[i, "audit_notes"] = _append_note(
                    out.at[i, "audit_notes"], f"swap_strict: WARN -> ABSTAIN ({c_note})"
                )
                out.at[i, "status"] = "ABSTAIN"
                out.at[i, "abstain_reason"] = ABSTAIN_CONTEXT_NONSPECIFIC
                _enforce_reason_vocab(out, i)
                continue

        if context_gate_mode != "off":
            if not bool(c_eval):
                out.at[i, "audit_notes"] = _append_note(
                    out.at[i, "audit_notes"], f"context_missing({context_gate_mode}): {c_note}"
                )
                if context_gate_mode == "hard":
                    out.at[i, "status"] = "ABSTAIN"
                    out.at[i, "abstain_reason"] = ABSTAIN_CONTEXT_MISSING
                    _enforce_reason_vocab(out, i)
                    continue
            else:
                if st_norm == "PASS":
                    pass
                elif st_norm == "WARN":
                    if context_gate_mode == "note":
                        out.at[i, "audit_notes"] = _append_note(
                            out.at[i, "audit_notes"], f"context_warn(note): {c_note}"
                        )
                    else:
                        out.at[i, "status"] = "ABSTAIN"
                        out.at[i, "abstain_reason"] = ABSTAIN_CONTEXT_NONSPECIFIC
                        out.at[i, "audit_notes"] = _append_note(
                            out.at[i, "audit_notes"], f"context_warn(hard): {c_note}"
                        )
                        _enforce_reason_vocab(out, i)
                        continue
                elif st_norm == "FAIL":
                    if context_gate_mode == "note":
                        out.at[i, "audit_notes"] = _append_note(
                            out.at[i, "audit_notes"], f"context_fail(note): {c_note}"
                        )
                    else:
                        out.at[i, "status"] = "FAIL"
                        out.at[i, "fail_reason"] = FAIL_CONTEXT
                        out.at[i, "audit_notes"] = _append_note(
                            out.at[i, "audit_notes"], f"context_fail(hard): {c_note}"
                        )
                        _enforce_reason_vocab(out, i)
                        continue
                else:
                    out.at[i, "audit_notes"] = _append_note(
                        out.at[i, "audit_notes"],
                        f"context_status_missing({context_gate_mode}): {c_note}",
                    )
                    if context_gate_mode == "hard":
                        out.at[i, "status"] = "ABSTAIN"
                        out.at[i, "abstain_reason"] = ABSTAIN_CONTEXT_MISSING
                        _enforce_reason_vocab(out, i)
                        continue

        if evidence_dropout_p > 0.0:
            out.at[i, "stress_evaluated"] = True

            kept, note = _apply_internal_evidence_dropout(
                genes=list(ev_union),
                evidence_key=str(out.at[i, "evidence_key"]),
                p=float(evidence_dropout_p),
            )

            # If dropout makes the evidence too small, it's inconclusive.
            if len(kept) < int(min_union):
                out.at[i, "status"] = "ABSTAIN"
                out.at[i, "stress_ok"] = False
                out.at[i, "abstain_reason"] = ABSTAIN_INCONCLUSIVE_STRESS
                out.at[i, "audit_notes"] = _append_note(
                    out.at[i, "audit_notes"],
                    f"dropout_inconclusive: {note} < min_union_genes={int(min_union)}",
                )
                _enforce_reason_vocab(out, i)
                continue

            # --- Jaccard-based robustness (Fix 1) ---
            # NOTE: dropout changes gene set by design, so hash drift is expected.
            # We judge stability by overlap (Jaccard) against the original union evidence set.
            ref_set = {_norm_gene_id(g) for g in ev_union if str(g).strip()}
            kept_set = {_norm_gene_id(g) for g in kept if str(g).strip()}

            # Guard against degenerate tiny unions (already checked min_union, but keep safe)
            if (len(ref_set) < int(min_union)) or (len(kept_set) < int(min_union)):
                out.at[i, "status"] = "ABSTAIN"
                out.at[i, "stress_ok"] = False
                out.at[i, "abstain_reason"] = ABSTAIN_INCONCLUSIVE_STRESS
                out.at[i, "audit_notes"] = _append_note(
                    out.at[i, "audit_notes"],
                    f"dropout_inconclusive: tiny_sets ref={len(ref_set)} kept={len(kept_set)}",
                )
                _enforce_reason_vocab(out, i)
                continue

            j_pass = _get_stress_jaccard_pass(card, default=0.8)
            j_soft = _get_stress_jaccard_soft(card, default=0.5)
            if j_soft > j_pass:
                # keep contract sane
                j_soft = max(0.0, min(j_pass, j_soft))

            j = _jaccard(ref_set, kept_set)
            inter = len(ref_set & kept_set)
            uni = len(ref_set | kept_set)

            if j >= float(j_pass):
                out.at[i, "stress_ok"] = True
                out.at[i, "audit_notes"] = _append_note(
                    out.at[i, "audit_notes"],
                    f"dropout_ok: {note} jaccard={j:.3f} inter={inter} union={uni}",
                )
            else:
                out.at[i, "status"] = "ABSTAIN"
                out.at[i, "stress_ok"] = False
                out.at[i, "abstain_reason"] = ABSTAIN_INCONCLUSIVE_STRESS

                bucket = "JACCARD_MID" if j >= float(j_soft) else "JACCARD_LOW"
                out.at[i, "audit_notes"] = _append_note(
                    out.at[i, "audit_notes"],
                    (
                        f"dropout_inconclusive({bucket}): {note} "
                        f"jaccard={j:.3f} inter={inter} union={uni} "
                        f"(pass>={float(j_pass):.2f}, soft>={float(j_soft):.2f})"
                    ),
                )
                _enforce_reason_vocab(out, i)
                continue

        if contradictory_p > 0.0 and str(out.at[i, "direction_norm"]).strip().lower() in {
            "up",
            "down",
        }:
            out.at[i, "stress_evaluated"] = True
            u = _deterministic_uniform_0_1(str(out.at[i, "evidence_key"]), salt="contradictory")
            if u < float(contradictory_p):
                out.at[i, "status"] = "ABSTAIN"
                out.at[i, "abstain_reason"] = ABSTAIN_INCONCLUSIVE_STRESS
                out.at[i, "contradiction_ok"] = False
                out.at[i, "audit_notes"] = _append_note(
                    out.at[i, "audit_notes"],
                    f"contradictory_probe_triggered(p={float(contradictory_p):.3f})",
                )
                _enforce_reason_vocab(out, i)
                continue

        if input_stress_cols:
            row_has_stress_value = False
            for c in input_stress_cols:
                v = row.get(c)
                if not _is_na_scalar(v) and str(v).strip():
                    row_has_stress_value = True
                    break

            if row_has_stress_value:
                out.at[i, "stress_evaluated"] = True
                _apply_external_stress(
                    out, i, row, gate_mode=stress_gate_mode, input_stress_cols=input_stress_cols
                )
                if str(out.at[i, "status"]).strip().upper() in {"FAIL", "ABSTAIN"}:
                    _enforce_reason_vocab(out, i)
                    continue

        if not bool(out.at[i, "stress_evaluated"]):
            if stress_gate_mode == "note":
                out.at[i, "audit_notes"] = _append_note(
                    out.at[i, "audit_notes"], "stress_missing(note_mode)"
                )
            elif stress_gate_mode == "hard":
                out.at[i, "status"] = "ABSTAIN"
                out.at[i, "stress_ok"] = False
                out.at[i, "abstain_reason"] = ABSTAIN_INCONCLUSIVE_STRESS
                out.at[i, "audit_notes"] = _append_note(
                    out.at[i, "audit_notes"], "stress_missing(hard_mode)"
                )
                _enforce_reason_vocab(out, i)
                continue

        if pass_note_enabled and (not str(out.at[i, "audit_notes"]).strip()):
            out.at[i, "audit_notes"] = "ok"

        _enforce_reason_vocab(out, i)

    # -------------------------
    # Diagnostic / derived context gate columns (run-wide, always computed)
    # -------------------------
    if "select_context_status_score" not in out.columns:
        out["select_context_status_score"] = -1
    if "context_gate_blocked" not in out.columns:
        out["context_gate_blocked"] = False
    if "eligible_context" not in out.columns:
        out["eligible_context"] = False

    st_all = out["context_status"].astype(str).str.strip().str.upper()
    ev_all = out["context_evaluated"].astype(bool)

    score = pd.Series([-1] * len(out), index=out.index)
    score[st_all == "PASS"] = 1
    score[st_all == "WARN"] = 0
    score[st_all == "FAIL"] = -1
    out["select_context_status_score"] = score.astype(int)

    gate_mode_eff = str(context_gate_mode or "").strip().lower()
    if gate_mode_eff == "hard":
        blocked = (~ev_all) | (st_all.isin(["WARN", "FAIL"])) | (st_all == "")
    else:
        blocked = pd.Series([False] * len(out), index=out.index)

    out["context_gate_blocked"] = blocked.astype(bool)
    out["eligible_context"] = (~blocked).astype(bool)

    # legacy: in this file "hit" == "blocked"
    if "context_gate_hit" in out.columns:
        out["context_gate_hit"] = blocked.astype(bool)

    # After audit_log.tsv is assembled (or right before writing it)
    ex = _get_extra(card)
    if str(ex.get("context_review_mode", "")).strip().lower() == "llm":
        cm = out["context_method"].fillna("").astype(str).str.lower()
        n_llm = int((cm == "llm").sum())
        n_proxy = int((cm == "proxy").sum())
        n_none = int((cm == "none").sum())
        if n_llm == 0 and (n_proxy + n_none) > 0:
            print(
                f"[SANITY][WARN] context_review_mode=llm but context_method has 0 llm "
                f"(proxy={n_proxy}, none={n_none}). Review likely never called; "
                f"falling back or skipping everywhere."
            )

    _apply_intra_run_contradiction(out)
    return out
