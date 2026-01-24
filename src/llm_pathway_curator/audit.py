# LLM-PathwayCurator/src/llm_pathway_curator/audit.py
from __future__ import annotations

import hashlib
import json
from typing import Any

import pandas as pd

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

_NA_TOKENS = {"", "na", "nan", "none", "NA"}
_NA_TOKENS_L = {t.lower() for t in _NA_TOKENS}


def _is_na_scalar(x: Any) -> bool:
    """pd.isna is unsafe for list-like; only treat scalars here."""
    if x is None:
        return True
    if isinstance(x, (list, tuple, set, dict)):
        return False
    try:
        v = pd.isna(x)
        return bool(v) if isinstance(v, bool) else False
    except Exception:
        return False


def _parse_ids(x: Any) -> list[str]:
    """
    Parse ids into list[str] with tolerant separators:
      - list/tuple/set: preserve order (dedup)
      - string: split on , ; | (fallback: whitespace if no commas)
      - NA -> []
    Safe against list-like inputs (won't call pd.isna on list).
    """
    if _is_na_scalar(x):
        return []

    if isinstance(x, (list, tuple, set)):
        items = [str(t).strip() for t in x if str(t).strip()]
    else:
        s = str(x).strip()
        if not s or s.lower() in _NA_TOKENS_L:
            return []

        s = s.replace(";", ",").replace("|", ",")
        s = s.replace("\n", " ").replace("\t", " ")
        s = " ".join(s.split()).strip()
        if not s or s.lower() in _NA_TOKENS_L:
            return []

        if "," in s:
            items = [t.strip() for t in s.split(",") if t.strip()]
        else:
            items = [t.strip() for t in s.split(" ") if t.strip()]

    seen: set[str] = set()
    uniq: list[str] = []
    for t in items:
        tt = str(t).strip()
        if not tt or tt.lower() in _NA_TOKENS_L:
            continue
        if tt not in seen:
            seen.add(tt)
            uniq.append(tt)
    return uniq


# -------------------------
# Gene canonicalization + hashing (compat-safe)
# -------------------------
def _clean_gene_id(g: Any) -> str:
    s = str(g).strip().strip('"').strip("'")
    s = " ".join(s.split())
    s = s.strip(",;|")
    return s


def _norm_gene_id(g: Any) -> str:
    """
    Canonical gene token for audit comparisons.

    IMPORTANT:
      - Do NOT force uppercasing here (align with schema/distill/modules trim-only).
      - We still provide a legacy upper hash path for backward compatibility.
    """
    return _clean_gene_id(g)


def _norm_gene_id_upper(g: Any) -> str:
    """Legacy HGNC-like normalization (deprecated; used only for backward-compat hash matching)."""
    return _clean_gene_id(g).upper()


def _hash_gene_set_trim12(genes: list[str]) -> str:
    """Set-stable hash over trim-only canonicalization."""
    uniq = sorted({_norm_gene_id(g) for g in genes if str(g).strip()})
    payload = ",".join(uniq)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def _hash_gene_set_upper12(genes: list[str]) -> str:
    """Set-stable hash over legacy uppercasing canonicalization."""
    uniq = sorted({_norm_gene_id_upper(g) for g in genes if str(g).strip()})
    payload = ",".join(uniq)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def _hash_term_ids(term_ids: list[str]) -> str:
    """Set-stable hash for referenced term IDs (fallback contradiction key)."""
    uniq = sorted({str(t).strip() for t in term_ids if str(t).strip()})
    payload = ",".join(uniq)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def _looks_like_hex_hash(x: Any) -> bool:
    """12-hex short hash (sha256[:12])."""
    if _is_na_scalar(x):
        return False
    s = str(x).strip().lower()
    if len(s) != 12:
        return False
    return all(ch in "0123456789abcdef" for ch in s)


# -------- SampleCard knob access (single boundary, with robust fallback) --------
def _get_extra(card: SampleCard) -> dict[str, Any]:
    try:
        ex = getattr(card, "extra", {}) or {}
        return ex if isinstance(ex, dict) else {}
    except Exception:
        return {}


def _as_bool(x: Any, default: bool) -> bool:
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
    try:
        return float(card.audit_tau())
    except Exception:
        return 0.8


def _get_min_overlap_default(card: SampleCard) -> int:
    try:
        return int(card.audit_min_gene_overlap())
    except Exception:
        return 1


def _get_min_union_genes(card: SampleCard, default: int = 3) -> int:
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


def _get_context_proxy_pass_min(card: SampleCard, default: int = 1) -> int:
    """
    If proxy context_score is present (no LLM review), convert it into:
      - score >= pass_min => PASS
      - score <  pass_min => WARN
    This is ONLY used when claim_json has no context review fields.
    """
    ex = _get_extra(card)
    v = ex.get("context_proxy_pass_min", None)
    try:
        return int(default) if v is None else int(v)
    except Exception:
        return int(default)


def _get_context_proxy_warn_p(card: SampleCard, default: float = 0.2) -> float:
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
    card: SampleCard, default: str = "context_keys,term_uid"
) -> list[str]:
    """
    Comma-separated list of fields to build deterministic proxy key.
    Allowed:
    context_keys,
    term_uid,
    module_id,
    gene_set_hash,
    comparison,
    cancer,
    disease,
    tissue,
    perturbation,
    condition
    """
    ex = _get_extra(card)
    s = ex.get("context_proxy_key_fields", default)
    if _is_na_scalar(s):
        s = default
    items = [t.strip().lower() for t in str(s).split(",") if t.strip()]
    return items or [t.strip().lower() for t in str(default).split(",") if t.strip()]


def _get_context_gate_mode(card: SampleCard, default: str = "note") -> str:
    """
    Tool contract (user-facing):
      - "off": ignore context gate
      - "note": annotate context issues, do not change status
      - "hard": evaluated-only gating (safe-side): unevaluated => ABSTAIN
    Backward compat:
      - "abstain" => "hard"
    """
    if hasattr(card, "context_gate_mode") and callable(card.context_gate_mode):
        try:
            s = str(card.context_gate_mode(default=default)).strip().lower()  # type: ignore[attr-defined]
        except Exception:
            s = ""
    else:
        ex = _get_extra(card)
        s = str(ex.get("context_gate_mode", default)).strip().lower()

    if s in {"off", "disable", "disabled", "none"}:
        return "off"
    if s in {"note", "warn", "warning"}:
        return "note"
    if s in {"abstain", "hard"}:
        return "hard"

    d = str(default).strip().lower()
    if d in {"off", "note", "hard"}:
        return d
    return "note"


def _get_stress_gate_mode(card: SampleCard, default: str = "off") -> str:
    """
    Tool contract (user-facing):
      - "off" (default): ignore missing stress entirely
      - "note": annotate stress results, do not change status
      - "hard": safe-side robustness gating:
          * missing stress => ABSTAIN_INCONCLUSIVE_STRESS
          * stress ABSTAIN/WARN/FAIL => ABSTAIN_INCONCLUSIVE_STRESS (do NOT FAIL)
        Rationale:
          Stress is a robustness probe; failure under stress is not proof of invalidity.
          This preserves monotonicity: stress PASS is a subset of non-stress PASS.
    Backward compat:
      - "abstain" => "hard"
    """
    if hasattr(card, "stress_gate_mode") and callable(card.stress_gate_mode):
        try:
            s = str(card.stress_gate_mode(default=default)).strip().lower()  # type: ignore[attr-defined]
        except Exception:
            s = ""
    else:
        ex = _get_extra(card)
        s = str(ex.get("stress_gate_mode", default)).strip().lower()

    if s in {"off", "none", "disable", "disabled"}:
        return "off"
    if s in {"note", "warn", "warning"}:
        return "note"
    if s in {"hard", "abstain"}:
        return "hard"

    d = str(default).strip().lower()
    if d in {"off", "note", "hard"}:
        return d
    return "off"


def _get_stability_gate_mode(card: SampleCard, default: str = "hard") -> str:
    """
    Tool contract (user-facing):
      - "off": do not gate on stability (research only)
      - "note": do not change status; annotate instability
      - "hard": agg < tau => ABSTAIN_UNSTABLE
    Backward compat:
      - "abstain" => "hard"
    """
    if hasattr(card, "stability_gate_mode") and callable(card.stability_gate_mode):
        try:
            s = str(card.stability_gate_mode(default=default)).strip().lower()  # type: ignore[attr-defined]
        except Exception:
            s = ""
    else:
        ex = _get_extra(card)
        s = str(ex.get("stability_gate_mode", default)).strip().lower()

    if s in {"off", "disable", "disabled", "none"}:
        return "off"
    if s in {"note", "warn", "warning"}:
        return "note"
    if s in {"abstain", "hard"}:
        return "hard"

    d = str(default).strip().lower()
    if d in {"off", "note", "hard"}:
        return d
    return "hard"


def _get_strict_evidence_check(card: SampleCard, default: bool = False) -> bool:
    """
    If True: missing evidence_genes in distilled => FAIL_SCHEMA_VIOLATION (strict contract).
    If False (default): missing evidence_genes => ABSTAIN (decision-grade).
    """
    ex = _get_extra(card)
    return _as_bool(ex.get("strict_evidence_check", None), default=default)


def _get_evidence_dropout_p(card: SampleCard, default: float = 0.0) -> float:
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
    ex = _get_extra(card)
    v = ex.get("contradictory_p", None)
    if v is None:
        return float(default)
    try:
        p = float(v)
    except Exception:
        return float(default)
    return min(1.0, max(0.0, p))


def _norm_status(x: Any) -> str:
    if _is_na_scalar(x):
        return ""
    return str(x).strip().upper()


def _append_note(old: str, msg: str) -> str:
    old = "" if _is_na_scalar(old) else str(old)
    return msg if not old else f"{old} | {msg}"


def _extract_claim_obj(cj: str) -> Claim | None:
    try:
        return Claim.model_validate_json(cj)
    except Exception:
        return None


def _extract_evidence_from_claim_json(cj: str) -> tuple[list[str], list[str], Any, Any]:
    """
    Source of truth: claim_json (Claim schema).
    Returns: (term_ids, gene_ids, gene_set_hash, module_id)
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

    term_ids = _parse_ids(ev.get("term_ids") or ev.get("term_id") or "")
    gene_ids = _parse_ids(ev.get("gene_ids") or ev.get("gene_id") or "")
    gsh = ev.get("gene_set_hash")
    mid = ev.get("module_id")
    if isinstance(mid, (list, tuple, set)):
        mid = next(iter(mid), None)
    return (term_ids, gene_ids, gsh, mid)


def _extract_direction_from_claim_json(cj: str) -> str:
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
    Primary source: Claim schema fields.
    Returns: (evaluated, status, note)
      - evaluated: bool
      - status: "PASS"|"WARN"|"FAIL"|"" (may be empty if missing/legacy)
      - note: compact description for audit_notes
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
    Row-level context fields (compat).
    Returns: (evaluated, status, note)

    Priority (most specific first):
      1) context_review_* (from select.py LLM probe)
      2) legacy context_* (pre-schema)
      3) context_score presence
    """
    # --- v1.1+ probe columns emitted by select._apply_context_review_llm ---
    if "context_review_evaluated" in row.index:
        v = row.get("context_review_evaluated")
        if not _is_na_scalar(v):
            try:
                ev = bool(v)
            except Exception:
                ev = False

            st = ""
            if "context_review_status" in row.index:
                st = _norm_status(row.get("context_review_status"))
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

            # Critical compat: if a status exists, treat as evaluated
            # even if evaluated flag is False
            if (not ev) and st in {"PASS", "WARN", "FAIL"}:
                ev = True

            note = f"context_review_evaluated={ev}"
            if st:
                note += f" status={st}"
            if rsn:
                note += f" reason={rsn}"
            return (ev, st, note)

    # --- legacy/older caller columns ---
    if "context_evaluated" in row.index:
        v = row.get("context_evaluated")
        if _is_na_scalar(v):
            return (False, "", "context_evaluated=NA")
        try:
            ev = bool(v)
        except Exception:
            return (False, "", "context_evaluated invalid")

        st = ""
        if "context_status" in row.index:
            st = _norm_status(row.get("context_status"))
            if st == "ABSTAIN":
                st = "WARN"
            if st and st not in {"PASS", "WARN", "FAIL"}:
                st = ""

        # Critical compat: if a status exists, treat as evaluated even if legacy flag is False
        if (not ev) and st in {"PASS", "WARN", "FAIL"}:
            ev = True

        return (ev, st, f"context_evaluated={ev}" + (f" status={st}" if st else ""))

    if "context_status" in row.index:
        st = _norm_status(row.get("context_status"))
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
    Backfill stress/contradiction columns on claims using distilled annotations.

    - If claims already include stress_* columns, do nothing (respect caller).
    - Else if distilled has stress_tag:
        non-empty tag => stress_status=ABSTAIN, stress_reason="STRESS_TAG"
    - If claims already include contradiction_* columns, do nothing.
    - Else if distilled has contradiction_flip:
        True => contradiction_status=FAIL, contradiction_reason="CONTRADICTION_FLIP"
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

    dist = distilled.copy()
    if "term_uid" not in dist.columns:
        if {"source", "term_id"}.issubset(set(dist.columns)):
            dist["term_uid"] = (
                dist["source"].astype(str).str.strip()
                + ":"
                + dist["term_id"].astype(str).str.strip()
            )
        elif "term_id" in dist.columns:
            dist["term_uid"] = dist["term_id"].astype(str).str.strip()
        else:
            return out2

    dist["term_uid"] = dist["term_uid"].astype(str).str.strip()

    stress_map: dict[str, str] = {}
    if (not claim_has_stress) and ("stress_tag" in dist.columns):
        for tu, tag in zip(dist["term_uid"].tolist(), dist["stress_tag"].tolist(), strict=False):
            t = "" if _is_na_scalar(tag) else str(tag).strip()
            if not t or not tu:
                continue
            if tu not in stress_map:
                stress_map[tu] = t

    contra_map: dict[str, bool] = {}
    if (not claim_has_contra) and ("contradiction_flip" in dist.columns):
        for tu, v in zip(
            dist["term_uid"].tolist(), dist["contradiction_flip"].tolist(), strict=False
        ):
            if not tu or tu in contra_map:
                continue
            if _is_na_scalar(v):
                continue
            try:
                contra_map[tu] = bool(v)
            except Exception:
                continue

    if not stress_map and not contra_map:
        return out2

    if stress_map and (not claim_has_stress):
        out2["stress_status"] = ""
        out2["stress_reason"] = ""
        out2["stress_notes"] = ""
    if contra_map and (not claim_has_contra):
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

        if stress_map and (not claim_has_stress):
            tags = [stress_map.get(t, "") for t in term_ids]
            tags = [t for t in tags if t]
            if tags:
                out2.at[i, "stress_status"] = "ABSTAIN"
                out2.at[i, "stress_reason"] = "STRESS_TAG"
                out2.at[i, "stress_notes"] = (
                    f"stress_tag={tags[0]}" if len(tags) == 1 else f"stress_tag={tags}"
                )

        if contra_map and (not claim_has_contra):
            flips = [contra_map.get(t, False) for t in term_ids]
            if any(bool(x) for x in flips):
                out2.at[i, "contradiction_status"] = "FAIL"
                out2.at[i, "contradiction_reason"] = "CONTRADICTION_FLIP"
                out2.at[i, "contradiction_notes"] = "distilled.contradiction_flip=True"

    return out2


def _apply_external_contradiction(out: pd.DataFrame, i: int, row: pd.Series) -> None:
    """
    External contradiction check (optional).
    Policy:
      - Missing contradiction_status => do NOT abstain; just note.
      - PASS => ok
      - ABSTAIN/WARN => abstain (inconclusive)
      - FAIL => fail (contradiction)
      - Unknown => schema violation
    """
    if "contradiction_status" not in out.columns:
        return

    st = _norm_status(row.get("contradiction_status"))
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
        out.at[i, "abstain_reason"] = ABSTAIN_INCONCLUSIVE_STRESS
        out.at[i, "audit_notes"] = _append_note(out.at[i, "audit_notes"], f"contradiction={st}")
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
    Apply stress results if present.

    Contract intent:
      - Stress is a robustness probe, not a validity proof.
      - Therefore, stress failures should NOT be escalated to FAIL.
      - In hard mode, "missing" or "failed/abstained" stress => ABSTAIN_INCONCLUSIVE_STRESS.
    """
    if not input_stress_cols:
        out.at[i, "stress_ok"] = pd.NA
        return

    gate_mode = (gate_mode or "").strip().lower()
    if gate_mode not in {"off", "note", "hard"}:
        gate_mode = "note"

    # If only stress_ok provided by caller
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

        ok = bool(v)
        out.at[i, "stress_ok"] = ok

        if ok:
            out.at[i, "audit_notes"] = _append_note(out.at[i, "audit_notes"], "stress_ok=True")
            return

        # stress_ok=False => robustness not supported; do NOT FAIL
        if gate_mode == "hard":
            out.at[i, "status"] = "ABSTAIN"
            out.at[i, "abstain_reason"] = ABSTAIN_INCONCLUSIVE_STRESS
            out.at[i, "audit_notes"] = _append_note(
                out.at[i, "audit_notes"], "stress_ok=False -> ABSTAIN (hard)"
            )
        else:
            out.at[i, "audit_notes"] = _append_note(out.at[i, "audit_notes"], "stress_ok=False")
        return

    st = _norm_status(row.get("stress_status"))
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
        # stress FAIL => not robust; do NOT escalate to FAIL
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
    Intra-run contradiction check:
      - Among PASS claims only
      - If same evidence key has both direction=up and direction=down => FAIL_CONTRADICTION
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
) -> tuple[bool, str, str]:
    """
    Deterministic proxy context review:
      - returns evaluated=True
      - status is PASS for most, WARN for a small fraction (p_warn)
    This avoids 'all WARN' collapse when context_score is uninformative.
    """
    p_warn = _get_context_proxy_warn_p(card, default=0.2)
    key_fields = _get_context_proxy_key_fields(card, default="context_keys,term_uid")

    # Build a stable key
    parts: list[str] = []
    for f in key_fields:
        if f == "context_keys":
            v = row.get("context_keys", "")
            if not _is_na_scalar(v):
                parts.append(str(v).strip())
        elif f == "term_uid":
            # resolved term_uids are preferred
            parts.append(",".join(sorted({str(t).strip() for t in term_ids if str(t).strip()})))
        elif f == "module_id":
            parts.append(str(module_id or "").strip())
        elif f == "gene_set_hash":
            parts.append(str(gene_set_hash or "").strip())
        elif f in {"comparison", "cancer", "disease", "tissue", "perturbation", "condition"}:
            parts.append(str(row.get(f, "")).strip())
        else:
            # ignore unknown fields silently
            continue

    key = "|".join([p for p in parts if p])
    if not key:
        key = str(gene_set_hash or "") or ",".join(term_ids)

    u = _deterministic_uniform_0_1(key, salt="proxy_context_v2")
    if u < float(p_warn):
        return True, "WARN", f"proxy_context_v2: u={u:.3f} < p_warn={p_warn:.3f} key={key_fields}"
    return True, "PASS", f"proxy_context_v2: u={u:.3f} >= p_warn={p_warn:.3f} key={key_fields}"


def _apply_internal_evidence_dropout(
    *,
    genes: list[str],
    evidence_key: str,
    p: float,
) -> tuple[list[str], str]:
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
    term_uids = dist["term_uid"].astype(str).str.strip().tolist()
    known = {t for t in term_uids if t and t.lower() not in _NA_TOKENS_L}

    raw_to_uids: dict[str, set[str]] = {}
    for tu in known:
        raw = tu.split(":", 1)[-1].strip()
        if not raw:
            continue
        raw_to_uids.setdefault(raw, set()).add(tu)

    raw_unique = {raw: next(iter(u)) for raw, u in raw_to_uids.items() if len(u) == 1}
    raw_amb = {raw for raw, u in raw_to_uids.items() if len(u) > 1}
    return known, raw_unique, raw_amb


def _resolve_term_ids_to_uids(
    term_ids: list[str],
    *,
    known: set[str],
    raw_unique: dict[str, str],
    raw_ambiguous: set[str],
) -> tuple[list[str], list[str], list[str]]:
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
    Mechanical audit (tool-facing).
    Status priority: FAIL > ABSTAIN > PASS
    """
    # --- IMPORTANT: capture caller-provided columns BEFORE compat injection ---
    caller_cols = set(claims.columns)

    out = claims.copy()

    # compat: inject stress/contradiction columns from distilled when absent
    out = _inject_stress_from_distilled(out, distilled)

    # Columns that truly came from the caller (not from compat injection)
    input_stress_cols = [
        c
        for c in ["stress_status", "stress_ok", "stress_reason", "stress_notes"]
        if c in caller_cols
    ]
    input_contra_cols = [
        c
        for c in ["contradiction_status", "contradiction_reason", "contradiction_notes"]
        if c in caller_cols
    ]

    # Stable audit fields (always exist after this point)
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

    # Optional debug columns (kept minimal; safe for TSV export)
    if "gene_set_hash_computed" not in out.columns:
        out["gene_set_hash_computed"] = ""
    if "gene_set_hash_match_mode" not in out.columns:
        out["gene_set_hash_match_mode"] = ""

    # always log parsed context review fields (even if caller didn't provide)
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

    # module_reason (debuggable / decision-grade)
    if "module_reason" not in out.columns:
        out["module_reason"] = ""

    # Prepare distilled lookup
    dist = distilled.copy()
    if "term_uid" not in dist.columns:
        if {"source", "term_id"}.issubset(set(dist.columns)):
            dist["term_uid"] = (
                dist["source"].astype(str).str.strip()
                + ":"
                + dist["term_id"].astype(str).str.strip()
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

    has_module = "module_id" in distilled.columns
    module_surv = None
    if has_survival and has_module:
        tmpm = distilled.copy()
        tmpm["module_id"] = tmpm["module_id"].astype(str).str.strip()
        tmpm["term_survival"] = pd.to_numeric(tmpm["term_survival"], errors="coerce")
        tmpm = tmpm[~tmpm["module_id"].str.lower().isin(_NA_TOKENS_L)]
        if not tmpm.empty:
            module_surv = tmpm.groupby("module_id")["term_survival"].min()

    tau_default = _get_tau_default(card)
    if tau is not None:
        try:
            tau_default = float(tau)
        except Exception:
            pass

    min_overlap_default = _get_min_overlap_default(card)

    # Precompute evidence genes per term_uid
    term_to_gene_set: dict[str, set[str]] = {}
    if "evidence_genes" in dist.columns:
        for tk, xs in zip(
            dist["term_uid"].astype(str).str.strip(),
            dist["evidence_genes"],
            strict=True,
        ):
            if not tk or tk.lower() in _NA_TOKENS_L:
                continue
            if isinstance(xs, (list, tuple, set)):
                genes = [_norm_gene_id(g) for g in xs if str(g).strip()]
            else:
                genes = [_norm_gene_id(g) for g in _parse_ids(xs)]
            gs = {g for g in genes if g}
            if not gs:
                continue
            term_to_gene_set.setdefault(tk, set()).update(gs)
    elif "evidence_genes_str" in dist.columns:
        for tk, s in zip(
            dist["term_uid"].astype(str).str.strip(),
            dist["evidence_genes_str"].astype(str),
            strict=True,
        ):
            if not tk or tk.lower() in _NA_TOKENS_L:
                continue
            genes = [_norm_gene_id(g) for g in _parse_ids(s)]
            gs = {g for g in genes if g}
            if not gs:
                continue
            term_to_gene_set.setdefault(tk, set()).update(gs)

    # Hub gene heuristic (degree on canonical tokens)
    gene_to_term_degree: dict[str, int] = {}
    for _tk, gs in term_to_gene_set.items():
        for g in gs:
            gene_to_term_degree[g] = gene_to_term_degree.get(g, 0) + 1

    hub_thr = _get_hub_term_degree(card, default=200)
    hub_genes = {g for g, deg in gene_to_term_degree.items() if deg > int(hub_thr)}

    pass_note_enabled = _get_pass_notes(card, default=True)
    context_gate_mode = _get_context_gate_mode(card, default="note")
    stability_gate_mode = _get_stability_gate_mode(card, default="hard")
    stress_gate_mode = _get_stress_gate_mode(card, default="off")
    strict_evidence = _get_strict_evidence_check(card, default=False)

    min_union = _get_min_union_genes(card, default=3)
    hub_frac_thr = _get_hub_frac_thr(card, default=0.5)

    evidence_dropout_p = _get_evidence_dropout_p(card, default=0.0)
    contradictory_p = _get_contradictory_p(card, default=0.0)

    for i, row in out.iterrows():
        # 0) schema validation (claim_json -> Claim)
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

        # normalize claim_json to one-line canonical JSON
        try:
            out.at[i, "claim_json"] = json.dumps(
                cobj.model_dump(),
                ensure_ascii=False,
                separators=(",", ":"),
            )
            cj_str = str(out.at[i, "claim_json"])  # keep cj_str consistent downstream
        except Exception:
            # keep original
            pass

        # direction (for contradiction check)
        d = str(row.get("direction", "")).strip().lower()
        if d not in {"up", "down", "na"}:
            d = _extract_direction_from_claim_json(cj_str)
        out.at[i, "direction_norm"] = d if d in {"up", "down"} else "na"

        # Per-claim tau override
        tau_row = tau_default
        if "tau" in out.columns and not _is_na_scalar(row.get("tau")):
            try:
                tau_row = float(row.get("tau"))
            except Exception:
                tau_row = tau_default
        out.at[i, "tau_used"] = float(tau_row)

        # Per-claim min_overlap override
        min_overlap = min_overlap_default
        if "min_overlap" in out.columns and not _is_na_scalar(row.get("min_overlap")):
            try:
                min_overlap = int(row.get("min_overlap"))
            except Exception:
                min_overlap = min_overlap_default

        # 1) evidence refs
        term_ids_raw, gene_ids, gsh, module_id = _extract_evidence_from_claim_json(cj_str)
        module_id = "" if module_id is None else str(module_id).strip()
        out.at[i, "module_id_effective"] = module_id

        # canonicalize gene_ids (trim-only) + also prepare upper for overlap fallback
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

        # union evidence genes across referenced terms (deterministic order)
        ev_union: list[str] = []
        ev_seen_trim: set[str] = set()
        for t in term_ids:
            for g in sorted(term_to_gene_set.get(t, set())):
                if g not in ev_seen_trim:
                    ev_seen_trim.add(g)
                    ev_union.append(g)

        # also keep an upper view for overlap/backward-compat comparisons
        ev_seen_upper: set[str] = {_norm_gene_id_upper(g) for g in ev_seen_trim if str(g).strip()}

        computed_gsh_trim = _hash_gene_set_trim12(ev_union) if ev_union else ""
        computed_gsh_upper = _hash_gene_set_upper12(ev_union) if ev_union else ""
        out.at[i, "term_ids_set_hash"] = _hash_term_ids(term_ids)

        gsh_norm = "" if _is_na_scalar(gsh) else str(gsh).strip().lower()
        if not _looks_like_hex_hash(gsh_norm):
            out.at[i, "status"] = "FAIL"
            out.at[i, "link_ok"] = False
            out.at[i, "fail_reason"] = FAIL_SCHEMA_VIOLATION
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

        # Backward-compat: accept either trim-hash or upper-hash match
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

        # Optional gene_ids overlap check (min_overlap). Allow case-insensitive fallback.
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

        # 2) stability gate
        if (not has_survival) or (term_surv is None):
            out.at[i, "status"] = "ABSTAIN"
            out.at[i, "stability_ok"] = False
            out.at[i, "abstain_reason"] = ABSTAIN_MISSING_SURVIVAL
            out.at[i, "audit_notes"] = _append_note(
                out.at[i, "audit_notes"], "term_survival column missing"
            )
            _enforce_reason_vocab(out, i)
            continue

        agg = float("nan")
        scope = "term"

        if module_surv is not None and module_id and module_id in module_surv.index:
            try:
                agg = float(module_surv.get(module_id, float("nan")))
                scope = "module"
            except Exception:
                agg = float("nan")
                scope = "term"

        if pd.isna(agg):
            vals: list[float] = []
            for t in term_ids:
                v = term_surv.get(t, float("nan"))
                try:
                    vals.append(float(v))
                except Exception:
                    vals.append(float("nan"))
            agg = float(pd.Series(vals).min(skipna=True)) if vals else float("nan")
            scope = "term"

        out.at[i, "term_survival_agg"] = agg
        out.at[i, "stability_scope"] = scope

        # module_reason: why module/term survival was used
        if scope == "module":
            out.at[i, "module_reason"] = f"module_survival_used(module_id={module_id})"
        else:
            if module_id and (
                module_surv is None or module_id not in getattr(module_surv, "index", [])
            ):
                out.at[i, "module_reason"] = (
                    f"term_survival_used(module_id_missing_or_unseen={module_id})"
                )
            else:
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
            msg = f"survival[{scope}]={agg:.3f} < tau={float(tau_row):.2f}"

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

        # 3) optional external contradiction (only if caller provided any contradiction values)
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

        # 4) deterministic abstain rules
        if len(ev_union) < int(min_union):
            out.at[i, "status"] = "ABSTAIN"
            out.at[i, "abstain_reason"] = ABSTAIN_UNDER_SUPPORTED
            msg = (
                "under_supported: |union_evidence_genes|="
                f"{len(ev_union)} < min_union_genes={int(min_union)}"
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

        # 4.5) context gate
        # PRIMARY = claim_json fields;
        # FALLBACK = row context_review_* (LLM probe) or legacy row fields.
        c_eval, c_status, c_note = _extract_context_review_from_claim_json(cj_str)

        # --- proxy mode: avoid "all WARN" collapse.
        # Use proxy ONLY when explicitly requested OR when context_score is a valid number.
        context_review_mode = str(row.get("context_review_mode", "")).strip().lower()

        cs_val = None
        if "context_score" in row.index:
            cs_raw = row.get("context_score")
            try:
                cs_val = None if _is_na_scalar(cs_raw) else float(cs_raw)
            except Exception:
                cs_val = None

        if (not c_eval) and (not str(c_status or "").strip()):
            if (context_review_mode == "proxy") or (cs_val is not None):
                c_eval, c_status, c_note = _proxy_context_status(
                    card=card,
                    row=row,
                    term_ids=term_ids,
                    module_id=str(module_id or ""),
                    gene_set_hash=str(gsh_norm or ""),
                )

        # Row fallback (covers select._apply_context_review_llm output
        # even when claim_json not patched)
        if (not c_eval) and (not str(c_status or "").strip()):
            ev2, st2, note2 = _context_eval_from_row(row)
            if ev2 or st2:
                c_eval, c_status, c_note = ev2, st2, f"row_fallback: {note2}"

        # Final normalization (CRITICAL): if we have a decision status, it is evaluated.
        st_norm = str(c_status or "").strip().upper()
        if (not c_eval) and st_norm in {"PASS", "WARN", "FAIL"}:
            c_eval = True

        # Always materialize context columns into audit log
        # Prefer claim_json if present; else fill from resolved c_eval/c_status/c_note.
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
                out.at[i, "context_method"] = (
                    "proxy"
                    if "proxy_context" in str(c_note)
                    else ("llm" if "context_review_" in str(c_note) else "row")
                )
                out.at[i, "context_status"] = st_norm
                out.at[i, "context_reason"] = (
                    "proxy_context_score"
                    if "proxy_context" in str(c_note)
                    else (
                        ""
                        if _is_na_scalar(out.at[i, "context_reason"])
                        else str(out.at[i, "context_reason"])
                    )
                )
                out.at[i, "context_notes"] = str(c_note)

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

        # 4.8) internal stress: evidence dropout
        if evidence_dropout_p > 0.0:
            out.at[i, "stress_evaluated"] = True
            kept, note = _apply_internal_evidence_dropout(
                genes=list(ev_union),
                evidence_key=str(out.at[i, "evidence_key"]),
                p=float(evidence_dropout_p),
            )

            if len(kept) < int(min_union):
                out.at[i, "status"] = "ABSTAIN"
                out.at[i, "stress_ok"] = False
                out.at[i, "abstain_reason"] = ABSTAIN_UNSTABLE
                out.at[i, "audit_notes"] = _append_note(
                    out.at[i, "audit_notes"],
                    f"dropout_unstable: {note} < min_union_genes={int(min_union)}",
                )
                _enforce_reason_vocab(out, i)
                continue

            gsh_after_trim = _hash_gene_set_trim12(list(kept))
            gsh_after_upper = _hash_gene_set_upper12(list(kept))
            gsh_eff = str(out.at[i, "gene_set_hash_effective"]).strip().lower()

            # Backward-compat: accept either trim/upper stability in the hash space
            if (gsh_after_trim.strip().lower() != gsh_eff) and (
                gsh_after_upper.strip().lower() != gsh_eff
            ):
                out.at[i, "status"] = "ABSTAIN"
                out.at[i, "stress_ok"] = False
                out.at[i, "abstain_reason"] = ABSTAIN_UNSTABLE
                out.at[i, "audit_notes"] = _append_note(
                    out.at[i, "audit_notes"],
                    (
                        f"dropout_hash_change: {note} "
                        f"after(trim={gsh_after_trim}, upper={gsh_after_upper})"
                    ),
                )
                _enforce_reason_vocab(out, i)
                continue

            out.at[i, "stress_ok"] = True
            out.at[i, "audit_notes"] = _append_note(out.at[i, "audit_notes"], f"dropout_ok: {note}")

        # 4.9) internal stress: contradiction injection
        if contradictory_p > 0.0 and str(out.at[i, "direction_norm"]).strip().lower() in {
            "up",
            "down",
        }:
            out.at[i, "stress_evaluated"] = True
            u = _deterministic_uniform_0_1(str(out.at[i, "evidence_key"]), salt="contradictory")
            if u < float(contradictory_p):
                out.at[i, "status"] = "FAIL"
                out.at[i, "fail_reason"] = FAIL_CONTRADICTION
                out.at[i, "contradiction_ok"] = False
                out.at[i, "audit_notes"] = _append_note(
                    out.at[i, "audit_notes"],
                    f"contradictory_injected(p={float(contradictory_p):.3f})",
                )
                _enforce_reason_vocab(out, i)
                continue

        # 5) external stress gate (if caller provided any stress columns with values)
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

    _apply_intra_run_contradiction(out)
    return out
