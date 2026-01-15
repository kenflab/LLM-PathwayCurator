# LLM-PathwayCurator/src/llm_pathway_curator/audit.py
from __future__ import annotations

from typing import Any

import pandas as pd

from .audit_reasons import ABSTAIN_MISSING_SURVIVAL, ABSTAIN_UNSTABLE, FAIL_EVIDENCE_DRIFT
from .sample_card import SampleCard


def _is_na_scalar(x: Any) -> bool:
    """pd.isna is unsafe for list-like; only treat scalars here."""
    if x is None:
        return True
    if isinstance(x, (list, tuple, set, dict)):
        return False
    try:
        return bool(pd.isna(x))
    except Exception:
        return False


def _parse_ids(x: Any) -> list[str]:
    """
    Parse comma/semicolon separated ids into list[str].
    Safe against list-like inputs (won't call pd.isna on list).
    """
    if _is_na_scalar(x):
        return []
    if isinstance(x, (list, tuple, set)):
        items = [str(t).strip() for t in x if str(t).strip()]
    else:
        s = str(x).strip()
        if not s or s.lower() in {"na", "nan", "none"}:
            return []
        s = s.replace(";", ",")
        items = [t.strip() for t in s.split(",") if t.strip()]

    # de-dup while preserving order
    seen: set[str] = set()
    uniq: list[str] = []
    for t in items:
        if t not in seen:
            seen.add(t)
            uniq.append(t)
    return uniq


def _default_tau(card: SampleCard) -> float:
    """
    v0: allow SampleCard.extra["audit_tau"] or attribute card.audit_tau (optional).
    If not present, fall back to conservative default.
    """
    # Prefer explicit attribute if user added one
    v = getattr(card, "audit_tau", None)
    if v is not None:
        try:
            return float(v)
        except Exception:
            pass

    # Also allow extra["audit_tau"]
    try:
        extra = getattr(card, "extra", {}) or {}
        if isinstance(extra, dict) and "audit_tau" in extra:
            return float(extra["audit_tau"])
    except Exception:
        pass

    return 0.8


def audit_claims(claims: pd.DataFrame, distilled: pd.DataFrame, card: SampleCard) -> pd.DataFrame:
    """
    Mechanical audit (v0, spec-aligned):
      1) Evidence-link integrity: term_ids must exist in distilled EvidenceTable
         - empty or unknown term_ids => FAIL (evidence_drift)
      2) Stability gate: aggregate(term_survival over referenced term_ids) >= tau
         - aggregate = min(term_survival)  (conservative)
         - if term_survival column missing => ABSTAIN (missing_survival)
         - if referenced terms have missing survival => ABSTAIN (missing_survival)
         - if agg < tau => ABSTAIN (unstable)

    Status priority: FAIL > ABSTAIN > PASS
    """
    out = claims.copy()

    # Explicit audit fields (stable columns)
    out["status"] = "PASS"
    out["link_ok"] = True
    out["stability_ok"] = True
    out["contradiction_ok"] = True
    out["stress_ok"] = True
    out["abstain_reason"] = ""
    out["fail_reason"] = ""
    out["audit_notes"] = ""

    # Optional numeric feature for calibration/reporting
    if "term_survival_agg" not in out.columns:
        out["term_survival_agg"] = pd.NA

    # Prepare distilled lookup
    if "term_id" not in distilled.columns:
        raise ValueError("audit_claims: distilled missing column term_id")

    term_id_str = distilled["term_id"].astype(str).str.strip()
    term_id_str = term_id_str[~term_id_str.str.lower().isin({"", "na", "nan", "none"})]
    known_terms = set(term_id_str.tolist())

    # Survival lookup
    has_survival = "term_survival" in distilled.columns
    surv: pd.Series | None
    if has_survival:
        tmp = distilled.copy()
        tmp["term_id_str"] = tmp["term_id"].astype(str).str.strip()
        tmp["term_survival"] = pd.to_numeric(tmp["term_survival"], errors="coerce")
        # Conservative per-term survival: min across duplicates
        surv = tmp.groupby("term_id_str")["term_survival"].min()
    else:
        surv = None

    tau_default = _default_tau(card)

    for i, row in out.iterrows():
        # Per-claim tau override (optional)
        tau = tau_default
        if "tau" in out.columns and not _is_na_scalar(row.get("tau")):
            try:
                tau = float(row.get("tau"))
            except Exception:
                tau = tau_default

        term_ids = _parse_ids(row.get("term_ids", ""))

        # 1) evidence-link integrity
        if not term_ids:
            out.at[i, "status"] = "FAIL"
            out.at[i, "link_ok"] = False
            out.at[i, "fail_reason"] = FAIL_EVIDENCE_DRIFT
            out.at[i, "audit_notes"] = "empty term_ids"
            continue

        missing = [t for t in term_ids if t not in known_terms]
        if missing:
            out.at[i, "status"] = "FAIL"
            out.at[i, "link_ok"] = False
            out.at[i, "fail_reason"] = FAIL_EVIDENCE_DRIFT
            out.at[i, "audit_notes"] = f"unknown term_ids={missing}"
            continue

        # 2) stability gate
        if not has_survival or surv is None:
            # Spec: do not "learn" stability implicitly; abstain if unavailable
            out.at[i, "status"] = "ABSTAIN"
            out.at[i, "stability_ok"] = False
            out.at[i, "abstain_reason"] = ABSTAIN_MISSING_SURVIVAL
            out.at[i, "audit_notes"] = "term_survival column missing"
            continue

        vals: list[float] = []
        for t in term_ids:
            v = surv.get(t, float("nan"))
            try:
                vals.append(float(v))
            except Exception:
                vals.append(float("nan"))

        # Conservative aggregation: min across referenced terms
        agg = float(pd.Series(vals).min(skipna=True)) if vals else float("nan")
        out.at[i, "term_survival_agg"] = agg

        if pd.isna(agg):
            out.at[i, "status"] = "ABSTAIN"
            out.at[i, "stability_ok"] = False
            out.at[i, "abstain_reason"] = ABSTAIN_MISSING_SURVIVAL
            out.at[i, "audit_notes"] = "term_survival missing for referenced terms"
            continue

        if agg < tau:
            out.at[i, "status"] = "ABSTAIN"
            out.at[i, "stability_ok"] = False
            out.at[i, "abstain_reason"] = ABSTAIN_UNSTABLE
            out.at[i, "audit_notes"] = f"term_survival_agg={agg:.3f} < tau={tau:.2f}"
            continue

    return out
