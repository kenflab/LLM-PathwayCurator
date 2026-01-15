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


_NA_TOKENS = {"", "na", "nan", "none", "NA"}


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
        if not s or s in _NA_TOKENS or s.lower() in _NA_TOKENS:
            return []
        s = s.replace(";", ",")
        items = [t.strip() for t in s.split(",") if t.strip()]

    seen: set[str] = set()
    uniq: list[str] = []
    for t in items:
        if t in _NA_TOKENS or t.lower() in _NA_TOKENS:
            continue
        if t not in seen:
            seen.add(t)
            uniq.append(t)
    return uniq


def _default_min_gene_overlap(card: SampleCard) -> int:
    v = None
    try:
        extra = getattr(card, "extra", {}) or {}
        if isinstance(extra, dict) and "audit_min_gene_overlap" in extra:
            v = extra["audit_min_gene_overlap"]
    except Exception:
        v = None
    try:
        return int(v) if v is not None else 1
    except Exception:
        return 1


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
    key_col = "term_uid" if "term_uid" in distilled.columns else "term_id"

    term_key = distilled[key_col].astype(str).str.strip()
    term_key = term_key[~term_key.isin(_NA_TOKENS)]
    term_key = term_key[~term_key.str.lower().isin(_NA_TOKENS)]
    known_terms = set(term_key.tolist())

    has_survival = "term_survival" in distilled.columns
    surv = None
    if has_survival:
        tmp = distilled.copy()
        tmp["term_key"] = tmp[key_col].astype(str).str.strip()
        tmp["term_survival"] = pd.to_numeric(tmp["term_survival"], errors="coerce")
        surv = tmp.groupby("term_key")["term_survival"].min()
    else:
        surv = None

    tau_default = _default_tau(card)

    # Precompute evidence genes per term_key for drift check
    term_to_gene_set: dict[str, set[str]] = {}
    if "evidence_genes" in distilled.columns:
        for tk, xs in zip(
            distilled[key_col].astype(str).str.strip(),
            distilled["evidence_genes"],
            strict=True,
        ):
            if tk in _NA_TOKENS or tk.lower() in _NA_TOKENS:
                continue
            if isinstance(xs, (list, tuple, set)):
                gs = {str(g).strip() for g in xs if str(g).strip()}
            else:
                gs = set()
            if not gs:
                continue
            if tk not in term_to_gene_set:
                term_to_gene_set[tk] = set()
            term_to_gene_set[tk].update(gs)
    elif "evidence_genes_str" in distilled.columns:
        for tk, s in zip(
            distilled[key_col].astype(str).str.strip(),
            distilled["evidence_genes_str"].astype(str),
            strict=True,
        ):
            if tk in _NA_TOKENS or tk.lower() in _NA_TOKENS:
                continue
            parts = [g.strip() for g in s.replace(";", ",").split(",") if g.strip()]
            gs = set(parts)
            if not gs:
                continue
            term_to_gene_set.setdefault(tk, set()).update(gs)

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

        gene_ids = _parse_ids(row.get("gene_ids", ""))

        min_overlap = _default_min_gene_overlap(card)

        gene_ids = _parse_ids(row.get("gene_ids", ""))
        if gene_ids:
            ev_set: set[str] = set()
            for t in term_ids:
                ev_set |= term_to_gene_set.get(t, set())

            if ev_set:
                n_hit = sum(1 for g in gene_ids if g in ev_set)
                if n_hit < min_overlap:
                    out.at[i, "status"] = "FAIL"
                    out.at[i, "link_ok"] = False
                    out.at[i, "fail_reason"] = FAIL_EVIDENCE_DRIFT
                    out.at[i, "audit_notes"] = (
                        f"gene_ids drift: hits={n_hit} < min_overlap={min_overlap}"
                    )
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
