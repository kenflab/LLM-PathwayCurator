# LLM-PathwayCurator/src/llm_pathway_curator/audit.py
from __future__ import annotations

import hashlib
from typing import Any

import pandas as pd

from .audit_reasons import (
    ABSTAIN_INCONCLUSIVE_STRESS,
    ABSTAIN_MISSING_SURVIVAL,
    ABSTAIN_UNSTABLE,
    FAIL_EVIDENCE_DRIFT,
    FAIL_SCHEMA_VIOLATION,
)
from .claim_schema import Claim
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


def _hash_gene_set(genes: list[str]) -> str:
    """
    Stable short hash for an ordered gene list (v0).
    NOTE: we keep order as produced by our union builder (deterministic).
    """
    payload = ",".join([str(g).strip() for g in genes if str(g).strip()])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def _get_tau_default(card: SampleCard) -> float:
    """
    Source of truth:
      - SampleCard.audit_tau() if present
      - else: legacy extra/attribute fallback
    """
    if hasattr(card, "audit_tau") and callable(card.audit_tau):
        try:
            return float(card.audit_tau())  # type: ignore[attr-defined]
        except Exception:
            pass

    v = getattr(card, "audit_tau", None)
    if v is not None:
        try:
            return float(v)
        except Exception:
            pass

    try:
        extra = getattr(card, "extra", {}) or {}
        if isinstance(extra, dict) and "audit_tau" in extra:
            return float(extra["audit_tau"])
    except Exception:
        pass

    return 0.8


def _get_min_overlap_default(card: SampleCard) -> int:
    """
    Source of truth:
      - SampleCard.audit_min_gene_overlap() if present
      - else: legacy extra fallback
    """
    if hasattr(card, "audit_min_gene_overlap") and callable(card.audit_min_gene_overlap):
        try:
            return int(card.audit_min_gene_overlap())  # type: ignore[attr-defined]
        except Exception:
            pass

    try:
        extra = getattr(card, "extra", {}) or {}
        if isinstance(extra, dict) and "audit_min_gene_overlap" in extra:
            return int(extra["audit_min_gene_overlap"])
    except Exception:
        pass

    return 1


def audit_claims(claims: pd.DataFrame, distilled: pd.DataFrame, card: SampleCard) -> pd.DataFrame:
    """
    Mechanical audit (v0, spec-aligned):
      0) Schema validation: claim_json must validate against Claim schema
         - missing/invalid claim_json => FAIL (schema_violation)
      1) Evidence-link integrity: term_ids must exist in distilled EvidenceTable
         - empty or unknown term_ids => FAIL (evidence_drift)
         - if gene_set_hash is provided: hash(union evidence genes) must match
           => FAIL (evidence_drift)
         - else: overlap(gene_ids, union evidence genes) >= min_overlap => FAIL (evidence_drift)
      2) Stability gate: aggregate(term_survival over referenced term_ids) >= tau
         - aggregate = min(term_survival)  (conservative)
         - if term_survival column missing => ABSTAIN (missing_survival)
         - if referenced terms have missing survival => ABSTAIN (missing_survival)
         - if agg < tau => ABSTAIN (unstable)
      3) v0 guardrail: if advanced audits (stress/contradiction) are not implemented,
         do NOT return PASS; ABSTAIN instead (inconclusive_stress)

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

    tau_default = _get_tau_default(card)
    min_overlap_default = _get_min_overlap_default(card)

    # Precompute evidence genes per term_key for drift check (robust to list OR string)
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
                genes = [str(g).strip() for g in xs if str(g).strip()]
            else:
                genes = _parse_ids(xs)
            gs = set(genes)
            if not gs:
                continue
            term_to_gene_set.setdefault(tk, set()).update(gs)

    elif "evidence_genes_str" in distilled.columns:
        for tk, s in zip(
            distilled[key_col].astype(str).str.strip(),
            distilled["evidence_genes_str"].astype(str),
            strict=True,
        ):
            if tk in _NA_TOKENS or tk.lower() in _NA_TOKENS:
                continue
            genes = _parse_ids(s)
            gs = set(genes)
            if not gs:
                continue
            term_to_gene_set.setdefault(tk, set()).update(gs)

    for i, row in out.iterrows():
        # 0) schema validation (claim_json -> Claim)
        cj = row.get("claim_json", None)
        if cj is None or _is_na_scalar(cj) or (not str(cj).strip()):
            out.at[i, "status"] = "FAIL"
            out.at[i, "link_ok"] = False
            out.at[i, "fail_reason"] = FAIL_SCHEMA_VIOLATION
            out.at[i, "audit_notes"] = "missing claim_json"
            continue
        try:
            Claim.model_validate_json(str(cj))
        except Exception as e:
            out.at[i, "status"] = "FAIL"
            out.at[i, "link_ok"] = False
            out.at[i, "fail_reason"] = FAIL_SCHEMA_VIOLATION
            out.at[i, "audit_notes"] = f"claim_json schema violation: {type(e).__name__}"
            continue

        # Per-claim tau override (optional)
        tau = tau_default
        if "tau" in out.columns and not _is_na_scalar(row.get("tau")):
            try:
                tau = float(row.get("tau"))
            except Exception:
                tau = tau_default

        # Per-claim min_overlap override (optional)
        min_overlap = min_overlap_default
        if "min_overlap" in out.columns and not _is_na_scalar(row.get("min_overlap")):
            try:
                min_overlap = int(row.get("min_overlap"))
            except Exception:
                min_overlap = min_overlap_default

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

        # union evidence genes across referenced terms (deterministic order)
        ev_union: list[str] = []
        ev_seen: set[str] = set()
        for t in term_ids:
            for g in sorted(term_to_gene_set.get(t, set())):
                if g not in ev_seen:
                    ev_seen.add(g)
                    ev_union.append(g)

        # 1b) stronger drift check if gene_set_hash is provided
        gsh = row.get("gene_set_hash", None)
        if gsh is not None and (not _is_na_scalar(gsh)) and str(gsh).strip():
            if ev_union:
                computed = _hash_gene_set(ev_union)
                gsh_norm = str(gsh).strip().lower()
                if computed != gsh_norm:
                    out.at[i, "status"] = "FAIL"
                    out.at[i, "link_ok"] = False
                    out.at[i, "fail_reason"] = FAIL_EVIDENCE_DRIFT
                    out.at[i, "audit_notes"] = f"gene_set_hash mismatch: {computed} != {gsh_norm}"
                    continue
            # if we cannot compute union, fall back to overlap

        # 1c) fallback overlap check (representative genes)
        if gene_ids and ev_seen:
            n_hit = sum(1 for g in gene_ids if g in ev_seen)
            if n_hit < int(min_overlap):
                out.at[i, "status"] = "FAIL"
                out.at[i, "link_ok"] = False
                out.at[i, "fail_reason"] = FAIL_EVIDENCE_DRIFT
                out.at[i, "audit_notes"] = (
                    f"gene_ids drift: hits={n_hit} < min_overlap={min_overlap}"
                )
                continue

        # 2) stability gate
        if (not has_survival) or (surv is None):
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

        if agg < float(tau):
            out.at[i, "status"] = "ABSTAIN"
            out.at[i, "stability_ok"] = False
            out.at[i, "abstain_reason"] = ABSTAIN_UNSTABLE
            out.at[i, "audit_notes"] = f"term_survival_agg={agg:.3f} < tau={float(tau):.2f}"
            continue

        # 3) v0 guardrail: if advanced audits are not implemented, do NOT return PASS
        # This keeps the paper's claim honest: "mechanical audits decide PASS/ABSTAIN/FAIL".
        if out.at[i, "status"] == "PASS":
            out.at[i, "status"] = "ABSTAIN"
            out.at[i, "stress_ok"] = False
            out.at[i, "abstain_reason"] = ABSTAIN_INCONCLUSIVE_STRESS
            out.at[i, "audit_notes"] = "v0: stress/contradiction audits not implemented"

    return out
