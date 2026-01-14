# LLM-PathwayCurator/src/llm_pathway_curator/audit.py
from __future__ import annotations

from typing import Any

import pandas as pd

from .audit_reasons import ABSTAIN_UNSTABLE, FAIL_EVIDENCE_DRIFT
from .sample_card import SampleCard


def _parse_ids(x: Any) -> list[str]:
    if x is None:
        return []
    s = str(x).strip()
    if not s or s.lower() in {"na", "nan", "none"}:
        return []
    s = s.replace(";", ",")
    out = [t.strip() for t in s.split(",") if t.strip()]
    # de-dup while preserving order
    seen: set[str] = set()
    uniq: list[str] = []
    for t in out:
        if t not in seen:
            seen.add(t)
            uniq.append(t)
    return uniq


def audit_claims(claims: pd.DataFrame, distilled: pd.DataFrame, card: SampleCard) -> pd.DataFrame:
    """
    Minimal mechanical audit (v0 scaffold):
      1) evidence-link integrity: term_ids must exist in EvidenceTable
      2) stability gate: term_survival >= tau else ABSTAIN
    """
    out = claims.copy()

    # defaults
    out["status"] = "PASS"
    out["link_ok"] = True
    out["stability_ok"] = True
    out["contradiction_ok"] = True
    out["stress_ok"] = True
    out["abstain_reason"] = ""
    out["fail_reason"] = ""
    out["audit_notes"] = ""

    # prepare distilled lookup
    if "term_id" not in distilled.columns:
        raise ValueError("audit_claims: distilled missing column term_id")

    term_ids_series = distilled["term_id"].astype(str).str.strip()
    known_terms = set(term_ids_series.tolist())

    # survival lookup (robust to duplicates)
    has_survival = "term_survival" in distilled.columns
    if has_survival:
        surv = (
            distilled.assign(term_id_str=term_ids_series)
            .groupby("term_id_str")["term_survival"]
            .max()  # conservative: if any replicate survives, keep high; adjust later if needed
        )
    else:
        surv = None

    tau = 0.8  # placeholder; real calibration later

    for i, row in out.iterrows():
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

        # 2) stability gate (ABSTAIN)
        if has_survival and surv is not None:
            t0 = term_ids[0]
            ts = float(surv.get(t0, float("nan")))
            if pd.isna(ts):
                out.at[i, "status"] = "ABSTAIN"
                out.at[i, "stability_ok"] = False
                out.at[i, "abstain_reason"] = ABSTAIN_UNSTABLE
                out.at[i, "audit_notes"] = "term_survival is NA"
                continue
            if ts < tau:
                out.at[i, "status"] = "ABSTAIN"
                out.at[i, "stability_ok"] = False
                out.at[i, "abstain_reason"] = ABSTAIN_UNSTABLE
                out.at[i, "audit_notes"] = f"term_survival={ts:.3f} < tau={tau:.2f}"
                continue

    return out
