from __future__ import annotations

import pandas as pd

from .audit_reasons import ABSTAIN_UNSTABLE, FAIL_EVIDENCE_DRIFT
from .sample_card import SampleCard


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

    known_terms = set(distilled["term_id"].astype(str))
    tau = 0.8  # placeholder; real calibration later

    # evidence-link integrity
    for i, row in out.iterrows():
        term_ids = [t for t in str(row.get("term_ids", "")).split(",") if t]
        if not term_ids:
            out.at[i, "status"] = "FAIL"
            out.at[i, "link_ok"] = False
            out.at[i, "fail_reason"] = FAIL_EVIDENCE_DRIFT
            continue
        if any(t not in known_terms for t in term_ids):
            out.at[i, "status"] = "FAIL"
            out.at[i, "link_ok"] = False
            out.at[i, "fail_reason"] = FAIL_EVIDENCE_DRIFT
            continue

        # stability gate (use per-term survival if available)
        # For now, approximate via the referenced term's survival (if columns exist).
        if "term_survival" in distilled.columns:
            t0 = term_ids[0]
            ts = float(
                distilled.loc[distilled["term_id"].astype(str) == t0, "term_survival"].iloc[0]
            )
            if ts < tau:
                out.at[i, "status"] = "ABSTAIN"
                out.at[i, "stability_ok"] = False
                out.at[i, "abstain_reason"] = ABSTAIN_UNSTABLE

    return out
