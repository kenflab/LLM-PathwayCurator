from __future__ import annotations

import pandas as pd

from .sample_card import SampleCard


def distill_evidence(evidence: pd.DataFrame, card: SampleCard) -> pd.DataFrame:
    """
    Minimal placeholder:
    - ensure required columns exist
    - add survival columns as stubs
    """
    required = {"term_id", "term_name", "source", "stat", "qval", "direction", "evidence_genes"}
    missing = sorted(required - set(evidence.columns))
    if missing:
        raise ValueError(f"EvidenceTable missing columns: {missing}")

    out = evidence.copy()
    # placeholder survival (real impl: LOO/jackknife)
    out["term_survival"] = 1.0
    out["gene_survival"] = 1.0
    out["module_survival"] = 1.0
    return out
