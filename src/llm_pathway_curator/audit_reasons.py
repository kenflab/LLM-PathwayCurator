# LLM-PathwayCurator/src/llm_pathway_curator/audit_reasons.py
from __future__ import annotations

# Keep these stable: they are part of the paper's reproducible output.
FAIL_EVIDENCE_DRIFT = "evidence_drift"
FAIL_SCHEMA_VIOLATION = "schema_violation"
FAIL_CONTRADICTION = "contradiction"

ABSTAIN_UNSTABLE = "unstable"
ABSTAIN_CONTEXT_NONSPECIFIC = "context_nonspecific"
ABSTAIN_UNDER_SUPPORTED = "under_supported"
ABSTAIN_HUB_BRIDGE = "hub_bridge"
ABSTAIN_INCONCLUSIVE_STRESS = "inconclusive_stress"
ABSTAIN_MISSING_SURVIVAL = "missing_survival"
