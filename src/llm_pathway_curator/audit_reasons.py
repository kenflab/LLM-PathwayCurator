# LLM-PathwayCurator/src/llm_pathway_curator/audit_reasons.py
from __future__ import annotations

# Keep these stable: they are part of the paper's reproducible output.

# FAIL reasons (auditable violations)
FAIL_EVIDENCE_DRIFT = "evidence_drift"
FAIL_SCHEMA_VIOLATION = "schema_violation"
FAIL_CONTRADICTION = "contradiction"

# ABSTAIN reasons (insufficient/unstable/unclear under audits)
ABSTAIN_UNSTABLE = "unstable"
ABSTAIN_MISSING_SURVIVAL = "missing_survival"
ABSTAIN_CONTEXT_NONSPECIFIC = "context_nonspecific"
ABSTAIN_UNDER_SUPPORTED = "under_supported"
ABSTAIN_HUB_BRIDGE = "hub_bridge"
ABSTAIN_INCONCLUSIVE_STRESS = "inconclusive_stress"

# Stable enumerations (useful for reports/tests)
FAIL_REASONS = (
    FAIL_EVIDENCE_DRIFT,
    FAIL_SCHEMA_VIOLATION,
    FAIL_CONTRADICTION,
)

ABSTAIN_REASONS = (
    ABSTAIN_UNSTABLE,
    ABSTAIN_MISSING_SURVIVAL,
    ABSTAIN_CONTEXT_NONSPECIFIC,
    ABSTAIN_UNDER_SUPPORTED,
    ABSTAIN_HUB_BRIDGE,
    ABSTAIN_INCONCLUSIVE_STRESS,
)

ALL_REASONS = FAIL_REASONS + ABSTAIN_REASONS
