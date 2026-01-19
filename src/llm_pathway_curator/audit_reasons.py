# LLM-PathwayCurator/src/llm_pathway_curator/audit_reasons.py
from __future__ import annotations

from typing import Final, Literal

# Keep these stable: they are part of the paper's reproducible output.

# FAIL reasons (auditable violations)
FAIL_EVIDENCE_DRIFT: Final[str] = "evidence_drift"
FAIL_SCHEMA_VIOLATION: Final[str] = "schema_violation"
FAIL_CONTRADICTION: Final[str] = "contradiction"

# ABSTAIN reasons (insufficient/unstable/unclear under audits)
ABSTAIN_UNSTABLE: Final[str] = "unstable"
ABSTAIN_MISSING_SURVIVAL: Final[str] = "missing_survival"
ABSTAIN_CONTEXT_NONSPECIFIC: Final[str] = "context_nonspecific"
ABSTAIN_UNDER_SUPPORTED: Final[str] = "under_supported"
ABSTAIN_HUB_BRIDGE: Final[str] = "hub_bridge"
ABSTAIN_INCONCLUSIVE_STRESS: Final[str] = "inconclusive_stress"

# Stable enumerations (useful for reports/tests)
FAIL_REASONS: Final[tuple[str, ...]] = (
    FAIL_EVIDENCE_DRIFT,
    FAIL_SCHEMA_VIOLATION,
    FAIL_CONTRADICTION,
)

ABSTAIN_REASONS: Final[tuple[str, ...]] = (
    ABSTAIN_UNSTABLE,
    ABSTAIN_MISSING_SURVIVAL,
    ABSTAIN_CONTEXT_NONSPECIFIC,
    ABSTAIN_UNDER_SUPPORTED,
    ABSTAIN_HUB_BRIDGE,
    ABSTAIN_INCONCLUSIVE_STRESS,
)

ALL_REASONS: Final[tuple[str, ...]] = FAIL_REASONS + ABSTAIN_REASONS

# Optional but useful: strict type for reason codes used in report/audit schema.
ReasonCode = Literal[
    "evidence_drift",
    "schema_violation",
    "contradiction",
    "unstable",
    "missing_survival",
    "context_nonspecific",
    "under_supported",
    "hub_bridge",
    "inconclusive_stress",
]


def is_fail_reason(code: str) -> bool:
    return code in FAIL_REASONS


def is_abstain_reason(code: str) -> bool:
    return code in ABSTAIN_REASONS


def is_known_reason(code: str) -> bool:
    return code in ALL_REASONS
