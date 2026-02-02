# LLM-PathwayCurator/src/llm_pathway_curator/audit_reasons.py
from __future__ import annotations

from typing import Final, Literal

# Keep these stable: they are part of the paper's reproducible output.

# FAIL reasons (auditable violations)
FAIL_EVIDENCE_DRIFT: Final[str] = "evidence_drift"
FAIL_SCHEMA_VIOLATION: Final[str] = "schema_violation"
FAIL_CONTRADICTION: Final[str] = "contradiction"
FAIL_CONTEXT: Final[str] = "context_fail"

# ABSTAIN reasons (insufficient/unstable/unclear under audits)
ABSTAIN_UNSTABLE: Final[str] = "unstable"
ABSTAIN_MISSING_SURVIVAL: Final[str] = "missing_survival"
ABSTAIN_MISSING_EVIDENCE_GENES: Final[str] = "missing_evidence_genes"  # NEW
ABSTAIN_CONTEXT_NONSPECIFIC: Final[str] = "context_nonspecific"
ABSTAIN_CONTEXT_MISSING: Final[str] = "context_missing"
ABSTAIN_UNDER_SUPPORTED: Final[str] = "under_supported"
ABSTAIN_HUB_BRIDGE: Final[str] = "hub_bridge"
ABSTAIN_INCONCLUSIVE_STRESS: Final[str] = "inconclusive_stress"

# Stable enumerations (useful for reports/tests)
FAIL_REASONS: Final[tuple[str, ...]] = (
    FAIL_EVIDENCE_DRIFT,
    FAIL_SCHEMA_VIOLATION,
    FAIL_CONTRADICTION,
    FAIL_CONTEXT,
)

ABSTAIN_REASONS: Final[tuple[str, ...]] = (
    ABSTAIN_UNSTABLE,
    ABSTAIN_MISSING_SURVIVAL,
    ABSTAIN_MISSING_EVIDENCE_GENES,  # NEW
    ABSTAIN_CONTEXT_NONSPECIFIC,
    ABSTAIN_CONTEXT_MISSING,
    ABSTAIN_UNDER_SUPPORTED,
    ABSTAIN_HUB_BRIDGE,
    ABSTAIN_INCONCLUSIVE_STRESS,
)

ALL_REASONS: Final[tuple[str, ...]] = FAIL_REASONS + ABSTAIN_REASONS

ReasonCode = Literal[
    "evidence_drift",
    "schema_violation",
    "contradiction",
    "context_fail",
    "unstable",
    "missing_survival",
    "missing_evidence_genes",  # NEW
    "context_nonspecific",
    "context_missing",
    "under_supported",
    "hub_bridge",
    "inconclusive_stress",
]

DecisionReasonCode = Literal[
    "ok",
    "evidence_drift",
    "schema_violation",
    "contradiction",
    "context_fail",
    "unstable",
    "missing_survival",
    "missing_evidence_genes",  # NEW
    "context_nonspecific",
    "context_missing",
    "under_supported",
    "hub_bridge",
    "inconclusive_stress",
]


def is_fail_reason(code: str) -> bool:
    """
    Check whether a reason code is a FAIL reason.

    Parameters
    ----------
    code : str
        Reason code string.

    Returns
    -------
    bool
        True if `code` is in `FAIL_REASONS`, otherwise False.

    Notes
    -----
    `FAIL_REASONS` is part of the paper's reproducible output contract and
    should remain stable.
    """
    return code in FAIL_REASONS


def is_abstain_reason(code: str) -> bool:
    """
    Check whether a reason code is an ABSTAIN reason.

    Parameters
    ----------
    code : str
        Reason code string.

    Returns
    -------
    bool
        True if `code` is in `ABSTAIN_REASONS`, otherwise False.

    Notes
    -----
    `ABSTAIN_REASONS` is part of the paper's reproducible output contract and
    should remain stable.
    """
    return code in ABSTAIN_REASONS


def is_known_reason(code: str) -> bool:
    """
    Check whether a reason code is known by this module.

    Parameters
    ----------
    code : str
        Reason code string.

    Returns
    -------
    bool
        True if `code` is in `ALL_REASONS`, otherwise False.

    Notes
    -----
    `ALL_REASONS` excludes "ok" by design. Use `is_decision_reason()` when
    you want to accept the "ok" sentinel.
    """
    return code in ALL_REASONS


def is_decision_reason(code: str) -> bool:
    """
    Check whether a string is a valid decision reason code.

    This includes the sentinel "ok" as well as all known FAIL/ABSTAIN
    reason codes.

    Parameters
    ----------
    code : str
        Decision reason code.

    Returns
    -------
    bool
        True if `code` is "ok" or is included in `ALL_REASONS`, otherwise
        False.
    """
    return (code == "ok") or (code in ALL_REASONS)
