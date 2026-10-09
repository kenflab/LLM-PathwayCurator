"""Limited, source-based checks on enrichment prose; no semantic truth labels.

All offsets refer to the unchanged submitted text. A clear limited check never
approves free prose. This module has no model transport or study-specific registry.
"""

from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation
from typing import Any

RULESET = "enrichment-source-checks/1.2"
_NUMBER = r"[-+−]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+\-−]?\d+)?"
_NUMERIC = re.compile(
    r"\b(?P<kind>NES|ES|q(?:[-_ ]?value)?|qval|padj|FDR|"
    r"adjusted\s+(?:p(?:[- ]?value)?|value)|stat(?:istic)?)\b\s*[\])]?[ ]*"
    r"(?:=|:|\bis\b|\bwas\b|\bof\b)?\s*"
    r"(?P<op>[<>≤≥]=?)?\s*(?P<value>" + _NUMBER + r""
    r"(?:\s*[×x]\s*10\s*(?:\^|\*\*)\s*[-+−]?\d+)?)"
    r"(?P<percent>[ \t]*%)?",
    re.I,
)
_SIGNIFICANCE = re.compile(
    r"\b(?P<neg>not\s+|non[- ]?)?"
    r"(?:statistically\s+significant|significant\s+"
    r"(?:enrichment|association|difference|change))\b",
    re.I,
)
_DIRECTION = re.compile(r"\b(?P<pol>positive(?:ly)?|negative(?:ly)?)\s+enrich(?:ment|ed)\b", re.I)
_CAUTION = re.compile(
    r"\b(?:causes?|caused|drives?|driven|leads?\s+to|results?\s+in|"
    r"abolishes?|activates?|inactivates?|mechanistically|proves?|"
    r"(?:up|down)[- ]regulated|(?:more|less)\s+active|"
    r"clinical(?:ly)?\s+(?:actionable|effective|beneficial)|"
    r"therapeutic\s+(?:target|benefit|efficacy))\b",
    re.I,
)
_BOUNDARY = re.compile(r"[;.!?]\s+|\b(?:but|however|and|whereas)\b", re.I)
_NEGATION = re.compile(
    r"\b(?:no|not(?!\s+only\b)|never|cannot|can't|doesn't|didn't|without|insufficient)\b",
    re.I,
)
_UNADJUSTED_BASIS = re.compile(r"\b(?:unadjusted|uncorrected|nominal(?:ly)?)\b", re.I)


def _local_clause(text: str, start: int, end: int) -> str:
    """Bound a significance cue by the existing conservative clause separators."""
    before = list(_BOUNDARY.finditer(text[:start]))
    after = _BOUNDARY.search(text, end)
    left = before[-1].end() if before else 0
    right = after.start() if after else len(text)
    return text[left:right]


def decimal_number(value: str) -> Decimal:
    cleaned = value.replace("−", "-").strip()
    match = re.fullmatch(r"(.+?)\s*[×x]\s*10\s*(?:\^|\*\*)\s*([-+]?\d+)", cleaned)
    if match:
        result = Decimal(match[1]) * Decimal(10) ** int(match[2])
    else:
        result = Decimal(cleaned)
    if not result.is_finite():
        raise ValueError("Expected a finite number")
    return result


def _negated(text: str, start: int) -> bool:
    prefix = text[:start]
    boundaries = list(_BOUNDARY.finditer(prefix))
    clause = prefix[boundaries[-1].end() :] if boundaries else prefix
    return bool(_NEGATION.search(clause[-100:]))


def _numeric_match(reported: Decimal, expected: Decimal, operator: str) -> bool:
    if operator in {"<", "<=", "≤", ">", ">=", "≥"}:
        return {
            "<": expected < reported,
            "<=": expected <= reported,
            "≤": expected <= reported,
            ">": expected > reported,
            ">=": expected >= reported,
            "≥": expected >= reported,
        }[operator]
    if reported == 0:
        return expected == 0
    half_unit = Decimal(1).scaleb(reported.as_tuple().exponent) / 2
    return abs(reported - expected) <= half_unit


def inspect_text(text: str, evidence: dict[str, Any], cutoff: float) -> dict[str, Any]:
    """Return span-level findings; retain original wording and missing coverage.

    Numeric rounding uses half a unit of the final displayed digit. Adjusted
    values written with a percent sign are scaled with their precision retained.
    Inequalities are checked as inequalities. Ambiguous negation asks for review
    instead of establishing a significance violation. Unqualified p values are
    never treated as q values.
    """
    findings: list[dict[str, Any]] = []
    covered: set[str] = set()

    def add(match, code: str, severity: str, message: str, *, expected=None):
        findings.append(
            {
                "code": code,
                "severity": severity,
                "start": match.start(),
                "end": match.end(),
                "quote": text[match.start() : match.end()],
                "message": message,
                "source_value": expected,
            }
        )

    for match in _NUMERIC.finditer(text):
        kind = match["kind"].lower()
        field = "stat" if kind in {"nes", "es"} or kind.startswith("stat") else "qval"
        if match["percent"] and field != "qval":
            add(
                match,
                "NUMERIC_UNIT_UNVERIFIED",
                "REVIEW",
                "A percent unit cannot be assumed for the source statistic.",
            )
            continue
        if kind in {"nes", "es"} and str(evidence.get("stat_kind", "")).upper() != kind.upper():
            add(
                match,
                "STATISTIC_TYPE_UNVERIFIED",
                "REVIEW",
                f"The source statistic is not identified as {kind.upper()}.",
            )
            continue
        value = evidence.get(field)
        if value is None:
            add(match, "SOURCE_VALUE_UNAVAILABLE", "REVIEW", "The source value is not estimable.")
            continue
        covered.add(field)
        try:
            reported = decimal_number(match["value"])
            if match["percent"]:
                reported = reported.scaleb(-2)
            expected = Decimal(str(value))
            satisfied = _numeric_match(reported, expected, match["op"] or "=")
        except (InvalidOperation, ValueError, OverflowError):
            add(
                match,
                "NUMERIC_EXPRESSION_UNRESOLVED",
                "REVIEW",
                "Could not resolve this numeric expression.",
            )
            continue
        if not satisfied:
            add(
                match,
                "NUMERIC_MISMATCH",
                "ERROR",
                "The stated number or bound disagrees with the source.",
                expected=value,
            )

    for match in _SIGNIFICANCE.finditer(text):
        negative = bool(match["neg"])
        qval = evidence.get("qval")
        if _UNADJUSTED_BASIS.search(_local_clause(text, match.start(), match.end())):
            add(
                match,
                "SIGNIFICANCE_BASIS_REQUIRES_REVIEW",
                "REVIEW",
                "This clause refers to nominal or unadjusted significance. "
                "The source adjusted value alone cannot verify that statement.",
                expected=qval,
            )
        elif qval is None:
            add(
                match,
                "SIGNIFICANCE_UNVERIFIED",
                "REVIEW",
                "No estimable adjusted value is available.",
            )
        elif not negative and _negated(text, match.start()):
            add(
                match,
                "SIGNIFICANCE_SCOPE_REQUIRES_REVIEW",
                "REVIEW",
                "An earlier negation makes the scope of the significance wording ambiguous; "
                "inspect the sentence instead of assigning a significance violation.",
                expected=qval,
            )
        elif (not negative) != (qval <= cutoff):
            add(
                match,
                "SIGNIFICANCE_MISMATCH",
                "ERROR",
                f"The wording disagrees with the declared adjusted-value cutoff {cutoff:g}.",
                expected=qval,
            )

    for match in _DIRECTION.finditer(text):
        if _negated(text, match.start()):
            continue
        expected = evidence.get("direction")
        if expected not in {"up", "down"}:
            add(
                match,
                "DIRECTION_UNVERIFIED",
                "REVIEW",
                "The source has no signed enrichment direction.",
            )
        elif (match["pol"].lower().startswith("positive")) != (expected == "up"):
            add(
                match,
                "DIRECTION_MISMATCH",
                "ERROR",
                "The signed enrichment wording disagrees with the source.",
                expected=expected,
            )

    for match in _CAUTION.finditer(text):
        if not _negated(text, match.start()):
            add(
                match,
                "INFERENTIAL_LANGUAGE_REQUIRES_REVIEW",
                "REVIEW",
                "Inspect this mechanistic, causal or clinical wording against the study design; "
                "enrichment alone does not establish it.",
            )

    return {
        "findings": findings,
        "explicit_numeric_fields": sorted(covered),
        "numeric_coverage": "COMPLETE_EXPLICIT_STAT_Q"
        if covered == {"stat", "qval"}
        else "PARTIAL_OR_ABSENT",
        "limited_checks_status": "VIOLATION_DETECTED"
        if any(x["severity"] == "ERROR" for x in findings)
        else "REVIEW_FLAGGED"
        if findings
        else "NO_LIMITED_FLAG",
        "semantic_correctness": "NOT_ESTABLISHED",
        "automatic_prose_acceptance": False,
    }


def factual_statement(evidence: dict[str, Any], comparison: str, cutoff: float) -> str:
    """Render source statistics without a mechanistic or gene-expression claim."""
    term = evidence["term_name"]
    stat, qval = evidence["stat"], evidence["qval"]
    label = evidence["stat_kind"]
    numeric = f"{label}={stat:.6g}" if stat is not None else f"{label}=not estimable"
    adjusted = f"adjusted value={qval:.6g}" if qval is not None else "adjusted value=not estimable"
    if evidence["direction"] in {"up", "down"}:
        sign = "positive" if evidence["direction"] == "up" else "negative"
        description = f"The source records {sign} enrichment"
    else:
        description = "The source records an enrichment result without a signed direction"
    if qval is None:
        threshold = "The adjusted-value cutoff cannot be evaluated."
    elif qval <= cutoff:
        threshold = f"This meets the declared adjusted-value cutoff {cutoff:g}."
    else:
        threshold = f"This does not meet the declared adjusted-value cutoff {cutoff:g}."
    return f"{term}: {description} for {comparison} ({numeric}; {adjusted}). {threshold}"
