"""Conservative checks of explicit numeric syntax; not a general prose parser."""

import re
from decimal import Decimal

from .models import Evidence

NUMBER = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
EXPLICIT = re.compile(
    rf"(?<![\w-])(?P<metric>NES|(?:BH\s+)?q(?:[- ]value)?)\s*"
    rf"(?P<op><=|>=|=|<|>|≤|≥)\s*(?P<value>{NUMBER})(?!\w|\.\d)",
    re.IGNORECASE,
)
# Only these complete forms have deterministic significance semantics.
SIGNIFICANCE = re.compile(
    rf"\b(?P<neg>not\s+)?statistically\s+significant\s+at\s+"
    rf"FDR\s*(?:=\s*)?(?P<value>{NUMBER})(?!\w|\.\d)",
    re.IGNORECASE,
)


def explicit_numeric_checks(text: str, evidence: Evidence) -> list[dict]:
    records = []
    for match in EXPLICIT.finditer(text):
        metric = "nes" if match["metric"].upper() == "NES" else "q_value"
        canonical = Decimal(str(getattr(evidence, metric)))
        written = Decimal(match["value"])
        op = match["op"].replace("≤", "<=").replace("≥", ">=")
        satisfied = {
            "=": canonical == written,
            "<": canonical < written,
            "<=": canonical <= written,
            ">": canonical > written,
            ">=": canonical >= written,
        }[op]
        # Nonidentical rounding is not silently accepted or called biological contradiction.
        records.append(
            {
                "code": "TEXT_EXPLICIT_NUMERIC_MISMATCH" if not satisfied else "MATCH",
                "metric": metric,
                "quote": match.group(),
                "start": match.start(),
                "end": match.end(),
                "canonical": str(canonical),
                "operator": op,
                "literal": match["value"],
                "satisfied": satisfied,
                "scope": "explicit_numeric_relation_exact_decimal_comparison",
            }
        )
    for match in SIGNIFICANCE.finditer(text):
        threshold = Decimal(match["value"])
        # The scientific FDR criterion is 0.05, not a threshold chosen by prose.
        if threshold != Decimal("0.05"):
            continue
        claims_significance = not bool(match["neg"])
        satisfied = claims_significance == (Decimal(str(evidence.q_value)) <= threshold)
        records.append(
            {
                "code": "TEXT_FDR_SIGNIFICANCE_MISMATCH" if not satisfied else "MATCH",
                "quote": match.group(),
                "start": match.start(),
                "end": match.end(),
                "canonical_q": str(evidence.q_value),
                "threshold": str(threshold),
                "satisfied": satisfied,
                "scope": "literal_statistically_significant_at_FDR_0_05_form",
            }
        )
    return records


def factual_statement(evidence: Evidence) -> str:
    if evidence.nes > 0:
        direction = "toward TP53-mutant samples"
    elif evidence.nes < 0:
        direction = "toward TP53-wild-type samples"
    else:
        direction = "with no enrichment direction (NES=0)"
    significance = "meets" if evidence.q_value <= 0.05 else "does not meet"
    return (
        f"In TCGA-{evidence.cohort_id} ({evidence.cohort_name}), "
        f"gene set {evidence.term_id} has NES={evidence.nes!r} {direction} "
        "in the observational TP53-mutant versus TP53-wild-type comparison "
        f"(BH q={evidence.q_value!r}; {significance} FDR <= 0.05). "
        "This enrichment record does not establish a causal TP53 effect, "
        "disease specificity, or occurrence of the event named by the gene set."
    )
