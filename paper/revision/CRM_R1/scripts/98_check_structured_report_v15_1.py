#!/usr/bin/env python3
"""Development contract for faithful structured reporting, not pathway plausibility."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd
from v15_1_provenance_common import (
    code_records,
    finish,
    fresh_output,
    read_json,
    record,
    require,
    write_table,
)


def number(value, label):
    require(
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and 0 <= value <= 1,
        f"Invalid {label}",
    )
    return value


def genes(value, label):
    require(
        isinstance(value, list)
        and bool(value)
        and all(isinstance(v, str) and v.strip() == v and v for v in value)
        and len(set(value)) == len(value),
        f"Invalid {label}",
    )
    return set(value)


def check_record(item):
    require(
        isinstance(item, dict) and set(item) == {"case_id", "evidence", "claim"},
        "Expected case_id, evidence, claim only",
    )
    require(
        isinstance(item["case_id"], str) and bool(item["case_id"].strip()),
        "Missing case ID",
    )
    evidence, claim = item["evidence"], item["claim"]
    required_e = {
        "evidence_id",
        "cohort_id",
        "comparison",
        "term_uid",
        "direction",
        "q_value",
        "evidence_genes",
    }
    required_c = {
        "evidence_id",
        "cohort_id",
        "comparison",
        "term_uid",
        "direction",
        "reported_q_value",
        "supporting_genes",
        "asserts_fdr_significance",
    }
    require(
        isinstance(evidence, dict) and set(evidence) == required_e,
        "Evidence schema mismatch",
    )
    require(isinstance(claim, dict) and set(claim) == required_c, "Claim schema mismatch")
    fields = ["evidence_id", "cohort_id", "comparison", "term_uid", "direction"]
    for source in (evidence, claim):
        for key in fields:
            value = source[key]
            require(
                isinstance(value, str) and value.strip() == value and bool(value),
                f"Invalid {key}",
            )
        require(source["direction"] in {"UP", "DOWN"}, "Direction must be UP or DOWN")
    q = number(evidence["q_value"], "evidence q-value")
    reported_q = number(claim["reported_q_value"], "reported q-value")
    truth_genes = genes(evidence["evidence_genes"], "evidence genes")
    claim_genes = genes(claim["supporting_genes"], "supporting genes")
    require(
        isinstance(claim["asserts_fdr_significance"], bool),
        "Significance assertion must be bool",
    )
    reasons = ["MISMATCH_" + key.upper() for key in fields if claim[key] != evidence[key]]
    if reported_q != q:
        reasons.append("Q_VALUE_TRANSCRIPTION")
    if not claim_genes <= truth_genes:
        reasons.append("GENE_NOT_IN_SUPPLIED_EVIDENCE")
    if claim["asserts_fdr_significance"] and q > 0.05:
        reasons.append("UNSUPPORTED_FDR_SIGNIFICANCE_AT_0_05")
    return {
        "case_id": item["case_id"],
        "contract_status": "CONTRACT_VIOLATION" if reasons else "NO_STRUCTURED_VIOLATION_DETECTED",
        "reason_codes": ";".join(reasons),
        "context_relevance": "NOT_ASSESSED",
        "biological_correctness": "NOT_ASSESSED",
        "free_text_overstatement": "NOT_ASSESSED",
        "membership_or_ranking_changed": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="JSON array; evidence fields must come from an external canonical source",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    inputs = [record(args.input)] + code_records(__file__)
    items = read_json(args.input)
    require(isinstance(items, list) and bool(items), "Expected a nonempty JSON array")
    result = pd.DataFrame([check_record(item) for item in items])
    require(not result.case_id.duplicated().any(), "Duplicate case ID")
    with fresh_output(args.output) as out:
        write_table(out / "structured_contract_checks.private.tsv", result)
        finish(
            out,
            inputs,
            {
                "rows": len(result),
                "scope": "structured reporting fidelity only",
                "gold_standard_biology": False,
                "canonical_source_independently_authenticated": False,
                "fdr_threshold": 0.05,
                "llm_context_plausibility_is_a_blocking_rule": False,
                "new_full_audit_method": False,
                "membership_exported": False,
            },
        )
    print("[PASS] Structured checks written; inspect each contract_status:", args.output)


if __name__ == "__main__":
    main()
