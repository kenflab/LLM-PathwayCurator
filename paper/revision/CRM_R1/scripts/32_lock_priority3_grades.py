#!/usr/bin/env python3
"""Validate and immutably lock completed, method-blinded Priority 3 grades."""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from v11_lock_common import (
    file_record,
    normalized,
    read_json,
    require,
    sha256_file,
    write_json,
)

CRM_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = CRM_DIR.parents[2]
DEFAULT_PROTOCOL = CRM_DIR / "config" / "priority3_protocol.json"
DEFAULT_CHECKER = Path(__file__).resolve().with_name("31_check_priority3_retrieval.py")

GRADING_COLUMNS = [
    "eligible",
    "exclusion_reason",
    "evidence_grade",
    "direction_match",
    "context_match",
    "study_design",
    "data_overlap",
    "contradiction",
    "supporting_note",
    "curator_id",
]
FORBIDDEN_METHOD_COLUMNS = {
    "claim_uid",
    "audit_status",
    "status_full",
    "method_membership",
    "full_audit_selected",
    "q_value_matched_selected",
    "stability_matched_selected",
    "term_survival",
    "term_survival_agg",
    "context_status",
}
GRADE_RANK = {"E0": 0, "E1": 1, "E2": 2, "E3": 3, "E4": 4}


def run_retrieval_gate(checker: Path, data_root: Path) -> None:
    result = subprocess.run(
        [sys.executable, str(checker), "--data-root", str(data_root)],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    output = result.stdout + result.stderr
    print(output, end="")
    require(result.returncode == 0, "Priority 3 retrieval gate failed")
    require(
        "[GO] P4 blinded packet preparation" in output,
        "P3 checker did not confirm blinding",
    )


def validate_completed(
    completed: pd.DataFrame,
    frozen_blank: pd.DataFrame,
    protocol: dict,
) -> pd.DataFrame:
    require(
        list(completed.columns) == list(frozen_blank.columns),
        "P3 column schema changed",
    )
    require(
        not (set(completed.columns) & FORBIDDEN_METHOD_COLUMNS),
        "Method fields leaked into P3",
    )
    require(
        "screening_id" in completed and "review_id" in completed,
        "P3 identifiers are missing",
    )
    require(completed["screening_id"].is_unique, "P3 screening_id is not unique")
    require(len(completed) == len(frozen_blank), "P3 screening row count changed")
    fixed = [column for column in completed.columns if column not in GRADING_COLUMNS]
    left = normalized(completed.sort_values("screening_id"), fixed).reset_index(drop=True)
    right = normalized(frozen_blank.sort_values("screening_id"), fixed).reset_index(drop=True)
    require(
        left.equals(right),
        "P3 retrieved records or blinded claim fields changed during grading",
    )

    result = completed.copy()
    for column in GRADING_COLUMNS:
        result[column] = result[column].fillna("").astype(str).str.strip()
        require(result[column].ne("").all(), f"P3 grading field is incomplete: {column}")
    options = protocol["record_grading_fields"]
    for column, allowed in options.items():
        require(set(result[column]) <= set(allowed), f"Invalid P3 values in {column}")
    require(
        set(result["evidence_grade"]) <= {"EXCLUDE", "E1", "E2", "E3", "E4"},
        "Bad grade",
    )
    yes = result["eligible"].eq("YES")
    no = result["eligible"].eq("NO")
    require(
        result.loc[yes, "evidence_grade"].isin({"E1", "E2", "E3", "E4"}).all(),
        "Eligible records require E1-E4",
    )
    require(
        result.loc[no, "evidence_grade"].eq("EXCLUDE").all(),
        "Ineligible records require EXCLUDE",
    )
    require(
        result.loc[yes, "exclusion_reason"].eq("NA").all(),
        "Eligible records require exclusion_reason=NA",
    )
    require(
        result["curator_id"].str.match(r"^[A-Za-z0-9_.-]+$").all(),
        "Use coded curator IDs only",
    )
    return result


def aggregate_claim_evidence(grades: pd.DataFrame, all_review_ids: pd.Series) -> pd.DataFrame:
    rows = []
    for review_id in sorted(all_review_ids.astype(str).unique()):
        group = grades.loc[grades["review_id"].astype(str).eq(review_id)].copy()
        eligible = group.loc[group["eligible"].eq("YES")].copy()
        eligible["grade_rank"] = eligible["evidence_grade"].map(GRADE_RANK)
        maximum = "E0"
        if not eligible.empty:
            maximum = str(eligible.loc[eligible["grade_rank"].idxmax(), "evidence_grade"])
        primary = eligible[
            eligible["evidence_grade"].isin({"E3", "E4"})
            & eligible["direction_match"].eq("MATCH")
            & eligible["data_overlap"].eq("INDEPENDENT")
        ]
        rows.append(
            {
                "review_id": review_id,
                "retrieved_screening_rows": len(group),
                "eligible_records": len(eligible),
                "maximum_evidence_grade": maximum,
                "primary_independent_support": bool(len(primary)),
                "direction_matched_support": bool(eligible["direction_match"].eq("MATCH").any()),
                "contradiction_recorded": bool(eligible["contradiction"].eq("YES").any()),
                "independent_supporting_records": len(primary),
            }
        )
    result = pd.DataFrame(rows)
    require(
        len(result) == 50 and result["review_id"].is_unique,
        "P3 claim census must be 50",
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--completed-screening", required=True, type=Path)
    parser.add_argument("--lock-label", required=True)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--retrieval-checker", type=Path, default=DEFAULT_CHECKER)
    args = parser.parse_args()

    data_root = args.data_root.expanduser().resolve()
    completed_path = args.completed_screening.expanduser().resolve()
    require(
        completed_path.is_file(),
        f"Missing completed P3 screening file: {completed_path}",
    )
    run_retrieval_gate(args.retrieval_checker, data_root)
    protocol = read_json(args.protocol)
    benchmark = str(protocol["parent_priority2_benchmark_id"])
    p3_root = data_root / "output" / "priority3" / benchmark
    retrieval_manifest_path = p3_root / "retrieval" / "priority3_retrieval_manifest.json"
    retrieval_manifest = read_json(retrieval_manifest_path)
    blank_path = Path(retrieval_manifest["outputs"]["screening"]["path"])
    claims_path = Path(retrieval_manifest["outputs"]["claims_blinded"]["path"])
    blank = pd.read_csv(blank_path, sep="\t", dtype={"pmid": str})
    completed = pd.read_csv(completed_path, sep="\t", dtype={"pmid": str})
    completed = validate_completed(completed, blank, protocol)
    claims = pd.read_csv(claims_path, sep="\t")
    claim_evidence = aggregate_claim_evidence(completed, claims["review_id"])

    outdir = p3_root / "grading_lock_v1"
    require(
        not outdir.exists(),
        f"P3 grading lock is immutable and already exists: {outdir}",
    )
    outdir.mkdir(parents=True)
    grades_path = outdir / "record_grades.private.tsv"
    claims_out = outdir / "claim_evidence.tsv"
    manifest_path = outdir / "priority3_grading_lock_manifest.json"
    digest_path = outdir / "priority3_grading_lock_manifest.sha256"
    completed.to_csv(grades_path, sep="\t", index=False, lineterminator="\n")
    claim_evidence.to_csv(claims_out, sep="\t", index=False, lineterminator="\n")
    manifest = {
        "schema_version": "CRM_R1_PRIORITY3_GRADING_LOCK_v1",
        "status": "P3_GRADES_LOCKED_BEFORE_METHOD_UNBLINDING",
        "benchmark_id": benchmark,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "lock_label": args.lock_label,
        "record_rows": len(completed),
        "claim_rows": len(claim_evidence),
        "method_membership_read": False,
        "inputs": {
            "retrieval_manifest": file_record(retrieval_manifest_path),
            "frozen_blank_screening": file_record(blank_path),
            "completed_screening_external": file_record(completed_path),
            "protocol": file_record(args.protocol),
        },
        "outputs": {
            "record_grades_private": file_record(grades_path),
            "claim_evidence": file_record(claims_out),
        },
    }
    write_json(manifest_path, manifest)
    digest_path.write_text(
        f"{sha256_file(manifest_path)}  {manifest_path.name}\n", encoding="utf-8"
    )
    print("[PASS] Priority 3 blinded grades locked before method unblinding")
    print(f"[INFO] Claims: {len(claim_evidence)}; records graded: {len(completed)}")
    print(f"[INFO] Lock label: {args.lock_label}")
    print("[NEXT] Run 33_check_priority3_grades_lock.py")


if __name__ == "__main__":
    main()
