#!/usr/bin/env python3
"""Join locked P3/P4 outcomes to frozen P2 membership and freeze Figure 2 sources."""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from v11_lock_common import (
    as_bool,
    file_record,
    overlap_aware_exact,
    read_json,
    require,
    sha256_file,
    wilson_interval,
    write_json,
)

CRM_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = CRM_DIR.parents[2]
DEFAULT_P2_CONFIG = CRM_DIR / "config" / "priorities2_5_protocol.json"
DEFAULT_P3_PROTOCOL = CRM_DIR / "config" / "priority3_protocol.json"
DEFAULT_P2_CHECKER = Path(__file__).resolve().with_name("21_check_priority2_freeze.py")
DEFAULT_P3_LOCK_CHECKER = Path(__file__).resolve().with_name("33_check_priority3_grades_lock.py")
DEFAULT_P4_LOCK_CHECKER = Path(__file__).resolve().with_name("43_check_priority4_ratings_lock.py")

METHODS = [
    ("raw_pool", "Raw candidate pool (descriptive)", "raw_pool_selected", "#6B7280"),
    ("q_value_matched", "q-value matched", "q_value_matched_selected", "#D55E00"),
    ("stability_matched", "Stability matched", "stability_matched_selected", "#009E73"),
    ("full_audit", "Full audit", "full_audit_selected", "#0072B2"),
]


def run_gate(command: list[str], expected: str, label: str) -> None:
    result = subprocess.run(command, cwd=REPO_ROOT, check=False, capture_output=True, text=True)
    output = result.stdout + result.stderr
    print(output, end="")
    require(result.returncode == 0, f"{label} failed")
    require(expected in output, f"{label} did not release Figure 2")


def summarize_method(
    claims: pd.DataFrame,
    *,
    method_id: str,
    method_label: str,
    membership_column: str,
    color: str,
    outcome: str,
    outcome_label: str,
) -> dict:
    selected = as_bool(claims[membership_column], name=membership_column)
    values = as_bool(claims.loc[selected, outcome], name=outcome)
    n = len(values)
    successes = int(values.sum())
    low, high = wilson_interval(successes, n)
    return {
        "method_id": method_id,
        "method_label": method_label,
        "membership_column": membership_column,
        "color": color,
        "outcome": outcome,
        "outcome_label": outcome_label,
        "n_selected": n,
        "n_outcome": successes,
        "fraction": successes / n,
        "ci_low": low,
        "ci_high": high,
        "reporting_coverage": n / len(claims),
        "comparison_role": "descriptive" if method_id == "raw_pool" else "coverage_matched",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--freeze-label", required=True)
    parser.add_argument("--p2-config", type=Path, default=DEFAULT_P2_CONFIG)
    parser.add_argument("--p3-protocol", type=Path, default=DEFAULT_P3_PROTOCOL)
    parser.add_argument("--p2-checker", type=Path, default=DEFAULT_P2_CHECKER)
    parser.add_argument("--p3-lock-checker", type=Path, default=DEFAULT_P3_LOCK_CHECKER)
    parser.add_argument("--p4-lock-checker", type=Path, default=DEFAULT_P4_LOCK_CHECKER)
    args = parser.parse_args()

    data_root = args.data_root.expanduser().resolve()
    run_gate(
        [
            sys.executable,
            str(args.p2_checker),
            "--data-root",
            str(data_root),
            "--config",
            str(args.p2_config),
        ],
        "[GO] P3 evidence retrieval",
        "P2 freeze checker",
    )
    run_gate(
        [
            sys.executable,
            str(args.p3_lock_checker),
            "--data-root",
            str(data_root),
            "--protocol",
            str(args.p3_protocol),
        ],
        "P3 outcomes may be joined",
        "P3 grading-lock checker",
    )
    run_gate(
        [
            sys.executable,
            str(args.p4_lock_checker),
            "--data-root",
            str(data_root),
            "--p3-protocol",
            str(args.p3_protocol),
        ],
        "P3 and P4 outcomes may now be joined",
        "P4 ratings-lock checker",
    )
    p3_protocol = read_json(args.p3_protocol)
    benchmark = str(p3_protocol["parent_priority2_benchmark_id"])
    p2_root = data_root / "output" / "priority2" / benchmark
    p2_manifest_path = p2_root / "metrics" / "priority2_freeze_manifest.json"
    p2_manifest = read_json(p2_manifest_path)
    p2_outputs = {label: Path(record["path"]) for label, record in p2_manifest["outputs"].items()}
    membership = pd.read_csv(p2_outputs["membership"], sep="\t")
    sampling = pd.read_csv(p2_outputs["sampling_frame"], sep="\t")
    p3_manifest_path = (
        data_root
        / "output"
        / "priority3"
        / benchmark
        / "grading_lock_v1"
        / "priority3_grading_lock_manifest.json"
    )
    p4_manifest_path = (
        data_root
        / "output"
        / "priority4"
        / benchmark
        / "ratings_lock_v1"
        / "priority4_ratings_lock_manifest.json"
    )
    p3_manifest = read_json(p3_manifest_path)
    p4_manifest = read_json(p4_manifest_path)
    p3_claims = pd.read_csv(Path(p3_manifest["outputs"]["claim_evidence"]["path"]), sep="\t")
    p4_consensus = pd.read_csv(Path(p4_manifest["outputs"]["rating_consensus"]["path"]), sep="\t")
    agreement = pd.read_csv(Path(p4_manifest["outputs"]["interrater_agreement"]["path"]), sep="\t")

    require({"claim_uid", "review_id"} <= set(sampling), "P2 sampling map is incomplete")
    required_memberships = {column for _, _, column, _ in METHODS}
    require(
        {"claim_uid", *required_memberships} <= set(membership),
        "P2 memberships are incomplete",
    )
    claims = (
        sampling[["claim_uid", "review_id"]]
        .merge(
            membership[["claim_uid", *required_memberships]],
            on="claim_uid",
            validate="one_to_one",
        )
        .merge(p3_claims, on="review_id", validate="one_to_one")
        .merge(p4_consensus, on="review_id", validate="one_to_one")
    )
    require(
        len(claims) == 50 and claims["claim_uid"].is_unique,
        "Figure 2 join census drift",
    )
    for column in required_memberships:
        claims[column] = as_bool(claims[column], name=column)
    require(int(claims["raw_pool_selected"].sum()) == 50, "Raw-pool census drift")
    matched_k = {
        int(claims[column].sum())
        for column in required_memberships
        if column != "raw_pool_selected"
    }
    require(len(matched_k) == 1, "Coverage-matched methods do not share K")

    outcome_specs = [
        ("primary_independent_support", "Independent direction-matched support"),
        ("major_overstatement_majority", "Major overstatement by majority"),
        (
            "minor_or_major_overstatement_majority",
            "Minor/major overstatement by majority",
        ),
    ]
    method_rows = []
    for outcome, label in outcome_specs:
        for method_id, method_label, membership_column, color in METHODS:
            method_rows.append(
                summarize_method(
                    claims,
                    method_id=method_id,
                    method_label=method_label,
                    membership_column=membership_column,
                    color=color,
                    outcome=outcome,
                    outcome_label=label,
                )
            )
    method_metrics = pd.DataFrame(method_rows)

    exact_rows = []
    for comparator_id, comparator_column in (
        ("q_value_matched", "q_value_matched_selected"),
        ("stability_matched", "stability_matched_selected"),
    ):
        for outcome, label, alternative in (
            (
                "primary_independent_support",
                "Independent direction-matched support",
                "greater",
            ),
            ("major_overstatement_majority", "Major overstatement by majority", "less"),
        ):
            result = overlap_aware_exact(
                claims[outcome],
                claims["full_audit_selected"],
                claims[comparator_column],
                alternative=alternative,
            )
            exact_rows.append(
                {
                    "method_a": "full_audit",
                    "method_b": comparator_id,
                    "outcome": outcome,
                    "outcome_label": label,
                    **result,
                }
            )
    exact = pd.DataFrame(exact_rows)
    design = pd.DataFrame(
        [
            {
                "step": 1,
                "label": "Frozen 50-claim HNSC pool",
                "detail": "Same candidates for every rule",
            },
            {
                "step": 2,
                "label": "Coverage-matched reporting",
                "detail": f"Full audit, q-value, and stability; K={next(iter(matched_k))}",
            },
            {
                "step": 3,
                "label": "Blinded independent evaluation",
                "detail": "Frozen PubMed grading + 3 raters",
            },
            {
                "step": 4,
                "label": "Unblind once",
                "detail": "Wilson intervals + overlap-aware exact reference",
            },
        ]
    )

    outdir = p2_root / "final_v11"
    require(not outdir.exists(), f"Figure 2 V11 source lock already exists: {outdir}")
    outdir.mkdir(parents=True)
    paths = {
        "panel_A": outdir / "figure2_panelA_design.tsv",
        "panel_BC": outdir / "figure2_panelsBC_method_metrics.tsv",
        "panel_D": outdir / "figure2_panelD_interrater_agreement.tsv",
        "exact": outdir / "figure2_overlap_aware_exact.tsv",
        "claim_source_private": outdir / "figure2_claim_source.private.tsv",
    }
    design.to_csv(paths["panel_A"], sep="\t", index=False, lineterminator="\n")
    method_metrics.to_csv(paths["panel_BC"], sep="\t", index=False, lineterminator="\n")
    agreement.to_csv(paths["panel_D"], sep="\t", index=False, lineterminator="\n")
    exact.to_csv(paths["exact"], sep="\t", index=False, lineterminator="\n")
    claims.to_csv(paths["claim_source_private"], sep="\t", index=False, lineterminator="\n")
    manifest_path = outdir / "figure2_source_manifest.json"
    digest_path = outdir / "figure2_source_manifest.sha256"
    manifest = {
        "schema_version": "CRM_R1_FIGURE2_SOURCE_v11",
        "status": "FIGURE2_SOURCE_FROZEN_AFTER_P3_P4_LOCK",
        "benchmark_id": benchmark,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "freeze_label": args.freeze_label,
        "candidate_claims": 50,
        "matched_k": next(iter(matched_k)),
        "inputs": {
            "p2_manifest": file_record(p2_manifest_path),
            "p3_grading_manifest": file_record(p3_manifest_path),
            "p4_ratings_manifest": file_record(p4_manifest_path),
        },
        "outputs": {label: file_record(path) for label, path in paths.items()},
        "private_outputs_not_for_public_redistribution": [paths["claim_source_private"].name],
    }
    write_json(manifest_path, manifest)
    digest_path.write_text(
        f"{sha256_file(manifest_path)}  {manifest_path.name}\n", encoding="utf-8"
    )
    print("[PASS] Figure 2 source tables frozen after both blinded locks")
    print(f"[INFO] Candidate claims: 50; matched K: {next(iter(matched_k))}")
    print(f"[INFO] Wrote: {manifest_path}")
    print(
        "[NEXT] Run 61_check_priority2_figure2_source.py, "
        "then render with 90_plot_priority2_figure2.py"
    )


if __name__ == "__main__":
    main()
