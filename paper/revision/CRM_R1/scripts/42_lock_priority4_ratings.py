#!/usr/bin/env python3
"""Validate three returned blinded rating files and freeze consensus outcomes."""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from v11_lock_common import (
    file_record,
    fleiss_kappa,
    read_json,
    require,
    sha256_file,
    write_json,
)

CRM_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = CRM_DIR.parents[2]
DEFAULT_PROTOCOL = CRM_DIR / "config" / "priority4_review_protocol.json"
DEFAULT_P3_PROTOCOL = CRM_DIR / "config" / "priority3_protocol.json"
DEFAULT_CHECKER = Path(__file__).resolve().with_name("41_check_priority4_packets.py")

RATING_FIELDS = [
    "q1_statistical_support",
    "q2_external_evidence",
    "q3_overstatement",
    "confidence_1_to_5",
    "concise_rationale",
]
FORBIDDEN_METHOD_COLUMNS = {
    "claim_uid",
    "claim_id",
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


def run_packet_gate(checker: Path, data_root: Path) -> None:
    result = subprocess.run(
        [sys.executable, str(checker), "--data-root", str(data_root)],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    output = result.stdout + result.stderr
    print(output, end="")
    require(result.returncode == 0, "Priority 4 packet gate failed")
    require("[GO] Distribute" in output, "P4 checker did not confirm the frozen packets")


def validate_returned_ratings(
    paths: list[Path],
    templates: dict[str, pd.DataFrame],
    protocol: dict,
) -> pd.DataFrame:
    require(
        len(paths) == int(protocol["minimum_independent_raters"]),
        "Exactly three ratings files are required",
    )
    returned: dict[str, pd.DataFrame] = {}
    for path in paths:
        require(path.is_file(), f"Missing returned ratings file: {path}")
        frame = pd.read_csv(path, sep="\t")
        require(
            not (set(frame.columns) & FORBIDDEN_METHOD_COLUMNS),
            f"Method fields leaked into {path.name}",
        )
        require(
            {"rater_id", "review_id", "packet_order", *RATING_FIELDS} <= set(frame.columns),
            f"Bad rating schema: {path}",
        )
        identifiers = frame["rater_id"].fillna("").astype(str).str.strip().unique()
        require(
            len(identifiers) == 1 and identifiers[0],
            f"One coded rater_id is required: {path}",
        )
        rater_id = str(identifiers[0])
        require(rater_id in templates, f"Unexpected rater_id: {rater_id}")
        require(rater_id not in returned, f"Duplicate returned rater_id: {rater_id}")
        expected = templates[rater_id][["rater_id", "review_id", "packet_order"]].copy()
        observed = frame[["rater_id", "review_id", "packet_order"]].copy()
        for table in (expected, observed):
            table["rater_id"] = table["rater_id"].astype(str).str.strip()
            table["review_id"] = table["review_id"].astype(str).str.strip()
            table["packet_order"] = pd.to_numeric(table["packet_order"], errors="raise").astype(int)
        require(
            observed.sort_values("packet_order")
            .reset_index(drop=True)
            .equals(expected.sort_values("packet_order").reset_index(drop=True)),
            f"Frozen P4 assignment changed: {rater_id}",
        )
        for question, allowed in protocol["questions"].items():
            frame[question] = frame[question].fillna("").astype(str).str.strip()
            require(frame[question].ne("").all(), f"Incomplete {question}: {rater_id}")
            require(set(frame[question]) <= set(allowed), f"Invalid {question}: {rater_id}")
        confidence = pd.to_numeric(frame["confidence_1_to_5"], errors="raise")
        require(
            confidence.isin(protocol["confidence_scale"]).all(),
            f"Invalid confidence: {rater_id}",
        )
        frame["confidence_1_to_5"] = confidence.astype(int)
        frame["concise_rationale"] = frame["concise_rationale"].fillna("").astype(str).str.strip()
        require(frame["concise_rationale"].ne("").all(), f"Missing rationale: {rater_id}")
        require(
            frame["review_id"].is_unique and len(frame) == 50,
            f"Rating census drift: {rater_id}",
        )
        frame["source_file"] = path.name
        returned[rater_id] = frame
    require(set(returned) == set(templates), "Not every assigned rater returned a file")
    return pd.concat([returned[rater] for rater in sorted(returned)], ignore_index=True)


def majority(values: pd.Series, categories: list[str]) -> str:
    counts = values.value_counts()
    winners = [category for category in categories if int(counts.get(category, 0)) >= 2]
    return winners[0] if len(winners) == 1 else "NO_MAJORITY"


def build_consensus(long: pd.DataFrame, protocol: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for review_id, group in long.groupby("review_id", sort=True):
        q1 = majority(
            group["q1_statistical_support"],
            protocol["questions"]["q1_statistical_support"],
        )
        q2 = majority(group["q2_external_evidence"], protocol["questions"]["q2_external_evidence"])
        q3 = majority(group["q3_overstatement"], protocol["questions"]["q3_overstatement"])
        rows.append(
            {
                "review_id": review_id,
                "q1_majority": q1,
                "q2_majority": q2,
                "q3_majority": q3,
                "major_overstatement_majority": int(
                    group["q3_overstatement"].eq("MAJOR_OVERSTATEMENT").sum()
                )
                >= 2,
                "minor_or_major_overstatement_majority": int(
                    group["q3_overstatement"]
                    .isin({"MINOR_OVERSTATEMENT", "MAJOR_OVERSTATEMENT"})
                    .sum()
                )
                >= 2,
                "direct_external_evidence_majority": int(
                    group["q2_external_evidence"].eq("DIRECT").sum()
                )
                >= 2,
                "statistically_supported_majority": int(
                    group["q1_statistical_support"].eq("SUPPORTED").sum()
                )
                >= 2,
                "mean_confidence": float(group["confidence_1_to_5"].mean()),
                "all_three_q1_agree": group["q1_statistical_support"].nunique() == 1,
                "all_three_q2_agree": group["q2_external_evidence"].nunique() == 1,
                "all_three_q3_agree": group["q3_overstatement"].nunique() == 1,
            }
        )
    consensus = pd.DataFrame(rows)
    agreement_rows = []
    for question, label in (
        ("q1_statistical_support", "Statistical support"),
        ("q2_external_evidence", "External evidence"),
        ("q3_overstatement", "Overstatement"),
    ):
        wide = long.pivot(index="review_id", columns="rater_id", values=question)
        pairwise = []
        columns = list(wide.columns)
        for index, first in enumerate(columns):
            for second in columns[index + 1 :]:
                pairwise.extend(wide[first].eq(wide[second]).astype(float).tolist())
        agreement_rows.append(
            {
                "question": question,
                "question_label": label,
                "n_claims": len(wide),
                "n_raters": len(columns),
                "fleiss_kappa": fleiss_kappa(wide, list(protocol["questions"][question])),
                "exact_unanimous_fraction": float(wide.nunique(axis=1).eq(1).mean()),
                "mean_pairwise_agreement": float(np.mean(pairwise)),
            }
        )
    require(
        len(consensus) == 50 and consensus["review_id"].is_unique,
        "P4 consensus census drift",
    )
    return consensus, pd.DataFrame(agreement_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--ratings", required=True, type=Path, action="append")
    parser.add_argument("--lock-label", required=True)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--p3-protocol", type=Path, default=DEFAULT_P3_PROTOCOL)
    parser.add_argument("--packet-checker", type=Path, default=DEFAULT_CHECKER)
    args = parser.parse_args()

    data_root = args.data_root.expanduser().resolve()
    run_packet_gate(args.packet_checker, data_root)
    protocol = read_json(args.protocol)
    p3_protocol = read_json(args.p3_protocol)
    benchmark = str(p3_protocol["parent_priority2_benchmark_id"])
    packet_root = data_root / "output" / "priority4" / benchmark / "packet_v1"
    packet_manifest_path = packet_root / "priority4_packet_manifest.json"
    packet_manifest = read_json(packet_manifest_path)
    templates = {}
    template_paths = {}
    for label, record in packet_manifest["outputs"].items():
        if not label.startswith("ratings_R"):
            continue
        path = Path(record["path"])
        frame = pd.read_csv(path, sep="\t")
        rater_id = str(frame["rater_id"].iloc[0]).strip()
        templates[rater_id] = frame
        template_paths[rater_id] = path
    ratings_paths = [path.expanduser().resolve() for path in args.ratings]
    long = validate_returned_ratings(ratings_paths, templates, protocol)
    consensus, agreement = build_consensus(long, protocol)

    outdir = data_root / "output" / "priority4" / benchmark / "ratings_lock_v1"
    require(
        not outdir.exists(),
        f"P4 ratings lock is immutable and already exists: {outdir}",
    )
    outdir.mkdir(parents=True)
    long_path = outdir / "ratings_long.private.tsv"
    consensus_path = outdir / "rating_consensus.tsv"
    agreement_path = outdir / "interrater_agreement.tsv"
    manifest_path = outdir / "priority4_ratings_lock_manifest.json"
    digest_path = outdir / "priority4_ratings_lock_manifest.sha256"
    long.to_csv(long_path, sep="\t", index=False, lineterminator="\n")
    consensus.to_csv(consensus_path, sep="\t", index=False, lineterminator="\n")
    agreement.to_csv(agreement_path, sep="\t", index=False, lineterminator="\n")
    manifest = {
        "schema_version": "CRM_R1_PRIORITY4_RATINGS_LOCK_v1",
        "status": "P4_RATINGS_LOCKED_BEFORE_METHOD_UNBLINDING",
        "benchmark_id": benchmark,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "lock_label": args.lock_label,
        "claims": 50,
        "raters": len(templates),
        "method_membership_read": False,
        "inputs": {
            "packet_manifest": file_record(packet_manifest_path),
            "protocol": file_record(args.protocol),
            **{
                f"blank_template_{rater}": file_record(path)
                for rater, path in template_paths.items()
            },
            **{
                f"returned_ratings_{index}": file_record(path)
                for index, path in enumerate(ratings_paths, start=1)
            },
        },
        "outputs": {
            "ratings_long_private": file_record(long_path),
            "rating_consensus": file_record(consensus_path),
            "interrater_agreement": file_record(agreement_path),
        },
    }
    write_json(manifest_path, manifest)
    digest_path.write_text(
        f"{sha256_file(manifest_path)}  {manifest_path.name}\n", encoding="utf-8"
    )
    print("[PASS] Priority 4 independent ratings locked before method unblinding")
    print(f"[INFO] Claims: 50; raters: {len(templates)}; label: {args.lock_label}")
    print("[NEXT] Run 43_check_priority4_ratings_lock.py")


if __name__ == "__main__":
    main()
