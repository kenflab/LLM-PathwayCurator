#!/usr/bin/env python3
"""Lock the existing P5 500-claim memberships before completing context review."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

BENCHMARK_ID = "PANCAN_TP53_v1_HNSC_R1_P5"
EXPECTED_K = 500
P2_PROTOCOL_SHA256 = "9e84af9c6f5282f08e4034d512efeb7efbcccf4183ee5c27319809f9a13a5086"
COLLECTIONS = {
    "C5_GO_BP": "HNSC.C5_GO_BP.evidence_table.tsv",
    "C2_CP_REACTOME": "HNSC.C2_CP_REACTOME.evidence_table.tsv",
}
MEMBERSHIP_COLUMNS = ["claim_id", "entity", "direction"]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def repo_root_from_script() -> Path:
    return Path(__file__).resolve().parents[4]


def run_priority2_gate(repo_root: Path, data_root: Path) -> None:
    checker = repo_root / "paper/revision/CRM_R1/scripts/21_check_priority2_freeze.py"
    subprocess.run(
        [sys.executable, str(checker), "--data-root", str(data_root)],
        cwd=repo_root,
        check=True,
    )


def require_clean_revision(repo_root: Path) -> str:
    for cached in (False, True):
        command = ["git", "diff", "--quiet"]
        if cached:
            command.append("--cached")
        command.extend(["--", "paper/revision/CRM_R1", "tests"])
        result = subprocess.run(command, cwd=repo_root, check=False)
        require(
            result.returncode == 0,
            "Commit tracked V10.3 code before locking the census",
        )
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_root, text=True).strip()


def file_record(path: Path, *, root: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve().relative_to(root.resolve())),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def load_membership_only(path: Path, collection: str) -> pd.DataFrame:
    require(path.is_file(), f"Missing membership-source audit: {path}")
    table = pd.read_csv(path, sep="\t", usecols=MEMBERSHIP_COLUMNS)
    require(len(table) == EXPECTED_K, f"{collection}: expected {EXPECTED_K} claims")
    require(table["claim_id"].nunique() == EXPECTED_K, f"{collection}: duplicate claim_id")
    require(table["entity"].nunique() == EXPECTED_K, f"{collection}: duplicate entity")
    direction = table["direction"].astype(str).str.strip().str.lower()
    require(set(direction) <= {"up", "down"}, f"{collection}: invalid direction")
    output = table.copy()
    output["claim_id"] = output["claim_id"].astype(str).str.strip()
    output["entity"] = output["entity"].astype(str).str.strip()
    output["direction"] = direction
    output.insert(0, "census_order", range(1, EXPECTED_K + 1))
    output.insert(0, "collection", collection)
    return output


def build_census_evidence(
    membership: pd.DataFrame, evidence_path: Path, collection: str
) -> pd.DataFrame:
    require(evidence_path.is_file(), f"Missing canonical EvidenceTable: {evidence_path}")
    evidence = pd.read_csv(evidence_path, sep="\t", low_memory=False)
    required = {"term_id", "direction"}
    require(required.issubset(evidence.columns), f"{collection}: EvidenceTable schema")
    require(evidence["term_id"].is_unique, f"{collection}: duplicate canonical term_id")
    indexed = evidence.assign(term_id=evidence["term_id"].astype(str).str.strip()).set_index(
        "term_id", drop=False
    )
    entities = membership["entity"].tolist()
    missing = sorted(set(entities) - set(indexed.index))
    require(
        not missing,
        f"{collection}: census entities missing from EvidenceTable: {missing[:5]}",
    )
    census = indexed.loc[entities].reset_index(drop=True)
    canonical_direction = census["direction"].astype(str).str.strip().str.lower()
    expected_direction = membership["direction"].reset_index(drop=True)
    require(
        canonical_direction.equals(expected_direction),
        f"{collection}: membership direction differs from canonical evidence",
    )
    require(len(census) == EXPECTED_K, f"{collection}: incomplete census evidence")
    return census[evidence.columns]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--lock-label", required=True)
    args = parser.parse_args()

    data_root = args.data_root.expanduser().resolve()
    repo_root = repo_root_from_script()
    p5_root = data_root / "output/priority5" / BENCHMARK_ID
    final_dir = p5_root / "candidate_census"
    freeze_manifest = p5_root / "freeze/priority5_input_manifest.json"
    hierarchy_output = p5_root / "ontology/hierarchy_metrics.tsv"
    require(not final_dir.exists(), f"Candidate census already exists: {final_dir}")
    require(not freeze_manifest.exists(), "P5 input freeze already exists")
    require(not hierarchy_output.exists(), "P5 hierarchy outcomes already exist")

    run_priority2_gate(repo_root, data_root)
    p2_protocol = repo_root / "paper/revision/CRM_R1/config/priorities2_5_protocol.json"
    require(sha256_file(p2_protocol) == P2_PROTOCOL_SHA256, "P2 protocol hash drift")
    commit = require_clean_revision(repo_root)

    evidence_root = repo_root / "paper/source_data/PANCAN_TP53_v1/evidence_tables"
    audit_root = p5_root / "audit_runs"
    p5_root.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".candidate_census.", dir=p5_root))
    try:
        membership_tables: list[pd.DataFrame] = []
        source_records: dict[str, Any] = {}
        output_paths: dict[str, Path] = {}
        for collection, filename in COLLECTIONS.items():
            audit_path = audit_root / collection / "HNSC/ours/gate_hard/tau_0.90/audit_log.tsv"
            evidence_path = evidence_root / filename
            membership = load_membership_only(audit_path, collection)
            census = build_census_evidence(membership, evidence_path, collection)
            output_path = temporary / f"{collection}.candidate_census.evidence_table.tsv"
            census.to_csv(output_path, sep="\t", index=False)
            membership_tables.append(membership)
            output_paths[collection] = output_path
            source_records[collection] = {
                "membership_source_audit": file_record(audit_path, root=data_root),
                "canonical_evidence": file_record(evidence_path, root=repo_root),
            }

        membership_path = temporary / "priority5_candidate_census.tsv"
        pd.concat(membership_tables, ignore_index=True).to_csv(
            membership_path, sep="\t", index=False
        )
        manifest_path = temporary / "priority5_candidate_census_manifest.json"
        manifest = {
            "schema_version": "CRM_R1_PRIORITY5_CANDIDATE_CENSUS_v1",
            "status": "LOCKED_BEFORE_COMPLETE_CONTEXT_REVIEW_AND_ONTOLOGY_EVALUATION",
            "benchmark_id": BENCHMARK_ID,
            "lock_label": str(args.lock_label),
            "created_utc": datetime.now(UTC).isoformat(),
            "git_commit": commit,
            "claims_per_collection": EXPECTED_K,
            "membership_source_columns_read": MEMBERSHIP_COLUMNS,
            "audit_status_columns_read": False,
            "hierarchy_outcomes_read": False,
            "priority3_priority4_outcomes_read": False,
            "membership_rule": (
                "retain all 500 entities from each pre-freeze P5 proposal run without "
                "status-dependent reselection"
            ),
            "sources": source_records,
            "outputs": {
                "membership": file_record(membership_path, root=temporary),
                **{
                    collection: file_record(path, root=temporary)
                    for collection, path in output_paths.items()
                },
            },
        }
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        (temporary / "priority5_candidate_census_manifest.sha256").write_text(
            f"{sha256_file(manifest_path)}  {manifest_path.name}\n", encoding="utf-8"
        )
        os.replace(temporary, final_dir)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise

    print("[PASS] Priority 5 candidate memberships locked without reading audit status")
    print(f"[INFO] Collections: {len(COLLECTIONS)}; claims per collection: {EXPECTED_K}")
    print("[INFO] Hierarchy and P3/P4 outcomes read: false")
    print(f"[INFO] Wrote: {final_dir}")
    print("[NEXT] Preserve the membership-source audit_runs, then rerun all 500 census claims")


if __name__ == "__main__":
    main()
