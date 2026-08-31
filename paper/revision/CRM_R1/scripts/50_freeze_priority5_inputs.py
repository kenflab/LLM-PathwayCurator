#!/usr/bin/env python3
"""Freeze P5 audit outputs and ontology snapshots before hierarchy evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import ssl
import subprocess
import sys
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

BENCHMARK_ID = "PANCAN_TP53_v1_HNSC_R1_P5"
EXPECTED_TAU = 0.90
EXPECTED_K = 500
COLLECTIONS = ("C5_GO_BP", "C2_CP_REACTOME")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def repo_root_from_script() -> Path:
    return Path(__file__).resolve().parents[4]


def load_protocol(repo_root: Path) -> tuple[Path, dict[str, Any]]:
    path = repo_root / "paper/revision/CRM_R1/config/priority5_protocol.json"
    require(path.is_file(), f"Missing Priority 5 protocol: {path}")
    protocol = json.loads(path.read_text(encoding="utf-8"))
    require(
        protocol["protocol_version"] == "CRM_R1_PRIORITY5_v10_3",
        "Wrong P5 protocol",
    )
    require(
        protocol["status"] == "PRESPECIFIED_AWAITING_INPUT_FREEZE",
        "P5 protocol is not in its pre-evaluation state",
    )
    return path, protocol


def run_priority2_gate(repo_root: Path, data_root: Path) -> None:
    checker = repo_root / "paper/revision/CRM_R1/scripts/21_check_priority2_freeze.py"
    require(checker.is_file(), f"Missing Priority 2 checker: {checker}")
    subprocess.run(
        [sys.executable, str(checker), "--data-root", str(data_root)],
        cwd=repo_root,
        check=True,
    )


def require_clean_tracked_revision(repo_root: Path) -> str:
    paths = ["paper/revision/CRM_R1", "tests"]
    for cached in (False, True):
        command = ["git", "diff", "--quiet"]
        if cached:
            command.append("--cached")
        command.extend(["--", *paths])
        result = subprocess.run(command, cwd=repo_root, check=False)
        require(
            result.returncode == 0,
            "Commit tracked V10.3 code before freezing P5 inputs",
        )
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_root, text=True).strip()


def ssl_context(use_system_trust: bool):
    if not use_system_trust:
        return None
    try:
        import truststore
    except ImportError as error:
        raise RuntimeError(
            "--use-system-trust requires: python -m pip install truststore==0.10.4"
        ) from error
    return truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)


def fetch_if_missing(*, url: str, destination: Path, use_system_trust: bool) -> None:
    if destination.is_file():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "LLM-PathwayCurator-CRM-R1-P5/1.0"},
    )
    temporary = destination.with_suffix(destination.suffix + ".download")
    try:
        with urllib.request.urlopen(
            request, timeout=180, context=ssl_context(use_system_trust)
        ) as response:
            temporary.write_bytes(response.read())
        require(temporary.stat().st_size > 100, f"Downloaded file is too small: {url}")
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def go_release_metadata(path: Path) -> dict[str, str | None]:
    metadata: dict[str, str | None] = {"data_version": None, "date": None}
    with path.open("r", encoding="utf-8") as handle:
        for _ in range(100):
            line = handle.readline()
            if not line or line == "[Term]\n":
                break
            if line.startswith("data-version:"):
                metadata["data_version"] = line.split(":", 1)[1].strip()
            elif line.startswith("date:"):
                metadata["date"] = line.split(":", 1)[1].strip()
    require(metadata["data_version"] is not None, "GO OBO lacks data-version header")
    return metadata


def validate_audit_log(
    path: Path, collection: str, expected_membership: pd.DataFrame
) -> dict[str, Any]:
    require(path.is_file(), f"Missing P5 audit log: {path}")
    table = pd.read_csv(path, sep="\t", low_memory=False)
    required = {
        "claim_id",
        "entity",
        "direction",
        "status",
        "gene_ids",
        "tau_used",
        "context_review_mode",
        "context_evaluated",
        "context_status",
        "context_method",
    }
    require(required.issubset(table.columns), f"{collection} audit log schema is incomplete")
    require(len(table) == EXPECTED_K, f"{collection}: expected {EXPECTED_K} claims")
    require(table["claim_id"].nunique() == EXPECTED_K, f"{collection}: duplicate claim_id")
    require(table["entity"].nunique() == EXPECTED_K, f"{collection}: duplicate entity")
    observed_membership = {
        (str(entity).strip(), str(direction).strip().lower())
        for entity, direction in zip(table["entity"], table["direction"], strict=True)
    }
    frozen_membership = {
        (str(entity).strip(), str(direction).strip().lower())
        for entity, direction in zip(
            expected_membership["entity"],
            expected_membership["direction"],
            strict=True,
        )
    }
    require(
        observed_membership == frozen_membership,
        f"{collection}: complete audit membership differs from locked census",
    )
    tau = pd.to_numeric(table["tau_used"], errors="raise")
    require((tau - EXPECTED_TAU).abs().max() < 1e-12, f"{collection}: tau drift")
    modes = set(table["context_review_mode"].astype(str).str.lower())
    require(modes == {"llm"}, f"{collection}: context review must be llm")
    evaluated = table["context_evaluated"].astype(str).str.lower().isin({"true", "1"})
    require(evaluated.all(), f"{collection}: missing context evaluations")
    statuses = table["context_status"].astype(str).str.strip().str.upper()
    require(
        statuses.isin({"PASS", "WARN", "FAIL"}).all(),
        f"{collection}: invalid context status",
    )
    methods = table["context_method"].astype(str).str.strip().str.lower()
    require(methods.eq("llm").all(), f"{collection}: context method must be llm")
    return {
        "rows": len(table),
        "pass": int(table["status"].astype(str).str.upper().eq("PASS").sum()),
        "abstain": int(table["status"].astype(str).str.upper().eq("ABSTAIN").sum()),
        "fail": int(table["status"].astype(str).str.upper().eq("FAIL").sum()),
    }


def file_record(path: Path, root: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve().relative_to(root.resolve())),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def validate_candidate_census(
    p5_root: Path,
) -> tuple[list[Path], dict[str, pd.DataFrame]]:
    census_dir = p5_root / "candidate_census"
    manifest_path = census_dir / "priority5_candidate_census_manifest.json"
    digest_path = census_dir / "priority5_candidate_census_manifest.sha256"
    membership_path = census_dir / "priority5_candidate_census.tsv"
    require(manifest_path.is_file(), f"Missing candidate census manifest: {manifest_path}")
    require(digest_path.is_file(), f"Missing candidate census digest: {digest_path}")
    expected_digest = digest_path.read_text(encoding="utf-8").split()[0]
    require(sha256_file(manifest_path) == expected_digest, "Candidate census digest drift")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    require(
        manifest.get("status") == "LOCKED_BEFORE_COMPLETE_CONTEXT_REVIEW_AND_ONTOLOGY_EVALUATION",
        "Candidate census is not locked",
    )
    require(manifest.get("audit_status_columns_read") is False, "Census used audit status")
    require(
        manifest.get("hierarchy_outcomes_read") is False,
        "Census used hierarchy outcomes",
    )
    require(membership_path.is_file(), "Missing candidate census membership")
    membership = pd.read_csv(membership_path, sep="\t")
    require(len(membership) == EXPECTED_K * len(COLLECTIONS), "Census row-count drift")
    output_paths = [membership_path]
    memberships: dict[str, pd.DataFrame] = {}
    for collection in COLLECTIONS:
        subset = membership.loc[membership["collection"].eq(collection)].copy()
        require(len(subset) == EXPECTED_K, f"{collection}: candidate census K drift")
        require(subset["entity"].nunique() == EXPECTED_K, f"{collection}: duplicate entity")
        memberships[collection] = subset
        evidence_path = census_dir / f"{collection}.candidate_census.evidence_table.tsv"
        require(evidence_path.is_file(), f"Missing census EvidenceTable: {evidence_path}")
        evidence = pd.read_csv(evidence_path, sep="\t", usecols=["term_id", "direction"])
        require(len(evidence) == EXPECTED_K, f"{collection}: census EvidenceTable K drift")
        require(
            evidence["term_id"].astype(str).tolist() == subset["entity"].astype(str).tolist(),
            f"{collection}: census membership/evidence drift",
        )
        output_paths.append(evidence_path)
    for label, record in manifest["outputs"].items():
        path = census_dir / str(record["path"])
        require(path.is_file(), f"Missing recorded census output {label}: {path}")
        require(sha256_file(path) == record["sha256"], f"Census output hash drift: {path}")
    return [manifest_path, digest_path, *output_paths], memberships


def build_manifest(
    *, data_root: Path, repo_root: Path, use_system_trust: bool, freeze_label: str
) -> tuple[Path, dict[str, Any]]:
    protocol_path, protocol = load_protocol(repo_root)
    run_priority2_gate(repo_root, data_root)
    commit = require_clean_tracked_revision(repo_root)

    p5_root = data_root / "output/priority5" / BENCHMARK_ID
    candidate_files, candidate_memberships = validate_candidate_census(p5_root)
    hierarchy_output = p5_root / "ontology/hierarchy_metrics.tsv"
    require(
        not hierarchy_output.exists(),
        "P5 hierarchy outcomes already exist; freeze refused",
    )
    freeze_dir = p5_root / "freeze"
    manifest_path = freeze_dir / "priority5_input_manifest.json"
    checksum_path = freeze_dir / "priority5_input_manifest.sha256"
    require(not manifest_path.exists(), f"P5 freeze already exists: {manifest_path}")
    require(
        not checksum_path.exists(),
        f"P5 freeze checksum already exists: {checksum_path}",
    )

    reference_dir = data_root / "reference/priority5"
    go_path = reference_dir / protocol["ontology_sources"]["go"]["expected_filename"]
    reactome_pathways = (
        reference_dir / protocol["ontology_sources"]["reactome"]["pathways_filename"]
    )
    reactome_relations = (
        reference_dir / protocol["ontology_sources"]["reactome"]["relations_filename"]
    )
    fetch_if_missing(
        url=protocol["ontology_sources"]["go"]["download_url"],
        destination=go_path,
        use_system_trust=use_system_trust,
    )
    fetch_if_missing(
        url=protocol["ontology_sources"]["reactome"]["pathways_url"],
        destination=reactome_pathways,
        use_system_trust=use_system_trust,
    )
    fetch_if_missing(
        url=protocol["ontology_sources"]["reactome"]["relations_url"],
        destination=reactome_relations,
        use_system_trust=use_system_trust,
    )

    external_files: list[dict[str, Any]] = []
    audit_summaries: dict[str, Any] = {}
    for collection in COLLECTIONS:
        audit_path = (
            p5_root
            / "audit_runs_complete"
            / collection
            / "HNSC/ours/gate_hard/tau_0.90/audit_log.tsv"
        )
        audit_summaries[collection] = validate_audit_log(
            audit_path, collection, candidate_memberships[collection]
        )
        external_files.append(file_record(audit_path, data_root))

    for path in (go_path, reactome_pathways, reactome_relations):
        external_files.append(file_record(path, data_root))
    for path in candidate_files:
        external_files.append(file_record(path, data_root))

    repository_inputs = [protocol_path]
    for item in protocol["audit_inputs"]["collections"]:
        repository_inputs.append(repo_root / item["evidence_table"])
    repository_inputs.append(repo_root / protocol["audit_inputs"]["sample_card"])
    scripts_dir = repo_root / "paper/revision/CRM_R1/scripts"
    repository_inputs.append(scripts_dir / "49_lock_priority5_candidate_census.py")
    repository_inputs.extend(sorted(scripts_dir.glob("5[0-4]_*.py")))
    require(all(path.is_file() for path in repository_inputs), "Missing repository P5 input")

    manifest = {
        "schema_version": "CRM_R1_PRIORITY5_INPUT_FREEZE_v1",
        "status": "FROZEN_BEFORE_HIERARCHY_EVALUATION",
        "benchmark_id": BENCHMARK_ID,
        "freeze_label": freeze_label,
        "created_utc": datetime.now(UTC).isoformat(),
        "git_commit": commit,
        "protocol_version": protocol["protocol_version"],
        "audit_configuration": protocol["audit_inputs"],
        "audit_summaries": audit_summaries,
        "go_release": go_release_metadata(go_path),
        "reactome_release": protocol["ontology_sources"]["reactome"]["release"],
        "safe_go_relations": protocol["ontology_sources"]["go"]["allowed_relations"],
        "external_files": external_files,
        "repository_files": [file_record(path, repo_root) for path in repository_inputs],
        "privacy_boundary": {
            "priority3_grades_read": False,
            "priority4_ratings_read": False,
            "hierarchy_used_by_audit": False,
        },
    }
    return manifest_path, manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--freeze-label", required=True)
    parser.add_argument("--use-system-trust", action="store_true")
    args = parser.parse_args()

    data_root = args.data_root.expanduser().resolve()
    require(data_root.is_dir(), f"Data root does not exist: {data_root}")
    manifest_path, manifest = build_manifest(
        data_root=data_root,
        repo_root=repo_root_from_script(),
        use_system_trust=bool(args.use_system_trust),
        freeze_label=str(args.freeze_label),
    )
    payload = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    atomic_write_text(manifest_path, payload)
    checksum_path = manifest_path.with_suffix(".sha256")
    atomic_write_text(checksum_path, f"{sha256_file(manifest_path)}  {manifest_path.name}\n")

    print("[PASS] Priority 5 audit inputs and ontology snapshots frozen")
    print(f"[INFO] GO release: {manifest['go_release']['data_version']}")
    print(f"[INFO] Reactome release: {manifest['reactome_release']}")
    print("[INFO] P3 grades and P4 ratings were not read")
    print(f"[INFO] Wrote: {manifest_path}")
    print("[NEXT] Run 51_check_priority5_freeze.py before hierarchy evaluation")


if __name__ == "__main__":
    main()
