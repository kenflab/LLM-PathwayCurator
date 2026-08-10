#!/usr/bin/env python3
"""Verify the immutable Priority 5 pre-hierarchy input freeze."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


BENCHMARK_ID = "PANCAN_TP53_v1_HNSC_R1_P5"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def repo_root_from_script() -> Path:
    return Path(__file__).resolve().parents[4]


def resolve_record(record: dict[str, Any], root: Path) -> Path:
    path = (root / record["path"]).resolve()
    require(path.is_file(), f"Frozen file is missing: {path}")
    require(path.stat().st_size == int(record["bytes"]), f"Size drift: {path}")
    require(sha256_file(path) == record["sha256"], f"SHA-256 drift: {path}")
    return path


def check_priority5_freeze(
    data_root: Path, *, allow_outputs: bool = False
) -> dict[str, Any]:
    repo_root = repo_root_from_script()
    freeze_dir = data_root / "output/priority5" / BENCHMARK_ID / "freeze"
    manifest_path = freeze_dir / "priority5_input_manifest.json"
    checksum_path = freeze_dir / "priority5_input_manifest.sha256"
    require(manifest_path.is_file(), f"Missing P5 manifest: {manifest_path}")
    require(checksum_path.is_file(), f"Missing P5 checksum: {checksum_path}")
    expected = checksum_path.read_text(encoding="utf-8").split()[0]
    require(sha256_file(manifest_path) == expected, "P5 manifest checksum mismatch")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    require(
        manifest["schema_version"] == "CRM_R1_PRIORITY5_INPUT_FREEZE_v1",
        "Wrong P5 freeze schema",
    )
    require(
        manifest["status"] == "FROZEN_BEFORE_HIERARCHY_EVALUATION",
        "P5 freeze status is not valid",
    )
    require(manifest["benchmark_id"] == BENCHMARK_ID, "P5 benchmark drift")
    require(manifest["audit_configuration"]["tau"] == 0.9, "P5 tau drift")
    require(
        manifest["audit_configuration"]["k_claims_per_collection"] == 500,
        "P5 claim-count drift",
    )
    require(
        manifest["safe_go_relations"] == ["is_a", "part_of"],
        "Unsafe or changed GO propagation relations",
    )
    boundary = manifest["privacy_boundary"]
    require(not boundary["priority3_grades_read"], "P3 outcome leakage")
    require(not boundary["priority4_ratings_read"], "P4 outcome leakage")
    require(not boundary["hierarchy_used_by_audit"], "Hierarchy leaked into audit")

    for record in manifest["external_files"]:
        resolve_record(record, data_root)
    for record in manifest["repository_files"]:
        resolve_record(record, repo_root)

    output = (
        data_root / "output/priority5" / BENCHMARK_ID / "ontology/hierarchy_metrics.tsv"
    )
    if not allow_outputs:
        require(not output.exists(), "Hierarchy output exists before the release gate")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--allow-outputs", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    manifest = check_priority5_freeze(
        args.data_root.expanduser().resolve(), allow_outputs=bool(args.allow_outputs)
    )
    print("[PASS] Priority 5 input freeze is complete and hash-consistent")
    print(
        "[INFO] "
        f"GO={manifest['go_release']['data_version']}; "
        f"Reactome={manifest['reactome_release']}; tau=0.90; k=500/collection"
    )
    if args.allow_outputs:
        print(
            "[INFO] Post-evaluation provenance check; existing P5 outputs were allowed"
        )
    else:
        print("[GO] Ontology hierarchy evaluation may now run exactly once")


if __name__ == "__main__":
    main()
