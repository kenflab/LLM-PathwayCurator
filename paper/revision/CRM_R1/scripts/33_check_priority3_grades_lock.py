#!/usr/bin/env python3
"""Verify the immutable Priority 3 grading lock."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from v11_lock_common import read_json, require, sha256_file

CRM_DIR = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL = CRM_DIR / "config" / "priority3_protocol.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    args = parser.parse_args()
    protocol = read_json(args.protocol)
    benchmark = str(protocol["parent_priority2_benchmark_id"])
    root = (
        args.data_root.expanduser().resolve()
        / "output"
        / "priority3"
        / benchmark
        / "grading_lock_v1"
    )
    manifest_path = root / "priority3_grading_lock_manifest.json"
    digest_path = root / "priority3_grading_lock_manifest.sha256"
    require(manifest_path.is_file() and digest_path.is_file(), "Missing P3 grading lock")
    require(
        sha256_file(manifest_path) == digest_path.read_text().split()[0],
        "P3 manifest digest mismatch",
    )
    manifest = read_json(manifest_path)
    require(
        manifest.get("status") == "P3_GRADES_LOCKED_BEFORE_METHOD_UNBLINDING",
        "P3 lock status drift",
    )
    require(
        manifest.get("method_membership_read") is False,
        "P3 method-unblinding flag drift",
    )
    paths = {}
    for label, record in {**manifest["inputs"], **manifest["outputs"]}.items():
        path = Path(record["path"])
        require(path.is_file(), f"Missing P3 lock file: {label}")
        require(sha256_file(path) == record["sha256"], f"P3 hash mismatch: {label}")
        paths[label] = path
    claims = pd.read_csv(paths["claim_evidence"], sep="\t")
    require(
        len(claims) == 50 and claims["review_id"].is_unique,
        "P3 locked claim census drift",
    )
    require(
        claims["maximum_evidence_grade"].isin({"E0", "E1", "E2", "E3", "E4"}).all(),
        "P3 claim-grade drift",
    )
    print("[PASS] Priority 3 grading lock is complete, blinded, and hash-consistent")
    print(f"[INFO] benchmark={benchmark}; claims=50; label={manifest['lock_label']}")
    print("[GO] P3 outcomes may be joined to method membership only after the P4 lock passes")


if __name__ == "__main__":
    main()
