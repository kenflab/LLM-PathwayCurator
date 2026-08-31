#!/usr/bin/env python3
"""Verify the immutable Priority 4 ratings lock."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from v11_lock_common import read_json, require, sha256_file

CRM_DIR = Path(__file__).resolve().parents[1]
DEFAULT_P3_PROTOCOL = CRM_DIR / "config" / "priority3_protocol.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--p3-protocol", type=Path, default=DEFAULT_P3_PROTOCOL)
    args = parser.parse_args()
    benchmark = str(read_json(args.p3_protocol)["parent_priority2_benchmark_id"])
    root = (
        args.data_root.expanduser().resolve()
        / "output"
        / "priority4"
        / benchmark
        / "ratings_lock_v1"
    )
    manifest_path = root / "priority4_ratings_lock_manifest.json"
    digest_path = root / "priority4_ratings_lock_manifest.sha256"
    require(manifest_path.is_file() and digest_path.is_file(), "Missing P4 ratings lock")
    require(
        sha256_file(manifest_path) == digest_path.read_text().split()[0],
        "P4 manifest digest mismatch",
    )
    manifest = read_json(manifest_path)
    require(
        manifest.get("status") == "P4_RATINGS_LOCKED_BEFORE_METHOD_UNBLINDING",
        "P4 lock status drift",
    )
    require(
        manifest.get("method_membership_read") is False,
        "P4 method-unblinding flag drift",
    )
    paths = {}
    for label, record in {**manifest["inputs"], **manifest["outputs"]}.items():
        path = Path(record["path"])
        require(path.is_file(), f"Missing P4 lock file: {label}")
        require(sha256_file(path) == record["sha256"], f"P4 hash mismatch: {label}")
        paths[label] = path
    consensus = pd.read_csv(paths["rating_consensus"], sep="\t")
    agreement = pd.read_csv(paths["interrater_agreement"], sep="\t")
    require(
        len(consensus) == 50 and consensus["review_id"].is_unique,
        "P4 consensus census drift",
    )
    require(
        set(agreement["question"])
        == {"q1_statistical_support", "q2_external_evidence", "q3_overstatement"},
        "P4 agreement inventory drift",
    )
    print("[PASS] Priority 4 ratings lock is complete, blinded, and hash-consistent")
    print(
        f"[INFO] benchmark={benchmark}; claims=50; "
        f"raters={manifest['raters']}; label={manifest['lock_label']}"
    )
    print("[GO] P3 and P4 outcomes may now be joined to the frozen Priority 2 memberships")


if __name__ == "__main__":
    main()
