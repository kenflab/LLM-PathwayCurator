#!/usr/bin/env python3
"""Verify the immutable Priority 2 claim-pool freeze before P3 or P4 starts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

CRM_DIR = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = CRM_DIR / "config" / "priorities2_5_protocol.json"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    require(isinstance(value, dict), f"JSON root must be an object: {path}")
    return value


def selected(values: pd.Series, *, column: str) -> pd.Series:
    if pd.api.types.is_bool_dtype(values.dtype):
        return values.fillna(False).astype(bool)
    normalized = values.fillna("").astype(str).str.strip().str.lower()
    require(set(normalized) <= {"true", "false"}, f"Invalid Boolean values in {column}")
    return normalized.eq("true")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()

    protocol = read_json(args.config)
    p2 = protocol["priority2"]
    benchmark_id = str(p2["benchmark_id"])
    root = args.data_root.resolve() / "output" / "priority2" / benchmark_id
    manifest_path = root / "metrics" / "priority2_freeze_manifest.json"
    digest_path = root / "metrics" / "priority2_freeze_manifest.sha256"
    require(manifest_path.is_file(), f"Missing Priority 2 manifest: {manifest_path}")
    require(digest_path.is_file(), f"Missing Priority 2 manifest digest: {digest_path}")
    expected_digest = digest_path.read_text(encoding="utf-8").split()[0]
    require(sha256_file(manifest_path) == expected_digest, "Priority 2 manifest digest mismatch")
    manifest = read_json(manifest_path)
    require(manifest.get("status") == "FROZEN", "Priority 2 manifest is not frozen")
    require(manifest.get("protocol_version") == protocol["protocol_version"], "Protocol drift")
    require(manifest.get("benchmark_id") == benchmark_id, "Benchmark drift")
    require(float(manifest.get("primary_tau")) == float(p2["primary_tau"]), "Tau drift")
    require(
        int(manifest.get("candidate_count")) == int(p2["candidate_count_expected"]),
        "Candidate count drift",
    )
    require(manifest.get("validation_outcomes_inspected") is False, "Outcome leakage flag")

    for group in ("inputs", "outputs"):
        records = manifest.get(group)
        require(isinstance(records, dict) and records, f"Manifest has no {group}")
        for label, record in records.items():
            path = Path(str(record["path"]))
            require(path.is_file(), f"Missing recorded {group} file {label}: {path}")
            require(sha256_file(path) == record["sha256"], f"Hash mismatch for {label}: {path}")

    output_paths = {key: Path(value["path"]) for key, value in manifest["outputs"].items()}
    claims = pd.read_csv(output_paths["claims"], sep="\t")
    membership = pd.read_csv(output_paths["membership"], sep="\t")
    sampling = pd.read_csv(output_paths["sampling_frame"], sep="\t")
    expected_n = int(p2["candidate_count_expected"])
    require(len(claims) == len(membership) == len(sampling) == expected_n, "Row-count drift")
    require(claims["claim_uid"].is_unique, "Claim UIDs are not unique")
    require(sampling["review_id"].is_unique, "Review IDs are not unique")
    require(set(claims["claim_uid"]) == set(membership["claim_uid"]), "Membership pool drift")
    require(set(claims["claim_uid"]) == set(sampling["claim_uid"]), "Sampling pool drift")
    matched_k = int(manifest["matched_k"])
    for column in (
        "q_value_matched_selected",
        "stability_matched_selected",
        "full_audit_selected",
    ):
        require(
            int(selected(membership[column], column=column).sum()) == matched_k,
            f"K drift: {column}",
        )
    require(
        int(selected(membership["raw_pool_selected"], column="raw_pool_selected").sum())
        == expected_n,
        "Raw-pool membership drift",
    )
    print("[PASS] Priority 2 freeze bundle is complete and hash-consistent")
    print(
        f"[INFO] benchmark={benchmark_id}; candidates={expected_n}; "
        f"matched K={matched_k}; tau={float(p2['primary_tau']):.2f}"
    )
    print("[GO] P3 evidence retrieval and P4 blinded packet preparation may begin")


if __name__ == "__main__":
    main()
