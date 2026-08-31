#!/usr/bin/env python3
"""Verify immutable method-blinded Priority 4 packets before distribution."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

CRM_DIR = Path(__file__).resolve().parents[1]
DEFAULT_P3_CONFIG = CRM_DIR / "config" / "priority3_protocol.json"
DEFAULT_P4_CONFIG = CRM_DIR / "config" / "priority4_review_protocol.json"

FORBIDDEN_COLUMNS = {
    "claim_uid",
    "claim_id",
    "audit_status",
    "method_membership",
    "term_survival",
    "context_status",
    "context_reason",
    "context_confidence",
    "status_full",
    "full_audit_selected",
    "q_value_matched_selected",
    "stability_matched_selected",
}
RATING_FIELDS = {
    "q1_statistical_support",
    "q2_external_evidence",
    "q3_overstatement",
    "confidence_1_to_5",
    "concise_rationale",
}


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


def require_hashed_file(record: dict[str, Any], *, label: str) -> Path:
    path = Path(str(record["path"]))
    require(path.is_file(), f"Missing recorded file {label}: {path}")
    require(sha256_file(path) == record["sha256"], f"Hash mismatch for {label}: {path}")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--p3-config", type=Path, default=DEFAULT_P3_CONFIG)
    parser.add_argument("--p4-config", type=Path, default=DEFAULT_P4_CONFIG)
    args = parser.parse_args()

    p3_protocol = read_json(args.p3_config)
    p4_protocol = read_json(args.p4_config)
    benchmark_id = str(p3_protocol["parent_priority2_benchmark_id"])
    root = args.data_root.resolve() / "output" / "priority4" / benchmark_id / "packet_v1"
    manifest_path = root / "priority4_packet_manifest.json"
    digest_path = root / "priority4_packet_manifest.sha256"
    require(manifest_path.is_file(), f"Missing P4 packet manifest: {manifest_path}")
    require(digest_path.is_file(), f"Missing P4 packet digest: {digest_path}")
    expected_digest = digest_path.read_text(encoding="utf-8").split()[0]
    require(sha256_file(manifest_path) == expected_digest, "P4 packet manifest digest mismatch")
    manifest = read_json(manifest_path)
    require(
        manifest.get("status") == "PACKETS_FROZEN_RATINGS_NOT_STARTED",
        "P4 packet is not frozen in its pre-rating state",
    )
    require(manifest.get("protocol_version") == p4_protocol["protocol_version"], "P4 drift")
    require(manifest.get("benchmark_id") == benchmark_id, "P4 benchmark drift")
    require(int(manifest.get("candidate_claims", -1)) == 50, "P4 claim-census drift")
    require(
        int(manifest.get("independent_raters", -1))
        == int(p4_protocol["minimum_independent_raters"]),
        "P4 rater-count drift",
    )
    require(manifest.get("method_membership_disclosed") is False, "P4 masking leakage flag")
    require(manifest.get("ratings_inspected") is False, "P4 ratings leakage flag")

    for label, record in manifest["inputs"].items():
        require_hashed_file(record, label=f"input:{label}")
    outputs = {
        label: require_hashed_file(record, label=f"output:{label}")
        for label, record in manifest["outputs"].items()
    }
    claims = pd.read_csv(outputs["claims_packet"], sep="\t")
    literature = pd.read_csv(outputs["literature_packet"], sep="\t", dtype={"pmid": str})
    require(len(claims) == 50, "P4 claim packet row-count drift")
    require(claims["review_id"].is_unique, "P4 review IDs are not unique")
    require(set(literature["review_id"]) <= set(claims["review_id"]), "P4 literature pool drift")
    require(
        literature.groupby(["review_id", "query_family"])
        .size()
        .le(int(p4_protocol["packet_literature_limit_per_query_family"]))
        .all(),
        "P4 literature limit drift",
    )

    rating_labels = sorted(label for label in outputs if label.startswith("ratings_R"))
    require(
        len(rating_labels) == int(p4_protocol["minimum_independent_raters"]),
        "P4 rating-template inventory drift",
    )
    packet_paths = [outputs["claims_packet"], outputs["literature_packet"]]
    for label in rating_labels:
        ratings = pd.read_csv(outputs[label], sep="\t")
        require(len(ratings) == 50, f"{label} row-count drift")
        require(set(ratings["review_id"]) == set(claims["review_id"]), f"{label} pool drift")
        require(RATING_FIELDS <= set(ratings), f"{label} lacks rating fields")
        for column in RATING_FIELDS:
            require(
                ratings[column].fillna("").astype(str).str.strip().eq("").all(),
                f"{label} was edited before packet release",
            )
        packet_paths.append(outputs[label])

    for path in packet_paths:
        columns = set(pd.read_csv(path, sep="\t", nrows=0).columns)
        leaked = columns & FORBIDDEN_COLUMNS
        require(not leaked, f"Method-blinding fields leaked into {path.name}: {sorted(leaked)}")
    print("[PASS] Priority 4 packets are complete, blank, masked, and hash-consistent")
    print(
        f"[INFO] benchmark={benchmark_id}; claims=50; "
        f"independent rating templates={len(rating_labels)}"
    )
    print("[GO] Distribute one copied ratings template per independent rater")


if __name__ == "__main__":
    main()
