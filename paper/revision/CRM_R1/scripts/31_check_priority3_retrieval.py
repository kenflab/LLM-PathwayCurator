#!/usr/bin/env python3
"""Verify the frozen Priority 3 PubMed retrieval before grading or P4 packets."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd

CRM_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = CRM_DIR.parents[2]
DEFAULT_P2_CONFIG = CRM_DIR / "config" / "priorities2_5_protocol.json"
DEFAULT_P3_CONFIG = CRM_DIR / "config" / "priority3_protocol.json"
DEFAULT_P2_CHECKER = Path(__file__).resolve().with_name("21_check_priority2_freeze.py")

FORBIDDEN_BLINDED_COLUMNS = {
    "claim_uid",
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
GRADING_COLUMNS = {
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


def run_p2_gate(*, checker: Path, data_root: Path, config: Path) -> None:
    result = subprocess.run(
        [sys.executable, str(checker), "--data-root", str(data_root), "--config", str(config)],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    output = result.stdout + result.stderr
    print(output, end="")
    require(result.returncode == 0, "Priority 2 freeze gate failed")
    require("[GO] P3 evidence retrieval" in output, "Priority 2 checker did not release P3")


def require_hashed_file(record: dict[str, Any], *, label: str) -> Path:
    path = Path(str(record["path"]))
    require(path.is_file(), f"Missing recorded file {label}: {path}")
    require(sha256_file(path) == record["sha256"], f"Hash mismatch for {label}: {path}")
    return path


def grading_is_blank(frame: pd.DataFrame) -> bool:
    for column in GRADING_COLUMNS:
        require(column in frame, f"Screening template lacks grading column: {column}")
        if frame[column].fillna("").astype(str).str.strip().ne("").any():
            return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--p2-config", type=Path, default=DEFAULT_P2_CONFIG)
    parser.add_argument("--p3-config", type=Path, default=DEFAULT_P3_CONFIG)
    parser.add_argument("--p2-checker", type=Path, default=DEFAULT_P2_CHECKER)
    args = parser.parse_args()

    data_root = args.data_root.resolve()
    run_p2_gate(checker=args.p2_checker, data_root=data_root, config=args.p2_config)
    protocol = read_json(args.p3_config)
    benchmark_id = str(protocol["parent_priority2_benchmark_id"])
    root = data_root / "output" / "priority3" / benchmark_id
    manifest_path = root / "retrieval" / "priority3_retrieval_manifest.json"
    digest_path = root / "retrieval" / "priority3_retrieval_manifest.sha256"
    require(manifest_path.is_file(), f"Missing Priority 3 manifest: {manifest_path}")
    require(digest_path.is_file(), f"Missing Priority 3 manifest digest: {digest_path}")
    expected_digest = digest_path.read_text(encoding="utf-8").split()[0]
    require(sha256_file(manifest_path) == expected_digest, "Priority 3 manifest digest mismatch")

    manifest = read_json(manifest_path)
    require(
        manifest.get("status") == "RETRIEVAL_FROZEN_GRADING_NOT_STARTED",
        "Priority 3 retrieval is not in the pre-grading frozen state",
    )
    require(manifest.get("protocol_version") == protocol["protocol_version"], "P3 protocol drift")
    require(manifest.get("benchmark_id") == benchmark_id, "P3 benchmark drift")
    require(manifest.get("grading_outcomes_inspected") is False, "P3 grading leakage flag")
    tls = manifest.get("tls_verification")
    require(isinstance(tls, dict), "P3 TLS verification metadata is missing")
    require(
        tls.get("mode") in {"python_default", "macos_native_system_trust"},
        "P3 TLS verification mode is invalid",
    )
    if tls.get("mode") == "macos_native_system_trust":
        require(tls.get("implementation") == "truststore", "P3 native trust implementation drift")
        require(bool(tls.get("truststore_version")), "P3 truststore version is missing")
    require(int(manifest.get("candidate_claims", -1)) == 50, "P3 claim-census drift")
    expected_queries = 50 * len(protocol["query_families"])
    require(int(manifest.get("queries", -1)) == expected_queries, "P3 query-count drift")

    input_paths = {
        label: require_hashed_file(record, label=f"input:{label}")
        for label, record in manifest["inputs"].items()
    }
    require(
        input_paths["p3_protocol"].resolve() == args.p3_config.resolve(),
        "P3 config path drift",
    )

    output_paths: dict[str, Path] = {}
    for label, record in manifest["outputs"].items():
        if label == "efetch_xml_batches":
            require(isinstance(record, list), "EFetch batch inventory is malformed")
            for index, item in enumerate(record, start=1):
                require_hashed_file(item, label=f"efetch_xml_batch:{index}")
            continue
        output_paths[label] = require_hashed_file(record, label=f"output:{label}")

    required_outputs = {
        "query_manifest",
        "links",
        "records",
        "claims_blinded",
        "screening",
        "grading_instructions",
    }
    require(required_outputs <= set(output_paths), "P3 manifest lacks required outputs")
    queries = pd.read_csv(output_paths["query_manifest"], sep="\t")
    links = pd.read_csv(output_paths["links"], sep="\t", dtype={"pmid": str})
    records = pd.read_csv(output_paths["records"], sep="\t", dtype={"pmid": str})
    claims = pd.read_csv(output_paths["claims_blinded"], sep="\t")
    screening = pd.read_csv(output_paths["screening"], sep="\t", dtype={"pmid": str})

    families = set(map(str, protocol["query_families"]))
    require(len(claims) == 50, "P3 blinded claim census is not 50")
    require(claims["review_id"].is_unique, "P3 review IDs are not unique")
    require(len(queries) == expected_queries, "P3 query manifest row-count drift")
    require(queries["query_id"].is_unique, "P3 query IDs are not unique")
    require(set(queries["review_id"]) == set(claims["review_id"]), "P3 query claim pool drift")
    per_claim = queries.groupby("review_id")["query_family"].agg(set)
    require(len(per_claim) == 50, "P3 does not contain every claim")
    require(per_claim.map(lambda observed: observed == families).all(), "P3 query-family drift")
    require(
        queries.groupby(["review_id", "query_family"]).size().eq(1).all(),
        "P3 contains duplicate claim-by-family queries",
    )
    require(set(queries["sort"].astype(str)) == {str(protocol["sort"])}, "P3 sort drift")
    require(
        pd.to_numeric(queries["retmax"], errors="raise")
        .eq(int(protocol["retmax_per_query"]))
        .all(),
        "P3 retmax drift",
    )
    require(pd.to_numeric(queries["total_hits"], errors="raise").ge(0).all(), "Negative hits")
    returned = pd.to_numeric(queries["returned_pmids"], errors="raise")
    require(returned.between(0, int(protocol["retmax_per_query"])).all(), "Bad returned count")
    require(queries["query"].str.contains("Publication Type", regex=False).all(), "No exclusions")

    require(set(links["query_id"]) <= set(queries["query_id"]), "P3 link has unknown query")
    require(not links.duplicated(["query_id", "pmid"]).any(), "Duplicate PMID within a query")
    if not links.empty:
        expected_link_counts = queries.set_index("query_id")["returned_pmids"].astype(int)
        observed_link_counts = (
            links.groupby("query_id").size().reindex(expected_link_counts.index, fill_value=0)
        )
        require(observed_link_counts.eq(expected_link_counts).all(), "P3 link-count drift")
    linked_pmids = set(links["pmid"].dropna().astype(str))
    fetched_pmids = set(records["pmid"].dropna().astype(str))
    missing_pmids = set(map(str, manifest.get("missing_pmids", [])))
    require(linked_pmids == fetched_pmids | missing_pmids, "P3 PMID inventory drift")
    require(fetched_pmids.isdisjoint(missing_pmids), "Fetched and missing PMIDs overlap")

    forbidden = FORBIDDEN_BLINDED_COLUMNS & (set(claims.columns) | set(screening.columns))
    require(
        not forbidden,
        f"Method-blinding fields leaked into P3 grading files: {sorted(forbidden)}",
    )
    require(grading_is_blank(screening), "P3 grading was started after retrieval freeze")
    require(set(screening["review_id"]) <= set(claims["review_id"]), "Screening claim drift")
    print("[PASS] Priority 3 retrieval bundle is complete, blinded, and hash-consistent")
    print(
        f"[INFO] benchmark={benchmark_id}; claims=50; queries={len(queries)}; "
        f"unique records={len(records)}"
    )
    print("[INFO] Grading fields remain blank; audit membership was not disclosed")
    print("[GO] P4 blinded packet preparation may begin; keep P3 grading fields blank")


if __name__ == "__main__":
    main()
