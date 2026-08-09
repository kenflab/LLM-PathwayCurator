#!/usr/bin/env python3
"""Build immutable, method-blinded Priority 4 review packets for three raters."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

CRM_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = CRM_DIR.parents[2]
DEFAULT_P2_CONFIG = CRM_DIR / "config" / "priorities2_5_protocol.json"
DEFAULT_P3_CONFIG = CRM_DIR / "config" / "priority3_protocol.json"
DEFAULT_P4_CONFIG = CRM_DIR / "config" / "priority4_review_protocol.json"
DEFAULT_P2_CHECKER = Path(__file__).resolve().with_name("21_check_priority2_freeze.py")
DEFAULT_P3_CHECKER = Path(__file__).resolve().with_name("31_check_priority3_retrieval.py")

FORBIDDEN_PACKET_COLUMNS = {
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


def git_value(*arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments], cwd=REPO_ROOT, check=True, capture_output=True, text=True
    )
    return result.stdout.strip()


def require_clean_tracked_worktree() -> None:
    status = git_value("status", "--porcelain", "--untracked-files=no")
    require(
        not status, f"Commit V9 and leave tracked files clean before P4 packet freeze: {status}"
    )


def run_gate(command: list[str], *, expected_release: str, name: str) -> None:
    result = subprocess.run(
        command,
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    output = result.stdout + result.stderr
    print(output, end="")
    require(result.returncode == 0, f"{name} failed")
    require(expected_release in output, f"{name} did not release the next step")


def top_gene_symbols(value: Any, *, limit: int = 20) -> str:
    values = [item.strip() for item in re.split(r"[;,]", str(value or "")) if item.strip()]
    return "; ".join(list(dict.fromkeys(values))[:limit])


def make_claim_packet(
    *,
    claims: pd.DataFrame,
    sampling: pd.DataFrame,
    mechanical_audit: pd.DataFrame,
) -> pd.DataFrame:
    required_claims = {
        "claim_uid",
        "claim_id_mechanical",
        "claim_text",
        "pathway_label",
        "direction",
        "statistic",
        "q_value",
    }
    require(required_claims <= set(claims), "Frozen claims lack P4 evidence fields")
    require({"review_id", "packet_order", "claim_uid"} <= set(sampling), "Bad sampling frame")
    require("claim_id" in mechanical_audit, "Mechanical audit lacks claim_id")
    gene_column = "gene_symbols_str" if "gene_symbols_str" in mechanical_audit else "gene_ids_str"
    require(gene_column in mechanical_audit, "Mechanical audit lacks supporting-gene fields")
    support = mechanical_audit[["claim_id", gene_column]].rename(
        columns={"claim_id": "claim_id_mechanical", gene_column: "supporting_genes"}
    )
    require(support["claim_id_mechanical"].is_unique, "Mechanical claim IDs are not unique")
    merged = sampling.merge(claims, on="claim_uid", how="left", validate="one_to_one").merge(
        support, on="claim_id_mechanical", how="left", validate="one_to_one"
    )
    require(merged["claim_text"].notna().all(), "P4 claim mapping failed")
    require(merged["supporting_genes"].notna().all(), "P4 supporting-gene mapping failed")
    merged["leading_edge_genes_top20"] = merged["supporting_genes"].map(top_gene_symbols)
    merged["leading_edge_gene_count"] = merged["supporting_genes"].map(
        lambda value: len({item.strip() for item in re.split(r"[;,]", str(value)) if item.strip()})
    )
    packet = merged[
        [
            "review_id",
            "packet_order",
            "claim_text",
            "pathway_label",
            "direction",
            "statistic",
            "q_value",
            "leading_edge_gene_count",
            "leading_edge_genes_top20",
        ]
    ].sort_values("packet_order")
    require(len(packet) == 50, "P4 claim packet is not the 50-claim census")
    require(packet["review_id"].is_unique, "P4 review IDs are not unique")
    return packet.reset_index(drop=True)


def make_literature_packet(
    *,
    links: pd.DataFrame,
    records: pd.DataFrame,
    claim_packet: pd.DataFrame,
    limit_per_family: int,
) -> pd.DataFrame:
    columns = [
        "review_id",
        "packet_order",
        "query_family",
        "query_rank",
        "pmid",
        "title",
        "publication_year",
        "journal",
        "publication_types",
        "doi",
        "pmcid",
        "abstract_available",
        "abstract",
    ]
    if links.empty:
        return pd.DataFrame(columns=columns)
    required_links = {"review_id", "query_family", "query_rank", "pmid"}
    require(required_links <= set(links), "P3 links lack P4 packet fields")
    links = links.copy()
    links["query_rank"] = pd.to_numeric(links["query_rank"], errors="raise").astype(int)
    links = links.loc[links["query_rank"].le(limit_per_family)]
    literature = links.merge(records, on="pmid", how="inner", validate="many_to_one").merge(
        claim_packet[["review_id", "packet_order"]],
        on="review_id",
        how="left",
        validate="many_to_one",
    )
    require(literature["title"].notna().all(), "P4 packet includes a record without a title")
    for column in columns:
        if column not in literature:
            literature[column] = ""
    literature = literature[columns].sort_values(
        ["packet_order", "query_family", "query_rank", "pmid"]
    )
    require(
        literature.groupby(["review_id", "query_family"]).size().le(limit_per_family).all(),
        "P4 literature limit drift",
    )
    return literature.reset_index(drop=True)


def make_rating_template(claim_packet: pd.DataFrame, *, rater_id: str) -> pd.DataFrame:
    template = claim_packet[["review_id", "packet_order"]].copy()
    template.insert(0, "rater_id", rater_id)
    template["q1_statistical_support"] = ""
    template["q2_external_evidence"] = ""
    template["q3_overstatement"] = ""
    template["confidence_1_to_5"] = ""
    template["concise_rationale"] = ""
    return template


def review_instructions(protocol: dict[str, Any]) -> str:
    questions = protocol["questions"]
    q1 = ", ".join(questions["q1_statistical_support"])
    q2 = ", ".join(questions["q2_external_evidence"])
    q3 = ", ".join(questions["q3_overstatement"])
    return f"""# Priority 4 independent review instructions

Review all 50 fixed pathway claims independently and in packet order. Do not coordinate with other
raters and do not perform additional literature searches. Use only `blinded_claims.tsv` and
`blinded_literature.private.tsv`. The literature file is private and must not be redistributed.

For each `review_id`, enter exactly one response for each question in your assigned ratings file:

1. Statistical support: `{q1}`.
2. External-evidence directness: `{q2}`.
3. Wording overstatement: `{q3}`.
4. Confidence: an integer from 1 through 5.
5. Concise rationale: one or two sentences tied to the provided evidence.

Judge the wording actually shown. A statistically enriched gene set does not by itself establish
mechanism, causality, or clinical utility. `DIRECT` external evidence requires a closely matched
HNSC/TP53/pathway relationship; broader TP53 or HNSC relevance is `INDIRECT`. Absence of support in
the frozen retrieval is `NO_SUPPORT`, not proof that the claim is false. Use `UNCERTAIN` when the
provided material is insufficient for a defensible choice.

Do not add names, initials, or free-text identifiers outside the assigned `rater_id`. Return only
your completed ratings TSV to the coordinating analyst.
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--p2-config", type=Path, default=DEFAULT_P2_CONFIG)
    parser.add_argument("--p3-config", type=Path, default=DEFAULT_P3_CONFIG)
    parser.add_argument("--p4-config", type=Path, default=DEFAULT_P4_CONFIG)
    parser.add_argument("--p2-checker", type=Path, default=DEFAULT_P2_CHECKER)
    parser.add_argument("--p3-checker", type=Path, default=DEFAULT_P3_CHECKER)
    args = parser.parse_args()

    require_clean_tracked_worktree()
    data_root = args.data_root.resolve()
    p3_protocol = read_json(args.p3_config)
    p4_protocol = read_json(args.p4_config)
    require(
        p4_protocol["status"] == "PACKET_SCHEMA_FROZEN_BEFORE_RATING",
        "P4 packet schema is not frozen",
    )
    run_gate(
        [
            sys.executable,
            str(args.p2_checker),
            "--data-root",
            str(data_root),
            "--config",
            str(args.p2_config),
        ],
        expected_release="[GO] P3 evidence retrieval",
        name="Priority 2 freeze gate",
    )
    run_gate(
        [
            sys.executable,
            str(args.p3_checker),
            "--data-root",
            str(data_root),
            "--p2-config",
            str(args.p2_config),
            "--p3-config",
            str(args.p3_config),
        ],
        expected_release="[GO] P4 blinded packet preparation",
        name="Priority 3 retrieval gate",
    )

    benchmark_id = str(p3_protocol["parent_priority2_benchmark_id"])
    p2_root = data_root / "output" / "priority2" / benchmark_id
    p3_root = data_root / "output" / "priority3" / benchmark_id
    p4_root = data_root / "output" / "priority4" / benchmark_id / "packet_v1"
    p2_manifest_path = p2_root / "metrics" / "priority2_freeze_manifest.json"
    p3_manifest_path = p3_root / "retrieval" / "priority3_retrieval_manifest.json"
    p2_manifest = read_json(p2_manifest_path)
    p3_manifest = read_json(p3_manifest_path)
    p2_outputs = {label: Path(item["path"]) for label, item in p2_manifest["outputs"].items()}
    p3_outputs = {
        label: Path(item["path"])
        for label, item in p3_manifest["outputs"].items()
        if label != "efetch_xml_batches"
    }
    mechanical_path = Path(p2_manifest["inputs"]["mechanical_audit"]["path"])
    claims = pd.read_csv(p2_outputs["claims"], sep="\t")
    sampling = pd.read_csv(p2_outputs["sampling_frame"], sep="\t")
    mechanical = pd.read_csv(mechanical_path, sep="\t")
    links = pd.read_csv(p3_outputs["links"], sep="\t", dtype={"pmid": str})
    records = pd.read_csv(p3_outputs["records"], sep="\t", dtype={"pmid": str})

    claim_packet = make_claim_packet(claims=claims, sampling=sampling, mechanical_audit=mechanical)
    literature_packet = make_literature_packet(
        links=links,
        records=records,
        claim_packet=claim_packet,
        limit_per_family=int(p4_protocol["packet_literature_limit_per_query_family"]),
    )
    ratings = {
        f"ratings_R{index}": make_rating_template(claim_packet, rater_id=f"P4_R{index}")
        for index in range(1, int(p4_protocol["minimum_independent_raters"]) + 1)
    }
    paths = {
        "claims_packet": p4_root / "blinded_claims.tsv",
        "literature_packet": p4_root / "blinded_literature.private.tsv",
        "instructions": p4_root / "REVIEW_INSTRUCTIONS.md",
        **{
            label: p4_root / f"ratings_template_{label.removeprefix('ratings_')}.tsv"
            for label in ratings
        },
        "manifest": p4_root / "priority4_packet_manifest.json",
        "manifest_sha256": p4_root / "priority4_packet_manifest.sha256",
    }
    collisions = [str(path) for path in paths.values() if path.exists()]
    if p4_root.exists() and any(p4_root.iterdir()):
        collisions.append(str(p4_root))
    require(not collisions, f"P4 packets are immutable; output collisions: {collisions}")
    p4_root.mkdir(parents=True, exist_ok=True)
    claim_packet.to_csv(paths["claims_packet"], sep="\t", index=False, lineterminator="\n")
    literature_packet.to_csv(paths["literature_packet"], sep="\t", index=False, lineterminator="\n")
    for label, table in ratings.items():
        table.to_csv(paths[label], sep="\t", index=False, lineterminator="\n")
    paths["instructions"].write_text(review_instructions(p4_protocol), encoding="utf-8")

    for label in ("claims_packet", "literature_packet", *ratings):
        columns = set(pd.read_csv(paths[label], sep="\t", nrows=0).columns)
        leaked = columns & FORBIDDEN_PACKET_COLUMNS
        require(not leaked, f"P4 method-blinding fields leaked into {label}: {sorted(leaked)}")
    input_paths = {
        "p2_manifest": p2_manifest_path,
        "p2_manifest_sha256": p2_root / "metrics" / "priority2_freeze_manifest.sha256",
        "p3_manifest": p3_manifest_path,
        "p3_manifest_sha256": p3_root / "retrieval" / "priority3_retrieval_manifest.sha256",
        "p2_claims": p2_outputs["claims"],
        "p2_sampling_frame": p2_outputs["sampling_frame"],
        "p2_mechanical_audit": mechanical_path,
        "p3_links": p3_outputs["links"],
        "p3_records_private": p3_outputs["records"],
        "p2_protocol": args.p2_config,
        "p3_protocol": args.p3_config,
        "p4_protocol": args.p4_config,
    }
    output_labels = ["claims_packet", "literature_packet", "instructions", *ratings]
    manifest = {
        "protocol_version": p4_protocol["protocol_version"],
        "status": "PACKETS_FROZEN_RATINGS_NOT_STARTED",
        "benchmark_id": benchmark_id,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "git_commit": git_value("rev-parse", "HEAD"),
        "candidate_claims": len(claim_packet),
        "independent_raters": len(ratings),
        "literature_rows": len(literature_packet),
        "method_membership_disclosed": False,
        "ratings_inspected": False,
        "inputs": {
            label: {"path": str(path.resolve()), "sha256": sha256_file(path)}
            for label, path in input_paths.items()
        },
        "outputs": {
            label: {"path": str(paths[label].resolve()), "sha256": sha256_file(paths[label])}
            for label in output_labels
        },
        "private_outputs_not_for_public_redistribution": ["blinded_literature.private.tsv"],
    }
    paths["manifest"].write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    paths["manifest_sha256"].write_text(
        f"{sha256_file(paths['manifest'])}  {paths['manifest'].name}\n", encoding="utf-8"
    )
    print("[PASS] Priority 4 method-blinded packets frozen before rating")
    print(f"[INFO] Claims: {len(claim_packet)}; literature rows: {len(literature_packet)}")
    print(f"[INFO] Independent rating templates: {len(ratings)}")
    print("[INFO] Claim UID, audit status, method membership, and stability are masked")
    print(f"[INFO] Wrote: {paths['manifest']}")
    print("[NEXT] Distribute one rating template per rater; do not reveal method membership")


if __name__ == "__main__":
    main()
