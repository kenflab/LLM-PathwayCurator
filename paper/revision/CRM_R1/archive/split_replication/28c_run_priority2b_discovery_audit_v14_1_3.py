#!/usr/bin/env python3
"""Run strict V14.1.3 discovery-only audits and export matched memberships."""

from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from v14_1_objective_common import (
    file_record,
    read_json,
    require,
    require_columns,
    sha256_file,
    software_versions,
    validate_manifest_outputs,
    write_json,
    write_manifest_digest,
)

import llm_pathway_curator
from llm_pathway_curator.pipeline import RunConfig, run_pipeline

STATS_SCHEMA = "CRM_R1_P2B_DISCOVERY_STATISTICS_v14_1_3"
HALLMARK_SCHEMA = "CRM_R1_P2B_HALLMARK_v14_1_2"
NUMERICAL_SCHEMA = "CRM_R1_P2B_FGSEA_NUMERICAL_LOCK_v14_1_3"
EXPORT_SCHEMA = "CRM_R1_P2B_DISCOVERY_MEMBERSHIP_EXPORT_v14_1_3"
METHODS = ["raw_pool", "q_value_matched", "stability_matched", "full_audit"]


def source_tree_digest(root: Path) -> str:
    digest = hashlib.sha256()
    files = sorted(root.rglob("*.py"))
    require(files, f"No Python package sources found under {root}")
    for path in files:
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def ollama_identity(model: str, expected_id: str) -> dict[str, str]:
    try:
        version_output = subprocess.check_output(
            ["ollama", "--version"], text=True, stderr=subprocess.STDOUT
        ).strip()
        list_output = subprocess.check_output(
            ["ollama", "list"], text=True, stderr=subprocess.STDOUT
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise ValueError("Ollama is unavailable; strict context review cannot start") from error
    match: list[str] | None = None
    for line in list_output.splitlines()[1:]:
        tokens = line.split()
        if tokens and tokens[0] == model:
            match = tokens
            break
    require(match is not None and len(match) >= 2, f"Ollama model is missing: {model}")
    model_id = match[1]
    require(
        model_id.startswith(expected_id),
        f"Ollama model ID drift: expected {expected_id}, observed {model_id}",
    )
    return {"version": version_output, "model": model, "model_id": model_id}


def validate_strict_llm_audit(audit: pd.DataFrame, *, expected_terms: set[str]) -> pd.DataFrame:
    required = {
        "entity",
        "direction",
        "status",
        "term_survival_agg",
        "context_review_mode",
        "context_method",
        "context_evaluated",
    }
    require_columns(audit, required, label="Audit log")
    result = audit.copy()
    require(len(result) == 50, f"Expected 50 audited claims; observed {len(result)}")
    result["entity"] = result["entity"].fillna("").astype(str).str.strip()
    require(set(result["entity"]) == expected_terms, "Audited candidate pool drift")
    require(not result["entity"].duplicated().any(), "Duplicate audited term")
    direction = result["direction"].fillna("").astype(str).str.strip().str.lower()
    require(set(direction) <= {"up", "down"}, "Audit direction vocabulary drift")
    result["direction"] = direction
    mode = result["context_review_mode"].fillna("").astype(str).str.strip().str.lower()
    method = result["context_method"].fillna("").astype(str).str.strip().str.lower()
    evaluated = (
        result["context_evaluated"]
        .fillna(False)
        .astype(str)
        .str.strip()
        .str.lower()
        .map({"true": True, "false": False, "1": True, "0": False})
    )
    require(mode.eq("llm").all(), "Context review did not remain in LLM mode")
    require(
        method.eq("llm").all(),
        "Proxy/off context result detected; fallback is forbidden",
    )
    require(
        evaluated.notna().all() and evaluated.all(),
        "Unevaluated context result detected",
    )
    result["term_survival_agg"] = pd.to_numeric(result["term_survival_agg"], errors="coerce")
    require(
        result["term_survival_agg"].between(0, 1).all(),
        "Audit stability must lie in [0, 1]",
    )
    require(
        set(result["status"].astype(str).str.upper()) <= {"PASS", "ABSTAIN", "FAIL"},
        "Unexpected audit status",
    )
    return result


def assemble_membership_block(statistics: pd.DataFrame, audit: pd.DataFrame) -> pd.DataFrame:
    require(len(statistics) == 50, "Each discovery block must contain 50 pathways")
    require(not statistics["term_id"].duplicated().any(), "Duplicate discovery term")
    require(statistics["cohort_id"].nunique() == 1, "Mixed cohorts in one block")
    require(statistics["split_id"].nunique() == 1, "Mixed splits in one block")
    expected_terms = set(statistics["term_id"].astype(str))
    checked = validate_strict_llm_audit(audit, expected_terms=expected_terms)
    audit_fields = checked.loc[:, ["entity", "direction", "status", "term_survival_agg"]].rename(
        columns={
            "entity": "term_id",
            "direction": "audit_direction",
            "status": "audit_status",
            "term_survival_agg": "discovery_stability_score",
        }
    )
    claims = statistics.merge(audit_fields, on="term_id", how="left", validate="one_to_one")
    require(claims["audit_status"].notna().all(), "Audit/statistics join failed")
    statistic_direction = claims["direction"].astype(str).str.strip().str.lower()
    require(
        claims["audit_direction"].eq(statistic_direction).all(),
        "Audit/statistics direction mismatch",
    )
    claims["claim_uid"] = claims["term_id"].astype(str)
    claims["discovery_direction"] = statistic_direction.str.upper()
    claims["discovery_q_value"] = pd.to_numeric(claims["qval"], errors="coerce")
    require(
        claims["discovery_q_value"].notna().all()
        and claims["discovery_q_value"].between(0, 1).all(),
        "Invalid locked discovery q-value",
    )
    claims["full_audit_pass"] = claims["audit_status"].astype(str).str.upper().eq("PASS")
    k = int(claims["full_audit_pass"].sum())
    cohort_id = str(claims["cohort_id"].iloc[0])
    split_id = str(claims["split_id"].iloc[0])
    require(k > 0, f"Matched K is zero: {cohort_id}/{split_id}")

    q_selected = set(
        claims.sort_values(["discovery_q_value", "claim_uid"], ascending=[True, True]).head(k)[
            "claim_uid"
        ]
    )
    stability_selected = set(
        claims.sort_values(
            ["discovery_stability_score", "claim_uid"], ascending=[False, True]
        ).head(k)["claim_uid"]
    )
    audit_selected = set(claims.loc[claims["full_audit_pass"], "claim_uid"])
    selected_by_method = {
        "raw_pool": set(claims["claim_uid"]),
        "q_value_matched": q_selected,
        "stability_matched": stability_selected,
        "full_audit": audit_selected,
    }
    rows: list[pd.DataFrame] = []
    keep = [
        "cohort_id",
        "split_id",
        "claim_uid",
        "discovery_direction",
        "discovery_q_value",
        "discovery_stability_score",
        "full_audit_pass",
    ]
    for method in METHODS:
        block = claims.loc[:, keep].copy()
        block["method_id"] = method
        block["selected"] = block["claim_uid"].isin(selected_by_method[method])
        rows.append(block)
    result = pd.concat(rows, ignore_index=True)
    return result[
        [
            "cohort_id",
            "split_id",
            "claim_uid",
            "method_id",
            "selected",
            "discovery_direction",
            "discovery_q_value",
            "discovery_stability_score",
            "full_audit_pass",
        ]
    ].sort_values(["cohort_id", "split_id", "claim_uid", "method_id"])


def completed_membership(
    path: Path, *, cohort_id: str, split_id: str, expected_terms: set[str]
) -> pd.DataFrame | None:
    if not path.is_file():
        return None
    try:
        frame = pd.read_csv(path, sep="\t", keep_default_na=False)
    except (OSError, UnicodeError, pd.errors.ParserError):
        return None
    required = {
        "cohort_id",
        "split_id",
        "claim_uid",
        "method_id",
        "selected",
        "discovery_direction",
        "discovery_q_value",
        "discovery_stability_score",
        "full_audit_pass",
    }
    if not required <= set(frame.columns):
        return None
    if len(frame) != 200:
        return None
    if frame.duplicated(["cohort_id", "split_id", "claim_uid", "method_id"]).any():
        return None
    if set(frame["claim_uid"].astype(str)) != expected_terms:
        return None
    if set(frame["method_id"].astype(str)) != set(METHODS):
        return None
    if set(frame["cohort_id"].astype(str)) != {cohort_id}:
        return None
    if set(frame["split_id"].astype(str)) != {split_id}:
        return None
    return frame


def write_evidence_table(block: pd.DataFrame, path: Path) -> None:
    evidence = block.loc[
        :,
        [
            "term_id",
            "term_name",
            "source",
            "stat",
            "qval",
            "direction",
            "evidence_genes",
        ],
    ].copy()
    evidence.to_csv(path, sep="\t", index=False, lineterminator="\n")


def write_sample_card(cohort_id: str, protocol: dict[str, object], path: Path) -> None:
    audit = protocol["audit"]
    stability = audit["stability"]
    card = {
        "comparison": "TP53_mut_vs_TP53_wt",
        "condition": cohort_id,
        "extra": {
            "audit_mode": "decision",
            "audit_tau": audit["tau"],
            "benchmark_id": "PANCAN_TP53_v1_P2B_v14_1_3",
            "context_gate_mode": audit["context_gate_mode"],
            "context_review_mode": audit["context_review_mode"],
            "distill_mode": "perturb",
            "evidence_jaccard_min": stability["evidence_jaccard_min"],
            "evidence_precision_min": stability["evidence_precision_min"],
            "evidence_recall_min": stability["evidence_recall_min"],
            "gene_dropout_p": stability["gene_dropout_p"],
            "gene_jitter_p": stability["gene_jitter_p"],
            "n_perturb": stability["n_perturb"],
            "stability_gate_mode": stability["gate_mode"],
            "stress_gate_mode": "off",
        },
        "k_claims": audit["candidate_claims"],
        "notes": "V14.1.3 discovery partition only; held-out validation not read.",
        "perturbation": "genotype",
        "tissue": "tumor",
    }
    write_json(path, card)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--work-root", required=True, type=Path)
    parser.add_argument("--memberships", required=True, type=Path)
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Validate locks, package source, and Ollama identity without auditing claims.",
    )
    args = parser.parse_args()
    data_root = args.data_root.expanduser().resolve()
    work_root = args.work_root.expanduser().resolve()
    membership_path = args.memberships.expanduser().resolve()
    priority2b = data_root / "output" / "priority2b"

    prohibited = [
        priority2b / "membership_lock_v14_1",
        priority2b / "validation_lock_v14_1",
        priority2b / "final_v14_1",
        priority2b / "membership_lock_v14_1_2",
        priority2b / "validation_lock_v14_1_2",
        priority2b / "final_v14_1_2",
        priority2b / "membership_lock_v14_1_3",
        priority2b / "validation_lock_v14_1_3",
        priority2b / "final_v14_1_3",
    ]
    present = [str(path) for path in prohibited if path.exists()]
    require(not present, "Downstream V14.1 output already exists: " + ", ".join(present))

    stats_manifest_path = (
        priority2b / "discovery_stats_lock_v14_1_3" / "discovery_statistics_manifest.json"
    )
    hallmark_manifest_path = priority2b / "hallmark_lock_v14_1_2" / "hallmark_manifest.json"
    numerical_manifest_path = (
        priority2b / "numerical_lock_v14_1_3" / "fgsea_numerical_manifest.json"
    )
    stats_outputs = validate_manifest_outputs(stats_manifest_path, STATS_SCHEMA)
    hallmark_outputs = validate_manifest_outputs(hallmark_manifest_path, HALLMARK_SCHEMA)
    validate_manifest_outputs(numerical_manifest_path, NUMERICAL_SCHEMA)
    protocol = read_json(hallmark_outputs["protocol_frozen"])
    require(protocol["status"] == "READY_TO_FREEZE_DISCOVERY", "Protocol status drift")
    require(
        protocol["audit"]["context_fallback_allowed"] is False,
        "Fallback must be forbidden",
    )
    require(protocol["audit"]["context_backend"] == "ollama", "Backend drift")

    os.environ["LLMPATH_BACKEND"] = "ollama"
    os.environ["LLMPATH_CLAIM_MODE"] = "deterministic"
    os.environ["LLMPATH_CONTEXT_REVIEW_MODE"] = "llm"
    os.environ["LLMPATH_CONTEXT_GATE_MODE"] = "hard"
    os.environ["LLMPATH_OLLAMA_MODEL"] = str(protocol["audit"]["context_model"])
    os.environ["LLMPATH_OLLAMA_TEMPERATURE"] = str(protocol["audit"]["context_temperature"])

    ollama = ollama_identity(
        str(protocol["audit"]["context_model"]),
        str(protocol["reference_environment"]["ollama_model_id"]),
    )
    package_root = Path(llm_pathway_curator.__file__).resolve().parent
    try:
        git_head = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        git_head = ""

    work_root.mkdir(parents=True, exist_ok=True)
    signature_path = work_root / "audit_run_signature.json"
    signature = {
        "schema_version": "CRM_R1_P2B_AUDIT_RUN_SIGNATURE_v14_1_3",
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "statistics_manifest_sha256": sha256_file(stats_manifest_path),
        "hallmark_manifest_sha256": sha256_file(hallmark_manifest_path),
        "numerical_manifest_sha256": sha256_file(numerical_manifest_path),
        "protocol_sha256": sha256_file(hallmark_outputs["protocol_frozen"]),
        "python": sys.version,
        "git_head": git_head,
        "package_root": str(package_root),
        "package_source_sha256": source_tree_digest(package_root),
        "context_backend": "ollama",
        "context_model": protocol["audit"]["context_model"],
        "context_temperature": protocol["audit"]["context_temperature"],
        "ollama": ollama,
    }
    if signature_path.exists():
        require(
            read_json(signature_path) == signature,
            "Audit work directory signature drift",
        )
    else:
        write_json(signature_path, signature)

    if args.preflight_only:
        print("[PASS] V14.1.3 discovery-audit preflight is complete")
        print("[INFO] Locked statistics, Hallmark terms, code, and Ollama identity verified")
        print("[GO] Run the same command without --preflight-only")
        return

    statistics = pd.read_csv(stats_outputs["statistics"], sep="\t", keep_default_na=False)
    terms = set(pd.read_csv(hallmark_outputs["terms"], sep="\t")["term_id"].astype(str))
    pair_groups = list(statistics.groupby(["cohort_id", "split_id"], sort=True))
    require(len(pair_groups) == 440, "Expected 440 discovery pairs")

    membership_blocks: list[pd.DataFrame] = []
    for job_index, ((cohort_id, split_id), block) in enumerate(pair_groups, start=1):
        cohort_id = str(cohort_id)
        split_id = str(split_id)
        job_dir = work_root / "jobs" / cohort_id / split_id
        job_dir.mkdir(parents=True, exist_ok=True)
        membership_job = job_dir / "membership_rows.tsv"
        completed = completed_membership(
            membership_job,
            cohort_id=cohort_id,
            split_id=split_id,
            expected_terms=terms,
        )
        if completed is not None:
            membership_blocks.append(completed)
            print(f"[SKIP] {cohort_id}/{split_id} ({job_index}/440)")
            continue

        evidence_path = job_dir / "discovery_evidence.tsv"
        card_path = job_dir / "sample_card.json"
        audit_dir = job_dir / "audit"
        existing_audit_path = audit_dir / "audit_log.tsv"
        if existing_audit_path.is_file():
            try:
                existing_audit = pd.read_csv(existing_audit_path, sep="\t", keep_default_na=False)
                recovered = assemble_membership_block(block, existing_audit)
            except (OSError, UnicodeError, pd.errors.ParserError, ValueError):
                recovered = None
            if recovered is not None:
                recovered.to_csv(membership_job, sep="\t", index=False, lineterminator="\n")
                membership_blocks.append(recovered)
                k = int(recovered.loc[recovered["method_id"].eq("full_audit"), "selected"].sum())
                print(f"[RECOVER] {cohort_id}/{split_id} K={k} ({job_index}/440)")
                continue

        write_evidence_table(block, evidence_path)
        write_sample_card(cohort_id, protocol, card_path)
        result = run_pipeline(
            RunConfig(
                evidence_table=str(evidence_path),
                sample_card=str(card_path),
                outdir=str(audit_dir),
                force=audit_dir.exists(),
                seed=int(protocol["audit"]["seed"]),
                tau=float(protocol["audit"]["tau"]),
                k_claims=int(protocol["audit"]["candidate_claims"]),
            )
        )
        audit_path = Path(result.artifacts.get("audit_log_tsv", audit_dir / "audit_log.tsv"))
        require(audit_path.is_file(), f"Missing audit log: {audit_path}")
        audit_log = pd.read_csv(audit_path, sep="\t", keep_default_na=False)
        membership = assemble_membership_block(block, audit_log)
        membership.to_csv(membership_job, sep="\t", index=False, lineterminator="\n")
        membership_blocks.append(membership)
        k = int(membership.loc[membership["method_id"].eq("full_audit"), "selected"].sum())
        print(f"[DONE] {cohort_id}/{split_id} K={k} ({job_index}/440)")

    memberships = pd.concat(membership_blocks, ignore_index=True).sort_values(
        ["cohort_id", "split_id", "claim_uid", "method_id"]
    )
    require(len(memberships) == 440 * 50 * 4, "Membership export row-count drift")
    membership_path.parent.mkdir(parents=True, exist_ok=True)
    memberships.to_csv(membership_path, sep="\t", index=False, lineterminator="\n")

    counts = (
        memberships.groupby(["cohort_id", "split_id", "method_id"])["selected"]
        .sum()
        .rename("n_selected")
        .reset_index()
    )
    counts_path = membership_path.with_name("discovery_membership_counts.tsv")
    counts.to_csv(counts_path, sep="\t", index=False, lineterminator="\n")
    manifest_path = membership_path.with_name("discovery_membership_export_manifest.json")
    manifest = {
        "schema_version": EXPORT_SCHEMA,
        "status": "DISCOVERY_MEMBERSHIP_EXPORT_COMPLETE_NOT_YET_LOCKED",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "cohort_split_pairs": 440,
        "terms_per_pair": 50,
        "methods": METHODS,
        "rows": len(memberships),
        "software_versions": software_versions(),
        "inputs": {
            "statistics_manifest": file_record(stats_manifest_path),
            "hallmark_manifest": file_record(hallmark_manifest_path),
            "numerical_manifest": file_record(numerical_manifest_path),
            "audit_run_signature": file_record(signature_path),
        },
        "outputs": {
            "memberships": file_record(membership_path),
            "counts": file_record(counts_path),
        },
        "context_review": {
            "mode": "llm",
            "backend": "ollama",
            "model": protocol["audit"]["context_model"],
            "fallback_allowed": False,
            "ollama": ollama,
        },
        "validation_outcomes_read": False,
        "human_ratings_read": False,
    }
    write_json(manifest_path, manifest)
    write_manifest_digest(manifest_path)
    print("[PASS] V14.1.3 strict discovery audit and membership export completed")
    print("[INFO] pairs=440; claims/pair=50; methods=4")
    print(f"[INFO] memberships: {membership_path}")
    print("[NEXT] Check with script 29c, then freeze with script 72c")


if __name__ == "__main__":
    main()
