#!/usr/bin/env python3
"""Freeze the Priority 2 same-pool claims and coverage-matched memberships."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

CRM_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = CRM_DIR.parents[2]
DEFAULT_CONFIG = CRM_DIR / "config" / "priorities2_5_protocol.json"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    require(isinstance(value, dict), f"JSON root must be an object: {path}")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def git_value(*arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def require_clean_tracked_worktree() -> None:
    status = git_value("status", "--porcelain", "--untracked-files=no")
    require(not status, f"Commit the P2 code and leave tracked files clean before freeze: {status}")


def normalized_strings(values: pd.Series) -> pd.Series:
    return values.fillna("").astype(str).str.strip()


def require_single_value(frame: pd.DataFrame, column: str, expected: str) -> None:
    require(column in frame.columns, f"Missing required audit column: {column}")
    observed = set(normalized_strings(frame[column]).str.lower())
    require(
        observed == {expected.lower()}, f"Expected {column}={expected}; observed {sorted(observed)}"
    )


def claim_key(entity: str, direction: str, *, protocol: dict[str, Any]) -> str:
    p2 = protocol["priority2"]
    context = p2["context"]
    components = [
        str(entity).strip(),
        str(context["condition"]).strip(),
        str(context["tissue"]).strip(),
        str(context["perturbation"]).strip(),
        str(p2["comparison"]).strip(),
        str(direction).strip().lower(),
        str(p2["claim_strength"]).strip(),
    ]
    return "claim_" + sha256_text("\x1f".join(components))[:16]


def pathway_label(entity: str) -> str:
    value = str(entity).strip()
    if value.startswith("HALLMARK_"):
        value = value.removeprefix("HALLMARK_")
    return value.replace("_", " ").title()


def claim_wording(entity: str, direction: str) -> str:
    group = "TP53-mutant" if direction == "up" else "TP53-wild-type"
    return (
        f"In HNSC tumors, {pathway_label(entity)} is enriched in {group} samples "
        "relative to the other TP53 group."
    )


def validate_audit(
    frame: pd.DataFrame,
    *,
    expected_n: int,
    expected_tau: float,
    review_mode: str,
    gate_mode: str,
) -> pd.DataFrame:
    required = {
        "claim_id",
        "entity",
        "direction",
        "status",
        "term_survival_agg",
        "tau_used",
        "claim_mode",
        "context_review_mode",
        "context_gate_mode",
        "context_evaluated",
    }
    missing = required - set(frame.columns)
    require(not missing, f"Audit log is missing columns: {sorted(missing)}")
    require(len(frame) == expected_n, f"Expected {expected_n} claims; observed {len(frame)}")
    require(frame["claim_id"].is_unique, "claim_id is not unique")
    keys = frame[["entity", "direction"]].astype(str).agg("\x1f".join, axis=1)
    require(keys.is_unique, "entity x direction is not unique")
    require(
        set(normalized_strings(frame["direction"]).str.lower()) <= {"up", "down"}, "Bad direction"
    )
    require(
        set(normalized_strings(frame["status"]).str.upper()) <= {"PASS", "ABSTAIN", "FAIL"},
        "Bad status",
    )
    tau_values = pd.to_numeric(frame["tau_used"], errors="raise")
    require(np.allclose(tau_values, expected_tau), "Audit tau differs from the protocol")
    require_single_value(frame, "claim_mode", "deterministic")
    require_single_value(frame, "context_review_mode", review_mode)
    require_single_value(frame, "context_gate_mode", gate_mode)
    evaluated = frame["context_evaluated"]
    if not pd.api.types.is_bool_dtype(evaluated.dtype):
        evaluated = normalized_strings(evaluated).str.lower().map({"true": True, "false": False})
    require(evaluated.notna().all(), "Invalid context_evaluated values")
    if review_mode == "off":
        require(not evaluated.astype(bool).any(), "Mechanical run unexpectedly evaluated context")
    else:
        require(evaluated.astype(bool).all(), "Full-audit run did not evaluate every claim")
        require("context_method" in frame, "Full-audit log lacks context_method")
        methods = set(normalized_strings(frame["context_method"]).str.lower())
        require(
            methods == {"llm"}, f"Full-audit context methods are not all llm: {sorted(methods)}"
        )
    frame = frame.copy()
    frame["entity"] = normalized_strings(frame["entity"])
    frame["direction"] = normalized_strings(frame["direction"]).str.lower()
    frame["status"] = normalized_strings(frame["status"]).str.upper()
    frame["term_survival_agg"] = pd.to_numeric(frame["term_survival_agg"], errors="raise")
    require(frame["term_survival_agg"].between(0, 1).all(), "Invalid term survival values")
    return frame


def validate_run_metadata(
    metadata: dict[str, Any],
    *,
    review_mode: str,
    expected_backend: str = "",
    expected_model: str = "",
) -> None:
    llm = metadata.get("inputs", {}).get("llm", {})
    require(isinstance(llm, dict), "run_meta inputs.llm is missing")
    review = llm.get("review", {})
    require(isinstance(review, dict), "run_meta LLM review metadata is missing")
    require(
        str(review.get("review_mode_effective_for_select", "")).lower() == review_mode,
        "run_meta effective context-review mode drift",
    )
    if review_mode == "llm":
        require(review.get("backend_enabled") is True, "Full-audit LLM backend was not enabled")
        require(
            str(review.get("backend_env", "")).lower() == expected_backend.lower(),
            "Full-audit backend differs from protocol",
        )
        identity = llm.get("backend_identity", {})
        require(isinstance(identity, dict), "run_meta backend identity is missing")
        require(
            str(identity.get("model_name", "")) == expected_model,
            "Full-audit model differs from protocol",
        )


def build_outputs(
    *,
    mechanical: pd.DataFrame,
    full_audit: pd.DataFrame,
    evidence: pd.DataFrame,
    protocol: dict[str, Any],
    freeze_label: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    p2 = protocol["priority2"]
    expected_n = int(p2["candidate_count_expected"])
    for column in ("term_id", "stat", "qval", "direction"):
        require(column in evidence, f"EvidenceTable is missing {column}")
    evidence = evidence.copy()
    evidence["entity"] = normalized_strings(evidence["term_id"])
    evidence["direction"] = normalized_strings(evidence["direction"]).str.lower()
    evidence["q_value"] = pd.to_numeric(evidence["qval"], errors="raise")
    evidence["statistic"] = pd.to_numeric(evidence["stat"], errors="raise")
    require(len(evidence) == expected_n, "EvidenceTable candidate count drift")
    require(evidence[["entity", "direction"]].duplicated().sum() == 0, "Duplicate evidence claims")

    join_columns = ["entity", "direction"]
    merged = mechanical.merge(
        full_audit,
        on=join_columns,
        how="outer",
        validate="one_to_one",
        suffixes=("_mechanical", "_full"),
        indicator=True,
    )
    require((merged["_merge"] == "both").all(), "Mechanical and full-audit candidate pools differ")
    merged = merged.drop(columns="_merge").merge(
        evidence[["entity", "direction", "q_value", "statistic"]],
        on=join_columns,
        how="left",
        validate="one_to_one",
    )
    require(merged["q_value"].notna().all(), "EvidenceTable did not match every candidate claim")

    merged["claim_uid"] = [
        claim_key(entity, direction, protocol=protocol)
        for entity, direction in merged[["entity", "direction"]].itertuples(index=False, name=None)
    ]
    require(merged["claim_uid"].is_unique, "Structured claim UID collision")
    merged["pathway_label"] = merged["entity"].map(pathway_label)
    merged["claim_text"] = [
        claim_wording(entity, direction)
        for entity, direction in merged[["entity", "direction"]].itertuples(index=False, name=None)
    ]
    merged["condition"] = str(p2["context"]["condition"])
    merged["tissue"] = str(p2["context"]["tissue"])
    merged["perturbation"] = str(p2["context"]["perturbation"])
    merged["comparison"] = str(p2["comparison"])
    merged["claim_strength"] = str(p2["claim_strength"])
    merged["abs_statistic"] = merged["statistic"].abs()
    merged["full_audit_selected"] = merged["status_full"].eq("PASS")
    selected_k = int(merged["full_audit_selected"].sum())
    require(0 < selected_k < expected_n, "Full audit produced degenerate coverage")

    q_order = merged.sort_values(
        ["q_value", "abs_statistic", "claim_uid"], ascending=[True, False, True]
    ).index
    stability_order = merged.sort_values(
        ["term_survival_agg_mechanical", "q_value", "claim_uid"],
        ascending=[False, True, True],
    ).index
    merged["raw_pool_selected"] = True
    merged["q_value_matched_selected"] = False
    merged.loc[q_order[:selected_k], "q_value_matched_selected"] = True
    merged["stability_matched_selected"] = False
    merged.loc[stability_order[:selected_k], "stability_matched_selected"] = True

    claims_columns = [
        "claim_uid",
        "claim_id_mechanical",
        "claim_id_full",
        "entity",
        "pathway_label",
        "direction",
        "condition",
        "tissue",
        "perturbation",
        "comparison",
        "claim_strength",
        "claim_text",
        "statistic",
        "q_value",
        "term_survival_agg_mechanical",
        "status_full",
        "context_status_full",
        "context_confidence_full",
        "abstain_reason_full",
        "fail_reason_full",
    ]
    for column in claims_columns:
        if column not in merged:
            merged[column] = pd.NA
    claims = merged[claims_columns].sort_values("claim_uid").reset_index(drop=True)

    membership_columns = [
        "claim_uid",
        "raw_pool_selected",
        "q_value_matched_selected",
        "stability_matched_selected",
        "full_audit_selected",
    ]
    membership = merged[membership_columns].sort_values("claim_uid").reset_index(drop=True)
    for method in membership_columns[1:]:
        membership[method] = membership[method].astype(bool)
    require(int(membership["q_value_matched_selected"].sum()) == selected_k, "q K drift")
    require(int(membership["stability_matched_selected"].sum()) == selected_k, "stability K drift")

    methods = {
        "raw_pool": "raw_pool_selected",
        "q_value_matched": "q_value_matched_selected",
        "stability_matched": "stability_matched_selected",
        "full_audit": "full_audit_selected",
    }
    metric_rows = []
    for method, column in methods.items():
        n_selected = int(membership[column].sum())
        metric_rows.append(
            {
                "method": method,
                "n_candidate": expected_n,
                "n_selected": n_selected,
                "coverage": n_selected / expected_n,
                "independent_support_fraction": math.nan,
                "overstatement_risk": math.nan,
                "outcome_status": "PENDING_P3_P4",
            }
        )
    metrics = pd.DataFrame(metric_rows)

    sampling = merged[["claim_uid"]].copy()
    seed = int(p2["review_sampling"]["seed"])
    rng = np.random.default_rng(seed)
    sampling["packet_order"] = rng.permutation(np.arange(1, len(sampling) + 1))
    sampling = sampling.sort_values("packet_order").reset_index(drop=True)
    sampling["review_id"] = [
        f"R1C{order:03d}_{sha256_text(f'{freeze_label}|{uid}')[:8]}"
        for order, uid in sampling[["packet_order", "claim_uid"]].itertuples(index=False, name=None)
    ]
    sampling = sampling[["review_id", "packet_order", "claim_uid"]]
    require(sampling["review_id"].is_unique, "Review ID collision")
    return claims, membership, metrics, sampling


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--mechanical-run", type=Path, default=None)
    parser.add_argument("--full-audit-run", type=Path, default=None)
    parser.add_argument("--evidence-table", type=Path, default=None)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--freeze-label", required=True)
    args = parser.parse_args()

    require_clean_tracked_worktree()
    protocol = read_json(args.config)
    p2 = protocol["priority2"]
    benchmark_id = str(p2["benchmark_id"])
    output_root = args.data_root.resolve() / "output" / "priority2" / benchmark_id
    mechanical_run = args.mechanical_run or (
        output_root / "runs_mechanical" / "HNSC" / "ours" / "gate_hard" / "tau_0.90"
    )
    full_audit_run = args.full_audit_run or (
        output_root / "runs_full_audit" / "HNSC" / "ours" / "gate_hard" / "tau_0.90"
    )
    evidence_path = args.evidence_table or (
        REPO_ROOT / "paper/source_data/PANCAN_TP53_v1/evidence_tables/HNSC.evidence_table.tsv"
    )
    input_paths = {
        "mechanical_audit": mechanical_run / "audit_log.tsv",
        "mechanical_run_meta": mechanical_run / "run_meta.json",
        "full_audit": full_audit_run / "audit_log.tsv",
        "full_audit_run_meta": full_audit_run / "run_meta.json",
        "evidence_table": evidence_path,
        "protocol": args.config,
    }
    for label, path in input_paths.items():
        require(path.is_file(), f"Missing {label}: {path}")
    final_paths = {
        "claims": output_root / "pool" / "claims.tsv",
        "membership": output_root / "membership" / "selection_membership.tsv",
        "risk_coverage": output_root / "metrics" / "risk_coverage_source.tsv",
        "sampling_frame": output_root / "review" / "sampling_frame.locked.tsv",
        "manifest": output_root / "metrics" / "priority2_freeze_manifest.json",
        "manifest_sha256": output_root / "metrics" / "priority2_freeze_manifest.sha256",
    }
    collisions = [str(path) for path in final_paths.values() if path.exists()]
    require(not collisions, f"Priority 2 freeze outputs are immutable; collisions: {collisions}")
    for later_priority in ("priority3", "priority4", "priority5"):
        later_root = args.data_root.resolve() / "output" / later_priority
        require(
            not later_root.exists() or not any(later_root.rglob("*")),
            f"{later_priority} outputs already exist; P2 must freeze first",
        )

    expected_n = int(p2["candidate_count_expected"])
    tau = float(p2["primary_tau"])
    mechanical = validate_audit(
        pd.read_csv(input_paths["mechanical_audit"], sep="\t"),
        expected_n=expected_n,
        expected_tau=tau,
        review_mode="off",
        gate_mode="note",
    )
    validate_run_metadata(
        read_json(input_paths["mechanical_run_meta"]),
        review_mode="off",
    )
    full_audit = validate_audit(
        pd.read_csv(input_paths["full_audit"], sep="\t"),
        expected_n=expected_n,
        expected_tau=tau,
        review_mode="llm",
        gate_mode="hard",
    )
    validate_run_metadata(
        read_json(input_paths["full_audit_run_meta"]),
        review_mode="llm",
        expected_backend=str(p2["full_audit_run"]["backend"]),
        expected_model=str(p2["full_audit_run"]["model"]),
    )
    evidence = pd.read_csv(input_paths["evidence_table"], sep="\t")
    claims, membership, metrics, sampling = build_outputs(
        mechanical=mechanical,
        full_audit=full_audit,
        evidence=evidence,
        protocol=protocol,
        freeze_label=args.freeze_label,
    )
    tables = {
        "claims": claims,
        "membership": membership,
        "risk_coverage": metrics,
        "sampling_frame": sampling,
    }
    for label, table in tables.items():
        path = final_paths[label]
        path.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(path, sep="\t", index=False, lineterminator="\n")

    selected_k = int(membership["full_audit_selected"].sum())
    overlap_q = int(
        (membership["full_audit_selected"] & membership["q_value_matched_selected"]).sum()
    )
    overlap_stability = int(
        (membership["full_audit_selected"] & membership["stability_matched_selected"]).sum()
    )
    manifest = {
        "protocol_version": protocol["protocol_version"],
        "status": "FROZEN",
        "priority": 2,
        "benchmark_id": benchmark_id,
        "freeze_label": args.freeze_label,
        "frozen_at_utc": datetime.now(UTC).isoformat(),
        "git_commit": git_value("rev-parse", "HEAD"),
        "primary_tau": tau,
        "candidate_count": expected_n,
        "matched_k": selected_k,
        "full_audit_q_value_overlap": overlap_q,
        "full_audit_stability_overlap": overlap_stability,
        "validation_outcomes_inspected": False,
        "inputs": {
            label: {"path": str(path.resolve()), "sha256": sha256_file(path)}
            for label, path in input_paths.items()
        },
        "outputs": {
            label: {
                "path": str(final_paths[label].resolve()),
                "sha256": sha256_file(final_paths[label]),
            }
            for label in tables
        },
    }
    final_paths["manifest"].write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    final_paths["manifest_sha256"].write_text(
        f"{sha256_file(final_paths['manifest'])}  {final_paths['manifest'].name}\n",
        encoding="utf-8",
    )
    print("[PASS] Priority 2 candidate pool and matched memberships frozen")
    print(f"[INFO] Candidate pool: {expected_n}; matched K: {selected_k}")
    print(f"[INFO] Full-audit/q-value overlap: {overlap_q}/{selected_k}")
    print(f"[INFO] Full-audit/stability overlap: {overlap_stability}/{selected_k}")
    print("[INFO] P3/P4 outcomes were not read or calculated")
    print(f"[INFO] Wrote: {final_paths['manifest']}")


if __name__ == "__main__":
    main()
