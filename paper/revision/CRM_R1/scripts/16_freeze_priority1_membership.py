#!/usr/bin/env python3
"""Freeze Priority 1 discovery memberships before any 72 h endpoint is calculated."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import re
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from llm_pathway_curator import _shared
from llm_pathway_curator.claim_schema import Claim

CRM_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = CRM_DIR.parents[2]
DEFAULT_CONFIG = CRM_DIR / "config" / "priority1_protocol.json"
FREEZE_LABEL_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{2,63}$")


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


def bool_series(values: pd.Series, *, column: str) -> pd.Series:
    if pd.api.types.is_bool_dtype(values.dtype):
        return values.fillna(False).astype(bool)
    normalized = values.fillna("").astype(str).str.strip().str.lower()
    allowed = {"", "0", "1", "false", "true", "no", "yes", "off", "on"}
    unexpected = sorted(set(normalized) - allowed)
    require(not unexpected, f"Unexpected Boolean values in {column}: {unexpected}")
    return normalized.isin({"1", "true", "yes", "on"})


def tau_tag(tau: float) -> str:
    return f"{tau:.2f}".replace(".", "p")


def q_value_order(table: pd.DataFrame) -> pd.DataFrame:
    ordered = table.copy()
    ordered["abs_NES"] = ordered["NES"].abs()
    return ordered.sort_values(
        ["padj", "pval", "abs_NES", "pathway"],
        ascending=[True, True, False, True],
    )


def add_size_strata(table: pd.DataFrame, *, n_strata: int = 4) -> pd.DataFrame:
    ordered = table.sort_values(["leading_edge_n", "pathway"]).copy()
    ordered["leading_edge_size_stratum"] = (
        np.floor(np.arange(len(ordered)) * n_strata / len(ordered)).astype(int) + 1
    )
    return table.merge(
        ordered[["claim_id", "leading_edge_size_stratum"]],
        on="claim_id",
        how="left",
        validate="one_to_one",
    )


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
    unstaged = subprocess.run(["git", "diff", "--quiet"], cwd=REPO_ROOT, check=False)
    staged = subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=REPO_ROOT, check=False)
    require(
        unstaged.returncode == 0 and staged.returncode == 0,
        "Commit the V5 freeze code first; tracked Git changes must be clean before freezing",
    )


def validate_claims(path: Path) -> int:
    proposed = pd.read_csv(path, sep="\t")
    require("claim_json" in proposed.columns, f"Missing claim_json: {path}")
    for value in proposed["claim_json"]:
        Claim.model_validate_json(value)
    return len(proposed)


def validate_audit(
    path: Path,
    claims_path: Path,
    *,
    expected_claims: int,
    expected_tau: float,
) -> pd.DataFrame:
    audit = pd.read_csv(path, sep="\t")
    required_columns = {
        "abstain_reason",
        "claim_id",
        "context_evaluated",
        "context_gate_blocked",
        "context_gate_hit",
        "entity",
        "fail_reason",
        "status",
        "tau_used",
        "term_survival_agg",
    }
    require(not (required_columns - set(audit.columns)), f"Audit columns missing: {path}")
    require(len(audit) == expected_claims, f"Unexpected claim count: {path}")
    require(audit["claim_id"].nunique() == len(audit), f"Duplicate claim_id: {path}")
    require(audit["entity"].nunique() == len(audit), f"Duplicate entity: {path}")
    require(validate_claims(claims_path) == len(audit), f"Claim/audit row mismatch: {path}")
    require(
        not bool_series(audit["context_evaluated"], column="context_evaluated").any(),
        f"Context was evaluated: {path}",
    )
    require(
        not bool_series(audit["context_gate_blocked"], column="context_gate_blocked").any(),
        f"Context blocked a claim: {path}",
    )
    require(
        not bool_series(audit["context_gate_hit"], column="context_gate_hit").any(),
        f"Context gate hit a claim: {path}",
    )
    require(
        not audit["abstain_reason"].fillna("").str.contains("context", case=False).any(),
        f"Context abstention found: {path}",
    )
    require(
        not audit["fail_reason"].fillna("").str.contains("context", case=False).any(),
        f"Context failure found: {path}",
    )
    if "distill_semantics" in audit.columns:
        semantics = set(audit["distill_semantics"].dropna().astype(str))
        require(semantics == {"replicates_proxy"}, f"Unexpected distill semantics: {semantics}")
    observed_tau = pd.to_numeric(audit["tau_used"], errors="raise")
    require(observed_tau.nunique() == 1, f"Multiple tau values in audit: {path}")
    require(
        math.isclose(float(observed_tau.iloc[0]), expected_tau, abs_tol=1e-12),
        f"tau directory/log mismatch: {path}",
    )
    survival = pd.to_numeric(audit["term_survival_agg"], errors="coerce")
    require(survival.notna().all(), f"Missing empirical survival: {path}")
    require(survival.between(0.0, 1.0).all(), f"Empirical survival outside [0,1]: {path}")
    audit = audit.copy()
    audit["status"] = audit["status"].astype(str).str.upper()
    audit["term_survival_agg"] = survival
    return audit


def inventory_entry(path: Path, *, scope: str, root: Path) -> dict[str, Any]:
    require(path.is_file(), f"Missing freeze input: {path}")
    return {
        "path": path.relative_to(root).as_posix(),
        "scope": scope,
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def table_text(table: pd.DataFrame) -> str:
    return table.to_csv(
        sep="\t",
        index=False,
        lineterminator="\n",
        na_rep="NA",
        float_format="%.17g",
    )


def write_freeze_bundle(
    *,
    membership_path: Path,
    membership_text: str,
    tau_grid_path: Path,
    tau_grid_text: str,
    manifest_path: Path,
    manifest: dict[str, Any],
    sidecar_path: Path,
) -> None:
    outputs = [membership_path, tau_grid_path, manifest_path, sidecar_path]
    existing = [path for path in outputs if path.exists()]
    require(not existing, f"Frozen outputs are immutable; existing files: {existing}")
    membership_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_paths = [path.with_name(f".{path.name}.tmp") for path in outputs]
    require(not any(path.exists() for path in temporary_paths), "Remove stale freeze .tmp files")
    try:
        temporary_paths[0].write_text(membership_text, encoding="utf-8")
        temporary_paths[1].write_text(tau_grid_text, encoding="utf-8")
        manifest_text = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        temporary_paths[2].write_text(manifest_text, encoding="utf-8")
        manifest_hash = sha256_file(temporary_paths[2])
        temporary_paths[3].write_text(
            f"{manifest_hash}  {manifest_path.name}\n",
            encoding="utf-8",
        )
        for source, destination in zip(temporary_paths, outputs, strict=True):
            os.replace(source, destination)
    finally:
        for path in temporary_paths:
            if path.exists():
                path.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--freeze-label", required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--audit-root", type=Path, default=None)
    parser.add_argument("--run-prefix", default="discovery_48h_empirical_ctxoff_note")
    args = parser.parse_args()

    require(
        FREEZE_LABEL_PATTERN.fullmatch(args.freeze_label) is not None,
        "--freeze-label must be 3-64 characters using letters, digits, dot, dash, or underscore",
    )
    require_clean_tracked_worktree()
    data_root = args.data_root.expanduser().resolve()
    config_path = args.config.expanduser().resolve()
    config = read_json(config_path)
    require(config["status"] == "FROZEN", "V5 protocol status must be FROZEN")
    require(config["protocol_version"] == "CRM_R1_PRIORITY1_v5", "Unexpected protocol version")
    primary = config["primary_analysis"]
    empirical = config["empirical_stability"]
    decision = config["freeze_decision"]
    primary_tau = float(empirical["primary_tau"])
    require(math.isclose(primary_tau, 0.8, abs_tol=1e-12), "Primary tau must be frozen at 0.80")
    require(primary["context_review_mode"] == "off", "Context review must be off")
    require(primary["context_gate_mode"] == "note", "Context gate must be note")
    require(primary["stability_gate_mode"] == "hard", "Stability gate must be hard")

    benchmark_id = str(config["benchmark_id"])
    benchmark_dir = data_root / "output" / "priority1" / benchmark_id
    audit_root = (
        args.audit_root.expanduser().resolve()
        if args.audit_root is not None
        else benchmark_dir / "out_audit"
    )
    metrics_dir = benchmark_dir / "metrics"
    validation_dir = benchmark_dir / "validation"
    known_72h_outputs = sorted(validation_dir.glob("*72h*")) if validation_dir.exists() else []
    require(
        not known_72h_outputs,
        f"72 h validation output already exists; freeze must precede it: {known_72h_outputs}",
    )

    tau_grid = [float(value) for value in empirical["calibration_tau_grid"]]
    require(tau_grid == sorted(set(tau_grid)), "Calibration tau grid must be sorted and unique")
    expected_rows = {float(row["tau"]): row for row in decision["calibration_counts"]}
    require(set(tau_grid) == set(expected_rows), "Frozen calibration summary/grid mismatch")
    expected_claims = int(primary["candidate_pool_size"])
    audits: dict[float, pd.DataFrame] = {}
    audit_files: dict[float, tuple[Path, Path, Path]] = {}
    memberships: dict[float, set[str]] = {}
    reference_map: pd.DataFrame | None = None

    for tau in tau_grid:
        run_dir = audit_root / f"{args.run_prefix}_tau_{tau_tag(tau)}_calibration_v1"
        audit_path = run_dir / "audit_log.tsv"
        claims_path = run_dir / "claims.proposed.tsv"
        run_meta_path = run_dir / "run_meta.json"
        for path in (audit_path, claims_path, run_meta_path):
            require(path.is_file(), f"Missing calibration artifact: {path}")
        audit = validate_audit(
            audit_path,
            claims_path,
            expected_claims=expected_claims,
            expected_tau=tau,
        )
        counts = audit["status"].value_counts()
        expected = expected_rows[tau]
        require(int(counts.get("PASS", 0)) == int(expected["pass"]), f"PASS drift at tau={tau}")
        require(
            int(counts.get("ABSTAIN", 0)) == int(expected["abstain"]),
            f"ABSTAIN drift at tau={tau}",
        )
        require(int(counts.get("FAIL", 0)) == int(expected["fail"]), f"FAIL drift at tau={tau}")
        nonpass = audit.loc[audit["status"].eq("ABSTAIN"), "abstain_reason"].fillna("")
        require(set(nonpass) <= {"unstable"}, f"Unexpected abstention reason at tau={tau}")
        memberships[tau] = set(audit.loc[audit["status"].eq("PASS"), "claim_id"].astype(str))

        mapping = audit[["claim_id", "entity", "term_survival_agg"]].copy()
        mapping["claim_id"] = mapping["claim_id"].astype(str)
        mapping = mapping.sort_values("claim_id").reset_index(drop=True)
        if reference_map is None:
            reference_map = mapping
        else:
            pd.testing.assert_frame_equal(reference_map, mapping, check_exact=True)
        audits[tau] = audit
        audit_files[tau] = (audit_path, claims_path, run_meta_path)

    for lower, higher in zip(tau_grid[:-1], tau_grid[1:], strict=True):
        require(
            memberships[higher].issubset(memberships[lower]),
            f"PASS membership is not monotone: {lower} -> {higher}",
        )

    primary_audit = audits[primary_tau]
    fgsea_path = benchmark_dir / "derived" / "fgsea" / "discovery_48h.tsv"
    require(fgsea_path.is_file(), f"Missing baseline fgsea: {fgsea_path}")
    fgsea = pd.read_csv(fgsea_path, sep="\t")
    required_fgsea = {"pathway", "NES", "pval", "padj", "size", "leadingEdge"}
    require(not (required_fgsea - set(fgsea.columns)), "Baseline fgsea columns are incomplete")
    require(len(fgsea) == expected_claims, "Baseline fgsea candidate-pool size drift")
    require(
        fgsea[["NES", "pval", "padj"]].notna().all().all(),
        "Baseline fgsea contains missing primary statistics",
    )

    audit_columns = ["claim_id", "entity", "status", "term_survival_agg"]
    if "direction" in primary_audit.columns:
        audit_columns.append("direction")
    membership = primary_audit[audit_columns].copy()
    membership["claim_id"] = membership["claim_id"].astype(str)
    membership = membership.merge(
        fgsea[["pathway", "NES", "pval", "padj", "size", "leadingEdge"]],
        left_on="entity",
        right_on="pathway",
        how="left",
        validate="one_to_one",
    )
    require(membership["pathway"].notna().all(), "Audit/fgsea pathway merge failed")
    membership["leading_edge_n"] = membership["leadingEdge"].map(
        lambda value: len(_shared.parse_genes(value))
    )
    require((membership["leading_edge_n"] > 0).all(), "Empty baseline leading edge")
    membership["empirical_selected"] = membership["status"].eq("PASS")
    selected_k = int(membership["empirical_selected"].sum())
    require(selected_k == int(decision["selected_k"]), "Frozen primary K drift")

    q_selected = set(q_value_order(membership).head(selected_k)["claim_id"].astype(str))
    membership["q_value_matched_selected"] = membership["claim_id"].isin(q_selected)
    membership = add_size_strata(membership, n_strata=4)
    size_matched_ids: set[str] = set()
    for stratum, group in membership.groupby("leading_edge_size_stratum", sort=True):
        target = int(group["empirical_selected"].sum())
        require(target <= len(group), f"Invalid size-stratum target: {stratum}")
        size_matched_ids.update(q_value_order(group).head(target)["claim_id"].astype(str))
    require(len(size_matched_ids) == selected_k, "Size-matched comparator K drift")
    membership["q_value_size_matched_selected"] = membership["claim_id"].isin(size_matched_ids)

    empirical_ids = set(membership.loc[membership["empirical_selected"], "claim_id"])
    overlap_q = len(empirical_ids & q_selected)
    overlap_size = len(empirical_ids & size_matched_ids)
    require(
        overlap_q == int(decision["expected_empirical_q_value_overlap"]),
        "Empirical/q-value overlap drift",
    )
    require(
        overlap_size == int(decision["expected_empirical_size_matched_overlap"]),
        "Empirical/size-matched overlap drift",
    )

    if "direction" not in membership.columns:
        membership["direction"] = np.where(membership["NES"] >= 0, "up", "down")
    membership = membership.rename(
        columns={
            "NES": "NES_48h",
            "direction": "direction_48h",
            "padj": "padj_48h",
            "pval": "pval_48h",
            "size": "pathway_size",
            "term_survival_agg": "empirical_survival_48h",
        }
    )
    membership["primary_tau"] = primary_tau
    membership["selection_status"] = "FROZEN"
    membership = membership[
        [
            "claim_id",
            "pathway",
            "entity",
            "direction_48h",
            "NES_48h",
            "pval_48h",
            "padj_48h",
            "pathway_size",
            "leadingEdge",
            "leading_edge_n",
            "leading_edge_size_stratum",
            "empirical_survival_48h",
            "empirical_selected",
            "q_value_matched_selected",
            "q_value_size_matched_selected",
            "primary_tau",
            "selection_status",
        ]
    ].sort_values("claim_id")

    tau_grid_table = pd.concat(
        [
            audit.assign(tau=tau)[
                ["tau", "claim_id", "entity", "status", "term_survival_agg", "abstain_reason"]
            ]
            for tau, audit in audits.items()
        ],
        ignore_index=True,
    )
    tau_grid_table["selection_status"] = "FROZEN"
    tau_grid_table = tau_grid_table.sort_values(["tau", "claim_id"])

    membership_path = metrics_dir / "selection_membership_frozen_tau0p80.tsv"
    tau_grid_path = metrics_dir / "selection_membership_tau_grid_frozen.tsv"
    manifest_path = metrics_dir / "priority1_freeze_manifest.json"
    sidecar_path = metrics_dir / "priority1_freeze_manifest.sha256"
    membership_text = table_text(membership)
    tau_grid_text = table_text(tau_grid_table)
    membership_hash = hashlib.sha256(membership_text.encode()).hexdigest()
    tau_grid_hash = hashlib.sha256(tau_grid_text.encode()).hexdigest()

    data_inputs = [
        data_root / "input" / str(config["dataset"]["expression_file"]),
        data_root / "input" / str(config["dataset"]["metadata_file"]),
        benchmark_dir / "preflight" / "input_manifest.json",
        benchmark_dir / "preflight" / "preflight_summary.json",
        benchmark_dir / "preflight" / "sample_metadata.normalized.tsv",
        benchmark_dir / "preflight" / "design_counts.tsv",
        benchmark_dir / "derived" / "rankings" / "discovery_48h.tsv",
        benchmark_dir / "derived" / "rankings" / "discovery_48h_gene_universe.tsv",
        benchmark_dir / "derived" / "fgsea" / "discovery_48h.tsv",
        benchmark_dir / "derived" / "fgsea" / "hallmark_gene_sets.tsv",
        benchmark_dir / "derived" / "empirical_resampling_48h" / "resample_manifest.tsv",
        benchmark_dir / "derived" / "empirical_resampling_48h" / "fgsea_resamples.tsv",
        benchmark_dir / "derived" / "empirical_resampling_48h" / "resample_qc.tsv",
        benchmark_dir / "evidence_tables" / "discovery_48h_empirical_replicates.tsv",
        benchmark_dir / "sample_cards" / "discovery_48h_empirical.sample_card.json",
        metrics_dir / "empirical_stability_calibration_preview.tsv",
        metrics_dir / "selection_membership_empirical_tau0p80_preview.tsv",
    ]
    for tau in tau_grid:
        data_inputs.extend(audit_files[tau])

    expression_path = data_inputs[0]
    metadata_path = data_inputs[1]
    require(
        sha256_file(expression_path) == config["dataset"]["expression_sha256"],
        "Count-matrix SHA-256 drift",
    )
    require(
        sha256_file(metadata_path) == config["dataset"]["metadata_sha256"],
        "Metadata SHA-256 drift",
    )

    code_inputs = [config_path, REPO_ROOT / "pyproject.toml"]
    code_inputs.extend(sorted((REPO_ROOT / "src" / "llm_pathway_curator").rglob("*.py")))
    code_inputs.extend(
        sorted(
            path
            for path in (CRM_DIR / "scripts").iterdir()
            if path.is_file() and re.match(r"^(00|1[0-7])_", path.name)
        )
    )
    inventory = [inventory_entry(path, scope="data_root", root=data_root) for path in data_inputs]
    inventory.extend(
        inventory_entry(path, scope="repository", root=REPO_ROOT) for path in code_inputs
    )

    correlation = membership[
        ["empirical_survival_48h", "pathway_size", "leading_edge_n", "padj_48h", "NES_48h"]
    ].copy()
    correlation["neglog10_padj_48h"] = -np.log10(
        correlation["padj_48h"].clip(lower=np.finfo(float).tiny)
    )
    correlation["abs_NES_48h"] = correlation["NES_48h"].abs()
    correlation_matrix = correlation.corr(method="spearman").round(12)

    manifest = {
        "benchmark_id": benchmark_id,
        "code": {
            "branch_at_freeze": git_value("branch", "--show-current"),
            "commit_at_freeze": git_value("rev-parse", "HEAD"),
            "tracked_worktree_clean": True,
        },
        "comparators": {
            "empirical_q_value_overlap": overlap_q,
            "empirical_size_matched_overlap": overlap_size,
            "leading_edge_size_strata": 4,
            "q_value_order": [
                "padj ascending",
                "pval ascending",
                "abs_NES descending",
                "pathway ascending",
            ],
            "random_matched_secondary": config["random_matched_secondary"],
        },
        "created_at_utc": datetime.now(UTC).isoformat(),
        "freeze_label": args.freeze_label,
        "held_out_expression_outcomes_calculated": False,
        "input_inventory": inventory,
        "manifest_schema_version": "CRM_R1_PRIORITY1_FREEZE_MANIFEST_v1",
        "memberships": {
            "empirical_stability_audit": sorted(empirical_ids),
            "q_value_matched": sorted(q_selected),
            "q_value_and_leading_edge_size_matched": sorted(size_matched_ids),
            "tau_grid_pass": {
                f"{tau:.2f}": sorted(claim_ids) for tau, claim_ids in memberships.items()
            },
        },
        "outputs": {
            "primary_membership": {
                "path": membership_path.relative_to(data_root).as_posix(),
                "sha256": membership_hash,
                "rows": len(membership),
            },
            "tau_grid_membership": {
                "path": tau_grid_path.relative_to(data_root).as_posix(),
                "sha256": tau_grid_hash,
                "rows": len(tau_grid_table),
            },
        },
        "primary_tau": primary_tau,
        "protocol": {
            "path": config_path.relative_to(REPO_ROOT).as_posix(),
            "sha256": sha256_file(config_path),
            "status": config["status"],
            "version": config["protocol_version"],
        },
        "python": {
            "implementation": platform.python_implementation(),
            "version": sys.version,
        },
        "selected_k": selected_k,
        "spearman_discovery_diagnostics": correlation_matrix.to_dict(),
        "validation_output_absent_at_freeze": True,
    }
    write_freeze_bundle(
        membership_path=membership_path,
        membership_text=membership_text,
        tau_grid_path=tau_grid_path,
        tau_grid_text=tau_grid_text,
        manifest_path=manifest_path,
        manifest=manifest,
        sidecar_path=sidecar_path,
    )

    print("[PASS] Priority 1 protocol and memberships frozen at tau=0.80")
    print(f"[INFO] Frozen K: {selected_k} of {expected_claims}")
    print(f"[INFO] Empirical/q-value overlap: {overlap_q}/{selected_k}")
    print(f"[INFO] Empirical/size-matched overlap: {overlap_size}/{selected_k}")
    print(f"[INFO] Freeze label: {args.freeze_label}")
    print(f"[INFO] Wrote: {manifest_path}")
    print("[INFO] No 72 h expression or pathway endpoint was read or calculated.")
    print("[NEXT] Run 17_check_priority1_freeze.py; do not calculate 72 h before it passes.")


if __name__ == "__main__":
    main()
