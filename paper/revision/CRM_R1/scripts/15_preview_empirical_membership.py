#!/usr/bin/env python3
"""Summarize 48 h empirical calibration and optionally write membership previews."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from llm_pathway_curator import _shared
from llm_pathway_curator.claim_schema import Claim

CRM_DIR = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = CRM_DIR / "config" / "priority1_protocol.json"


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


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


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


def validate_claims(path: Path) -> int:
    proposed = pd.read_csv(path, sep="\t")
    require("claim_json" in proposed.columns, f"Missing claim_json: {path}")
    for value in proposed["claim_json"]:
        Claim.model_validate_json(value)
    return len(proposed)


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--audit-root", type=Path, default=None)
    parser.add_argument(
        "--run-prefix",
        default="discovery_48h_empirical_ctxoff_note",
    )
    parser.add_argument("--primary-tau", type=float, default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    data_root = args.data_root.expanduser().resolve()
    config_path = args.config.expanduser().resolve()
    config = read_json(config_path)
    primary = config["primary_analysis"]
    empirical = config["empirical_stability"]
    benchmark_id = str(config["benchmark_id"])
    benchmark_dir = data_root / "output" / "priority1" / benchmark_id
    audit_root = (
        args.audit_root.expanduser().resolve()
        if args.audit_root is not None
        else benchmark_dir / "out_audit"
    )
    metrics_dir = benchmark_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    require(
        config["status"] in {"DRAFT_NOT_FROZEN", "FROZEN"},
        "Preview requires a draft or frozen Priority 1 protocol",
    )
    require(primary["context_review_mode"] == "off", "Context review must be off")
    require(primary["context_gate_mode"] == "note", "Context gate must be note")
    frozen_tau = empirical["primary_tau"]
    if config["status"] == "DRAFT_NOT_FROZEN":
        require(frozen_tau is None, "Draft config primary_tau must remain null")
    else:
        require(frozen_tau is not None, "Frozen config primary_tau must be set")
    tau_grid = [float(value) for value in empirical["calibration_tau_grid"]]
    require(len(tau_grid) >= 2, "Calibration grid requires at least two tau values")
    require(tau_grid == sorted(set(tau_grid)), "Calibration tau grid must be sorted and unique")

    rows: list[dict[str, Any]] = []
    memberships: dict[float, set[str]] = {}
    audit_paths: dict[float, Path] = {}

    for tau in tau_grid:
        run_dir = audit_root / f"{args.run_prefix}_tau_{tau_tag(tau)}_calibration_v1"
        audit_path = run_dir / "audit_log.tsv"
        claims_path = run_dir / "claims.proposed.tsv"
        require(audit_path.is_file(), f"Missing calibration audit: {audit_path}")
        require(claims_path.is_file(), f"Missing proposed claims: {claims_path}")
        audit_paths[tau] = audit_path

        audit = pd.read_csv(audit_path, sep="\t")
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
        require(not (required_columns - set(audit.columns)), f"Audit columns missing: {audit_path}")
        require(len(audit) == int(primary["candidate_pool_size"]), "Unexpected claim count")
        require(audit["claim_id"].nunique() == len(audit), "claim_id values are not unique")
        require(audit["entity"].nunique() == len(audit), "entity values are not unique")
        require(validate_claims(claims_path) == len(audit), "Claim/audit row count mismatch")

        require(
            not bool_series(audit["context_evaluated"], column="context_evaluated").any(),
            "Context was evaluated in context-off run",
        )
        require(
            not bool_series(audit["context_gate_blocked"], column="context_gate_blocked").any(),
            "Context blocked a context-off run",
        )
        require(
            not bool_series(audit["context_gate_hit"], column="context_gate_hit").any(),
            "Context gate hit in context-off run",
        )
        require(
            not audit["abstain_reason"].fillna("").str.contains("context", case=False).any(),
            "Context abstention found in context-off run",
        )
        require(
            not audit["fail_reason"].fillna("").str.contains("context", case=False).any(),
            "Context failure found in context-off run",
        )
        if "distill_semantics" in audit.columns:
            semantics = set(audit["distill_semantics"].dropna().astype(str))
            require(semantics == {"replicates_proxy"}, f"Unexpected distill semantics: {semantics}")

        observed_tau = float(pd.to_numeric(audit["tau_used"], errors="raise").iloc[0])
        require(math.isclose(observed_tau, tau, abs_tol=1e-12), "tau directory/log mismatch")
        require(
            pd.to_numeric(audit["term_survival_agg"], errors="coerce").notna().all(),
            "Missing empirical survival",
        )
        status = audit["status"].astype(str).str.upper()
        counts = status.value_counts()
        memberships[tau] = set(audit.loc[status.eq("PASS"), "claim_id"].astype(str))
        rows.append(
            {
                "tau": tau,
                "PASS": int(counts.get("PASS", 0)),
                "ABSTAIN": int(counts.get("ABSTAIN", 0)),
                "FAIL": int(counts.get("FAIL", 0)),
                "coverage": float(counts.get("PASS", 0) / len(audit)),
                "abstain_reasons": ";".join(
                    f"{key}:{value}"
                    for key, value in audit.loc[status.eq("ABSTAIN"), "abstain_reason"]
                    .fillna("")
                    .value_counts()
                    .items()
                ),
            }
        )

    calibration = pd.DataFrame(rows).sort_values("tau").reset_index(drop=True)
    for lower, higher in zip(tau_grid[:-1], tau_grid[1:], strict=True):
        require(
            memberships[higher].issubset(memberships[lower]),
            f"PASS membership is not monotone: {lower} -> {higher}",
        )

    calibration_path = metrics_dir / "empirical_stability_calibration_preview.tsv"
    require(
        args.force or not calibration_path.exists(),
        f"Output exists; use --force: {calibration_path}",
    )
    calibration.to_csv(calibration_path, sep="\t", index=False)

    output_paths: dict[str, Path] = {"calibration_preview": calibration_path}
    membership: pd.DataFrame | None = None
    primary_tau = args.primary_tau
    if primary_tau is not None:
        matching_tau = [tau for tau in tau_grid if math.isclose(tau, primary_tau, abs_tol=1e-12)]
        require(len(matching_tau) == 1, "--primary-tau must be in the prespecified grid")
        primary_tau = matching_tau[0]
        if frozen_tau is not None:
            require(
                math.isclose(primary_tau, float(frozen_tau), abs_tol=1e-12),
                "Preview tau cannot differ from the frozen primary tau",
            )
        audit = pd.read_csv(audit_paths[primary_tau], sep="\t")
        fgsea_path = benchmark_dir / "derived" / "fgsea" / "discovery_48h.tsv"
        require(fgsea_path.is_file(), f"Missing baseline fgsea: {fgsea_path}")
        fgsea = pd.read_csv(fgsea_path, sep="\t")
        require(
            {"pathway", "NES", "pval", "padj", "size", "leadingEdge"}.issubset(fgsea.columns),
            "Baseline fgsea columns are incomplete",
        )

        membership = audit[["claim_id", "entity", "status", "term_survival_agg"]].copy()
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
        require(0 < selected_k < len(membership), "Primary tau has degenerate coverage")

        ordered = q_value_order(membership)
        q_selected = set(ordered.head(selected_k)["claim_id"].astype(str))
        membership["q_value_matched_selected"] = membership["claim_id"].astype(str).isin(q_selected)

        membership = add_size_strata(membership, n_strata=4)
        size_matched_ids: set[str] = set()
        for stratum, group in membership.groupby("leading_edge_size_stratum", sort=True):
            target = int(group["empirical_selected"].sum())
            require(target <= len(group), f"Invalid size-stratum target: {stratum}")
            chosen = q_value_order(group).head(target)["claim_id"].astype(str)
            size_matched_ids.update(chosen)
        require(len(size_matched_ids) == selected_k, "Size-matched comparator K drift")
        membership["q_value_size_matched_selected"] = (
            membership["claim_id"].astype(str).isin(size_matched_ids)
        )
        membership["primary_tau_preview"] = primary_tau
        membership["selection_status"] = "PREVIEW_NOT_FROZEN"

        membership_path = metrics_dir / (
            f"selection_membership_empirical_tau{tau_tag(primary_tau)}_preview.tsv"
        )
        require(args.force or not membership_path.exists(), f"Output exists: {membership_path}")
        membership.sort_values(
            ["empirical_selected", "term_survival_agg", "padj", "pathway"],
            ascending=[False, False, True, True],
        ).to_csv(membership_path, sep="\t", index=False)
        output_paths["membership_preview"] = membership_path

    meta_path = metrics_dir / "empirical_stability_preview.run_meta.json"
    require(args.force or not meta_path.exists(), f"Output exists; use --force: {meta_path}")
    metadata = {
        "analysis_scope": "ENDO 48 h discovery-only empirical-stability preview",
        "audit_inputs": {
            str(tau): {"path": str(path), "sha256": sha256_file(path)}
            for tau, path in audit_paths.items()
        },
        "benchmark_id": benchmark_id,
        "held_out_expression_outcomes_calculated": False,
        "outputs": {
            name: {"path": str(path), "sha256": sha256_file(path)}
            for name, path in output_paths.items()
        },
        "primary_tau_preview": primary_tau,
        "protocol": {"path": str(config_path), "sha256": sha256_file(config_path)},
        "protocol_status": config["status"],
        "protocol_version": config["protocol_version"],
        "python": {"implementation": platform.python_implementation(), "version": sys.version},
        "script": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256_file(Path(__file__).resolve()),
        },
    }
    write_json(meta_path, metadata)

    print("[PASS] Empirical calibration claims are schema-valid and context is non-blocking")
    print("[PASS] PASS membership is monotone across tau")
    print("\n=== 48 H EMPIRICAL CALIBRATION PREVIEW ===")
    print(calibration.to_string(index=False))
    if membership is not None and primary_tau is not None:
        empirical_ids = set(
            membership.loc[membership["empirical_selected"], "claim_id"].astype(str)
        )
        q_ids = set(membership.loc[membership["q_value_matched_selected"], "claim_id"].astype(str))
        size_q_ids = set(
            membership.loc[membership["q_value_size_matched_selected"], "claim_id"].astype(str)
        )
        correlation = membership[
            ["term_survival_agg", "size", "leading_edge_n", "padj", "NES"]
        ].copy()
        correlation["neglog10_padj"] = -np.log10(
            correlation["padj"].clip(lower=np.finfo(float).tiny)
        )
        correlation["abs_NES"] = correlation["NES"].abs()
        print(f"\n=== PRIMARY TAU PREVIEW: {primary_tau:.2f} ===")
        print(f"K: {len(empirical_ids)}")
        print(f"empirical/q-value overlap: {len(empirical_ids & q_ids)}/{len(empirical_ids)}")
        print(
            "empirical/q-value-size-matched overlap: "
            f"{len(empirical_ids & size_q_ids)}/{len(empirical_ids)}"
        )
        print("\n=== SPEARMAN CORRELATIONS ===")
        print(
            correlation[
                [
                    "term_survival_agg",
                    "size",
                    "leading_edge_n",
                    "neglog10_padj",
                    "abs_NES",
                ]
            ]
            .corr(method="spearman")
            .round(3)
            .to_string()
        )
    print(f"\n[INFO] Wrote: {calibration_path}")
    print("[INFO] Preview only; protocol and membership remain unfrozen.")
    print("[INFO] No 72 h expression or pathway result was read.")


if __name__ == "__main__":
    main()
