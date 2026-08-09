#!/usr/bin/env python3
"""Evaluate the frozen Priority 1 held-out 72 h replication endpoint exactly once."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import platform
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from statistics import NormalDist
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import norm, rankdata, spearmanr

CRM_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = CRM_DIR.parents[2]
DEFAULT_CONFIG = CRM_DIR / "config" / "priority1_protocol.json"
DEFAULT_FREEZE_CHECK = Path(__file__).resolve().with_name("17_check_priority1_freeze.py")


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


def selected_mask(values: pd.Series, *, column: str) -> pd.Series:
    if pd.api.types.is_bool_dtype(values.dtype):
        return values.fillna(False).astype(bool)
    normalized = values.fillna("").astype(str).str.strip().str.lower()
    require(set(normalized) <= {"true", "false"}, f"Invalid Boolean values in {column}")
    return normalized.eq("true")


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
    require(not status, f"Commit V6 and leave tracked files clean before evaluation: {status}")


def run_freeze_gate(*, checker: Path, data_root: Path, config_path: Path) -> list[str]:
    result = subprocess.run(
        [
            sys.executable,
            str(checker),
            "--data-root",
            str(data_root),
            "--config",
            str(config_path),
            "--allow-post-validation",
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    output = [line for line in (result.stdout + result.stderr).splitlines() if line]
    for line in output:
        print(line)
    require(result.returncode == 0, "Post-validation freeze integrity check failed")
    require(
        any(line.startswith("[PASS] Priority 1 freeze bundle") for line in output),
        "Freeze checker did not confirm the immutable bundle",
    )
    return output


def wilson_interval(successes: int, total: int, *, alpha: float = 0.05) -> tuple[float, float]:
    require(total > 0, "Wilson interval requires total > 0")
    require(0 <= successes <= total, "Wilson interval successes outside [0,total]")
    z = NormalDist().inv_cdf(1.0 - alpha / 2.0)
    proportion = successes / total
    denominator = 1.0 + z**2 / total
    center = (proportion + z**2 / (2.0 * total)) / denominator
    half_width = (
        z
        * math.sqrt(proportion * (1.0 - proportion) / total + z**2 / (4.0 * total**2))
        / denominator
    )
    return center - half_width, center + half_width


def binary_auc(scores: np.ndarray, outcomes: np.ndarray) -> float:
    scores = np.asarray(scores, dtype=float)
    outcomes = np.asarray(outcomes, dtype=int)
    require(scores.ndim == outcomes.ndim == 1 and len(scores) == len(outcomes), "AUC shape drift")
    require(np.isfinite(scores).all(), "AUC scores must be finite")
    require(set(np.unique(outcomes)) <= {0, 1}, "AUC outcomes must be binary")
    positives = outcomes == 1
    n_positive = int(positives.sum())
    n_negative = int((~positives).sum())
    require(n_positive > 0 and n_negative > 0, "AUC requires both outcome classes")
    ranks = rankdata(scores, method="average")
    rank_sum = float(ranks[positives].sum())
    return (rank_sum - n_positive * (n_positive + 1) / 2.0) / (n_positive * n_negative)


def auc_resampling(
    scores: np.ndarray,
    outcomes: np.ndarray,
    *,
    bootstrap_draws: int,
    bootstrap_seed: int,
    permutation_draws: int,
    permutation_seed: int,
) -> dict[str, float | int | str]:
    scores = np.asarray(scores, dtype=float)
    outcomes = np.asarray(outcomes, dtype=int)
    if len(np.unique(outcomes)) < 2:
        return {
            "status": "NON_ESTIMABLE_SINGLE_OUTCOME_CLASS",
            "auroc": float("nan"),
            "bootstrap_ci_low": float("nan"),
            "bootstrap_ci_high": float("nan"),
            "bootstrap_draws": bootstrap_draws,
            "bootstrap_seed": bootstrap_seed,
            "bootstrap_type": "not estimable",
            "permutation_p_one_sided": float("nan"),
            "permutation_draws": permutation_draws,
            "permutation_seed": permutation_seed,
        }
    observed = binary_auc(scores, outcomes)
    positive_scores = scores[outcomes == 1]
    negative_scores = scores[outcomes == 0]

    bootstrap_rng = np.random.default_rng(bootstrap_seed)
    bootstrap = np.empty(bootstrap_draws, dtype=float)
    for index in range(bootstrap_draws):
        sampled_positive = bootstrap_rng.choice(
            positive_scores, size=len(positive_scores), replace=True
        )
        sampled_negative = bootstrap_rng.choice(
            negative_scores, size=len(negative_scores), replace=True
        )
        sampled_scores = np.concatenate([sampled_positive, sampled_negative])
        sampled_outcomes = np.concatenate(
            [np.ones(len(sampled_positive), dtype=int), np.zeros(len(sampled_negative), dtype=int)]
        )
        bootstrap[index] = binary_auc(sampled_scores, sampled_outcomes)

    permutation_rng = np.random.default_rng(permutation_seed)
    exceedances = 0
    for _ in range(permutation_draws):
        permuted = permutation_rng.permutation(outcomes)
        exceedances += binary_auc(scores, permuted) >= observed - 1e-15

    return {
        "status": "ESTIMABLE",
        "auroc": observed,
        "bootstrap_ci_low": float(np.quantile(bootstrap, 0.025)),
        "bootstrap_ci_high": float(np.quantile(bootstrap, 0.975)),
        "bootstrap_draws": bootstrap_draws,
        "bootstrap_seed": bootstrap_seed,
        "bootstrap_type": "stratified percentile bootstrap within outcome class",
        "permutation_p_one_sided": (exceedances + 1.0) / (permutation_draws + 1.0),
        "permutation_draws": permutation_draws,
        "permutation_seed": permutation_seed,
    }


def overlap_exact_randomization(
    pathway: pd.DataFrame,
    *,
    empirical_ids: set[str],
    q_value_ids: set[str],
) -> tuple[dict[str, Any], pd.DataFrame]:
    require(len(empirical_ids) == len(q_value_ids), "Exact comparison requires equal K")
    selected_k = len(empirical_ids)
    common = empirical_ids & q_value_ids
    empirical_only = sorted(empirical_ids - common)
    q_value_only = sorted(q_value_ids - common)
    require(len(empirical_only) == len(q_value_only), "Symmetric-difference sizes differ")
    require(empirical_only, "Exact comparison has an empty symmetric difference")

    outcomes = pathway.set_index("claim_id")["replicated_primary"].astype(int).to_dict()
    empirical_fraction = sum(outcomes[value] for value in empirical_ids) / selected_k
    q_value_fraction = sum(outcomes[value] for value in q_value_ids) / selected_k
    observed = empirical_fraction - q_value_fraction

    union = empirical_only + q_value_only
    union_outcomes = np.asarray([outcomes[value] for value in union], dtype=int)
    unique_k = len(empirical_only)
    null_differences: list[float] = []
    all_indices = set(range(len(union)))
    for assigned_empirical in itertools.combinations(range(len(union)), unique_k):
        empirical_index = set(assigned_empirical)
        q_index = all_indices - empirical_index
        difference = (
            int(union_outcomes[list(empirical_index)].sum())
            - int(union_outcomes[list(q_index)].sum())
        ) / selected_k
        null_differences.append(difference)

    null_values = np.asarray(null_differences, dtype=float)
    one_sided = float(np.mean(null_values >= observed - 1e-15))
    two_sided = float(np.mean(np.abs(null_values) >= abs(observed) - 1e-15))
    null_table = (
        pd.Series(null_values, name="replication_fraction_difference")
        .value_counts(sort=False)
        .rename_axis("replication_fraction_difference")
        .reset_index(name="assignments")
        .sort_values("replication_fraction_difference")
    )
    null_table["probability"] = null_table["assignments"] / len(null_values)

    summary = {
        "common_claims": len(common),
        "empirical_only_claims": len(empirical_only),
        "q_value_only_claims": len(q_value_only),
        "assignments": len(null_values),
        "observed_replication_fraction_difference": observed,
        "p_one_sided_empirical_greater": one_sided,
        "p_two_sided": two_sided,
        "interpretation": (
            "descriptive conditional exact reference; pathways are biologically dependent"
        ),
    }
    return summary, null_table


def fit_logistic_sensitivity(pathway: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    predictors = pd.DataFrame(
        {
            "empirical_survival_z": pathway["empirical_survival_48h"].astype(float),
            "neglog10_padj_48h_z": -np.log10(
                pathway["padj_48h"].astype(float).clip(lower=np.finfo(float).tiny)
            ),
            "log1p_leading_edge_n_z": np.log1p(pathway["leading_edge_n"].astype(float)),
        }
    )
    standard_deviation = predictors.std(axis=0, ddof=0)
    require((standard_deviation > 0).all(), "Logistic predictor has zero variance")
    predictors = (predictors - predictors.mean(axis=0)) / standard_deviation
    matrix = np.column_stack([np.ones(len(predictors)), predictors.to_numpy(dtype=float)])
    outcome = pathway["replicated_primary"].astype(int).to_numpy()
    names = ["intercept", *predictors.columns]
    if len(np.unique(outcome)) < 2:
        status = "NON_ESTIMABLE_SINGLE_OUTCOME_CLASS"
        coefficient_table = pd.DataFrame(
            {
                "term": names,
                "coefficient": np.nan,
                "standard_error": np.nan,
                "z": np.nan,
                "p_value": np.nan,
                "model_status": status,
            }
        )
        return coefficient_table, {
            "status": status,
            "optimizer_success": False,
            "optimizer_message": "binary outcome has one observed class",
            "hessian_condition_number": float("nan"),
            "model_auroc_apparent": float("nan"),
            "n": len(outcome),
            "events": int(outcome.sum()),
            "role": "exploratory adjusted sensitivity; no model repair permitted",
        }

    def objective(coefficients: np.ndarray) -> float:
        linear = matrix @ coefficients
        return float(np.logaddexp(0.0, linear).sum() - outcome @ linear)

    result = minimize(objective, np.zeros(matrix.shape[1]), method="BFGS")
    coefficients = np.asarray(result.x, dtype=float)
    probabilities = expit(matrix @ coefficients)
    weights = probabilities * (1.0 - probabilities)
    hessian = matrix.T @ (matrix * weights[:, None])
    condition = float(np.linalg.cond(hessian))
    separated = bool(
        np.max(np.abs(coefficients)) > 25
        or np.min(probabilities) < 1e-8
        or np.max(probabilities) > 1.0 - 1e-8
        or not np.isfinite(condition)
        or condition > 1e12
    )
    estimable = bool(result.success and not separated)
    if estimable:
        covariance = np.linalg.inv(hessian)
        standard_errors = np.sqrt(np.diag(covariance))
        z_values = coefficients / standard_errors
        p_values = 2.0 * norm.sf(np.abs(z_values))
        status = "ESTIMABLE_EXPLORATORY"
    else:
        standard_errors = np.full(len(coefficients), np.nan)
        z_values = np.full(len(coefficients), np.nan)
        p_values = np.full(len(coefficients), np.nan)
        status = "NON_ESTIMABLE_OR_SEPARATED_REPORTED_WITHOUT_MODEL_CHANGE"

    coefficient_table = pd.DataFrame(
        {
            "term": names,
            "coefficient": coefficients,
            "standard_error": standard_errors,
            "z": z_values,
            "p_value": p_values,
            "model_status": status,
        }
    )
    model_auc = binary_auc(probabilities, outcome)
    summary = {
        "status": status,
        "optimizer_success": bool(result.success),
        "optimizer_message": str(result.message),
        "hessian_condition_number": condition,
        "model_auroc_apparent": model_auc,
        "n": len(outcome),
        "events": int(outcome.sum()),
        "role": "exploratory adjusted sensitivity; no model repair permitted",
    }
    return coefficient_table, summary


def method_summary(pathway: pd.DataFrame) -> pd.DataFrame:
    methods = [
        ("raw_pool", "descriptive_reference", pd.Series(True, index=pathway.index)),
        (
            "empirical_stability_audit",
            "primary_method",
            pathway["empirical_selected"],
        ),
        ("q_value_matched", "primary_comparator", pathway["q_value_matched_selected"]),
        (
            "q_value_and_leading_edge_size_matched",
            "size_matched_sensitivity",
            pathway["q_value_size_matched_selected"],
        ),
    ]
    rows: list[dict[str, Any]] = []
    for method, role, mask in methods:
        selected = pathway.loc[mask]
        total = len(selected)
        replicated = int(selected["replicated_primary"].sum())
        same_direction = int(selected["same_direction"].sum())
        ci_low, ci_high = wilson_interval(replicated, total)
        direction_low, direction_high = wilson_interval(same_direction, total)
        rows.append(
            {
                "method": method,
                "role": role,
                "n_selected": total,
                "n_replicated": replicated,
                "replication_fraction": replicated / total,
                "replication_ci_low": ci_low,
                "replication_ci_high": ci_high,
                "replication_interval": "Wilson 95%",
                "n_same_direction": same_direction,
                "same_direction_fraction": same_direction / total,
                "same_direction_ci_low": direction_low,
                "same_direction_ci_high": direction_high,
                "nonreplication_risk": 1.0 - replicated / total,
            }
        )
    return pd.DataFrame(rows)


def tau_grid_summary(tau_grid: pd.DataFrame, pathway: pd.DataFrame) -> pd.DataFrame:
    merged = tau_grid.merge(
        pathway[["claim_id", "replicated_primary"]],
        on="claim_id",
        how="left",
        validate="many_to_one",
    )
    require(merged["replicated_primary"].notna().all(), "Tau grid/pathway merge failed")
    rows: list[dict[str, Any]] = []
    for tau, group in merged.groupby("tau", sort=True):
        selected = group.loc[group["status"].astype(str).str.upper().eq("PASS")]
        total = len(selected)
        require(total > 0, f"Degenerate frozen membership at tau={tau}")
        replicated = int(selected["replicated_primary"].sum())
        ci_low, ci_high = wilson_interval(replicated, total)
        rows.append(
            {
                "tau": float(tau),
                "n_candidate": len(group),
                "n_selected": total,
                "coverage": total / len(group),
                "n_replicated": replicated,
                "replication_fraction": replicated / total,
                "replication_ci_low": ci_low,
                "replication_ci_high": ci_high,
                "nonreplication_risk": 1.0 - replicated / total,
            }
        )
    return pd.DataFrame(rows)


def random_matched_reference(
    pathway: pd.DataFrame,
    *,
    selected_k: int,
    draws: int,
    seed: int,
) -> tuple[dict[str, Any], pd.DataFrame]:
    ordered = pathway.sort_values("claim_id").reset_index(drop=True)
    outcomes = ordered["replicated_primary"].astype(int).to_numpy()
    rng = np.random.default_rng(seed)
    fractions = np.empty(draws, dtype=float)
    for index in range(draws):
        selected = rng.choice(len(outcomes), size=selected_k, replace=False)
        fractions[index] = float(outcomes[selected].mean())
    distribution = (
        pd.Series(fractions, name="replication_fraction")
        .value_counts(sort=False)
        .rename_axis("replication_fraction")
        .reset_index(name="draws")
        .sort_values("replication_fraction")
    )
    distribution["probability"] = distribution["draws"] / draws
    summary = {
        "draws": draws,
        "seed": seed,
        "k": selected_k,
        "mean_replication_fraction": float(fractions.mean()),
        "quantile_0_025": float(np.quantile(fractions, 0.025)),
        "quantile_0_975": float(np.quantile(fractions, 0.975)),
        "role": "secondary random matched reference",
    }
    return summary, distribution


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if not math.isfinite(float(value)) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def write_table(path: Path, table: pd.DataFrame) -> None:
    table.to_csv(
        path,
        sep="\t",
        index=False,
        lineterminator="\n",
        na_rep="NA",
        float_format="%.17g",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--freeze-check", type=Path, default=DEFAULT_FREEZE_CHECK)
    args = parser.parse_args()

    require_clean_tracked_worktree()
    data_root = args.data_root.expanduser().resolve()
    config_path = args.config.expanduser().resolve()
    checker_path = args.freeze_check.expanduser().resolve()
    config = read_json(config_path)
    require(config["status"] == "FROZEN", "Priority 1 protocol must be FROZEN")
    require(config["protocol_version"] == "CRM_R1_PRIORITY1_v5", "Protocol drift")
    require(float(config["empirical_stability"]["primary_tau"]) == 0.8, "Frozen tau drift")
    freeze_gate_output = run_freeze_gate(
        checker=checker_path,
        data_root=data_root,
        config_path=config_path,
    )

    benchmark_id = str(config["benchmark_id"])
    benchmark_dir = data_root / "output" / "priority1" / benchmark_id
    metrics_dir = benchmark_dir / "metrics"
    validation_dir = benchmark_dir / "validation"
    source_dir = benchmark_dir / "source_data"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    source_dir.mkdir(parents=True, exist_ok=True)

    output_paths = {
        "pathway": metrics_dir / "replication_by_pathway.tsv",
        "method": metrics_dir / "replication_by_method.tsv",
        "tau_grid": metrics_dir / "replication_tau_grid.tsv",
        "continuous": metrics_dir / "replication_continuous.tsv",
        "logistic": metrics_dir / "replication_logistic_sensitivity.tsv",
        "exact_null": metrics_dir / "replication_overlap_exact_null.tsv",
        "random_null": metrics_dir / "replication_random_matched_null.tsv",
        "summary": metrics_dir / "priority1_replication_summary.json",
        "figure4": source_dir / "figure4.tsv",
        "run_meta": metrics_dir / "priority1_replication.run_meta.json",
    }
    existing = [path for path in output_paths.values() if path.exists()]
    require(not existing, f"Replication outputs are immutable; existing: {existing}")
    staging_dir = benchmark_dir / ".priority1_replication_staging"
    require(not staging_dir.exists(), f"Remove stale staging directory: {staging_dir}")

    membership_path = metrics_dir / "selection_membership_frozen_tau0p80.tsv"
    tau_grid_path = metrics_dir / "selection_membership_tau_grid_frozen.tsv"
    validation_path = validation_dir / "pathway_statistics_72h.tsv"
    validation_meta_path = validation_dir / "validation_72h.run_meta.json"
    freeze_manifest_path = metrics_dir / "priority1_freeze_manifest.json"
    freeze_sidecar_path = metrics_dir / "priority1_freeze_manifest.sha256"
    for path in (
        membership_path,
        tau_grid_path,
        validation_path,
        validation_meta_path,
        freeze_manifest_path,
        freeze_sidecar_path,
    ):
        require(path.is_file(), f"Missing Priority 1 input: {path}")

    membership = pd.read_csv(membership_path, sep="\t", dtype={"claim_id": "string"})
    tau_grid = pd.read_csv(tau_grid_path, sep="\t", dtype={"claim_id": "string"})
    validation_meta = read_json(validation_meta_path)
    require(
        validation_meta.get("protocol_version") == config["protocol_version"],
        "72 h metadata protocol version drift",
    )
    require(validation_meta.get("protocol_status") == "FROZEN", "72 h protocol was not frozen")
    require(
        validation_meta.get("validation_endpoint_calculated") is True,
        "72 h metadata does not record endpoint calculation",
    )
    require(
        validation_meta.get("discovery_expression_columns_loaded") is False,
        "72 h analysis loaded discovery expression columns",
    )
    samples_loaded = validation_meta.get("samples_loaded", {})
    require(
        samples_loaded.get("cell_state") == config["primary_analysis"]["cell_state"]
        and int(samples_loaded.get("time_h", -1))
        == int(config["primary_analysis"]["validation_time_h"])
        and int(samples_loaded.get("n", -1)) == 12,
        "72 h sample-scope metadata drift",
    )
    require(
        validation_meta["inputs"]["config"]["sha256"] == sha256_file(config_path),
        "72 h/config hash drift",
    )
    require(
        validation_meta["outputs"]["pathway_statistics_72h"]["sha256"]
        == sha256_file(validation_path),
        "72 h pathway table hash does not match its run metadata",
    )

    validation = pd.read_csv(validation_path, sep="\t")
    required_membership = {
        "claim_id",
        "pathway",
        "NES_48h",
        "padj_48h",
        "leading_edge_n",
        "empirical_survival_48h",
        "empirical_selected",
        "q_value_matched_selected",
        "q_value_size_matched_selected",
        "selection_status",
    }
    require(not (required_membership - set(membership.columns)), "Frozen membership columns drift")
    require(len(membership) == 50 and membership["claim_id"].nunique() == 50, "Pool size drift")
    require(set(membership["selection_status"].astype(str)) == {"FROZEN"}, "Membership not frozen")
    for column in (
        "empirical_selected",
        "q_value_matched_selected",
        "q_value_size_matched_selected",
    ):
        membership[column] = selected_mask(membership[column], column=column)
        require(int(membership[column].sum()) == 23, f"Frozen K drift: {column}")

    required_validation = {"pathway", "NES", "pval", "padj", "size", "leadingEdge"}
    require(not (required_validation - set(validation.columns)), "72 h pathway columns drift")
    require(len(validation) == 50 and validation["pathway"].nunique() == 50, "72 h pool drift")
    require(set(validation["pathway"]) == set(membership["pathway"]), "48/72 h pathway set drift")

    validation_for_merge = validation[
        ["pathway", "NES", "pval", "padj", "size", "leadingEdge"]
    ].rename(
        columns={
            "NES": "NES_72h",
            "pval": "pval_72h",
            "padj": "padj_72h",
            "size": "pathway_size_72h",
            "leadingEdge": "leading_edge_72h",
        }
    )
    pathway = membership.merge(
        validation_for_merge,
        on="pathway",
        how="left",
        validate="one_to_one",
    )
    numeric_columns = [
        "NES_48h",
        "padj_48h",
        "leading_edge_n",
        "empirical_survival_48h",
        "NES_72h",
        "pval_72h",
        "padj_72h",
    ]
    for column in numeric_columns:
        pathway[column] = pd.to_numeric(pathway[column], errors="raise")
        require(np.isfinite(pathway[column]).all(), f"Non-finite values in {column}")
    require(pathway["pval_72h"].between(0.0, 1.0).all(), "72 h p-values outside [0,1]")
    require(pathway["padj_72h"].between(0.0, 1.0).all(), "72 h FDR outside [0,1]")
    require(
        np.array_equal(
            pd.to_numeric(pathway["pathway_size"], errors="raise").to_numpy(),
            pd.to_numeric(pathway["pathway_size_72h"], errors="raise").to_numpy(),
        ),
        "48/72 h tested pathway sizes differ despite the frozen universe",
    )
    threshold = float(config["primary_analysis"]["replication_fdr_threshold"])
    pathway["same_direction"] = pathway["NES_48h"] * pathway["NES_72h"] > 0
    pathway["significant_72h"] = pathway["padj_72h"] < threshold
    pathway["replicated_primary"] = pathway["same_direction"] & pathway["significant_72h"]
    pathway["direction_72h"] = np.select(
        [pathway["NES_72h"] > 0, pathway["NES_72h"] < 0],
        ["up", "down"],
        default="neutral",
    )
    pathway = pathway.sort_values("claim_id").reset_index(drop=True)

    methods = method_summary(pathway)
    method_index = methods.set_index("method")
    empirical_fraction = float(
        method_index.loc["empirical_stability_audit", "replication_fraction"]
    )
    q_value_fraction = float(method_index.loc["q_value_matched", "replication_fraction"])
    size_fraction = float(
        method_index.loc["q_value_and_leading_edge_size_matched", "replication_fraction"]
    )
    primary_difference = empirical_fraction - q_value_fraction
    stop_gate = (
        "PASS_EMPIRICAL_POINT_ESTIMATE_IMPROVED"
        if primary_difference > 0
        else "STOP_NO_POINT_ESTIMATE_IMPROVEMENT"
    )

    empirical_ids = set(pathway.loc[pathway["empirical_selected"], "claim_id"].astype(str))
    q_value_ids = set(pathway.loc[pathway["q_value_matched_selected"], "claim_id"].astype(str))
    exact_summary, exact_null = overlap_exact_randomization(
        pathway,
        empirical_ids=empirical_ids,
        q_value_ids=q_value_ids,
    )
    require(
        math.isclose(
            exact_summary["observed_replication_fraction_difference"],
            primary_difference,
            abs_tol=1e-12,
        ),
        "Exact/primary risk-difference drift",
    )

    resampling = config["held_out_inference"]["continuous_resampling"]
    continuous = auc_resampling(
        pathway["empirical_survival_48h"].to_numpy(),
        pathway["replicated_primary"].astype(int).to_numpy(),
        bootstrap_draws=int(resampling["bootstrap_draws"]),
        bootstrap_seed=int(resampling["bootstrap_seed"]),
        permutation_draws=int(resampling["label_permutations"]),
        permutation_seed=int(resampling["permutation_seed"]),
    )
    spearman = spearmanr(pathway["NES_48h"], pathway["NES_72h"])
    continuous.update(
        {
            "n_pathways": len(pathway),
            "n_replicated": int(pathway["replicated_primary"].sum()),
            "spearman_nes_rho": float(spearman.statistic),
            "spearman_nes_p_two_sided": float(spearman.pvalue),
        }
    )
    continuous_table = pd.DataFrame([{"analysis": "empirical_survival_continuous", **continuous}])

    logistic_table, logistic_summary = fit_logistic_sensitivity(pathway)
    tau_summary = tau_grid_summary(tau_grid, pathway)
    random_config = config["random_matched_secondary"]
    random_summary, random_null = random_matched_reference(
        pathway,
        selected_k=int(random_config["k"]),
        draws=int(random_config["draws"]),
        seed=int(random_config["seed"]),
    )
    methods = pd.concat(
        [
            methods,
            pd.DataFrame(
                [
                    {
                        "method": "random_matched_secondary",
                        "role": "secondary_random_reference",
                        "n_selected": int(random_config["k"]),
                        "n_replicated": np.nan,
                        "replication_fraction": random_summary["mean_replication_fraction"],
                        "replication_ci_low": random_summary["quantile_0_025"],
                        "replication_ci_high": random_summary["quantile_0_975"],
                        "replication_interval": "central 95% of 10,000 random matched draws",
                        "n_same_direction": np.nan,
                        "same_direction_fraction": np.nan,
                        "same_direction_ci_low": np.nan,
                        "same_direction_ci_high": np.nan,
                        "nonreplication_risk": 1.0 - random_summary["mean_replication_fraction"],
                    }
                ]
            ),
        ],
        ignore_index=True,
    )

    primary_summary = {
        "benchmark_id": benchmark_id,
        "endpoint": config["held_out_inference"]["primary_binary_endpoint"],
        "empirical_replication_fraction": empirical_fraction,
        "q_value_matched_replication_fraction": q_value_fraction,
        "size_matched_replication_fraction": size_fraction,
        "primary_replication_fraction_difference_empirical_minus_q_value": primary_difference,
        "size_sensitivity_difference_empirical_minus_size_matched": empirical_fraction
        - size_fraction,
        "stop_gate_p1": stop_gate,
        "stop_gate_basis": "frozen point-estimate direction only",
        "overlap_exact_reference": exact_summary,
        "continuous_secondary": continuous,
        "logistic_sensitivity": logistic_summary,
        "random_matched_secondary": random_summary,
        "interpretation_boundary": (
            "held-out temporal replication of one objective audit component; not an independent "
            "cohort, full semantic-audit validation, mechanism, causality, or clinical utility"
        ),
    }

    workflow_rows = pd.DataFrame(
        [
            {
                "panel": "A",
                "record_type": "workflow",
                "metric": "discovery_samples",
                "value": 12,
            },
            {
                "panel": "A",
                "record_type": "workflow",
                "metric": "balanced_resamples",
                "value": 81,
            },
            {
                "panel": "A",
                "record_type": "workflow",
                "metric": "primary_tau",
                "value": 0.8,
            },
            {
                "panel": "A",
                "record_type": "workflow",
                "metric": "frozen_k",
                "value": 23,
            },
            {
                "panel": "A",
                "record_type": "workflow",
                "metric": "validation_samples",
                "value": 12,
            },
        ]
    )
    panel_b = pathway.copy()
    panel_b.insert(0, "record_type", "pathway")
    panel_b.insert(0, "panel", "B")
    panel_c = methods.copy()
    panel_c.insert(0, "record_type", "method")
    panel_c.insert(0, "panel", "C")
    panel_d = tau_summary.copy()
    panel_d.insert(0, "record_type", "tau_grid")
    panel_d.insert(0, "panel", "D")
    figure4 = pd.concat([workflow_rows, panel_b, panel_c, panel_d], ignore_index=True, sort=False)

    staging_dir.mkdir()
    staging_paths = {name: staging_dir / path.name for name, path in output_paths.items()}
    try:
        write_table(staging_paths["pathway"], pathway)
        write_table(staging_paths["method"], methods)
        write_table(staging_paths["tau_grid"], tau_summary)
        write_table(staging_paths["continuous"], continuous_table)
        write_table(staging_paths["logistic"], logistic_table)
        write_table(staging_paths["exact_null"], exact_null)
        write_table(staging_paths["random_null"], random_null)
        write_table(staging_paths["figure4"], figure4)
        staging_paths["summary"].write_text(
            json.dumps(json_safe(primary_summary), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        metadata = {
            "analysis_scope": "frozen Priority 1 held-out 72 h replication evaluation",
            "benchmark_id": benchmark_id,
            "code": {
                "branch": git_value("branch", "--show-current"),
                "commit": git_value("rev-parse", "HEAD"),
                "tracked_worktree_clean": True,
            },
            "created_at_utc": datetime.now(UTC).isoformat(),
            "freeze_gate_output": freeze_gate_output,
            "inputs": {
                "config": {"path": str(config_path), "sha256": sha256_file(config_path)},
                "freeze_manifest": {
                    "path": str(freeze_manifest_path),
                    "sha256": sha256_file(freeze_manifest_path),
                },
                "freeze_sidecar": {
                    "path": str(freeze_sidecar_path),
                    "sha256": sha256_file(freeze_sidecar_path),
                },
                "frozen_membership": {
                    "path": str(membership_path),
                    "sha256": sha256_file(membership_path),
                },
                "frozen_tau_grid": {
                    "path": str(tau_grid_path),
                    "sha256": sha256_file(tau_grid_path),
                },
                "validation_72h": {
                    "path": str(validation_path),
                    "sha256": sha256_file(validation_path),
                },
                "validation_72h_meta": {
                    "path": str(validation_meta_path),
                    "sha256": sha256_file(validation_meta_path),
                },
            },
            "outputs": {
                name: {"path": str(output_paths[name]), "sha256": sha256_file(staging_paths[name])}
                for name in output_paths
                if name != "run_meta"
            },
            "protocol_status": config["status"],
            "protocol_version": config["protocol_version"],
            "python": {
                "implementation": platform.python_implementation(),
                "version": sys.version,
            },
            "script": {
                "path": str(Path(__file__).resolve()),
                "sha256": sha256_file(Path(__file__).resolve()),
            },
            "stop_gate_p1": stop_gate,
        }
        staging_paths["run_meta"].write_text(
            json.dumps(json_safe(metadata), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        for name, destination in output_paths.items():
            os.replace(staging_paths[name], destination)
    finally:
        if staging_dir.exists():
            shutil.rmtree(staging_dir)

    print("[PASS] Priority 1 frozen held-out replication evaluation complete")
    print(f"[RESULT] Empirical replication fraction: {empirical_fraction:.4f}")
    print(f"[RESULT] q-value-matched replication fraction: {q_value_fraction:.4f}")
    print(f"[RESULT] Frozen replication-fraction difference: {primary_difference:+.4f}")
    print(f"[RESULT] Continuous empirical-survival AUROC: {continuous['auroc']:.4f}")
    if stop_gate.startswith("PASS_"):
        print("[P1 PASS] Empirical selection improved the held-out replication point estimate.")
    else:
        print("[P1 STOP] No empirical point-estimate improvement; narrow the manuscript claim.")
    print(f"[INFO] Wrote: {output_paths['summary']}")
    print(
        "[INFO] Do not change tau, membership, endpoint, or matched comparators after this result."
    )


if __name__ == "__main__":
    main()
