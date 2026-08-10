#!/usr/bin/env python3
"""Evaluate utility-ranking sensitivity only after P3 and P4 are locked."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


BENCHMARK_ID = "PANCAN_TP53_v1_HNSC_R1_P5"
COMPONENTS = (
    "statistical_support",
    "stability_support",
    "independent_evidence",
    "wording_safety",
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def locked_manifest(path: Path, label: str) -> dict[str, Any]:
    require(path.is_file(), f"Missing {label} lock manifest: {path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    status = str(manifest.get("status", "")).upper()
    require(
        "LOCKED" in status or "FROZEN" in status,
        f"{label} manifest is not locked: status={status!r}",
    )
    return manifest


def priority2_claim_ids(path: Path) -> set[str]:
    manifest = locked_manifest(path, "Priority 2")
    record = manifest.get("outputs", {}).get("membership", {})
    membership_path = Path(str(record.get("path", ""))).expanduser().resolve()
    require(membership_path.is_file(), "Priority 2 membership file is missing")
    require(
        sha256_file(membership_path) == record.get("sha256"),
        "Priority 2 membership hash drift",
    )
    membership = pd.read_csv(membership_path, sep="\t")
    require("claim_uid" in membership, "Priority 2 membership lacks claim_uid")
    claim_ids = set(membership["claim_uid"].astype(str))
    require(len(claim_ids) == 50, "Priority 2 membership must contain 50 claims")
    return claim_ids


def weight_grid(step: float = 0.25) -> list[tuple[float, ...]]:
    units = round(1.0 / step)
    require(abs(units * step - 1.0) < 1e-12, "Weight step must divide one exactly")
    weights = []
    for values in itertools.product(range(units + 1), repeat=len(COMPONENTS)):
        if sum(values) == units:
            weights.append(tuple(value / units for value in values))
    return weights


def utility_scores(table: pd.DataFrame, epsilon: float = 1e-6) -> pd.DataFrame:
    values = table.loc[:, COMPONENTS].to_numpy(dtype=float)
    rows: list[pd.DataFrame] = []

    def add(method: str, score: np.ndarray, weights: tuple[float, ...]) -> None:
        frame = pd.DataFrame(
            {
                "claim_id": table["claim_id"].astype(str),
                "aggregation": method,
                "score": score,
                "w_statistical": weights[0],
                "w_stability": weights[1],
                "w_independent_evidence": weights[2],
                "w_wording_safety": weights[3],
            }
        )
        frame["rank"] = frame["score"].rank(method="average", ascending=False)
        rows.append(frame)

    equal = (0.25, 0.25, 0.25, 0.25)
    add("multiplicative", np.prod(values, axis=1), equal)
    add("equal_weight_arithmetic", np.mean(values, axis=1), equal)
    add("minimum", np.min(values, axis=1), equal)
    for weights in weight_grid(0.25):
        weight_array = np.asarray(weights)
        score = np.exp(
            np.sum(np.log(np.clip(values, epsilon, 1.0)) * weight_array, axis=1)
        )
        tag = "_".join(str(int(round(weight * 100))) for weight in weights)
        add(f"log_linear_w_{tag}", score, weights)
    return pd.concat(rows, ignore_index=True)


def summarize_sensitivity(
    ranks: pd.DataFrame, *, top_k: int, material_shift: int
) -> pd.DataFrame:
    primary = ranks.loc[ranks["aggregation"].eq("multiplicative")].set_index("claim_id")
    primary_top = set(primary.nsmallest(top_k, "rank").index)
    rows: list[dict[str, Any]] = []
    for method, group in ranks.groupby("aggregation", sort=False):
        candidate = group.set_index("claim_id")
        aligned = primary[["rank"]].join(
            candidate[["rank"]], lsuffix="_primary", rsuffix="_candidate"
        )
        shift = (aligned["rank_candidate"] - aligned["rank_primary"]).abs()
        candidate_top = set(candidate.nsmallest(top_k, "rank").index)
        rows.append(
            {
                "aggregation": method,
                "spearman_vs_multiplicative": aligned.corr(method="spearman").iloc[
                    0, 1
                ],
                "top_k": top_k,
                "top_k_overlap_n": len(primary_top & candidate_top),
                "top_k_overlap_fraction": len(primary_top & candidate_top) / top_k,
                "maximum_absolute_rank_shift": float(shift.max()),
                "n_material_rank_shifts": int(shift.ge(material_shift).sum()),
                "material_rank_shift_threshold": material_shift,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--components", type=Path, required=True)
    parser.add_argument("--priority2-freeze-manifest", type=Path, required=True)
    parser.add_argument("--priority3-lock-manifest", type=Path, required=True)
    parser.add_argument("--priority4-lock-manifest", type=Path, required=True)
    args = parser.parse_args()

    data_root = args.data_root.expanduser().resolve()
    component_path = args.components.expanduser().resolve()
    p2_path = args.priority2_freeze_manifest.expanduser().resolve()
    p3_path = args.priority3_lock_manifest.expanduser().resolve()
    p4_path = args.priority4_lock_manifest.expanduser().resolve()
    frozen_claim_ids = priority2_claim_ids(p2_path)
    locked_manifest(p3_path, "Priority 3")
    locked_manifest(p4_path, "Priority 4")

    table = pd.read_csv(component_path, sep="\t")
    require(
        {"claim_id", *COMPONENTS}.issubset(table.columns),
        "Component table schema drift",
    )
    require(
        len(table) == 50 and table["claim_id"].nunique() == 50, "Expected 50 claims"
    )
    require(
        set(table["claim_id"].astype(str)) == frozen_claim_ids,
        "Utility component claims differ from the frozen Priority 2 census",
    )
    for column in COMPONENTS:
        values = pd.to_numeric(table[column], errors="raise")
        require(values.between(0.0, 1.0).all(), f"{column} must be in [0,1]")
        table[column] = values

    output_dir = data_root / "output/priority5" / BENCHMARK_ID / "utility"
    require(not output_dir.exists(), f"Utility output already exists: {output_dir}")
    output_dir.mkdir(parents=True)
    ranks = utility_scores(table)
    summary = summarize_sensitivity(ranks, top_k=25, material_shift=10)
    ranks_path = output_dir / "claim_utility_ranks.tsv"
    summary_path = output_dir / "sensitivity_metrics.tsv"
    ranks.to_csv(ranks_path, sep="\t", index=False)
    summary.to_csv(summary_path, sep="\t", index=False)
    meta = {
        "schema_version": "CRM_R1_PRIORITY5_UTILITY_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "benchmark_id": BENCHMARK_ID,
        "components_sha256": sha256_file(component_path),
        "priority2_freeze_sha256": sha256_file(p2_path),
        "priority3_lock_sha256": sha256_file(p3_path),
        "priority4_lock_sha256": sha256_file(p4_path),
        "ranks_sha256": sha256_file(ranks_path),
        "sensitivity_sha256": sha256_file(summary_path),
        "weight_step": 0.25,
        "epsilon": 1e-6,
        "top_k": 25,
        "material_rank_shift": 10,
    }
    temporary = output_dir / "utility_sensitivity.run_meta.json.tmp"
    final = output_dir / "utility_sensitivity.run_meta.json"
    temporary.write_text(
        json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, final)
    print("[PASS] Priority 5 utility sensitivity complete after P3/P4 lock")
    print(f"[INFO] Aggregations evaluated: {summary['aggregation'].nunique()}")
    print(f"[INFO] Wrote: {summary_path}")


if __name__ == "__main__":
    main()
