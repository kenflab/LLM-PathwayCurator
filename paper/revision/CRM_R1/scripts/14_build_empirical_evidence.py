#!/usr/bin/env python3
"""Build a replicate-stacked EvidenceTable from full and 81 resampled fgsea runs."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from llm_pathway_curator import _shared
from llm_pathway_curator.adapters.fgsea import FgseaAdapterConfig, fgsea_to_evidence_table

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


def adapt_one(raw: pd.DataFrame, *, replicate_id: str, resample_index: int) -> pd.DataFrame:
    adapted = fgsea_to_evidence_table(raw, config=FgseaAdapterConfig(source_name="fgsea"))
    adapted.insert(0, "resample_index", int(resample_index))
    adapted.insert(0, "replicate_id", str(replicate_id))
    return adapted


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--baseline-fgsea", type=Path, default=None)
    parser.add_argument("--resampled-fgsea", type=Path, default=None)
    parser.add_argument("--resample-manifest", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    data_root = args.data_root.expanduser().resolve()
    config_path = args.config.expanduser().resolve()
    config = read_json(config_path)
    primary = config["primary_analysis"]
    empirical = config["empirical_stability"]
    benchmark_id = str(config["benchmark_id"])
    benchmark_dir = data_root / "output" / "priority1" / benchmark_id

    require(config["status"] == "DRAFT_NOT_FROZEN", "V4 evidence build expects draft protocol")
    require(empirical["distill_mode"] == "replicates_proxy", "Unexpected distill mode")
    expected_resamples = int(empirical["expected_resamples"])
    expected_pathways = int(primary["candidate_pool_size"])

    baseline_path = (
        args.baseline_fgsea.expanduser().resolve()
        if args.baseline_fgsea is not None
        else benchmark_dir / "derived" / "fgsea" / "discovery_48h.tsv"
    )
    resampled_path = (
        args.resampled_fgsea.expanduser().resolve()
        if args.resampled_fgsea is not None
        else benchmark_dir / "derived" / "empirical_resampling_48h" / "fgsea_resamples.tsv"
    )
    manifest_path = (
        args.resample_manifest.expanduser().resolve()
        if args.resample_manifest is not None
        else benchmark_dir / "derived" / "empirical_resampling_48h" / "resample_manifest.tsv"
    )
    output_path = (
        args.output.expanduser().resolve()
        if args.output is not None
        else benchmark_dir / "evidence_tables" / "discovery_48h_empirical_replicates.tsv"
    )
    meta_path = output_path.with_name("discovery_48h_empirical_replicates.run_meta.json")

    for path in (baseline_path, resampled_path, manifest_path):
        require(path.is_file(), f"Missing input: {path}")
    existing = [path for path in (output_path, meta_path) if path.exists()]
    require(args.force or not existing, f"Outputs already exist; use --force: {existing}")

    baseline_raw = pd.read_csv(baseline_path, sep="\t")
    resampled_raw = pd.read_csv(resampled_path, sep="\t")
    manifest = pd.read_csv(manifest_path, sep="\t", dtype={"replicate_id": "string"})

    require("pathway" in baseline_raw.columns, "Baseline fgsea is missing pathway")
    require(
        {"replicate_id", "resample_index", "pathway"}.issubset(resampled_raw.columns),
        "Resampled fgsea is missing identifiers",
    )
    require(
        {"replicate_id", "resample_index"}.issubset(manifest.columns),
        "Resample manifest is missing identifiers",
    )
    require(len(baseline_raw) == expected_pathways, "Unexpected baseline pathway count")
    require(
        baseline_raw["pathway"].astype(str).nunique() == expected_pathways,
        "Baseline pathways are not unique",
    )
    require(len(manifest) == expected_resamples, "Unexpected resample-manifest row count")
    require(manifest["replicate_id"].nunique() == expected_resamples, "Resample IDs not unique")

    baseline_pathways = set(baseline_raw["pathway"].astype(str))
    frames = [
        adapt_one(
            baseline_raw,
            replicate_id=str(empirical["baseline_replicate_id"]),
            resample_index=0,
        )
    ]

    observed_ids: list[str] = []
    for replicate_id, raw_group in resampled_raw.groupby("replicate_id", sort=True):
        replicate_id = str(replicate_id)
        observed_ids.append(replicate_id)
        indices = pd.to_numeric(raw_group["resample_index"], errors="raise").astype(int).unique()
        require(len(indices) == 1, f"Multiple resample indices for {replicate_id}")
        require(len(raw_group) == expected_pathways, f"Unexpected rows for {replicate_id}")
        require(
            set(raw_group["pathway"].astype(str)) == baseline_pathways,
            f"Pathway membership drift for {replicate_id}",
        )
        frames.append(
            adapt_one(raw_group, replicate_id=replicate_id, resample_index=int(indices[0]))
        )

    require(len(observed_ids) == expected_resamples, "Unexpected number of fgsea resamples")
    require(
        set(observed_ids) == set(manifest["replicate_id"].astype(str)),
        "fgsea and manifest resample IDs differ",
    )

    evidence = pd.concat(frames, ignore_index=True)
    require(
        len(evidence) == (expected_resamples + 1) * expected_pathways,
        "Unexpected stacked EvidenceTable row count",
    )
    require(
        not evidence.duplicated(["replicate_id", "source", "term_id"]).any(),
        "Duplicate replicate/source/term rows",
    )
    baseline_terms = set(
        evidence.loc[evidence["replicate_id"].eq(empirical["baseline_replicate_id"]), "term_id"]
    )
    require(len(baseline_terms) == expected_pathways, "Adapted baseline term count drift")
    for replicate_id, group in evidence.groupby("replicate_id", sort=False):
        require(set(group["term_id"]) == baseline_terms, f"Adapted term drift: {replicate_id}")

    evidence_out = evidence.copy()
    evidence_out["evidence_genes"] = evidence_out["evidence_genes"].map(_shared.join_genes_tsv)
    evidence_out = evidence_out.sort_values(
        ["resample_index", "qval", "term_id"], ascending=[True, True, True]
    ).reset_index(drop=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    evidence_out.to_csv(output_path, sep="\t", index=False)

    metadata = {
        "analysis_scope": "ENDO 48 h discovery-only empirical resampling",
        "benchmark_id": benchmark_id,
        "dimensions": {
            "baseline_pathways": expected_pathways,
            "nonbaseline_resamples": expected_resamples,
            "stacked_rows": len(evidence_out),
        },
        "distill_contract": {
            "baseline_replicate_id": empirical["baseline_replicate_id"],
            "direction_match": bool(empirical["require_direction_match"]),
            "mode": empirical["distill_mode"],
        },
        "held_out_expression_outcomes_calculated": False,
        "inputs": {
            "baseline_fgsea": {"path": str(baseline_path), "sha256": sha256_file(baseline_path)},
            "config": {"path": str(config_path), "sha256": sha256_file(config_path)},
            "resample_manifest": {
                "path": str(manifest_path),
                "sha256": sha256_file(manifest_path),
            },
            "resampled_fgsea": {
                "path": str(resampled_path),
                "sha256": sha256_file(resampled_path),
            },
        },
        "output": {"path": str(output_path), "sha256": sha256_file(output_path)},
        "protocol_status": config["status"],
        "protocol_version": config["protocol_version"],
        "python": {"implementation": platform.python_implementation(), "version": sys.version},
        "script": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256_file(Path(__file__).resolve()),
        },
    }
    write_json(meta_path, metadata)

    print("[PASS] Priority 1 V4 replicate-stacked EvidenceTable")
    print(f"[INFO] Baseline pathways: {expected_pathways}")
    print(f"[INFO] Nonbaseline empirical resamples: {expected_resamples}")
    print(f"[INFO] Stacked rows: {len(evidence_out)}")
    print("[INFO] Held-out expression outcomes calculated: false")
    print(f"[INFO] Wrote: {output_path}")


if __name__ == "__main__":
    main()
