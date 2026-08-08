#!/usr/bin/env python3
"""Create the V4 empirical-resampling Priority 1 discovery Sample Card."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from llm_pathway_curator.sample_card import SampleCard

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


def read_metadata(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        rows = list(reader)
    required = {"geo_accession", "genotype", "cell_state", "treatment", "time_h"}
    require(reader.fieldnames is not None, f"Missing metadata header: {path}")
    require(not (required - set(reader.fieldnames)), f"Invalid normalized metadata: {path}")
    return rows


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    data_root = args.data_root.expanduser().resolve()
    config_path = args.config.expanduser().resolve()
    config = read_json(config_path)
    primary = config["primary_analysis"]
    empirical = config["empirical_stability"]
    benchmark_id = str(config["benchmark_id"])
    discovery_time_h = int(primary["discovery_time_h"])
    primary_state = str(primary["cell_state"])

    require(discovery_time_h == 48, "Priority 1 discovery time must remain 48 h")
    metadata_path = (
        data_root
        / "output"
        / "priority1"
        / benchmark_id
        / "preflight"
        / "sample_metadata.normalized.tsv"
    )
    require(metadata_path.is_file(), f"Run 00_preflight.py first: {metadata_path}")

    rows = read_metadata(metadata_path)
    discovery = [
        row
        for row in rows
        if row["cell_state"] == primary_state and int(row["time_h"]) == discovery_time_h
    ]
    cells = Counter((row["genotype"], row["treatment"]) for row in discovery)
    expected_cells = {
        ("WT", "UT"): 3,
        ("WT", "MMS"): 3,
        ("TP53_KO", "UT"): 3,
        ("TP53_KO", "MMS"): 3,
    }
    require(len(discovery) == 12, "Expected exactly 12 ENDO discovery samples at 48 h")
    require(cells == expected_cells, f"Unexpected 48 h factorial design: {dict(cells)}")

    output_path = (
        args.output.expanduser().resolve()
        if args.output is not None
        else data_root
        / "output"
        / "priority1"
        / benchmark_id
        / "sample_cards"
        / "discovery_48h_empirical.sample_card.json"
    )
    meta_path = output_path.with_name("discovery_48h_empirical.sample_card.run_meta.json")
    existing = [path for path in (output_path, meta_path) if path.exists()]
    require(args.force or not existing, f"Outputs already exist; use --force: {existing}")

    contrast = str(primary["contrast"])
    positive_direction = str(primary["positive_direction"])
    card: dict[str, Any] = {
        "condition": "DNA-damage response in human iPSC-derived definitive endoderm",
        "tissue": "in-vitro definitive endoderm",
        "perturbation": "MMS exposure with TP53 knockout",
        "comparison": contrast,
        "k_claims": int(primary["candidate_pool_size"]),
        "notes": (
            "GSE146225 discovery analysis restricted to ENDO at 48 h. Empirical stability is "
            "calculated from 81 balanced delete-one-per-factorial-cell reanalyses. Positive "
            f"enrichment means {positive_direction}. The 72 h endpoint remains held out."
        ),
        "context_tokens_text": (
            "human induced pluripotent stem cell definitive endoderm DNA damage MMS TP53 knockout"
        ),
        "extra": {
            "accession": str(config["dataset"]["accession"]),
            "audit_tau": float(primary["audit_tau"]),
            "audit_tau_status": str(primary["audit_tau_status"]),
            "benchmark_id": benchmark_id,
            "cell_state": primary_state,
            "context_gate_mode": str(primary["context_gate_mode"]),
            "context_review_mode": str(primary["context_review_mode"]),
            "contrast": contrast,
            "discovery_only": True,
            "discovery_time_h": discovery_time_h,
            "distill_evidence_jaccard_min": float(empirical["evidence_jaccard_min"]),
            "distill_evidence_precision_min": float(empirical["evidence_precision_min"]),
            "distill_evidence_recall_min": float(empirical["evidence_recall_min"]),
            "distill_loo_baseline_id": str(empirical["baseline_replicate_id"]),
            "distill_loo_direction_match": bool(empirical["require_direction_match"]),
            "distill_mode": str(empirical["distill_mode"]),
            "empirical_resampling_expected": int(empirical["expected_resamples"]),
            "empirical_resampling_method": str(empirical["method_id"]),
            "gene_id_map_tsv": "resources/gene_id_maps/id_map.tsv.gz",
            "gene_id_type": str(primary["gene_id_type"]),
            "gene_set_collection": str(primary["gene_set_collection"]),
            "goal": "auditable reporting of context-specific pathway enrichment claims",
            "held_out_outcomes_calculated": False,
            "positive_direction": positive_direction,
            "proposal_mode": str(primary["primary_proposal_mode"]),
            "protocol_status": str(config["status"]),
            "protocol_version": str(config["protocol_version"]),
            "stability_gate_mode": str(primary["stability_gate_mode"]),
            "validation_time_h": int(primary["validation_time_h"]),
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(output_path, card)
    parsed = SampleCard.from_json(output_path)
    require(parsed.k_claims() == int(primary["candidate_pool_size"]), "Sample Card k mismatch")
    require(parsed.audit_tau() == float(primary["audit_tau"]), "Sample Card tau mismatch")
    require(
        parsed.extra.get("distill_mode") == "replicates_proxy",
        "Sample Card must use replicates_proxy",
    )
    require(
        parsed.extra.get("context_review_mode") == "off",
        "Priority 1 context review must be off",
    )

    metadata = {
        "benchmark_id": benchmark_id,
        "held_out_expression_outcomes_calculated": False,
        "inputs": {
            "config": {"path": str(config_path), "sha256": sha256_file(config_path)},
            "normalized_metadata": {
                "path": str(metadata_path),
                "sha256": sha256_file(metadata_path),
            },
        },
        "output": {"path": str(output_path), "sha256": sha256_file(output_path)},
        "protocol_status": str(config["status"]),
        "python": {"implementation": platform.python_implementation(), "version": sys.version},
        "script": {"path": str(Path(__file__).resolve()), "sha256": sha256_file(Path(__file__))},
    }
    write_json(meta_path, metadata)

    print("[PASS] Priority 1 V4 empirical-resampling Sample Card")
    print(f"[INFO] Discovery samples encoded: {len(discovery)} ({primary_state}, 48 h)")
    print(f"[INFO] Expected balanced resamples: {int(empirical['expected_resamples'])}")
    print("[INFO] Held-out expression outcomes calculated: false")
    print(f"[INFO] Wrote: {output_path}")


if __name__ == "__main__":
    main()
