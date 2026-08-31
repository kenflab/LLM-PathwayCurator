#!/usr/bin/env python3
"""Validate Priority 1 inputs without calculating held-out biological outcomes."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

CRM_DIR = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = CRM_DIR / "config" / "priority1_protocol.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    require(isinstance(value, dict), f"JSON root must be an object: {path}")
    return value


def parse_metadata(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        rows = list(reader)

    required = {
        "Sample_title",
        "geo_accession",
        "genotype",
        "BioSample",
        "SRA",
        "Condition",
        "Group",
        "Sample_ID",
    }
    require(reader.fieldnames is not None, f"Missing metadata header: {path}")
    missing = required - set(reader.fieldnames)
    require(not missing, f"Metadata columns missing: {sorted(missing)}")

    normalized: list[dict[str, Any]] = []
    for row in rows:
        group_tokens = row["Group"].split("_")
        require(len(group_tokens) == 5, f"Unexpected Group: {row['Group']}")
        genotype_token, study_token, cell_state, treatment, time_token = group_tokens
        require(study_token == "BOB", f"Unexpected study token: {row['Group']}")
        require(time_token.endswith("h"), f"Unexpected time token: {time_token}")

        derived_genotype = "TP53_KO" if genotype_token == "TP53" else genotype_token
        require(
            derived_genotype == row["genotype"],
            f"Genotype mismatch for {row['geo_accession']}",
        )
        replicate_token = row["Sample_ID"].rsplit("_", maxsplit=1)[-1]
        require(replicate_token.isdigit(), f"Invalid replicate: {row['Sample_ID']}")

        normalized.append(
            {
                "geo_accession": row["geo_accession"],
                "sample_id": row["Sample_ID"],
                "sample_title": row["Sample_title"],
                "genotype": row["genotype"],
                "cell_state": cell_state,
                "treatment": treatment,
                "time_h": int(time_token.removesuffix("h")),
                "replicate": int(replicate_token),
                "biosample": row["BioSample"],
                "sra": row["SRA"],
            }
        )
    return normalized


def inspect_expression(path: Path) -> dict[str, Any]:
    gene_ids: set[str] = set()
    n_rows = 0
    n_zero = 0

    if path.suffix == ".gz":
        handle_context = gzip.open(path, mode="rt", encoding="utf-8", newline="")
    else:
        handle_context = path.open(encoding="utf-8", newline="")

    with handle_context as handle:
        reader = csv.reader(handle, delimiter="\t")
        header = next(reader)
        require(header and header[0] == "GeneID", "Count matrix first column must be GeneID")
        sample_ids = header[1:]
        require(len(sample_ids) == len(set(sample_ids)), "Count-matrix sample IDs are duplicated")

        for line_number, row in enumerate(reader, start=2):
            require(
                len(row) == len(header),
                f"Count-matrix row width mismatch at line {line_number}",
            )
            gene_id = row[0].strip()
            require(gene_id, f"Missing GeneID at line {line_number}")
            require(gene_id not in gene_ids, f"Duplicated GeneID: {gene_id}")
            gene_ids.add(gene_id)

            for token in row[1:]:
                try:
                    value = int(token)
                except ValueError as error:
                    raise ValueError(
                        f"Non-integer count at line {line_number}: {token!r}"
                    ) from error
                require(value >= 0, f"Negative count at line {line_number}")
                n_zero += int(value == 0)
            n_rows += 1

    return {
        "sample_ids": sample_ids,
        "n_samples": len(sample_ids),
        "n_gene_rows": n_rows,
        "n_zero_values": n_zero,
        "value_contract": "non-negative integers",
    }


def write_tsv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    data_root = args.data_root.expanduser().resolve()
    config_path = args.config.expanduser().resolve()
    config = load_json(config_path)
    dataset = config["dataset"]
    primary = config["primary_analysis"]
    benchmark_id = str(config["benchmark_id"])

    input_dir = data_root / "input"
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else data_root / "output" / "priority1" / benchmark_id / "preflight"
    )
    expression_path = input_dir / dataset["expression_file"]
    metadata_path = input_dir / dataset["metadata_file"]
    require(expression_path.is_file(), f"Missing expression input: {expression_path}")
    require(metadata_path.is_file(), f"Missing metadata input: {metadata_path}")

    expression_sha256 = sha256_file(expression_path)
    metadata_sha256 = sha256_file(metadata_path)
    expected_expression_sha256 = dataset.get("expression_sha256")
    if expected_expression_sha256 is None:
        require(
            config["status"] != "FROZEN",
            "Frozen protocol must specify the count-matrix SHA-256",
        )
    else:
        require(
            expression_sha256 == expected_expression_sha256,
            "Count-matrix SHA-256 does not match the protocol",
        )
    require(
        metadata_sha256 == dataset["metadata_sha256"],
        "Metadata SHA-256 does not match the frozen config",
    )

    metadata = parse_metadata(metadata_path)
    expression = inspect_expression(expression_path)
    metadata_samples = [str(row["geo_accession"]) for row in metadata]

    require(len(metadata) == dataset["expected_samples"], "Unexpected metadata sample count")
    require(
        expression["n_samples"] == dataset["expected_samples"],
        "Unexpected expression sample count",
    )
    expected_gene_rows = dataset.get("expected_gene_rows")
    if expected_gene_rows is None:
        require(
            config["status"] != "FROZEN",
            "Frozen protocol must specify the expected count-matrix gene rows",
        )
    else:
        require(
            expression["n_gene_rows"] == expected_gene_rows,
            "Unexpected count-matrix gene count",
        )
    require(len(metadata_samples) == len(set(metadata_samples)), "Metadata sample IDs duplicated")
    require(
        expression["sample_ids"] == metadata_samples,
        "Expression and metadata sample IDs or order differ",
    )

    counts = Counter(
        (
            int(row["time_h"]),
            str(row["cell_state"]),
            str(row["genotype"]),
            str(row["treatment"]),
        )
        for row in metadata
    )
    design_rows = [
        {
            "time_h": key[0],
            "cell_state": key[1],
            "genotype": key[2],
            "treatment": key[3],
            "n": value,
        }
        for key, value in sorted(counts.items())
    ]

    primary_state = str(primary["cell_state"])
    primary_times = [int(primary["discovery_time_h"]), int(primary["validation_time_h"])]
    for time_h in primary_times:
        for genotype in ["WT", "TP53_KO"]:
            for treatment in ["UT", "MMS"]:
                key = (time_h, primary_state, genotype, treatment)
                require(counts[key] == 3, f"Expected three replicates for design cell {key}")

    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_fields = [
        "geo_accession",
        "sample_id",
        "sample_title",
        "genotype",
        "cell_state",
        "treatment",
        "time_h",
        "replicate",
        "biosample",
        "sra",
    ]
    write_tsv(output_dir / "sample_metadata.normalized.tsv", metadata, metadata_fields)
    write_tsv(
        output_dir / "design_counts.tsv",
        design_rows,
        ["time_h", "cell_state", "genotype", "treatment", "n"],
    )

    manifest = {
        "protocol_version": config["protocol_version"],
        "protocol_status": config["status"],
        "benchmark_id": benchmark_id,
        "config_path": str(config_path),
        "inputs": {
            "expression": {
                "path": str(expression_path),
                "sha256": expression_sha256,
                "bytes": expression_path.stat().st_size,
            },
            "metadata": {
                "path": str(metadata_path),
                "sha256": metadata_sha256,
                "bytes": metadata_path.stat().st_size,
            },
        },
    }
    summary = {
        "status": "PASS",
        "protocol_status": config["status"],
        "benchmark_id": benchmark_id,
        "n_gene_rows": expression["n_gene_rows"],
        "n_samples": expression["n_samples"],
        "expression_value_contract": expression["value_contract"],
        "n_zero_counts": expression["n_zero_values"],
        "sample_order_identical": True,
        "primary_cell_state": primary_state,
        "discovery_time_h": primary_times[0],
        "validation_time_h": primary_times[1],
        "primary_factorial_cells": 8,
        "replicates_per_primary_factorial_cell": 3,
        "held_out_expression_outcomes_calculated": False,
    }
    (output_dir / "input_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "preflight_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )

    print(
        f"[PASS] Priority 1 preflight: {expression['n_gene_rows']} genes x "
        f"{expression['n_samples']} samples; non-negative integer counts"
    )
    print(f"[PASS] Exact metadata match and balanced {primary_state} design at 48 h and 72 h")
    print(f"[INFO] Protocol status: {config['status']}")
    print(f"[INFO] Wrote: {output_dir}")


if __name__ == "__main__":
    main()
