#!/usr/bin/env python3
"""Export immutable Figure 3 source tables from completed P5 ontology metrics."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd

BENCHMARK_ID = "PANCAN_TP53_v1_HNSC_R1_P5"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def repo_root_from_script() -> Path:
    return Path(__file__).resolve().parents[4]


def verify_ontology_outputs(ontology_dir: Path) -> None:
    meta_path = ontology_dir / "ontology_evaluation.run_meta.json"
    require(meta_path.is_file(), f"Missing ontology metadata: {meta_path}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    for filename, expected in meta["output_sha256"].items():
        path = ontology_dir / filename
        require(path.is_file(), f"Missing ontology output: {path}")
        require(sha256_file(path) == expected, f"Ontology output hash drift: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    args = parser.parse_args()
    data_root = args.data_root.expanduser().resolve()
    repo_root = repo_root_from_script()
    checker = repo_root / "paper/revision/CRM_R1/scripts/51_check_priority5_freeze.py"
    subprocess.run(
        [
            sys.executable,
            str(checker),
            "--data-root",
            str(data_root),
            "--allow-outputs",
        ],
        cwd=repo_root,
        check=True,
    )

    p5_root = data_root / "output/priority5" / BENCHMARK_ID
    ontology_dir = p5_root / "ontology"
    final_dir = p5_root / "final"
    require(not final_dir.exists(), f"Figure 3 source output already exists: {final_dir}")
    verify_ontology_outputs(ontology_dir)

    metrics = pd.read_csv(ontology_dir / "hierarchy_metrics.tsv", sep="\t")
    pairs = pd.read_csv(ontology_dir / "hierarchy_pairs.tsv", sep="\t")
    terms = pd.read_csv(ontology_dir / "ontology_term_metrics.tsv", sep="\t")
    mapping = pd.read_csv(ontology_dir / "mapping_qc.tsv", sep="\t")
    temporary = Path(tempfile.mkdtemp(prefix="priority5-final-", dir=p5_root))
    try:
        design = pd.DataFrame(
            [
                ("collections", "GO Biological Process; Reactome"),
                ("claims", "500 per collection"),
                ("audit", "deterministic proposals; LLM context review; tau=0.90"),
                ("hierarchy_role", "external evaluation only"),
                ("go_relations", "is_a; part_of"),
                ("primary_scope", "direct parent-child"),
                ("sensitivity_scope", "safe ancestor-descendant"),
                ("matched_reference", "depth- and evidence-size-matched nonedges"),
                ("priority3_priority4_outcomes", "not read"),
            ],
            columns=["item", "value"],
        )
        panel_b_columns = [
            "collection",
            "relation_scope",
            "n_pairs",
            "estimate_status",
            "directional_contradiction_fraction",
            "directional_contradiction_ci_low",
            "directional_contradiction_ci_high",
            "contradiction_null_mean",
            "contradiction_p_one_sided_lower",
        ]
        panel_c_columns = [
            "collection",
            "relation_scope",
            "parent_id",
            "child_id",
            "parent_status",
            "child_status",
            "leading_edge_jaccard",
            "child_covered_by_parent",
            "parent_covered_by_child",
        ]
        panel_d_columns = [
            "collection",
            "claim_id",
            "ontology_id",
            "ontology_name",
            "depth",
            "status",
            "direction",
            "gene_n",
        ]
        tables = {
            "figure3_panel_a_design.tsv": design,
            "figure3_panel_b_contradiction.tsv": metrics.loc[:, panel_b_columns],
            "figure3_panel_c_gene_support.tsv": pairs.loc[:, panel_c_columns],
            "figure3_panel_d_depth_status.tsv": terms.loc[:, panel_d_columns],
            "figure3_mapping_qc.tsv": mapping,
        }
        for filename, table in tables.items():
            table.to_csv(temporary / filename, sep="\t", index=False)

        manifest_rows = []
        for filename in tables:
            panel = filename.split("_panel_")[1][0].upper() if "_panel_" in filename else "QC"
            manifest_rows.append(
                {
                    "figure": "Figure 3" if panel != "QC" else "Supplement",
                    "panel": panel,
                    "source_table": filename,
                    "sha256": sha256_file(temporary / filename),
                    "analytical_endpoint_recomputed_by_plot": False,
                    "status": "READY",
                }
            )
        manifest_rows.append(
            {
                "figure": "Supplement",
                "panel": "utility",
                "source_table": "output/priority5/utility/sensitivity_metrics.tsv",
                "sha256": "PENDING_P3_P4_LOCK",
                "analytical_endpoint_recomputed_by_plot": False,
                "status": "PENDING_P3_P4_LOCK",
            }
        )
        manifest = pd.DataFrame(manifest_rows)
        manifest.to_csv(temporary / "figure_manifest.tsv", sep="\t", index=False)
        os.replace(temporary, final_dir)
    except Exception:
        for path in temporary.glob("*"):
            path.unlink()
        temporary.rmdir()
        raise

    print("[PASS] Priority 5 Figure 3 source tables frozen")
    print("[INFO] Ontology panels A-D: READY")
    print("[INFO] Utility supplement: PENDING_P3_P4_LOCK")
    print(f"[INFO] Wrote: {final_dir / 'figure_manifest.tsv'}")


if __name__ == "__main__":
    main()
