"""R01: saved-result inventory and source-checked deterministic contract export.

This is a development preflight, not a new biological or semantic benchmark.
All generated files stay under the external CRM_R1 data root.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import re
import subprocess
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

VERSION = "CRM_R1_REVISION_v17_R01"
REPO = Path(__file__).resolve().parents[4]
P2B = Path("output/priority2b")
STATS = P2B / "discovery_stats_lock_v14_1_3/discovery_statistics.tsv"
STATS_MANIFEST = STATS.parent / "discovery_statistics_manifest.json"
HALLMARK = P2B / "hallmark_lock_v14_1_2/hallmark_gene_sets.tsv"
HALLMARK_MANIFEST = HALLMARK.parent / "hallmark_manifest.json"
FINAL_MANIFEST = P2B / "final_v14_1_3/figure2_source_manifest.json"
JOB_ROOT = P2B / "discovery_work_v14_1_3/audit_final_v14_1_3_1/jobs"
GRADING = (
    Path("output/priority3/PANCAN_TP53_v1_HNSC_R1")
    / "grading_working/record_screening_P3C1.private.tsv"
)
P4 = Path("output/priority4/PANCAN_TP53_v1_HNSC_R1")
GRADING_FIELDS = (
    "eligible",
    "exclusion_reason",
    "evidence_grade",
    "direction_match",
    "context_match",
    "study_design",
    "data_overlap",
    "contradiction",
    "supporting_note",
    "curator_id",
)
REQUIRED_STATS = (
    "cohort_id",
    "split_id",
    "term_id",
    "term_name",
    "stat",
    "qval",
    "direction",
    "evidence_genes",
    "source",
    "n_discovery_mutant",
    "n_discovery_wild_type",
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def read_json(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, f"Duplicate JSON key: {key}")
            result[key] = value
        return result

    return json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=unique)


def write_json(path, obj):
    with Path(path).open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def read_table(path):
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)


def write_table(path, rows):
    pd.DataFrame(rows).to_csv(path, sep="\t", index=False, lineterminator="\n")


class Inputs:
    """Hash every accessed input and verify it again before completion."""

    def __init__(self):
        self.files = {}

    def add(self, path):
        path = Path(path).resolve(strict=True)
        current = sha(path)
        if str(path) in self.files:
            require(current == self.files[str(path)]["sha256"], f"Input changed: {path}")
        else:
            self.files[str(path)] = {
                "path": str(path),
                "sha256": current,
                "size_bytes": path.stat().st_size,
            }
        return current

    def verify(self):
        for item in self.files.values():
            require(sha(item["path"]) == item["sha256"], f"Input changed: {item['path']}")


def data_path(root, relative):
    path = (root / relative).resolve()
    require(path.is_relative_to(root), f"Path escapes CRM_R1 data root: {relative}")
    return path


def declared_path(root, value):
    """Rebase historical absolute CRM_R1 paths, never match only a basename."""
    path = Path(value)
    if not path.is_absolute():
        return data_path(root, path)
    if path.is_relative_to(root):
        return data_path(root, path.relative_to(root))
    indexes = [i for i, part in enumerate(path.parts) if part == "CRM_R1"]
    require(len(indexes) == 1, f"Cannot rebase historical path: {value}")
    return data_path(root, Path(*path.parts[indexes[0] + 1 :]))


def verify_manifest(root, relative, inputs, *, sections=("inputs", "outputs")):
    path = data_path(root, relative)
    digest = inputs.add(path)
    manifest = read_json(path)
    checks = []
    companion = path.with_suffix(".sha256")
    if companion.exists():
        inputs.add(companion)
        lines = companion.read_text(encoding="utf-8").strip().splitlines()
        require(len(lines) == 1, f"Expected one manifest digest: {companion}")
        fields = lines[0].split()
        require(fields and fields[0] == digest, f"Manifest digest mismatch: {companion}")
    for section in sections:
        require(isinstance(manifest.get(section), dict), f"Missing {section}: {path}")
        for key, item in manifest[section].items():
            require(isinstance(item, dict), f"Invalid manifest record: {section}/{key}")
            expected = item.get("sha256", "")
            require(re.fullmatch(r"[0-9a-f]{64}", expected) is not None, "Invalid SHA256")
            actual_path = declared_path(root, item["path"])
            actual = inputs.add(actual_path)
            require(actual == expected, f"Frozen input digest mismatch: {actual_path}")
            checks.append(
                {
                    "manifest": str(relative),
                    "section": section,
                    "key": key,
                    "relative_path": str(actual_path.relative_to(root)),
                    "sha256": actual,
                    "status": "MATCH",
                }
            )
    return manifest, checks


def git(args, path=REPO, *, check=True):
    result = subprocess.run(
        ["git", "-C", str(path), *args],
        text=True,
        capture_output=True,
        check=False,
    )
    if check:
        require(result.returncode == 0, f"git {' '.join(args)} failed: {result.stderr.strip()}")
    return result.stdout.strip()


def git_snapshot(path=REPO):
    top = git(["rev-parse", "--show-toplevel"], path, check=False)
    if not top or Path(top).resolve() != Path(path).resolve():
        return {"path": str(path), "is_repository_root": False}
    changes = git(["status", "--porcelain=v1", "--untracked-files=all"], path)
    return {
        "path": str(Path(path).resolve()),
        "is_repository_root": True,
        "head": git(["rev-parse", "HEAD"], path),
        "branch": git(["branch", "--show-current"], path),
        "dirty": bool(changes),
        "changed_paths": changes.splitlines(),
        "common_git_dir": git(["rev-parse", "--git-common-dir"], path),
    }


def project_inventory(repo):
    rows = []
    for path in sorted(repo.parent.glob("LLM-PathwayCurator*")):
        if path.is_dir():
            item = git_snapshot(path)
            item["working_repository"] = path.resolve() == repo.resolve()
            rows.append(item)
    return rows


def numeric(value, name):
    require(isinstance(value, str) and value.strip() == value and value, f"Blank {name}")
    number = float(value)
    require(math.isfinite(number), f"Nonfinite {name}")
    return number


def genes(value):
    require(isinstance(value, str) and value, "Missing leading-edge genes")
    result = value.split(",")
    require(all(g and g == g.strip() for g in result), "Blank/padded leading-edge gene")
    require(len(result) == len(set(result)), "Duplicate leading-edge genes")
    return result


def saved_number_alignment(frozen, job, name):
    """Permit at most one binary64 step from TSV reserialization, record it.

    The frozen value is always exported. Even one step cannot cross the FDR
    decision boundary or change the NES sign. This is not statistical tolerance.
    """
    a, b = numeric(frozen, name), numeric(job, name)
    if name == "qval":
        require(0 <= a <= 1 and 0 <= b <= 1, "q-value outside [0,1]")
        require((a <= 0.05) == (b <= 0.05), "Job/frozen FDR decision mismatch")
    else:
        require((a > 0) == (b > 0) and (a < 0) == (b < 0), "Job/frozen NES sign mismatch")
    if a == b:
        return "EXACT_FLOAT"
    require(abs(a - b) <= max(math.ulp(a), math.ulp(b)), f"Job/frozen {name} mismatch")
    return "ONE_ULP_RESERIALIZATION"


def validate_stats(table, gene_sets, cohort_names, *, expected_terms=50):
    """Check all saved discovery partitions without reading validation statistics."""
    require(set(REQUIRED_STATS).issubset(table.columns), "Missing discovery columns")
    require(not table.empty, "Empty discovery census")
    require(not table.duplicated(["cohort_id", "split_id", "term_id"]).any(), "Duplicate term")
    require(len(gene_sets) == expected_terms, "Wrong Hallmark census")
    groups = []
    for (cohort, split), group in table.groupby(["cohort_id", "split_id"], sort=True):
        require(cohort in cohort_names, f"Unknown TCGA code: {cohort}")
        require(re.fullmatch(r"S\d{3}", split) is not None, f"Invalid split ID: {split}")
        require(set(group.term_id) == set(gene_sets), f"Incomplete terms: {cohort}/{split}")
        for name in ("n_discovery_mutant", "n_discovery_wild_type"):
            require(group[name].nunique() == 1, f"Inconsistent group size: {cohort}/{split}")
            value = group.iloc[0][name]
            require(value.isdigit() and int(value) > 0, f"Invalid {name}")
        significant = 0
        directions = Counter()
        for row in group.to_dict("records"):
            nes = numeric(row["stat"], "NES")
            q = numeric(row["qval"], "q-value")
            require(0 <= q <= 1, "q-value outside [0,1]")
            direction = "up" if nes > 0 else "down" if nes < 0 else "zero"
            require(row["direction"] == direction, "Saved direction disagrees with NES")
            require(
                row["term_name"].strip() == row["term_name"] and row["term_name"],
                "Missing/padded term name",
            )
            support = genes(row["evidence_genes"])
            require(
                set(support).issubset(gene_sets[row["term_id"]]),
                f"Leading edge outside frozen gene set: {cohort}/{split}/{row['term_id']}",
            )
            significant += q <= 0.05
            directions[direction] += 1
        groups.append(
            {
                "cohort_id": cohort,
                "split_id": split,
                "rows": len(group),
                "fdr_le_0_05": significant,
                "up": directions["up"],
                "down": directions["down"],
                "zero": directions["zero"],
                "schema_status": "VALID",
                "n_discovery_mutant": int(group.iloc[0].n_discovery_mutant),
                "n_discovery_wild_type": int(group.iloc[0].n_discovery_wild_type),
            }
        )
    return groups


def export_adapter(root, out, inputs, *, cohort="ACC", split="S001"):
    from llm_pathway_curator.contract_pipeline import ContractRunConfig, run_contract_pipeline
    from llm_pathway_curator.contract_v161.models import Claim, Evidence
    from llm_pathway_curator.contract_v161.registry import TCGA_NAMES
    from llm_pathway_curator.contract_v161.text_checks import factual_statement

    stats_manifest, checks = verify_manifest(root, STATS_MANIFEST, inputs)
    hallmark_manifest, extra = verify_manifest(root, HALLMARK_MANIFEST, inputs)
    checks += extra
    stats_path, hallmark_path = data_path(root, STATS), data_path(root, HALLMARK)
    for path, manifest, key in (
        (stats_path, stats_manifest, "statistics"),
        (hallmark_path, hallmark_manifest, "gene_sets"),
    ):
        require(
            declared_path(root, manifest["outputs"][key]["path"]) == path,
            f"Frozen manifest points to a different source: {path}",
        )
    stats_digest, hallmark_digest = inputs.add(stats_path), inputs.add(hallmark_path)
    table = read_table(stats_path)
    gene_table = read_table(hallmark_path)
    require(set(gene_table.columns) == {"term_id", "gene_id"}, "Unexpected gene-set columns")
    require(not gene_table.duplicated(["term_id", "gene_id"]).any(), "Duplicate gene-set pair")
    gene_sets = {term: set(group.gene_id) for term, group in gene_table.groupby("term_id")}
    group_qc = validate_stats(table, gene_sets, TCGA_NAMES)
    require(len(table) == stats_manifest["rows"], "Discovery row count differs from freeze")
    require(len(group_qc) == stats_manifest["cohort_split_pairs"], "Partition count drift")
    require(len(gene_table) == hallmark_manifest["term_gene_pairs"], "Gene-set count drift")
    selected = table.loc[table.cohort_id.eq(cohort) & table.split_id.eq(split)].copy()
    require(len(selected) == 50, f"Expected all 50 terms for {cohort}/{split}")
    job = data_path(root, JOB_ROOT / cohort / split)
    card_path, job_path = job / "sample_card.json", job / "discovery_evidence.tsv"
    card_digest = inputs.add(card_path)
    inputs.add(job_path)
    card, job_table = read_json(card_path), read_table(job_path)
    require(card.get("condition") == cohort, "Sample Card cohort mismatch")
    require(card.get("comparison") == "TP53_mut_vs_TP53_wt", "Unsupported contrast")
    require(card.get("perturbation") == "genotype", "Unexpected perturbation")
    require(not job_table.term_id.duplicated().any(), "Duplicate job term")
    require(set(job_table.term_id) == set(selected.term_id), "Job term census differs from freeze")
    indexed = job_table.set_index("term_id")
    alignment = []
    for row in selected.to_dict("records"):
        other = indexed.loc[row["term_id"]]
        for name in ("stat", "qval"):
            status = saved_number_alignment(row[name], other[name], name)
            alignment.append(
                {
                    "term_id": row["term_id"],
                    "field": name,
                    "frozen_raw": row[name],
                    "job_raw": other[name],
                    "status": status,
                    "export_source": "frozen_statistics",
                }
            )
        for name in ("term_name", "direction", "evidence_genes", "source"):
            require(row[name] == other[name], f"Job/frozen {name} mismatch")
    metadata_path = data_path(root, HALLMARK.parent / "hallmark_export_metadata.tsv")
    inputs.add(metadata_path)
    meta = read_table(metadata_path).set_index("field").value.to_dict()
    require(
        meta.get("gene_identifier") == "HGNC gene symbol (msigdbr gene_symbol)",
        "Wrong gene identifier",
    )
    gene_version = f"Hallmark gene-symbol snapshot; msigdbr {meta['msigdbr_version']}"
    registry_path = REPO / "src/llm_pathway_curator/contract_v161/registry.py"
    inputs.add(registry_path)
    evidence, claims, mapping = [], [], []
    for index, row in selected.sort_values("term_id").iterrows():
        evidence_id = f"TCGA_{cohort}_{split}_{row.term_id}"
        record = {
            "schema_version": "CRM_R1_EVIDENCE_v16",
            "evidence_id": evidence_id,
            "cohort_id": cohort,
            "cohort_name": TCGA_NAMES[cohort],
            "split_id": split,
            "contrast": {
                "comparison_id": "TP53_MUTANT_vs_TP53_WILD_TYPE",
                "positive_group": "TP53_MUTANT",
                "reference_group": "TP53_WILD_TYPE",
                "study_design": "observational",
            },
            "term_id": row.term_id,
            "term_name": row.term_name,
            "gene_set_version": gene_version,
            "gene_set_sha256": hallmark_digest,
            "nes": float(row.stat),
            "q_value": float(row.qval),
            "direction": "UP" if float(row.stat) > 0 else "DOWN" if float(row.stat) < 0 else "ZERO",
            "leading_edge_genes": genes(row.evidence_genes),
            "metadata": {},
            "source": {
                "artifact": str(stats_path),
                "sha256": stats_digest,
                "locator": (
                    f"cohort_id={cohort};split_id={split};term_id={row.term_id};line={index + 2}"
                ),
            },
            "stability": None,
        }
        if card.get("tissue"):
            record["metadata"]["tissue"] = {
                "value": card["tissue"],
                "source": {
                    "artifact": str(card_path),
                    "sha256": card_digest,
                    "locator": "/tissue",
                },
            }
        canonical = Evidence.model_validate(record)
        text = factual_statement(canonical)
        claim = {
            "schema_version": "CRM_R1_CLAIM_v16",
            "claim_id": f"STANDARD_{evidence_id}",
            "evidence_id": evidence_id,
            "text": text,
            "cohort_id": cohort,
            "cohort_name": TCGA_NAMES[cohort],
            "split_id": split,
            "comparison_id": canonical.contrast.comparison_id,
            "term_id": row.term_id,
            "gene_set_version": gene_version,
            "gene_set_sha256": record["gene_set_sha256"],
            "reported_nes": canonical.nes,
            "reported_q_value": canonical.q_value,
            "direction": canonical.direction,
            "supporting_genes": canonical.leading_edge_genes,
            "metadata_assertions": {},
            "significance_assertion": "SIGNIFICANT"
            if canonical.q_value <= 0.05
            else "NOT_SIGNIFICANT",
            "causal_assertion": False,
            "disease_specificity_assertion": False,
            "literal_pathway_event_assertion": False,
        }
        Claim.model_validate(claim)
        evidence.append(record)
        claims.append(claim)
        mapping.append(
            {
                "evidence_id": evidence_id,
                "term_id": row.term_id,
                "source_line": index + 2,
                "raw_nes": row.stat,
                "raw_q_value": row.qval,
                "direction": canonical.direction,
                "leading_edge_count": len(canonical.leading_edge_genes),
                "standard_statement": text,
            }
        )
    out.mkdir()
    write_table(out / "all_discovery_schema_qc.tsv", group_qc)
    write_table(out / "frozen_hash_checks.tsv", checks)
    write_table(out / "job_numeric_alignment.private.tsv", alignment)
    write_table(out / "source_mapping.private.tsv", mapping)
    evidence_file, claims_file = out / "evidence.private.json", out / "claims.private.json"
    write_json(evidence_file, evidence)
    write_json(claims_file, claims)
    result = run_contract_pipeline(
        ContractRunConfig(
            str(evidence_file),
            str(claims_file),
            str(out / "contract_run"),
            mode="deterministic",
        )
    )
    require(result.summary["technical_or_input_errors"] == 0, "Contract input/export errors")
    require(result.summary["retained_statistical_candidates"] == 50, "Candidate retention drift")
    summary = {
        "scope": "source_checked_standard_statement_adapter_not_semantic_performance",
        "cohort": cohort,
        "cohort_name": TCGA_NAMES[cohort],
        "split": split,
        "all_discovery_rows_checked": len(table),
        "all_partitions_checked": len(group_qc),
        "exported_records": len(evidence),
        "source_hash_checks": len(checks),
        "job_numeric_alignment": dict(Counter(item["status"] for item in alignment)),
        "fdr_le_0_05": sum(item["q_value"] <= 0.05 for item in evidence),
        "group_counts": {
            "mutant": int(selected.iloc[0].n_discovery_mutant),
            "wild_type": int(selected.iloc[0].n_discovery_wild_type),
        },
        "canonical_names_source": str(registry_path),
        "stability_exported": False,
        "context_score_exported": False,
        "legacy_claim_wording_changed": False,
        "model_calls": 0,
        "biological_validation_performed": False,
        "contract_summary": result.summary,
    }
    write_json(out / "summary.json", summary)
    return summary


def saved_result_inventory(root, out, inputs):
    out.mkdir()
    _, checks = verify_manifest(root, FINAL_MANIFEST, inputs)
    write_table(out / "p2b_source_hash_checks.tsv", checks)
    summary = {"p2b_source_hash_checks": len(checks), "p2b_source_hash_status": "MATCH"}
    paths = {
        "p1": Path("output/priority1/GSE146225_TP53_v1/metrics/priority1_replication_summary.json"),
        "p2b_methods": P2B / "final_v14_1_3/figure2_panelB_method_summary.tsv",
        "p2b_contrast": P2B / "final_v14_1_3/figure2_panelC_contrast_summary.tsv",
    }
    for name, relative in paths.items():
        path = data_path(root, relative)
        if not path.exists():
            summary[name] = {"status": "MISSING", "relative_path": str(relative)}
            continue
        inputs.add(path)
        if path.suffix == ".json":
            data = read_json(path)
            summary[name] = {
                key: data.get(key)
                for key in (
                    "benchmark_id",
                    "empirical_replication_fraction",
                    "q_value_matched_replication_fraction",
                    "primary_replication_fraction_difference_empirical_minus_q_value",
                    "interpretation_boundary",
                )
            }
        else:
            summary[name] = read_table(path).to_dict("records")
    grading = data_path(root, GRADING)
    if grading.exists():
        inputs.add(grading)
        table = read_table(grading)
        require(set(GRADING_FIELDS).issubset(table), "Missing P3 grading columns")
        summary["p3"] = {
            "rows": len(table),
            "unique_pmids": table.pmid.nunique(),
            "claims_with_records": table.review_id.nunique(),
            "unfilled_by_field": {
                name: int(table[name].str.strip().eq("").sum()) for name in GRADING_FIELDS
            },
            "grading_performed_by_this_run": False,
        }
    else:
        summary["p3"] = {"status": "MISSING", "relative_path": str(GRADING)}
    ratings = data_path(root, P4 / "ratings_lock_v1/ratings_long.private.tsv")
    claims_path = data_path(root, P4 / "packet_v1/blinded_claims.tsv")
    if ratings.exists() and claims_path.exists():
        inputs.add(ratings)
        inputs.add(claims_path)
        table, claims = read_table(ratings), read_table(claims_path)
        require(not table.duplicated(["rater_id", "review_id"]).any(), "Duplicate P4 rating")
        require(set(table.review_id) == set(claims.review_id), "P4 claim census mismatch")
        summary["p4"] = {
            "ratings": len(table),
            "raters": table.rater_id.nunique(),
            "claims": len(claims),
            "claims_q_lt_0_05": int(pd.to_numeric(claims.q_value).lt(0.05).sum()),
            "new_ratings_or_relabeling": False,
        }
    else:
        summary["p4"] = {"status": "MISSING"}
    legacy = []
    for dataset, relative in (
        (
            "HNSC",
            "paper/source_data/PANCAN_TP53_v1/out_fig2/HNSC/ours/gate_hard/tau_0.90/run_meta.json",
        ),
        (
            "BeatAML",
            "paper/source_data/BEATAML_TP53_v1/out_figS4/BEATAML/ours/gate_hard/tau_0.90/run_meta.json",
        ),
    ):
        path = REPO / relative
        row = {"dataset": dataset, "repository_path": relative, "status": "MISSING"}
        if path.exists():
            inputs.add(path)
            data = read_json(path)
            claims_meta = data.get("inputs", {}).get("claims", {})
            row.update(
                status="FOUND",
                context_score_source=claims_meta.get("context_score_source"),
                scope="metadata_inventory_only_not_complete_figure_provenance",
            )
        legacy.append(row)
    summary["legacy_runs"] = legacy
    summary["full_figure_provenance_completed"] = False
    summary["new_biological_endpoints_calculated"] = False
    write_json(out / "summary.json", summary)
    return summary


def output_directory(root, value=None, *, repo=REPO):
    root = Path(root).expanduser().resolve(strict=True)
    require(
        (root / "input").is_dir() and (root / "output").is_dir(),
        "CRM_R1 data root must contain input/ and output/",
    )
    allowed = data_path(root, Path("output/revision_v17"))
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    out = Path(value).expanduser().resolve() if value else allowed / f"r01_{stamp}"
    require(
        out.is_relative_to(allowed) and out != allowed,
        "Output must be a new child of CRM_R1/output/revision_v17/",
    )
    require(not out.is_relative_to(Path(repo).resolve()), "Analysis output cannot be inside Git")
    require(not out.exists(), "Output already exists; use a new run directory")
    return root, out


def run_r01(root, output=None):
    root, out = output_directory(root, output)
    out.mkdir(parents=True)
    inputs = Inputs()
    for path in [Path(__file__), REPO / "paper/revision/CRM_R1/scripts/60_revision_r01.py"]:
        inputs.add(path)
    for path in sorted((REPO / "src/llm_pathway_curator/contract_v161").glob("*.py")):
        inputs.add(path)
    inputs.add(REPO / "src/llm_pathway_curator/contract_pipeline.py")
    versions = {name: importlib.metadata.version(name) for name in ("pandas", "numpy", "pydantic")}
    started = {
        "schema": VERSION,
        "started_utc": datetime.now(UTC).isoformat(),
        "data_root": str(root),
        "outdir": str(out),
        "git": git_snapshot(),
        "python": platform.python_version(),
        "packages": versions,
        "model_calls": 0,
        "historical_memberships_modified": False,
        "confirmatory_validation": False,
    }
    write_json(out / "STARTED.json", started)
    try:
        print(f"[R01] output: {out}", flush=True)
        inventory = saved_result_inventory(root, out / "inventory", inputs)
        write_json(out / "project_inventory.private.json", project_inventory(REPO))
        adapter = export_adapter(root, out / "adapter", inputs)
        inputs.verify()
        summary = {
            "schema": VERSION,
            "status": "COMPLETE",
            "outdir": str(out),
            "model_calls": 0,
            "scope": "development_preflight_and_adapter",
            "retained_candidates": adapter["contract_summary"]["retained_statistical_candidates"],
            "exported_records": adapter["exported_records"],
            "discovery_rows_checked": adapter["all_discovery_rows_checked"],
            "partitions_checked": adapter["all_partitions_checked"],
            "input_hashes_unchanged": True,
            "p2b_hash_checks": inventory["p2b_source_hash_checks"],
            "biological_or_semantic_performance_estimated": False,
            "full_figure_provenance_completed": False,
            "next_step": "finish_figure_provenance_and_external_dataset_metadata_preflight",
        }
        write_json(out / "summary.json", summary)
        write_json(out / "INPUT_MANIFEST.private.json", list(inputs.files.values()))
        artifacts = {
            str(path.relative_to(out)): {"sha256": sha(path), "size_bytes": path.stat().st_size}
            for path in sorted(out.rglob("*"))
            if path.is_file()
        }
        write_json(
            out / "OUTPUT_MANIFEST.json",
            {
                "schema": VERSION,
                "finished_utc": datetime.now(UTC).isoformat(),
                "outputs": artifacts,
            },
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
        return summary
    except Exception as error:
        write_json(
            out / "FAILED.json", {"completed": False, "error": str(error), "schema": VERSION}
        )
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default=os.environ.get("CRM_R1_DATA_ROOT"))
    parser.add_argument("--out", help="New directory under DATA_ROOT/output/revision_v17/")
    args = parser.parse_args()
    if not args.data_root:
        parser.error("Provide --data-root or CRM_R1_DATA_ROOT")
    try:
        run_r01(args.data_root, args.out)
    except (OSError, ValueError) as error:
        print(f"[STOP] {error}", file=sys.stderr)
        return 2
    return 0
