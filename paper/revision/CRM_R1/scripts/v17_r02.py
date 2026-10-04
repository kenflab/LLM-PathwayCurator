"""Inspect historical figure sources and metadata, without estimating new outcomes.

No network calls, LLM calls, expression loading, enrichment, or expert grading.
The metadata snapshot is supplied separately under the external CRM_R1 root.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
import re
import sys
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path

from v17_revision import (
    REPO,
    Inputs,
    data_path,
    git_snapshot,
    output_directory,
    read_json,
    read_table,
    require,
    sha,
    write_json,
    write_table,
)

VERSION = "CRM_R1_REVISION_v17_R02"
CONFIG = Path("paper/revision/CRM_R1/config/r02_reviewed_sources.json")
RUN_GROUPS = (
    ("PANCAN_TP53_v1", "out_fig2"),
    ("PANCAN_TP53_v1", "out_figS2"),
    ("PANCAN_TP53_v1", "out_figS3"),
    ("BEATAML_TP53_v1", "out_figS4"),
)


def local_path(root, relative):
    """Resolve an explicitly named source; never repair a path using its basename."""
    p = Path(relative)
    require(not p.is_absolute() and ".." not in p.parts, f"Unsafe source path: {relative}")
    return data_path(root, p)


def load_function(filename, name):
    p = Path(__file__).parent / filename
    spec = importlib.util.spec_from_file_location(f"crm_r02_{p.stem}", p)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, name)


def check_source_lock(repo, config, inputs):
    checked = []
    for relative, expected in config["repository_sources"].items():
        p = local_path(repo, relative)
        actual = inputs.add(p)
        require(actual == expected, f"Reviewed source changed: {relative}")
        checked.append({"repository_path": relative, "sha256": actual, "status": "MATCH"})
    observed = {
        str(p.relative_to(repo))
        for dataset, folder in RUN_GROUPS
        for p in (repo / "paper/source_data" / dataset / folder).rglob("run_meta.json")
    }
    require(observed == set(config["run_metadata_paths"]), "Historical run census changed")
    return checked


def proxy_hash(ctx_id, keys, term_uid):
    import hashlib

    payload = json.dumps(
        {
            "ctx_id": ctx_id.strip(),
            "context_keys": [k.strip() for k in keys if k.strip()],
            "term_uid": term_uid.strip(),
        },
        sort_keys=True,
        ensure_ascii=False,
    )
    return int(hashlib.sha256(payload.encode("utf-8")).hexdigest()[:13], 16) / (1 << 52)


def inspect_run(path, repo, inputs):
    meta = read_json(path)
    audit_path = path.parent / "audit_log.tsv"
    audit = read_table(audit_path)
    require(not audit.duplicated("claim_id").any(), f"Duplicate claim: {audit_path}")
    require(not audit.duplicated("term_uid").any(), f"Duplicate term: {audit_path}")
    require(audit.status.isin(["PASS", "ABSTAIN", "FAIL"]).all(), "Invalid historical decision")
    claims = meta.get("inputs", {}).get("claims", {})
    llm = meta.get("inputs", {}).get("llm", {})
    mode = claims.get("context_review_mode")
    row = {
        "run_metadata_path": str(path.relative_to(repo)),
        "run_id": meta.get("run_id", ""),
        "pipeline_status": meta.get("status", ""),
        "last_step": meta.get("step", ""),
        "context_review_mode": mode or "UNRECORDED",
        "context_score_source": claims.get("context_score_source", "UNRECORDED"),
        "claim_mode": llm.get("claim", {}).get("claim_mode_effective_for_select", "UNRECORDED"),
        "backend_attached": llm.get("select_entrypoint", {}).get("backend_attached", "UNRECORDED"),
        "rows": len(audit),
        **{
            f"final_{s.lower()}": int(audit.status.eq(s).sum()) for s in ("PASS", "ABSTAIN", "FAIL")
        },
        "proxy_rows_checked": 0,
        "proxy_hash_matches": 0,
        "normalized_input_hashes": "NOT_CHECKED",
        "technical_llm_receipt": "NOT_APPLICABLE",
        "technical_llm_stop_reason": "",
        "execution_lineage_resolved": meta.get("status") == "ok",
        "biological_validity_established": False,
    }
    if mode != "llm":
        required = {"context_ctx_id", "context_keys", "context_score_proxy_u01_norm"}
        if required <= set(audit):
            values = [
                proxy_hash(r.context_ctx_id, r.context_keys.split(","), r.term_uid)
                for r in audit.itertuples()
            ]
            observed = [float(v) for v in audit.context_score_proxy_u01_norm]
            require(all(math.isfinite(v) for v in observed), "Nonfinite historical hash value")
            row["proxy_rows_checked"] = len(values)
            row["proxy_hash_matches"] = sum(
                math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-15)
                for a, b in zip(values, observed, strict=True)
            )
        if meta.get("status") != "ok":
            row["interpretation"] = "ABORTED_METADATA_WITH_RETAINED_ARTIFACTS_LINEAGE_UNRESOLVED"
        else:
            row["interpretation"] = "RECORDED_PROXY_ENGINEERING_DEMONSTRATION"
    else:
        row["interpretation"] = "RECORDED_LLM_ARTIFACTS_REQUIRE_TECHNICAL_VERIFICATION"
        for f in path.parent.glob("context_review_cache*.json"):
            inputs.add(f)
        checker = load_function("96_verify_llm_audit_v15_1.py", "check_bundle")
        try:
            _, receipt, records = checker(path.parent, "llama3.1:8b")
            for r in records:
                inputs.add(r["path"])
            row["technical_llm_receipt"] = receipt["technical_status"]
        except (OSError, ValueError, KeyError) as error:
            row["technical_llm_receipt"] = "NOT_CERTIFIED"
            row["technical_llm_stop_reason"] = str(error)
            row["execution_lineage_resolved"] = False
    recorded = meta.get("inputs", {})
    hashes = (
        ("evidence.normalized.tsv", "evidence_normalized_sha256"),
        ("sample_card.normalized.json", "sample_card_normalized_sha256"),
    )
    results = []
    for name, key in hashes:
        expected = recorded.get(key)
        if expected:
            actual = inputs.add(path.parent / name)
            results.append(actual == expected)
    row["normalized_input_hashes"] = (
        "MATCH" if len(results) == 2 and all(results) else "MISMATCH_OR_UNRECORDED"
    )
    if False in results:
        row["execution_lineage_resolved"] = False
    return row


def declared_dependencies(repo, inputs):
    path = repo / "paper/FIGURE_MAP.csv"
    inputs.add(path)
    raw = path.read_bytes()
    try:
        text, encoding = raw.decode("utf-8"), "utf-8"
    except UnicodeDecodeError:
        text, encoding = raw.decode("cp1252"), "cp1252"
    rows = []
    for index, record in enumerate(csv.DictReader(text.splitlines()), start=2):
        for field in ("script", "input_paths", "output_paths", "source_data_file"):
            for value in record[field].split(";"):
                value = value.strip()
                if value in {"", "NA"}:
                    state = "NOT_APPLICABLE"
                elif "<" in value or "*" in value:
                    state = "TEMPLATE_NOT_A_CONCRETE_FILE"
                else:
                    target = local_path(repo, value)
                    state = "FOUND" if target.exists() else "DECLARED_PATH_MISSING"
                rows.append(
                    {
                        "map_csv_line": index,
                        "figure_id": record["figure_id"],
                        "panel": record["panel"],
                        "kind": record["kind"],
                        "field": field,
                        "declared_path": value,
                        "status": state,
                        "map_encoding": encoding,
                    }
                )
    return rows


def parse_soft(path, accession):
    attrs = defaultdict(list)
    expected_type = "SERIES" if accession.startswith("GSE") else "SAMPLE"
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    require(lines and lines[0] == f"^{expected_type} = {accession}", "Wrong GEO record identity")
    for line in lines[1:]:
        require(
            not line.startswith(
                ("!sample_table_begin", "!platform_table_begin", "!series_matrix_table_begin")
            ),
            "Expression table in metadata",
        )
        if not line or line.startswith("#"):
            continue
        require(line.startswith("!") and " = " in line, "Non-metadata row in SOFT prefix")
        key, value = line[1:].split(" = ", 1)
        attrs[key].append(value)
    require(
        attrs[f"{expected_type.title()}_geo_accession"] == [accession], "GEO accession mismatch"
    )
    return dict(attrs)


def one(attrs, key):
    values = attrs.get(key, [])
    require(len(values) == 1 and values[0].strip(), f"Missing/ambiguous metadata: {key}")
    return values[0]


def characteristics(attrs):
    result = {}
    for value in attrs.get("Sample_characteristics_ch1", []):
        require(":" in value, "Malformed sample characteristic")
        key, text = value.split(":", 1)
        key = key.strip().lower()
        require(key not in result, f"Duplicate sample characteristic: {key}")
        result[key] = text.strip()
    return result


def metadata_preflight(snapshot, config, inputs):
    manifest_path = snapshot / "METADATA_MANIFEST.json"
    require(
        inputs.add(manifest_path) == config["metadata_manifest_sha256"],
        "Reviewed metadata snapshot manifest changed",
    )
    manifest = read_json(manifest_path)
    require(manifest.get("expression_rows_saved") == 0, "Snapshot includes expression rows")
    records = {}
    sources = []
    for row in manifest["files"]:
        acc = row["accession"]
        require(acc not in records, "Duplicate metadata accession")
        require(re.fullmatch(r"GSE\d+|GSM\d+", acc) is not None, "Invalid GEO accession")
        require(row["scope"] == "metadata_prefix_only_no_expression_rows", "Wrong snapshot scope")
        path = local_path(snapshot, row["path"])
        require(inputs.add(path) == row["sha256"], f"Metadata snapshot changed: {acc}")
        records[acc] = parse_soft(path, acc)
        sources.append(
            {
                "accession": acc,
                "sha256": row["sha256"],
                "source_url": row["source_url"],
                "retrieved_utc": row["retrieved_utc"],
            }
        )
    samples, pairs = [], []
    for series in ("GSE52778", "GSE34313"):
        require(series in records, f"Series metadata missing: {series}")
        census = records[series].get("Series_sample_id", [])
        require(len(census) == len(set(census)), "Duplicate sample in series census")
        require(set(census) == set(config["metadata_sample_census"][series]), "Sample census drift")
        for acc in census:
            require(acc in records, f"Sample metadata missing: {acc}")
            a = records[acc]
            require(a.get("Sample_series_id") == [series], "Sample/series relation changed")
            require(one(a, "Sample_organism_ch1") == "Homo sapiens", "Species mismatch")
            c = characteristics(a)
            title = one(a, "Sample_title")
            if series == "GSE52778":
                m = re.fullmatch(r"(N\d+)_(untreated|Dex|Alb|Alb_Dex)", title)
                require(m is not None, f"Unknown discovery sample title: {title}")
                unit, role = m.groups()
                require(c.get("cell line") == unit, "Title/cell-line mismatch")
                expected = {
                    "untreated": "Untreated",
                    "Dex": "Dexamethasone",
                    "Alb": "Albuterol",
                    "Alb_Dex": "Albuterol_Dexamethasone",
                }[role]
                require(c.get("treatment") == expected, "Title/treatment mismatch")
                hours = 18
                require(
                    "18 h" in one(a, "Sample_treatment_protocol_ch1"), "Discovery timing changed"
                )
                require(one(a, "Sample_platform_id") == "GPL11154", "Discovery platform changed")
            else:
                m = re.fullmatch(r"(nodex|dex4hr|dex24hr)_(\d+)", title)
                require(m is not None, f"Unknown validation sample title: {title}")
                role, replicate = m.groups()
                unit = "HASM1_culture_replicate_" + replicate
                expected = {
                    "nodex": "none",
                    "dex4hr": "dexamethasone for 4 hr",
                    "dex24hr": "dexamethasone for 24 hr",
                }[role]
                require(c.get("treatment") == expected, "Validation title/treatment mismatch")
                hours = {"nodex": "CONTROL_SYNCHRONIZED_HARVEST", "dex4hr": 4, "dex24hr": 24}[role]
                require(one(a, "Sample_platform_id") == "GPL6480", "Validation platform changed")
            samples.append(
                {
                    "series": series,
                    "accession": acc,
                    "title": title,
                    "role": role,
                    "exposure_hours": hours,
                    "unit_label": unit,
                    "platform": one(a, "Sample_platform_id"),
                    "source_locator": f"{acc}.metadata.soft.txt:Sample_characteristics_ch1",
                    "donor_pairing_verified": series == "GSE52778",
                }
            )
    discovery = [s for s in samples if s["series"] == "GSE52778"]
    for unit in sorted({r["unit_label"] for r in discovery}):
        rows = [r for r in discovery if r["unit_label"] == unit]
        require(
            Counter(r["role"] for r in rows) == Counter(["untreated", "Dex", "Alb", "Alb_Dex"]),
            "Incomplete/duplicate discovery donor treatments",
        )
        pairs.append(
            {
                "cell_line": unit,
                "control_accession": next(r["accession"] for r in rows if r["role"] == "untreated"),
                "dex_accession": next(r["accession"] for r in rows if r["role"] == "Dex"),
                "contrast": "dexamethasone_vs_control_vehicle",
                "hours": 18,
            }
        )
    validation = [s for s in samples if s["series"] == "GSE34313"]
    counts = Counter(s["role"] for s in validation)
    require(
        counts == Counter({"nodex": 4, "dex4hr": 3, "dex24hr": 3}),
        "Validation group census changed",
    )
    require(
        not ({r["accession"] for r in discovery} & {r["accession"] for r in validation}),
        "Cross-study sample overlap",
    )
    summary = {
        "status": "METADATA_CHECKED_PROTOCOL_NOT_FROZEN",
        "metadata_records": len(records),
        "sample_records": len(samples),
        "discovery": {
            "series": "GSE52778",
            "paired_donors": len(pairs),
            "primary_eligible_samples": 2 * len(pairs),
            "hours": 18,
            "excluded_roles": ["Alb", "Alb_Dex"],
        },
        "validation": {
            "series": "GSE34313",
            "primary_candidate_hours": 24,
            "public_controls": counts["nodex"],
            "public_24h_dex": counts["dex24hr"],
            "public_4h_dex": counts["dex4hr"],
            "primary_cell_lines": 1,
            "cell_line_source": "PMID:21257922 array experiment used HASM1",
            "donor_pairing_verified": False,
            "replicate_suffix_is_not_a_donor_id": True,
            "original_study_excluded_samples": 2,
        },
        "different_accessions_platforms_and_studies": True,
        "donor_disjointness_between_studies_verified": False,
        "independent_cultures_are_not_independent_donors": True,
        "expression_or_pathway_outcomes_loaded": False,
        "biological_validation_performed": False,
        "gene_universe_and_probe_mapping_checked": False,
        "fallback_GSE96583": "SERIES_METADATA_ONLY_DONOR_ANNOTATIONS_NOT_QUALIFIED",
        "decision": "Retain dex pair as a limited case. Donor replication is unverified.",
        "before_expression_analysis": [
            "Specify GPL6480 gene mapping, normalization, duplicates and measured universe.",
            "Fix the contrast and culture-level model. Suffixes do not establish pairing.",
            "Freeze the method, candidate census, reporting endpoints and replication protocol.",
            "Document published result exposure and unresolved donor overlap.",
        ],
    }
    return samples, pairs, sources, summary


def figure_preflight(repo, root, config, out, inputs):
    checked = check_source_lock(repo, config, inputs)
    write_table(out / "source_hash_checks.tsv", checked)
    runs = [inspect_run(local_path(repo, p), repo, inputs) for p in config["run_metadata_paths"]]
    write_table(out / "historical_run_inventory.tsv", runs)
    write_table(out / "declared_dependencies.tsv", declared_dependencies(repo, inputs))
    ranked = load_function("95_reconstruct_legacy_proxy_v15_1.py", "reconstruct")
    ranking_checks = []
    for item in config["ranked_examples"]:
        base = local_path(repo, item["run_dir"])
        checks, _, summary = ranked(
            read_table(base / "audit_log.tsv"),
            read_table(local_path(repo, item["ranked_path"])),
            read_json(base / "run_meta.json"),
        )
        write_table(out / f"{item['dataset']}_hash_utility_checks.tsv", checks)
        ranking_checks.append({"dataset": item["dataset"], **summary})
    write_json(out / "ranked_source_summary.json", ranking_checks)
    reviewed_docs = []
    for item in config["reviewed_documents"]:
        p = data_path(root, item["relative_path"])
        state = "MISSING"
        if p.exists():
            state = (
                "MATCH_REVIEWED_COPY"
                if inputs.add(p) == item["sha256"]
                else "DIFFERENT_COPY_REVIEW_STALE"
            )
        reviewed_docs.append({**item, "status": state})
    write_table(out / "reviewed_document_checks.tsv", reviewed_docs)
    all_documents_match = all(r["status"] == "MATCH_REVIEWED_COPY" for r in reviewed_docs)
    panels = []
    for item in config["reviewed_panels"]:
        panels.append(
            {
                **item,
                "layout_review_status": "MATCH_REVIEWED_COPY"
                if all_documents_match
                else "REVIEW_STALE_OR_MISSING_DOCUMENT",
                "exact_pdf_assembly_lineage_verified": False,
            }
        )
    write_table(out / "reviewed_panel_routes.tsv", panels)
    public_export_errors = []
    for relative in config["public_ranked_csvs"]:
        p = local_path(repo, relative)
        rows = list(csv.DictReader(p.read_text(encoding="utf-8").splitlines()))
        for i, row in enumerate(rows, start=2):
            for column, value in row.items():
                if value in {"#NAME?", "#REF!", "#VALUE!", "#DIV/0!"}:
                    public_export_errors.append(
                        {
                            "repository_path": relative,
                            "csv_line": i,
                            "claim_id": row.get("claim_id", ""),
                            "column": column,
                            "value": value,
                        }
                    )
    write_table(out / "public_csv_export_errors.tsv", public_export_errors)
    working = []
    for priority in ("priority1", "priority2", "priority2b", "priority5"):
        for p in sorted(data_path(root, f"output/{priority}").rglob("fig/*.run_meta.json")):
            inputs.add(p)
            working.append(
                {
                    "relative_path": str(p.relative_to(root)),
                    "sha256": sha(p),
                    "role": "R1_WORKING_FIGURE_SEPARATE_FROM_SUBMITTED_LEGACY_PANELS",
                }
            )
    write_table(out / "r1_working_figure_inventory.tsv", working)
    return {
        "source_files_checked": len(checked),
        "historical_runs": len(runs),
        "pipeline_status_counts": dict(Counter(r["pipeline_status"] for r in runs)),
        "context_mode_counts": dict(Counter(r["context_review_mode"] for r in runs)),
        "proxy_artifact_rows_checked": sum(r["proxy_rows_checked"] for r in runs),
        "proxy_artifact_hash_matches": sum(r["proxy_hash_matches"] for r in runs),
        "llm_receipts": [
            {
                "path": r["run_metadata_path"],
                "status": r["technical_llm_receipt"],
                "reason": r["technical_llm_stop_reason"],
            }
            for r in runs
            if r["context_review_mode"] == "llm"
        ],
        "reviewed_document_copies_match": all_documents_match,
        "reviewed_panel_routes": len(panels),
        "public_csv_error_cells": len(public_export_errors),
        "r1_working_figure_records": len(working),
        "full_publication_provenance_resolved": False,
        "remaining_findings": config["findings"],
    }


def run_r02(root, snapshot, output=None, *, repo=REPO):
    root = Path(root).expanduser().resolve(strict=True)
    snapshot = Path(snapshot).expanduser().resolve(strict=True)
    require(
        snapshot.is_relative_to(root / "input"), "Metadata snapshot must be under CRM_R1/input/"
    )
    require(not snapshot.is_relative_to(repo), "Metadata snapshot cannot be inside Git")
    if output is None:
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
        output = root / "output/revision_v17" / f"r02_{stamp}"
    root, out = output_directory(root, output, repo=repo)
    out.mkdir(parents=True)
    inputs = Inputs()
    config_path = local_path(repo, CONFIG)
    inputs.add(config_path)
    config = read_json(config_path)
    for name in (
        "v17_r02.py",
        "61_revision_r02.py",
        "v17_revision.py",
        "95_reconstruct_legacy_proxy_v15_1.py",
        "96_verify_llm_audit_v15_1.py",
        "v15_1_provenance_common.py",
    ):
        inputs.add(repo / "paper/revision/CRM_R1/scripts" / name)
    write_json(
        out / "STARTED.json",
        {
            "schema": VERSION,
            "git": git_snapshot(),
            "data_root": str(root),
            "metadata_snapshot": str(snapshot),
            "model_calls": 0,
            "scope": "historical_figure_provenance_and_external_metadata_only",
        },
    )
    try:
        print(f"[R02] output: {out}", flush=True)
        figure_out, metadata_out = out / "figures", out / "metadata"
        figure_out.mkdir()
        metadata_out.mkdir()
        figures = figure_preflight(repo, root, config, figure_out, inputs)
        samples, pairs, sources, metadata = metadata_preflight(snapshot, config, inputs)
        write_table(metadata_out / "sample_design.tsv", samples)
        write_table(metadata_out / "discovery_pairs.tsv", pairs)
        write_table(metadata_out / "metadata_sources.tsv", sources)
        write_json(metadata_out / "summary.json", metadata)
        write_json(figure_out / "summary.json", figures)
        inputs.verify()
        summary = {
            "schema": VERSION,
            "status": "COMPLETE_WITH_FINDINGS",
            "outdir": str(out),
            "input_hashes_unchanged": True,
            "model_calls": 0,
            "new_biological_or_semantic_performance_estimated": False,
            "figures": figures,
            "external_metadata": metadata,
            "next_step": "bounded_semantic_development_then_separate_protocol_freeze",
        }
        write_json(out / "summary.json", summary)
        write_json(out / "INPUT_MANIFEST.private.json", list(inputs.files.values()))
        write_json(
            out / "OUTPUT_MANIFEST.json",
            {
                "schema": VERSION,
                "outputs": {
                    str(p.relative_to(out)): {"sha256": sha(p), "size_bytes": p.stat().st_size}
                    for p in sorted(out.rglob("*"))
                    if p.is_file()
                },
            },
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
        return summary
    except Exception as error:
        write_json(
            out / "FAILED.json", {"schema": VERSION, "error": str(error), "completed": False}
        )
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", default=os.environ.get("CRM_R1_DATA_ROOT"))
    p.add_argument("--metadata-snapshot", required=True)
    p.add_argument("--out")
    args = p.parse_args()
    if not args.data_root:
        p.error("Provide --data-root or CRM_R1_DATA_ROOT")
    try:
        run_r02(args.data_root, args.metadata_snapshot, args.out)
    except (OSError, ValueError, KeyError) as error:
        print(f"[STOP] {error}", file=sys.stderr)
        return 2
    return 0
