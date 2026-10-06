#!/usr/bin/env python3
"""Collect saved revision evidence and current manuscript, without refitting.

Outputs are private review materials under the existing CRM_R1/output tree.
Missing/unverified evidence is reported explicitly, never filled or estimated.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
import stat
import sys
import zipfile
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]
BENCHMARK = "PANCAN_TP53_v1_HNSC_R1"
R06_PATH = "output/revision_v17/r06_20261005T163843024404Z.zip"
R06_SHA = "e69c13965ec0828de5486bf05a2fb5fd4362f16f70985c8091ebcb1b8fbd788a"
R07_PATH = "output/revision_v17/r07_20261005T190838308028Z.zip"
R07_SHA = "f66c9a355083ca725e4783ff10f7465ff63b3e45a2d7044e46f40712052fd28b"
DEX_SHA = "07a18f1618a802ed953d11f5153a6f42b73d178246bbc8a908c8f20841be9222"
P3_FIELDS = ("eligible", "exclusion_reason", "evidence_grade", "direction_match", "context_match", "study_design", "data_overlap", "contradiction", "supporting_note", "curator_id")
GUIDES = ("README.md", "manuscript/MANUSCRIPT_METHODS_RESULTS_DRAFT.md", "manuscript/REBUTTAL_RESPONSE_MATRIX.md", "manuscript/FIGURE2_LAYOUT_AND_LEGEND.md")
W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def table(raw):
    return list(csv.DictReader(io.StringIO(raw.decode("utf-8-sig")), delimiter="\t"))


def obj(raw):
    return json.loads(raw.decode("utf-8"))


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def write_table(path, rows, fields):
    with Path(path).open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def local_path(root, recorded):
    """Rebase an original full CRM_R1 path; never search by basename."""
    path = Path(recorded)
    if path.is_absolute() and path.is_relative_to(root):
        relative = path.relative_to(root)
    elif path.is_absolute():
        markers = [i for i, part in enumerate(path.parts) if part == "CRM_R1"]
        require(len(markers) == 1, "Recorded path cannot be unambiguously rebased")
        relative = Path(*path.parts[markers[0] + 1:])
    else:
        relative = path
    require(relative.parts and ".." not in relative.parts, "Unsafe source path")
    result = root / relative
    require(result.resolve().is_relative_to(root), "Source path escapes CRM_R1")
    require(not any(p.is_symlink() for p in (result, *result.parents)), "Symlink in source path")
    return result


def archive_files(path, kind):
    with zipfile.ZipFile(path) as archive:
        infos = archive.infolist()
        require(len(infos) == len({i.filename for i in infos}), "Duplicate archive entry")
        require(sum(i.file_size for i in infos) < 100 * 1024 * 1024, "Result archive unexpectedly large")
        for item in infos:
            name = Path(item.filename)
            require(not name.is_absolute() and ".." not in name.parts and not item.is_dir() and not stat.S_ISLNK(item.external_attr >> 16), "Unsafe result archive entry")
        if kind == "r06":
            roots = {Path(i.filename).parts[0] for i in infos}
            require(len(roots) == 1, "R06 archive has ambiguous roots")
            prefix = next(iter(roots)) + "/"
            raw_manifest = archive.read(prefix + "OUTPUT_MANIFEST.json")
            manifest = obj(raw_manifest)["files"]
            require(set(archive.namelist()) == {prefix + name for name in manifest} | {prefix + "OUTPUT_MANIFEST.json"}, "R06 manifest inventory mismatch")
            files = {}
            for name, record in manifest.items():
                raw = archive.read(prefix + name)
                require(digest(raw) == record["sha256"] and len(raw) == record["size_bytes"], "R06 export hash/size mismatch: " + name)
                files[name] = raw
            files["OUTPUT_MANIFEST.json"] = raw_manifest
        else:
            raw_manifest = archive.read("EXPORT_SHA256.json")
            manifest_payload = obj(raw_manifest)
            manifest = manifest_payload if kind == "r07" else manifest_payload["files"]
            require(set(archive.namelist()) == set(manifest) | {"EXPORT_SHA256.json"}, "Archive manifest inventory mismatch")
            files = {}
            for name, record in manifest.items():
                raw = archive.read(name)
                expected = record["sha256"] if isinstance(record, dict) else record
                require(digest(raw) == expected, "Archive export hash mismatch: " + name)
                files[name] = raw
            files["EXPORT_SHA256.json"] = raw_manifest
    return files


class Collection:
    def __init__(self, root, out):
        self.root, self.out = root, out
        self.inputs, self.status = {}, {}

    def read(self, relative):
        path = local_path(self.root, relative)
        raw = path.read_bytes()
        self.inputs[str(path.relative_to(self.root))] = digest(raw)
        return raw

    def copy(self, relative, destination):
        raw = self.read(relative)
        path = self.out / destination
        require(path.resolve().is_relative_to(self.out), "Snapshot escapes new output")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
        return raw

    def snapshot(self, directory, files, names):
        for name in names:
            require(name in files, "Saved result file missing: " + name)
            target = self.out / "saved_results" / directory / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(files[name])

    def optional(self, name, function):
        try:
            value = function()
            self.status[name] = {"status": "SAVED_OUTPUTS_CHECKED", **value}
        except (OSError, ValueError, KeyError, zipfile.BadZipFile, ET.ParseError) as error:
            self.status[name] = {"status": "MISSING_OR_NOT_VERIFIED", "reason": str(error)}

    def verify_output_manifest(self, relative, *, hash_key="outputs"):
        raw = self.read(relative)
        manifest = obj(raw)
        sha_companion = Path(relative).with_suffix(".sha256")
        if local_path(self.root, sha_companion).is_file():
            require(self.read(sha_companion).decode().split()[0] == digest(raw), "Output manifest companion hash mismatch")
        records = manifest[hash_key]
        require(isinstance(records, dict) and records, "Empty saved output hash inventory")
        verified = []
        for name, record in records.items():
            if isinstance(record, str):
                recorded, expected = str(Path(relative).parent / name), record
            else:
                recorded, expected = record["path"], record["sha256"]
            path = local_path(self.root, recorded)
            raw_output = self.read(path)
            require(digest(raw_output) == expected, "Saved output hash mismatch: " + str(path.relative_to(self.root)))
            require(len(raw_output) <= 8 * 1024 * 1024, "Saved output too large for compact source collection")
            destination = self.out / "frozen_source_data" / path.relative_to(self.root)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(raw_output)
            verified.append(str(path.relative_to(self.root)))
        manifest_copy = self.out / "frozen_source_data" / relative
        manifest_copy.parent.mkdir(parents=True, exist_ok=True)
        manifest_copy.write_bytes(raw)
        return {"recorded_output_files_checked": len(verified), "paths": verified,
                "scope": "Recorded output hashes only; raw inputs and statistical models not rerun"}


def collect_r06(collection):
    raw = collection.read(R06_PATH)
    require(digest(raw) == R06_SHA, "Original reviewed R06 archive differs")
    files = archive_files(local_path(collection.root, R06_PATH), "r06")
    summary = obj(files["summary.json"])
    require(summary["status"] == "COMPLETE_EXPLORATORY_P4_COMPARISON" and summary["source_linkage_verified"], "R06 linkage not complete")
    require((summary["candidate_claims"], summary["raters"], summary["ratings"]) == (50, 3, 150), "R06 census drift")
    require(not summary["candidate_wording_changed"] and not summary["naive_llm_prose_comparison_performed"], "R06 text scope drift")
    names = ("summary.json", "rater_method_endpoints.tsv", "paired_method_differences.tsv", "interrater_agreement.tsv", "rater_category_counts.tsv", "method_overlap.tsv", "ratings_linked.private.tsv", "linkage_checks.private.tsv", "OUTPUT_MANIFEST.json", "Fig_R06_P4_major_overstatement_by_rater.pdf", "Fig_R06_P4_major_overstatement_by_rater.png")
    collection.snapshot("r06", files, names)
    agreement = table(files["interrater_agreement.tsv"])
    endpoints = table(files["rater_method_endpoints.tsv"])
    require(len(agreement) == 3 and len(endpoints) == 96, "R06 exported table census drift")
    return {"archive_sha256": digest(raw), "claims": 50, "raters": 3, "ratings": 150,
            "interrater_agreement": agreement,
            "major_overstatement_by_rater": [r for r in endpoints if r["endpoint"] == "confirmed_major_overstatement" and r["rater_id"] in {"P4_R1", "P4_R2", "P4_R3"}],
            "scope": "Existing standardized statements; post hoc fixed-rater comparison; no new text accuracy labels"}


def collect_r07(collection):
    raw = collection.read(R07_PATH)
    require(digest(raw) == R07_SHA, "Original reviewed R07 archive differs")
    files = archive_files(local_path(collection.root, R07_PATH), "r07")
    summary = obj(files["results/summary.json"])
    require(summary["status"] == "NATURAL_TEXT_COMPARISON_COMPLETE" and summary["candidate_count"] == 50 and summary["model_backend_authenticated"], "R07 generation/scope drift")
    require(summary["numeric_gate_status_counts"] == {"EXPLICIT_NUMERIC_COVERAGE_INCOMPLETE": 49, "EXPLICIT_NES_Q_CHECKED": 1}, "Frozen R07 check counts drift")
    names = ("results/summary.json", "results/method_coverage.tsv", "results/canonical_evidence.private.json", "results/first_outcomes.private.json", "results/comparisons.private.json", "results/source_fact_check.optional.private.tsv", "results/model_config.private.json", "EXPORT_SHA256.json")
    collection.snapshot("r07", files, names)
    return {"archive_sha256": digest(raw), "candidate_count": 50, "checked_prose_count": 1,
            "explicit_numeric_coverage_incomplete": 49, "scope": "Restricted numeric coverage; not semantic accuracy or revised/full audit performance"}


def collect_dex(collection):
    receipt_path = "output/revision_v17/external_dex_design_v1/COMPLETED_ANALYSIS.json"
    receipt = obj(collection.read(receipt_path))
    path = str(Path(receipt["outdir_relative_path"])) + ".zip"
    raw = collection.read(path)
    require(digest(raw) == receipt["archive_sha256"] == DEX_SHA, "Original reviewed dex archive differs")
    files = archive_files(local_path(collection.root, path), "dex")
    summary = obj(files["SUMMARY.json"])
    require(summary["status"] == "COMPLETE_LIMITED_CROSS_STUDY_CASE" and summary["candidate_count"] == 50, "Dex result scope/census drift")
    selected = [set(summary["comparisons"][i]["selected_ids"].split(";")) for i in (2, 3)]
    require(selected[0] == selected[1] and len(selected[0]) == 10, "Dex size-matched selection identity drift")
    names = ("SUMMARY.json", "METHOD_COMPARISON.tsv", "TERM_LEDGER.tsv", "DESIGN_LOCK.json", "protocol.json", "RUNTIME_IDENTITY.json", "INPUT_MANIFEST.json", "PROBE_FILTER_DIAGNOSTIC.json", "TECHNICAL_AMENDMENT_R08_1.json", "EXPORT_SHA256.json")
    collection.snapshot("external_dex", files, names)
    return {"archive_sha256": digest(raw), "candidate_count": 50, "same_selected_ids": True,
            "comparisons": summary["comparisons"], "scope": summary["scope"]}


def docx_text(raw):
    """Expose final-view prose, deleted text, and comments separately."""
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        require(sum(i.file_size for i in archive.infolist()) < 100 * 1024 * 1024, "Word file unexpectedly large")
        document = ET.fromstring(archive.read("word/document.xml"))
        paragraphs, deleted = [], []
        insertions, deletions = 0, 0
        for paragraph in document.iter(W + "p"):
            pieces = []
            def visit(node, removed=False):
                nonlocal insertions, deletions
                if node is not paragraph and node.tag == W + "p":
                    return
                if node.tag == W + "del":
                    deletions += 1
                    removed = True
                if node.tag == W + "ins":
                    insertions += 1
                if node.tag in (W + "t", W + "delText"):
                    if removed or node.tag == W + "delText":
                        deleted.append(node.text or "")
                    else:
                        pieces.append(node.text or "")
                elif node.tag == W + "tab" and not removed:
                    pieces.append("\t")
                elif node.tag in (W + "br", W + "cr") and not removed:
                    pieces.append("\n")
                for child in node:
                    visit(child, removed)
            visit(paragraph)
            # Do not strip paragraph content or merge separate table-cell paragraphs.
            if pieces:
                paragraphs.append("".join(pieces))
        comments = []
        if "word/comments.xml" in archive.namelist():
            tree = ET.fromstring(archive.read("word/comments.xml"))
            for comment in tree.iter(W + "comment"):
                comments.append({"id": comment.get(W + "id", ""), "text": "".join(n.text or "" for n in comment.iter(W + "t"))})
        return {"paragraphs": paragraphs, "deleted_text_fragments": deleted,
                "tracked_insertions": insertions, "tracked_deletions": deletions, "comments": comments,
                "scope": "Word document.xml final-view text only; headers, footnotes, textboxes and rendered layout still require visual review"}


def manuscript_flags(paragraphs):
    patterns = {
        "DECISION_GRADE_TERM": r"decision[-– ]grade",
        "HISTORICAL_RATER_CENSUS": r"\b(?:two|2)\s+raters?\b|\b100\s+claims?\b",
        "SUPERIORITY_OR_ACCURACY_LANGUAGE": r"\b(?:outperform\w*|superior\w*|biologically correct|biological accuracy|decision[- ]quality advantage)\b",
        "LITERATURE_GRADING_CLAIM": r"\bE[34]\b|independent.{0,35}(?:support|evidence)|(?:grading|graded).{0,35}(?:complete|locked)",
        "INDEPENDENT_VALIDATION_LANGUAGE": r"independent.{0,30}(?:biological|donor|validation)",
        "LEGACY_UTILITY_LANGUAGE": r"context[_ ](?:score|confidence)|multiplicative|utility.{0,30}(?:rank|score)",
    }
    flags = []
    for index, paragraph in enumerate(paragraphs, 1):
        for name, pattern in patterns.items():
            if re.search(pattern, paragraph, re.I):
                flags.append({"paragraph_index": index, "review_topic": name,
                              "text": paragraph, "status": "AUTHOR_REVIEW_REQUIRED_NOT_AUTOMATIC_ERROR_LABEL"})
    return flags


def collect_manuscript(collection):
    name = "LLM-PathwayCurator_CRM_R1.docx"
    raw = collection.copy(name, "current_manuscript/" + name)
    parsed = docx_text(raw)
    directory = collection.out / "current_manuscript"
    write_json(directory / "R1_TEXT_AND_TRACKED_CHANGES.private.json", parsed)
    (directory / "R1_TEXT.private.txt").write_text("\n".join(f"[paragraph {i}] {p}" for i, p in enumerate(parsed["paragraphs"], 1)) + "\n", encoding="utf-8")
    flags = manuscript_flags(parsed["paragraphs"])
    write_table(directory / "CLAIM_REVIEW_FLAGS.private.tsv", flags, ["paragraph_index", "review_topic", "text", "status"])
    for other in ("LLM-PathwayCurator_CRM.docx", "Supplementary_Information_CRM.docx"):
        if local_path(collection.root, other).is_file():
            collection.copy(other, "current_manuscript/" + other)
    return {"source_sha256": digest(raw), "paragraphs": len(parsed["paragraphs"]),
            "review_flags": len(flags), "tracked_insertions": parsed["tracked_insertions"],
            "tracked_deletions": parsed["tracked_deletions"], "comments": len(parsed["comments"]),
            "scope": parsed["scope"], "manuscript_edited": False}


def collect_p3(collection):
    relative = f"output/priority3/{BENCHMARK}/grading_working/record_screening_P3C1.private.tsv"
    raw = collection.copy(relative, "p3_grading/record_screening_P3C1.private.tsv")
    rows = table(raw)
    require(rows and set(P3_FIELDS) <= set(rows[0]), "P3 grading columns missing")
    missing = {name: sum(not r[name].strip() for r in rows) for name in P3_FIELDS}
    complete = sum(all(r[name].strip() for name in ("eligible", "curator_id", "supporting_note")) for r in rows)
    return {"rows": len(rows), "blank_by_field": missing,
            "rows_with_nonblank_eligible_curator_and_note": complete,
            "grading_performed_by_this_run": False, "complete_locked_grading_authenticated": False,
            "scope": "Completeness inventory only; blank cells are not negative grades; no external-support accuracy endpoint is generated"}


def response_matrix(states):
    evidence = lambda name: states[name]["status"]
    return [
        ("Editor", "Substantial additional evidence", "P1 temporal, P2B multi-cohort, existing-rater, ontology and dex analyses; retain null/adverse findings", "Improved interpretive/decision quality of the current full audit is not established", "OPEN_SCIENTIFIC_GAP"),
        ("R1 major 1; R2 major 3", "Biological reliability and human agreement", "R06: 50 unchanged statements, 3 raters, 150 ratings; separate raters and uncertainty", "Low agreement cannot establish biological correctness; original 100/2 description needs reconciliation", evidence("r06")),
        ("R1 major 2; R2 major 1-2", "Unaudited and simple baselines", "R06 same-text, same-K legacy selection comparisons; R07 first natural prose and source-template control", "R06 labels do not apply to R07 prose; numeric coverage does not establish full-audit utility", "PARTIALLY_ADDRESSED"),
        ("R1 major 3", "GO/Reactome hierarchy", "P5 direct/ancestor pairs, leading-edge overlap and depth", "Opposite NES signs are directional discordance, not proof of biological contradiction or correctness", evidence("p5")),
        ("R1 minor 1; R2 major 2", "Additional systems and external validity", "P1 temporal perturbation; dex non-cancer, cross-study culture-level example", "One validation cell line and unknown donor overlap; no full-audit superiority", evidence("external_dex")),
        ("R1 minor 2", "Utility formulation", "Identify empirical evidence, sample stability and independently supported context separately", "Legacy hash-based context is not biological fit; weight changes do not rescue the old score", "OPEN_PROVENANCE_AND_SCOPE"),
        ("R1 minor 3", "Human uncertainty", "R06 rater-specific Wilson descriptions, joint claim resampling and kappa intervals", "Fixed-rater, dependent-pathway intervals; preserve UNCERTAIN and degenerate CI status", evidence("r06")),
        ("R2 novelty; R2 major 5", "Prior propose/verify work", "Cite Khan et al., PMID 41071041; focus on a pathway-specific evidence contract and deterministic dispositions", "The cited work uses LLM faithfulness evaluators; do not describe it as a non-LLM verifier", "TEXT_READY_FOR_INTEGRATION"),
        ("R2 major 4", "PASS/ABSTAIN operating behavior", "Describe the actual legacy stability/context filter with contract/contradiction backstops", "Keep production behavior, synthetic controls and revised development prototypes distinct", "TEXT_READY_WITH_CURRENT_MANUSCRIPT"),
        ("R2 major 6", "A practical before/after example", "Present original text, source row, performed checks and limits; R07 numeric-checked text can still overinterpret", "Do not present unsupported WNT/HNSC context rejection as a verified biological success", "WORKED_EXAMPLE_REQUIRES_FINAL_SCOPE"),
        ("All", "Source Data and final manuscript", "Current Word, P1/P2B/P5 frozen outputs, R06/R07/dex snapshots and unresolved figure routes", "This collection is not final assembly or a readiness certificate", evidence("current_manuscript")),
    ]


def draft_text(states):
    text = ["DRAFT — source-linked revision text; final figure/page references require manuscript assembly.",
            "This file does not certify that the editor's substantive evidence requirement has been met.", ""]
    if states["r06"]["status"] == "SAVED_OUTPUTS_CHECKED":
        k = states["r06"]["interrater_agreement"]
        values = "; ".join(f"{r['question']}: κ={float(r['fleiss_kappa']):.3f}" for r in k)
        text.extend(["Results — existing-rater comparison",
            f"We linked the original 50 standardized HNSC statements to all 150 locked ratings from three evaluators, retaining the original wording and fixed method memberships. Agreement was low ({values}). We therefore report each evaluator separately, retain uncertain responses and report descriptive uncertainty without treating majority labels as biological ground truth. This post hoc analysis compares the legacy full-audit selection with the same-pool and size-matched reporting rules. It does not score newly generated natural-language prose or the revised audit, and it did not demonstrate a legacy audit advantage.", ""])
    if states["r07"]["status"] == "SAVED_OUTPUTS_CHECKED":
        text.extend(["Results — natural-language reporting scope",
            "All 50 first local-model descriptions from the previously inspected HNSC discovery partition were retained. The frozen explicit-number checker retained one description (2% coverage), with 49 classified as explicit numeric coverage incomplete. Its size-matched q comparator also retained one description. These are coverage and execution observations, not estimates of semantic accuracy. The existing expert ratings apply only to their original standardized statements and were not transferred to the new descriptions. The source-derived template remains a simple reporting baseline.", ""])
    if states["external_dex"]["status"] == "SAVED_OUTPUTS_CHECKED":
        text.extend(["Results — non-cancer worked case",
            "In the primary 24-h external dexamethasone contrast, 10/50 Hallmark candidates met the same-direction, validation-q≤0.05 endpoint. Discovery q filtering selected 11 terms, with 3/11 meeting the endpoint. Adding donor stability retained 10 terms, with 3/10 meeting the endpoint, identical to the size-matched q comparator. The latter methods selected exactly the same ten terms. This limited cross-study example provides no observed incremental selection benefit from donor stability. Validation used cultures from one HASM1 cell line; cross-study donor disjointness remains unverified. The experiment evaluates the statistical component and does not measure semantic accuracy or full-audit superiority.", ""])
    text.extend(["Discussion / novelty text",
        "Prior gene-prioritization work combined LLM screening with literature-grounded faithfulness evaluation (Khan et al., 2025; PMID: 41071041; doi:10.1093/bioinformatics/btaf541). We do not claim invention of proposal–verification separation. The contribution considered here is its pathway-specific implementation through typed evidence records, provenance links and deterministic disposition rules. Verification of these defined conditions should not be equated with biological correctness. The current data do not establish a general interpretive advantage of the full audit over simpler reporting rules.", "",
        "P1/P2B/P5 integration",
        "Use the collected original tables and metadata to insert exact results. Preserve the adverse P2B full-minus-baseline contrasts and the cohort-level analysis unit. Distinguish P1 temporal empirical resampling from synthetic gene perturbations. Describe P5 opposite-sign pairs as directional discordance; do not equate these with biological errors. Do not insert an independent PubMed-support endpoint without completed authenticated grading.", "",
        "Citation to verify in the final reference list",
        "Khan T, et al. Automating candidate gene prioritization with large language models: from naive scoring to literature-grounded validation. Bioinformatics. 2025;41(10):btaf541. https://doi.org/10.1093/bioinformatics/btaf541",
        "https://pmc.ncbi.nlm.nih.gov/articles/PMC12548045/",
        "Methods §2.3.3 describes Phi-4 and GPT-o3-mini evaluators; this is not a non-LLM verification stage.", ""])
    return "\n".join(text)


def build(root, out, repo=REPO):
    require((root / "input").is_dir() and (root / "output").is_dir(), "Use the existing CRM_R1 data root")
    require(not root.is_relative_to(repo), "Keep CRM_R1 data outside the repository")
    require(out.resolve().is_relative_to(root / "output") and not out.exists(), "New collection must be under output/ and must not overwrite a folder")
    require(not any(p.is_symlink() for p in (out, *out.parents)), "Symlink in output path")
    out.mkdir(parents=True)
    collection = Collection(root, out)
    try:
        collection.optional("r06", lambda: collect_r06(collection))
        collection.optional("r07", lambda: collect_r07(collection))
        collection.optional("external_dex", lambda: collect_dex(collection))
        collection.optional("current_manuscript", lambda: collect_manuscript(collection))
        collection.optional("p1", lambda: collection.verify_output_manifest("output/priority1/GSE146225_TP53_v1/metrics/priority1_replication.run_meta.json"))
        collection.optional("p2b", lambda: collection.verify_output_manifest("output/priority2b/final_v14_1_3/figure2_source_manifest.json"))
        collection.optional("p5", lambda: collection.verify_output_manifest(f"output/priority5/{BENCHMARK}_P5/ontology/ontology_evaluation.run_meta.json", hash_key="output_sha256"))
        collection.optional("p3_completeness", lambda: collect_p3(collection))
        # Templates/guides remain distinct from the author's current Word.
        for guide in GUIDES:
            source = repo / "paper/revision/CRM_R1" / guide
            target = out / "current_code_guides" / guide
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(source.read_bytes())
        matrix = response_matrix(collection.status)
        write_table(out / "RESPONSE_MATRIX.private.tsv", [dict(zip(("comment", "topic", "evidence", "remaining_limit", "status"), row)) for row in matrix], ["comment", "topic", "evidence", "remaining_limit", "status"])
        (out / "RESULTS_AND_DISCUSSION_DRAFT_EN.txt").write_text(draft_text(collection.status), encoding="utf-8")
        actions = [
            ("Legacy Fig2 e/f", "Reversed panel route; verify the current Word/PDF and revise or remove affected utility panels", "OPEN"),
            ("Legacy FigS2 b/c", "Reversed CSV/source-PDF labels; use the actual source data and legend", "OPEN"),
            ("Legacy FigS4 f", "Packed-circle display versus bar description; reconcile the displayed plot and legend", "OPEN"),
            ("Legacy BRCA GO tau0.20", "Aborted run with downstream artifacts; execution lineage is unresolved", "OPEN"),
            ("Historical FigS3 LLM receipt", "Recorded backend/model identity is uncertified", "OPEN"),
            ("Legacy utility/context", "Hash-derived score is not biological context fit; disclose and remove the unsupported interpretation", "OPEN"),
            ("Legacy ranked CSV exports", "Recorded #NAME? score_source cells require export/source reconciliation; do not silently replace with missing values or zeros", "OPEN"),
            ("P3 support panel", "Do not use retrieval hits or blank grades as independent evidence support", "NO_ENDPOINT_UNTIL_GRADING_AUTHENTICATED"),
            ("Existing R2 rater workbook", "Final Aug24 source confirmed separately; retain low agreement and original locked ratings", "SOURCE_VERSION_RESOLVED_SEPARATELY"),
            ("Dex probe correction", "Original design retained; R08.1 amendment and completed result preserved", collection.status["external_dex"]["status"]),
        ]
        write_table(out / "PUBLICATION_ACTIONS.private.tsv", [dict(zip(("item", "action", "status"), row)) for row in actions], ["item", "action", "status"])
        (out / "FIGURE_PLAN.private.txt").write_text("""Draft four-main-figure structure; final numbers require the current Word.
1. Workflow, source linkage and an explicitly bounded worked example.
2. Main comparative evidence: adverse P2B cohort-level results and R06 rater-specific comparisons/agreement. Raw-pool and matched-K contrasts are distinct.
3. Empirical statistical component: P1 temporal replication plus the primary dex worked case; include equal/null selection comparisons and denominator labels.
4. GO/Reactome hierarchy: leading-edge support, depth and directional discordance, without biological correctness labels.
Supplement: R07 first-prose coverage/limits, all candidates, secondary dex 4h, legacy stress tests and corrected source routes where provenance supports retention.
Do not allocate an independent PubMed-support panel while its grading is incomplete. The figure plan does not certify that the current full audit improves interpretive quality.
""", encoding="utf-8")
        missing = [name for name, value in collection.status.items() if value["status"] != "SAVED_OUTPUTS_CHECKED"]
        summary = {"schema": "CRM_R1_SUBMISSION_ASSEMBLY_R10", "status": "SOURCE_COLLECTION_COMPLETE_WITH_OPEN_ITEMS",
                   "outdir": str(out), "source_status": collection.status, "missing_or_unverified_components": missing,
                   "original_input_file_count": len(collection.inputs), "model_calls": 0, "new_expert_ratings": 0,
                   "R_fitting_or_enrichment_rerun": False, "author_manuscript_edited": False,
                   "submission_ready": False, "current_full_audit_advantage_established": False,
                   "raw_expression_data_included": False, "collection_code_sha256": file_sha(Path(__file__)),
                   "scope": "Private source collection and draft integration; unresolved evidence and manuscript/figure checks remain explicit"}
        write_json(out / "SUMMARY.json", summary)
        write_json(out / "INPUT_HASHES.private.json", {"files": collection.inputs})
        ja = ["R10 引き継ぎと原稿統合用の資料", "",
              "この実行は保存済み出力と現行Wordの収集です。新しい発現解析・LLM・専門家評価は実行していません。元データ、設計、評価、本文は変更していません。", ""]
        ja.extend(f"{name}: {value['status']}" for name, value in collection.status.items())
        ja.extend(["", "先に読むファイル: SUMMARY.json、RESPONSE_MATRIX.private.tsv、PUBLICATION_ACTIONS.private.tsv、RESULTS_AND_DISCUSSION_DRAFT_EN.txt。",
                   "current_manuscript/は現行R1 Wordのcopyと読取textです。tracked deletionは本文と分離し、commentsも保存しています。本文の改稿や組版検査は未実施です。",
                   "frozen_source_data/は宣言された出力hashが一致した原表です。保存出力の一致と、raw入力・解析の独立再検証は別です。missing/unverifiedを完了扱いしません。",
                   "旧R2ラベルの版問題は著者指定の最終返却Excelとの照合で解決済みです。低一致度を改善したことにはなりません。",
                   "P3は記入状況だけを確認します。blankをnegative gradeへ変更しません。620行全量採点や専門家Round2は今回の工程に含めません。",
                   "今月の投稿に向け、次は実物Wordと元表に基づいてResults/Methods、四つの主図、Source Data、point-by-pointを整合させます。",
                   "旧full auditの不利なP2B比較、既存評価者の不一致、R07のcoverage不足、dexの同一選択集合を保持します。現在のfull auditの解釈品質改善は未実証です。",
                   "新しいdataset・閾値・有料モデルを結果に応じて追加することを、既定の次工程にはしません。投稿準備が完了したとは判定していません。",
                   "この収集ZIPは原稿・個別評価・local pathを含むprivate作業用です。public Gitへ追加するのはcodeとガイドのみです。"])
        (out / "START_HERE_JA.txt").write_text("\n".join(ja) + "\n", encoding="utf-8")
        require(all(file_sha(local_path(root, name)) == value for name, value in collection.inputs.items()), "An original source changed during collection")
        exported = {str(p.relative_to(out)): file_sha(p) for p in sorted(out.rglob("*")) if p.is_file()}
        write_json(out / "EXPORT_SHA256.json", {"files": exported})
        archive = Path(str(out) + ".zip")
        with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED) as zipped:
            for name in [*exported, "EXPORT_SHA256.json"]:
                zipped.write(out / name, name)
        return {**summary, "results_archive": str(archive)}
    except Exception as error:
        write_json(out / "STOPPED.private.json", {"error": str(error), "original_inputs_modified": False})
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--outdir", type=Path)
    args = parser.parse_args(argv)
    try:
        root = args.data_root.expanduser().resolve()
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        out = args.outdir.expanduser().resolve() if args.outdir else root / ("output/revision_v17/submission_assembly_" + stamp)
        result = build(root, out)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except (ValueError, OSError, KeyError, zipfile.BadZipFile) as error:
        print(f"[STOP] {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
