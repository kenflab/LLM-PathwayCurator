"""Reassess unchanged R07 text using the public source workflow.

This is an exposed development corpus, not an independent accuracy benchmark.
No model transport, expression fitting or expert-label transfer is performed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import stat
import sys
import zipfile
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath

REPO = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPO / "src"))

from llm_pathway_curator import RunConfig, run_pipeline  # noqa: E402
from llm_pathway_curator.grounding import RULESET  # noqa: E402

PROTOCOL = Path(__file__).with_name("protocol.json")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def write_tsv(path, rows):
    with Path(path).open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, list(rows[0]), delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def implementation_hashes():
    paths = [Path(__file__), PROTOCOL]
    paths += [
        REPO / "src/llm_pathway_curator" / n
        for n in ("review.py", "grounding.py", "pipeline.py", "_shared.py")
    ]
    return {p.relative_to(REPO).as_posix(): sha(p.read_bytes()) for p in paths}


def load_corpus(path, expected_count):
    """Verify either the original R07 ZIP or a complete submission-source ZIP."""
    path = Path(path)
    require(path.is_file() and path.stat().st_size < 50_000_000, "Missing or oversized source ZIP")
    raw_archive = path.read_bytes()
    with zipfile.ZipFile(io.BytesIO(raw_archive)) as archive:
        entries = archive.infolist()
        names = [e.filename for e in entries]
        require(len(names) == len(set(names)), "Duplicate ZIP paths")
        require(sum(e.file_size for e in entries) < 100_000_000, "Uncompressed ZIP limit")
        for entry in entries:
            name = PurePosixPath(entry.filename)
            require(
                not name.is_absolute()
                and ".." not in name.parts
                and "\\" not in entry.filename
                and not entry.is_dir()
                and not stat.S_ISLNK(entry.external_attr >> 16),
                "Unsafe ZIP path",
            )
        payload = {n: archive.read(n) for n in names}
    manifest = json.loads(payload["EXPORT_SHA256.json"])
    exports = manifest.get("files", manifest)
    require(isinstance(exports, dict), "Invalid export manifest")
    require(
        set(payload) == set(exports) | {"EXPORT_SHA256.json"}, "Missing or unmanifested exports"
    )
    for name, expected in exports.items():
        require(sha(payload[name]) == expected, "Export hash mismatch: " + name)
    if "SUMMARY.json" in payload:
        assembly = json.loads(payload["SUMMARY.json"])
        require(assembly["schema"] == "CRM_R1_SUBMISSION_ASSEMBLY_R10", "Unsupported collection")
        require(not assembly["missing_or_unverified_components"], "Incomplete collection")
        prefix = "saved_results/r07/results/"
    else:
        prefix = "results/"
    summary = json.loads(payload[prefix + "summary.json"])
    require(summary["schema"] == "CRM_R1_REVISION_v17_R07", "Unsupported saved-text source")
    evidence = json.loads(payload[prefix + "canonical_evidence.private.json"])
    texts = json.loads(payload[prefix + "comparisons.private.json"])
    require(len(evidence) == len(texts) == expected_count, "Unexpected candidate/text census")
    by_id = {e["evidence_id"]: e for e in evidence}
    by_text_id = {t["evidence_id"]: t for t in texts}
    require(len(by_id) == len(by_text_id) == expected_count, "Duplicate evidence/text identities")
    require(set(by_id) == set(by_text_id), "Evidence/text identity mismatch")
    require(len({e["term_id"] for e in evidence}) == expected_count, "Duplicate term IDs")
    contexts = {
        json.dumps((e["contrast"], e["cohort_name"], e["metadata"]), sort_keys=True)
        for e in evidence
    }
    require(len(contexts) == 1, "Use separate reports for different study contexts")
    for item in texts:
        source = by_id[item["evidence_id"]]
        require(item["term_id"] == source["term_id"], "Text term identity mismatch")
        for lhs, rhs in (("canonical_q_value", "q_value"), ("canonical_nes", "nes")):
            require(
                math.isclose(item[lhs], source[rhs], rel_tol=0, abs_tol=0),
                "Saved text/source statistic mismatch",
            )
        require(isinstance(item["unaudited_text"], str), "Saved text must be a string")
    return evidence, texts, sha(raw_archive), exports


def new_folder(root, name):
    root = Path(root).resolve()
    require(root.is_dir(), "Use the existing CRM_R1 data root")
    out = root / "output/revision_v17" / (name + datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ"))
    out.mkdir(parents=True, exist_ok=False)
    return out


def prepare(root, source):
    protocol = json.loads(PROTOCOL.read_bytes())
    evidence, texts, archive_sha, _ = load_corpus(source, protocol["candidate_count"])
    rows = [
        {
            "term_id": e["term_id"],
            "term_name": e["term_name"],
            "source": "fgsea",
            "stat": e["nes"],
            "stat_kind": "NES",
            "qval": e["q_value"],
            "direction": e["direction"].lower(),
            "evidence_genes": ";".join(e["leading_edge_genes"]),
        }
        for e in evidence
    ]
    drafts = [
        {"term_id": t["term_id"], "claim_id": t["evidence_id"], "text": t["unaudited_text"]}
        for t in texts
    ]
    first = evidence[0]
    contrast = first["contrast"]
    card = {
        "comparison": f"{contrast['positive_group']} (positive group) versus "
        f"{contrast['reference_group']} (reference group)",
        "condition": first["cohort_name"],
        "tissue": first["metadata"]["tissue"]["value"],
        "study_design": contrast["study_design"],
        "notes": "Previously seen R07 development corpus; user-supplied historical metadata.",
    }
    out = new_folder(root, "tool_reassessment_design_")
    write_tsv(out / "evidence.tsv", rows)
    write_tsv(out / "drafts.private.tsv", drafts)
    write_json(out / "sample_card.json", card)
    write_json(out / "protocol.json", protocol)
    links = [
        {
            "term_id": e["term_id"],
            "evidence_id": e["evidence_id"],
            "original_source": e["source"]["artifact"],
            "original_source_sha256": e["source"]["sha256"],
            "original_locator": e["source"]["locator"],
        }
        for e in evidence
    ]
    write_tsv(out / "original_links.private.tsv", links)
    hashes = implementation_hashes()
    for relative in hashes:
        snapshot = out / "implementation_snapshot" / relative
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        snapshot.write_bytes((REPO / relative).read_bytes())
    outputs = {
        p.relative_to(out).as_posix(): sha(p.read_bytes()) for p in out.rglob("*") if p.is_file()
    }
    write_json(
        out / "DESIGN.json",
        {
            "schema": protocol["schema"],
            "scope": protocol["scope"],
            "source_path": str(Path(source).resolve()),
            "source_sha256": archive_sha,
            "ruleset": RULESET,
            "implementation_sha256": hashes,
            "outputs": outputs,
            "candidate_count": len(evidence),
            "independent_validation": False,
            "original_inputs_modified": False,
        },
    )
    return out


def load_design(folder):
    folder = Path(folder).resolve()
    design = json.loads((folder / "DESIGN.json").read_bytes())
    require(design["ruleset"] == RULESET, "Ruleset changed; prepare a new development design")
    require(
        design["implementation_sha256"] == implementation_hashes(),
        "Implementation changed; prepare a new development design",
    )
    for name, expected in design["outputs"].items():
        require((folder / name).resolve().is_relative_to(folder), "Unsafe design path")
        require(sha((folder / name).read_bytes()) == expected, "Prepared input changed: " + name)
    protocol = json.loads((folder / "protocol.json").read_bytes())
    evidence, texts, source_sha, _ = load_corpus(design["source_path"], protocol["candidate_count"])
    require(source_sha == design["source_sha256"], "Original source ZIP changed")
    return folder, design, protocol, evidence, texts


def run(root, folder):
    folder, design, protocol, evidence, texts = load_design(folder)
    out = new_folder(root, "tool_reassessment_")
    result = run_pipeline(
        RunConfig(
            evidence_table=str(folder / "evidence.tsv"),
            sample_card=str(folder / "sample_card.json"),
            claims_file=str(folder / "drafts.private.tsv"),
            outdir=str(out / "source_report"),
            workflow="source",
            q_threshold=protocol["q_threshold"],
        )
    )
    records = [
        json.loads(s) for s in Path(result.artifacts["report_jsonl"]).read_text().splitlines()
    ]
    prior = {t["term_id"]: t for t in texts}
    paired = []
    flags = Counter()
    dispositions = Counter()
    coverage = Counter()
    for record in records:
        (checked,) = record["submitted_reviews"]
        original = prior[record["evidence"]["term_id"]]
        require(checked["text"] == original["unaudited_text"], "Original draft wording changed")
        require(
            checked["text_sha256"] == sha(original["unaudited_text"].encode()),
            "Original draft text hash changed",
        )
        flags.update(f["code"] for f in checked["findings"])
        dispositions[checked["prose_disposition"]] += 1
        coverage[checked["numeric_coverage"]] += 1
        paired.append(
            {
                "term_id": record["evidence"]["term_id"],
                "text_sha256": checked["text_sha256"],
                "original_numeric_gate_status": original["numeric_gate_status"],
                "new_numeric_coverage": checked["numeric_coverage"],
                "limited_check_status": checked["limited_checks_status"],
                "prose_disposition": checked["prose_disposition"],
                "flag_codes": ";".join(sorted({f["code"] for f in checked["findings"]})),
            }
        )
    require(len(paired) == design["candidate_count"] == len(evidence), "Output census changed")
    require(
        sha(Path(design["source_path"]).read_bytes()) == design["source_sha256"],
        "Source changed during reporting",
    )
    write_tsv(out / "paired_checks.private.tsv", paired)
    meta = json.loads(Path(result.meta_path).read_bytes())
    summary = {
        "schema": "saved-text-tool-reassessment-results/1",
        "status": "COMPLETE",
        "scope": protocol["scope"],
        "candidate_count": len(paired),
        "candidate_retained_count": len(records),
        "original_texts_unchanged": True,
        "original_inputs_modified": False,
        "design_dir": str(folder),
        "design_sha256": sha((folder / "DESIGN.json").read_bytes()),
        "source_archive_sha256": design["source_sha256"],
        "ruleset": RULESET,
        "selected_source_statement_count": meta["selected_count"],
        "numeric_coverage_counts": dict(coverage),
        "prose_disposition_counts": dict(dispositions),
        "finding_occurrence_counts": dict(flags),
        "model_calls": meta["model_calls"],
        "paid_model_calls": 0,
        "new_expert_ratings": 0,
        "semantic_accuracy_estimated": False,
        "biological_accuracy_estimated": False,
        "independent_validation": False,
        "full_audit_superiority_estimated": False,
        "upstream_expression_or_enrichment_refit": False,
        "interpretation": (
            "Limited source checks on exposed saved text. Flags are not expert truth labels."
        ),
        "outdir": str(out),
        "report_html": result.artifacts["report_html"],
    }
    write_json(out / "SUMMARY.json", summary)
    write_json(
        out / "EXPORT_SHA256.json",
        {p.relative_to(out).as_posix(): sha(p.read_bytes()) for p in out.rglob("*") if p.is_file()},
    )
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--source", help="Exact original R07 or submission collection ZIP")
    parser.add_argument("--design", help="An existing prepared design directory")
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    try:
        require(args.prepare or args.run, "Specify --prepare and/or --run")
        require(
            not (args.prepare and args.design), "Choose a new preparation or an existing design"
        )
        if args.prepare:
            require(args.source is not None, "--prepare requires an exact --source ZIP")
            folder = prepare(args.data_root, args.source)
            print(
                json.dumps(
                    {
                        "status": "DEVELOPMENT_DESIGN_RECORDED",
                        "design_dir": str(folder),
                        "independent_validation": False,
                    },
                    indent=2,
                )
            )
        else:
            require(args.design is not None, "--run requires --design or --prepare")
            folder = Path(args.design)
        if args.run:
            print(json.dumps(run(args.data_root, folder), indent=2))
    except (ValueError, KeyError, OSError, zipfile.BadZipFile) as error:
        raise SystemExit("[STOP] " + str(error)) from error


if __name__ == "__main__":
    main()
