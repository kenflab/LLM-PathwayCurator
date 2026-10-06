"""One local proposal per frozen candidate, with unchanged-prose comparators.

The existing explicit-number checker is a limited comparator, not a semantic
judge or validated full audit. No paid backends, new expert reviews or P3 grades.
"""

# The existing checkout supplies the package without changing editable installs.
# ruff: noqa: E402

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import json
import os
import sys
import time
import zipfile
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / "src"))

from v17_revision import (
    Inputs,
    declared_path,
    export_adapter,
    output_directory,
    require,
    sha,
    write_json,
)

from revision_tools.legacy_contract.models import Evidence, ModelConfig
from revision_tools.legacy_contract.prompt import canonical_json, digest, strict_json
from revision_tools.legacy_contract.runtime import (
    AttemptStore,
    OllamaTransport,
    TechnicalError,
    WireResponse,
    now,
    write_new,
)
from revision_tools.legacy_contract.text_checks import (
    explicit_numeric_checks,
    factual_statement,
)

VERSION = "CRM_R1_REVISION_v17_R07"
POLICY = REPO / "paper/revision/CRM_R1/config/r07_natural_text_policy.json"
DESIGN_REL = Path("output/revision_v17/r07_natural_text_design_v1")


def write_tsv(path, rows, fields=None):
    with Path(path).open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fields or list(rows[0]), delimiter="\t", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def code_paths():
    result = [
        Path(__file__),
        Path(__file__).with_name("66_revision_r07.py"),
        Path(__file__).with_name("v17_revision.py"),
        POLICY,
    ]
    result.extend(sorted((REPO / "paper/revision/CRM_R1/experiments/revision_tools/legacy_contract").glob("*.py")))
    result.append(REPO / "paper/revision/CRM_R1/experiments/revision_tools/legacy_pipeline.py")
    return result


def load_design(root):
    folder = Path(root) / DESIGN_REL
    manifest = strict_json((folder / "DESIGN_MANIFEST.json").read_bytes())
    inputs = Inputs()
    for item in manifest["inputs"]:
        require(
            inputs.add(item["path"]) == item["sha256"], "R07 frozen input changed: " + item["path"]
        )
    for relative, expected in manifest["outputs"].items():
        path = folder / relative
        require(path.resolve().is_relative_to(folder.resolve()), "Unsafe design output path")
        require(inputs.add(path) == expected, "R07 design output changed: " + relative)
    require(
        sha(POLICY) == manifest["policy_sha256"],
        "R07 policy changed; do not regenerate existing text",
    )
    policy = strict_json((folder / "policy.json").read_bytes())
    evidence = strict_json((folder / "adapter/evidence.private.json").read_bytes())
    require(len(evidence) == policy["candidate_count"] == 50, "Expected complete 50-term census")
    require(len({r["evidence_id"] for r in evidence}) == 50, "Duplicate canonical ID")
    require(
        [r["term_id"] for r in evidence] == sorted(r["term_id"] for r in evidence),
        "Candidate order changed",
    )
    for record in evidence:
        Evidence.model_validate(record)
        require(
            record["cohort_id"] == policy["cohort"] and record["split_id"] == policy["split"],
            "Wrong frozen cohort/split",
        )
        require(
            inputs.add(record["source"]["artifact"]) == record["source"]["sha256"],
            "Canonical source differs",
        )
    return folder, manifest, policy, evidence, inputs


def prepare_design(root):
    root = Path(root).expanduser().resolve(strict=True)
    folder = root / DESIGN_REL
    if folder.exists():
        require(
            (root / "input").is_dir() and (root / "output").is_dir(), "Invalid CRM_R1 data root"
        )
        require(not root.is_relative_to(REPO), "Data root must stay outside Git")
        load_design(root)
        return {
            "schema": VERSION,
            "status": "DESIGN_ALREADY_FROZEN_UNCHANGED",
            "design": str(folder),
            "model_calls": 0,
        }
    output_directory(root, folder)
    inputs = Inputs()
    for path in code_paths():
        inputs.add(path)
    policy = strict_json(POLICY.read_bytes())
    require(
        policy["schema"] == "CRM_R1_R07_NATURAL_TEXT_POLICY_v1" and policy["candidate_count"] == 50,
        "Wrong R07 policy",
    )
    folder.mkdir(parents=True)
    try:
        adapter = export_adapter(
            root, folder / "adapter", inputs, cohort=policy["cohort"], split=policy["split"]
        )
        write_json(folder / "policy.json", policy)
        inputs.verify()
        summary = {
            "schema": VERSION,
            "status": "FROZEN_FOR_ONE_LOCAL_GENERATION",
            "design": str(folder),
            "candidate_count": 50,
            "adapter": adapter,
            "model_calls": 0,
            "new_expert_ratings": 0,
            "expert_ratings_reused": False,
            "historical_selection_changed": False,
            "independent_validation_performed": False,
            "biological_accuracy_estimated": False,
        }
        write_json(folder / "summary.json", summary)
        outputs = {
            str(p.relative_to(folder)): sha(p) for p in sorted(folder.rglob("*")) if p.is_file()
        }
        write_json(
            folder / "DESIGN_MANIFEST.json",
            {
                "schema": "CRM_R1_R07_DESIGN_v1",
                "policy_sha256": sha(POLICY),
                "inputs": list(inputs.files.values()),
                "outputs": outputs,
            },
        )
        return summary
    except Exception as exc:
        write_json(folder / "FAILED.json", {"error": str(exc), "model_calls": 0})
        raise


def model_config(policy):
    return ModelConfig(
        host=policy["host"],
        model=policy["model"],
        model_digest=policy["model_digest"],
        server_version=policy["server_version"],
        options=policy["generation_options"],
        timeout_seconds=policy["timeout_seconds"],
    )


def make_request(evidence, policy, model, freeze_sha, *, software_test=False):
    supplied = {
        k: evidence[k]
        for k in [
            "evidence_id",
            "cohort_id",
            "cohort_name",
            "split_id",
            "term_id",
            "term_name",
            "nes",
            "q_value",
            "direction",
            "contrast",
        ]
    }
    supplied["tissue"] = evidence.get("metadata", {}).get("tissue", {}).get("value")
    body = {
        "model": model.model,
        "system": policy["system_prompt"],
        "prompt": policy["instruction"] + "\n\nEVIDENCE_RECORD:\n" + canonical_json(supplied),
        "stream": False,
        "options": model.options.model_dump(),
        "keep_alive": "5m",
    }
    envelope = {
        "prompt_version": policy["prompt_version"],
        "freeze_sha256": freeze_sha,
        "evidence_sha256": digest(evidence),
        "model_config": model.model_dump(),
        "body": body,
        "software_transport_test": software_test,
    }
    return {"key": digest(envelope), "envelope": envelope}


def parse_prose(raw, request):
    wire = strict_json(raw)
    if (
        not isinstance(wire, dict)
        or wire.get("model") != request["envelope"]["model_config"]["model"]
    ):
        raise TechnicalError("RESPONSE_MODEL_MISMATCH", "Wrong response model", raw=raw)
    if wire.get("done") is not True or wire.get("done_reason") != "stop":
        raise TechnicalError(
            "GENERATION_NOT_COMPLETE", "Stopped by length or missing stop receipt", raw=raw
        )
    text = wire.get("response")
    if not isinstance(text, str) or not text.strip():
        raise TechnicalError("EMPTY_PROSE", "Missing prose response", raw=raw)
    # Preserve the returned string, including original whitespace and formatting.
    return text


class ProseStore(AttemptStore):
    """Commit every first response/error; an interruption is not permission to retry."""

    def run(self, request, transport, allow_new):
        with self.locked(request) as folder:
            result, seal, started = (
                folder / "FIRST_RESULT.json",
                folder / "FIRST_RESULT_SHA256.json",
                folder / "STARTED.json",
            )
            if started.exists() and not result.exists():
                require(
                    strict_json(started.read_bytes())["request_key"] == request["key"],
                    "Interrupted request key differs",
                )
                record = {
                    "request_key": request["key"],
                    "status": "INTERRUPTED",
                    "text": None,
                    "error_code": "INTERRUPTED",
                    "raw_response_base64": "",
                    "raw_response_sha256": hashlib.sha256(b"").hexdigest(),
                    "observations": {},
                }
                write_new(result, record)
                write_new(seal, {"sha256": digest(record)})
            if result.exists():
                record = strict_json(result.read_bytes())
                require(
                    seal.exists() and strict_json(seal.read_bytes()) == {"sha256": digest(record)},
                    "First outcome seal differs",
                )
                require(record["request_key"] == request["key"], "First outcome key differs")
                raw = base64.b64decode(record["raw_response_base64"], validate=True)
                require(
                    hashlib.sha256(raw).hexdigest() == record["raw_response_sha256"],
                    "First raw response changed",
                )
                require(
                    record["status"] in {"SUCCEEDED", "TECHNICAL_ERROR", "INTERRUPTED"},
                    "Unknown first outcome status",
                )
                if record["status"] == "SUCCEEDED":
                    require(
                        parse_prose(raw, request) == record["text"],
                        "Saved prose differs from wire response",
                    )
                    self.validate_identity(record["observations"], request)
                else:
                    require(record["text"] is None, "Failed response contains accepted prose")
                return record, False
            if not allow_new:
                return {
                    "request_key": request["key"],
                    "status": "NOT_RUN_BUDGET",
                    "text": None,
                    "error_code": "BUDGET_LIMIT",
                }, False
            write_new(started, {"request_key": request["key"], "started_utc": now()})
            raw, observations, text, error = b"", {}, None, None
            try:
                response = transport(request)
                if not isinstance(response, WireResponse) or not isinstance(response.raw, bytes):
                    raise TechnicalError("TRANSPORT_RESPONSE_INVALID", "Expected exact wire bytes")
                raw, observations = response.raw, response.observations
                self.validate_identity(observations, request)
                text = parse_prose(raw, request)
            except TechnicalError as exc:
                error, raw = exc, exc.raw or raw
            except Exception as exc:
                error = TechnicalError("TRANSPORT_ERROR", str(exc))
            record = {
                "request_key": request["key"],
                "ended_utc": now(),
                "status": "SUCCEEDED" if error is None else "TECHNICAL_ERROR",
                "text": text if error is None else None,
                "error_code": error.code if error else None,
                "error_message": str(error) if error else None,
                "raw_response_base64": base64.b64encode(raw).decode("ascii"),
                "raw_response_sha256": hashlib.sha256(raw).hexdigest(),
                "observations": observations,
            }
            write_new(result, record)
            write_new(seal, {"sha256": digest(record)})
            return record, True

    @staticmethod
    def validate_identity(observations, request):
        model = request["envelope"]["model_config"]
        expected = {
            "model": model["model"],
            "digest": model["model_digest"],
            "version": model["server_version"],
        }
        if any(observations.get(k) != expected for k in ["identity_before", "identity_after"]):
            raise TechnicalError(
                "BACKEND_IDENTITY_MISMATCH", "Local weight/version receipt differs"
            )


def numeric_gate(text, evidence):
    if text is None:
        return {"retained": False, "status": "GENERATION_UNAVAILABLE", "checks": []}
    checks = explicit_numeric_checks(text, Evidence.model_validate(evidence))
    metrics = {r.get("metric") for r in checks}
    mismatch = any(not r["satisfied"] for r in checks)
    covered = {"nes", "q_value"} <= metrics
    return {
        "retained": covered and not mismatch,
        "status": "EXPLICIT_RELATION_MISMATCH"
        if mismatch
        else "EXPLICIT_NES_Q_CHECKED"
        if covered
        else "EXPLICIT_NUMERIC_COVERAGE_INCOMPLETE",
        "checks": checks,
        "semantic_safety_assessed": False,
    }


def compare(evidence, outcomes):
    rows = []
    for record, outcome in zip(evidence, outcomes, strict=True):
        gate = numeric_gate(outcome["text"], record)
        rows.append(
            {
                "evidence_id": record["evidence_id"],
                "term_id": record["term_id"],
                "canonical_q_value": record["q_value"],
                "canonical_nes": record["nes"],
                "generation_status": outcome["status"],
                "request_key": outcome["request_key"],
                "unaudited_text": outcome["text"],
                "numeric_gate_status": gate["status"],
                "numeric_gate_retained": gate["retained"],
                "numeric_checks": gate["checks"],
                "source_template_text": factual_statement(Evidence.model_validate(record)),
                "statistical_candidate_retained": True,
            }
        )
    k = sum(row["numeric_gate_retained"] for row in rows)
    available = sorted(
        [r for r in rows if r["generation_status"] == "SUCCEEDED"],
        key=lambda r: (r["canonical_q_value"], r["term_id"]),
    )
    selected = {r["evidence_id"] for r in available[:k]}
    for row in rows:
        row["matched_q_selected"] = row["evidence_id"] in selected
    counts = {
        "UNAUDITED_LLM_PROSE": len(available),
        "EXPLICIT_NUMERIC_GATE": k,
        "MATCHED_Q_VALUE": len(selected),
        "SOURCE_TEMPLATE": len(rows),
    }
    require(len(selected) == k, "Matched baseline count differs")
    table = [
        {
            "method": method,
            "canonical_candidates": len(rows),
            "reports_available": count,
            "coverage_all_candidates": count / len(rows),
            "biological_accuracy": "NOT_ESTIMATED",
            "source_faithfulness_accuracy": "NOT_ESTIMATED_REQUIRES_SEPARATE_ANNOTATION",
        }
        for method, count in counts.items()
    ]
    return rows, table


def annotation_packet(rows, evidence):
    lookup = {r["evidence_id"]: r for r in evidence}
    packet = []
    for row in rows:
        if row["generation_status"] != "SUCCEEDED":
            continue
        record = lookup[row["evidence_id"]]
        packet.append(
            {
                "evidence_id": row["evidence_id"],
                "text_sha256": hashlib.sha256(row["unaudited_text"].encode()).hexdigest(),
                "unaudited_text": row["unaudited_text"],
                "source_nes": repr(record["nes"]),
                "source_q": repr(record["q_value"]),
                "source_direction": record["direction"],
                "source_cohort": record["cohort_name"],
                "source_comparison": record["contrast"]["comparison_id"],
                "source_artifact": record["source"]["artifact"],
                "source_sha256": record["source"]["sha256"],
                "source_locator": record["source"]["locator"],
                "source_faithfulness": "",
                "error_quote_or_unclear_reason": "",
                "annotator_id": "",
            }
        )
    # The packet does not disclose gate/method decisions, and labels start blank.
    return packet


def export_results(out, cache, outcomes):
    files = {"results/" + str(p.relative_to(out)): p for p in sorted(out.rglob("*")) if p.is_file()}
    for record in outcomes:
        for path in sorted((cache / record["request_key"]).glob("*.json")):
            files["cache/" + record["request_key"] + "/" + path.name] = path
    hashes = {name: sha(path) for name, path in files.items()}
    target = out.with_suffix(".zip")
    with zipfile.ZipFile(target, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, path in files.items():
            raw = path.read_bytes()
            require(hashlib.sha256(raw).hexdigest() == hashes[name], "Result changed during export")
            archive.writestr(name, raw)
        archive.writestr("EXPORT_SHA256.json", json.dumps(hashes, indent=2) + "\n")
    return target


def evaluate_archive(root, archive_path, annotation_path, *, output=None):
    """Use separately entered source-fact labels; never infer a label from a gate."""
    inputs = Inputs()
    inputs.add(archive_path)
    inputs.add(annotation_path)
    inputs.add(Path(__file__))
    with zipfile.ZipFile(archive_path) as archive:
        hashes = strict_json(archive.read("EXPORT_SHA256.json"))
        for name, expected in hashes.items():
            require(
                hashlib.sha256(archive.read(name)).hexdigest() == expected,
                "R07 archive record changed: " + name,
            )
        for name in [
            "results/summary.json",
            "results/comparisons.private.json",
            "results/canonical_evidence.private.json",
            "results/first_outcomes.private.json",
        ]:
            require(name in hashes, "Required archive record is not authenticated: " + name)
        summary = strict_json(archive.read("results/summary.json"))
        rows = strict_json(archive.read("results/comparisons.private.json"))
        evidence = strict_json(archive.read("results/canonical_evidence.private.json"))
        outcomes = strict_json(archive.read("results/first_outcomes.private.json"))
    require(
        summary["schema"] == VERSION and len(rows) == len(evidence) == len(outcomes) == 50,
        "Wrong source census",
    )
    recomputed, _ = compare(evidence, outcomes)
    require(recomputed == rows, "Current implementation differs from the recorded comparison")
    for record in evidence:
        require(
            inputs.add(declared_path(Path(root).resolve(), record["source"]["artifact"]))
            == record["source"]["sha256"],
            "Independent original source changed",
        )
    by_id = {r["evidence_id"]: r for r in rows}
    require(len(by_id) == 50, "Duplicate comparison ID")
    labels = {}
    with Path(annotation_path).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {
            "evidence_id",
            "text_sha256",
            "source_faithfulness",
            "error_quote_or_unclear_reason",
            "annotator_id",
        }
        require(
            reader.fieldnames is not None and required <= set(reader.fieldnames),
            "Annotation headers missing",
        )
        require(
            len(reader.fieldnames) == len(set(reader.fieldnames)), "Duplicate annotation headers"
        )
        seen = set()
        for record in reader:
            require(
                None not in record and all(v is not None for v in record.values()),
                "Malformed annotation row",
            )
            uid = record["evidence_id"]
            require(uid in by_id and uid not in seen, "Unknown/duplicate annotation ID")
            seen.add(uid)
            row = by_id[uid]
            require(
                row["generation_status"] == "SUCCEEDED", "Cannot annotate an unavailable paragraph"
            )
            require(
                record["text_sha256"] == hashlib.sha256(row["unaudited_text"].encode()).hexdigest(),
                "Annotation belongs to different text",
            )
            label = record["source_faithfulness"].strip()
            if not label:
                continue
            require(
                label in {"CHECKED_FACTS_MATCH", "CHECKED_FACT_ERROR", "UNCLEAR"},
                "Unknown source-fact label",
            )
            require(record["annotator_id"].strip(), "Annotation requires author identifier")
            reason = record["error_quote_or_unclear_reason"].strip()
            if label == "CHECKED_FACT_ERROR":
                require(
                    reason and reason in row["unaudited_text"],
                    "Error requires exact quote from original prose",
                )
            if label == "UNCLEAR":
                require(reason, "UNCLEAR requires a reason")
            labels[uid] = label
    selectors = {
        "UNAUDITED_LLM_PROSE": lambda r: r["generation_status"] == "SUCCEEDED",
        "EXPLICIT_NUMERIC_GATE": lambda r: r["numeric_gate_retained"],
        "MATCHED_Q_VALUE": lambda r: r["matched_q_selected"],
    }
    table = []
    for method, selected in selectors.items():
        offered = [r for r in rows if selected(r)]
        n = len(offered)
        errors = sum(labels.get(r["evidence_id"]) == "CHECKED_FACT_ERROR" for r in offered)
        matches = sum(labels.get(r["evidence_id"]) == "CHECKED_FACTS_MATCH" for r in offered)
        unknown = n - errors - matches
        withheld_matches = sum(
            labels.get(r["evidence_id"]) == "CHECKED_FACTS_MATCH" and not selected(r) for r in rows
        )
        table.append(
            {
                "method": method,
                "all_canonical_candidates": 50,
                "reports_offered": n,
                "coverage_all_candidates": n / 50,
                "confirmed_source_fact_errors": errors,
                "confirmed_source_fact_matches": matches,
                "unknown_offered": unknown,
                "source_fact_error_fraction_lower": errors / n if n else None,
                "source_fact_error_fraction_upper": (errors + unknown) / n if n else None,
                "source_fact_matches_withheld": withheld_matches,
                "scope": (
                    "descriptive_author_checked_source_facts_"
                    "not_biological_or_semantic_accuracy"
                ),
            }
        )
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    root, out = output_directory(
        root, output or Path(root) / "output/revision_v17" / f"r07_evaluation_{stamp}"
    )
    out.mkdir(parents=True)
    write_tsv(out / "source_fact_comparison.tsv", table)
    write_json(out / "source_fact_labels.private.json", labels)
    result = {
        "schema": VERSION,
        "status": "SOFTWARE_TRANSPORT_TEST_ONLY"
        if summary["status"] == "SOFTWARE_TRANSPORT_TEST_ONLY"
        else "SOURCE_FACT_COMPARISON_DESCRIPTIVE",
        "annotation_labels": dict(Counter(labels.values())),
        "annotations_supplied_separately": True,
        "annotator_independence_authenticated": False,
        "source_faithfulness_accuracy_inferred_from_gate": False,
        "source_template_accuracy_estimated": False,
        "biological_accuracy_estimated": False,
        "semantic_safety_estimated": False,
        "bootstrap_or_significance_tests_performed": False,
        "model_calls": 0,
        "new_expert_ratings": 0,
        "source_archive_sha256": sha(archive_path),
        "annotation_sha256": sha(annotation_path),
        "outdir": str(out),
        "results_archive": str(out.with_suffix(".zip")),
        "interpretation": (
            "Unknown includes UNCLEAR and unannotated; bounds are partial-information "
            "bounds, not confidence intervals."
        ),
    }
    inputs.verify()
    write_json(out / "summary.json", result)
    write_json(out / "INPUT_MANIFEST.private.json", {"files": list(inputs.files.values())})
    export_results(out, Path(root) / "output/revision_v17/r07_first_prose_cache", [])
    return result


def run_live(root, *, calls=50, seconds=1800, output=None, transport=None):
    require(type(calls) is int and 0 <= calls <= 50, "At most 50 new local requests")
    require(0 < seconds <= 1800, "Wall budget must be positive and <=1800 seconds")
    folder, manifest, policy, evidence, inputs = load_design(root)
    model = model_config(policy)
    injected = transport is not None
    if transport is None:
        transport = OllamaTransport(model)
        transport.identity()  # Read-only local checks before starting any generation.
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    root, out = output_directory(
        root, output or Path(root) / "output/revision_v17" / f"r07_{stamp}"
    )
    out.mkdir(parents=True)
    cache = root / "output/revision_v17/r07_first_prose_cache"
    store = ProseStore(cache)
    freeze_sha = sha(folder / "DESIGN_MANIFEST.json")
    write_json(
        out / "STARTED.json",
        {
            "schema": VERSION,
            "design_sha256": freeze_sha,
            "software_transport_test": injected,
            "max_new_requests": calls,
            "wall_budget_seconds": seconds,
        },
    )
    outcomes, started, new = [], time.monotonic(), 0
    try:
        for index, record in enumerate(evidence, 1):
            request = make_request(record, policy, model, freeze_sha, software_test=injected)
            result, attempted = store.run(
                request, transport, new < calls and time.monotonic() - started < seconds
            )
            new += int(attempted)
            outcomes.append(result)
            print(
                f"[R07] {index}/50 {record['term_id']} {result['status']}; "
                f"new local requests {new}/{calls}",
                flush=True,
            )
        rows, methods = compare(evidence, outcomes)
        write_json(out / "canonical_evidence.private.json", evidence)
        write_json(out / "first_outcomes.private.json", outcomes)
        write_json(out / "comparisons.private.json", rows)
        write_tsv(out / "method_coverage.tsv", methods)
        packet = annotation_packet(rows, evidence)
        write_tsv(
            out / "source_fact_check.optional.private.tsv",
            packet,
            fields=list(packet[0])
            if packet
            else [
                "evidence_id",
                "text_sha256",
                "source_faithfulness",
                "error_quote_or_unclear_reason",
                "annotator_id",
            ],
        )
        write_json(out / "policy.json", policy)
        write_json(out / "model_config.private.json", model.model_dump())
        snapshot = out / "source_snapshot"
        snapshot.mkdir()
        for path in [Path(__file__), POLICY]:
            (snapshot / path.name).write_bytes(path.read_bytes())
        inputs.verify()
        summary = {
            "schema": VERSION,
            "scope": policy["scope"],
            "status": "SOFTWARE_TRANSPORT_TEST_ONLY"
            if injected
            else "NATURAL_TEXT_COMPARISON_COMPLETE"
            if all(r["status"] != "NOT_RUN_BUDGET" for r in outcomes)
            else "NATURAL_TEXT_COMPARISON_PARTIAL",
            "outdir": str(out),
            "results_archive": str(out.with_suffix(".zip")),
            "candidate_count": 50,
            "candidate_retained_count": 50,
            "generation_status_counts": dict(Counter(r["status"] for r in outcomes)),
            "numeric_gate_status_counts": dict(Counter(r["numeric_gate_status"] for r in rows)),
            "new_local_model_requests": new,
            "paid_model_requests": 0,
            "new_expert_ratings": 0,
            "model_backend_authenticated": not injected,
            "expert_ratings_reused_for_new_text": False,
            "new_semantic_reviews": 0,
            "input_hashes_unchanged": True,
            "design_sha256": freeze_sha,
            "source_faithfulness_accuracy_estimated": False,
            "biological_accuracy_estimated": False,
            "source_template_is_simple_baseline": True,
            "full_audit_performance_estimated": False,
            "next_step": (
                "review_first_saved_outputs_then_optional_author_source_fact_check_"
                "before_accuracy_claim"
            ),
        }
        write_json(out / "summary.json", summary)
        write_json(out / "INPUT_MANIFEST.private.json", {"files": list(inputs.files.values())})
        (out / "READOUT_JA.txt").write_text(
            "未監査原文・明示数値チェック・同数q比較・単純な元データテンプレートを出力しました。\n候補50件はすべて保持しています。coverageの分母も50件です。\n数値チェックの通過は文章全体の正しさや安全性を意味しません。\n同じチェックを正解ラベルとして使わず、精度は未推定です。任意の著者による元データ照合表のラベルは空欄です。\n以前のP4評価は新しい文章には転用していません。追加の専門家評価・生物学的検証は未実施です。\nテンプレートは単純baselineです。その数字の一致を独自auditの優位性として扱いません。\n未試行の候補は同じコマンドで続行できます。失敗済み・完了済み候補は再生成しません。\n",
            encoding="utf-8",
        )
        export_results(out, cache, outcomes)
        return summary
    except BaseException as exc:
        write_json(
            out / "FAILED.json",
            {"error_type": type(exc).__name__, "error": str(exc), "new_local_requests": new},
        )
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default=os.environ.get("CRM_R1_DATA_ROOT"), type=Path)
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--evaluate-archive", type=Path)
    parser.add_argument("--annotations", type=Path)
    parser.add_argument("--max-new-requests", type=int, default=50)
    parser.add_argument("--wall-budget-seconds", type=float, default=1800)
    args = parser.parse_args()
    require(args.data_root is not None, "Set CRM_R1_DATA_ROOT or --data-root")
    require(
        not (args.live and args.evaluate_archive),
        "Generation and annotation evaluation are separate runs",
    )
    require(
        bool(args.evaluate_archive) == bool(args.annotations),
        "Evaluation requires both archive and annotations",
    )
    try:
        result = (
            evaluate_archive(args.data_root, args.evaluate_archive, args.annotations)
            if args.evaluate_archive
            else run_live(
                args.data_root, calls=args.max_new_requests, seconds=args.wall_budget_seconds
            )
            if args.live
            else prepare_design(args.data_root)
        )
    except (OSError, ValueError, TechnicalError) as exc:
        print(f"[STOP] {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
