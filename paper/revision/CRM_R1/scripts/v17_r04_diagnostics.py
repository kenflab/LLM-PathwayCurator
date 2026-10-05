"""Authenticate and replay returned R03 outcomes; never repair or overwrite them."""

from __future__ import annotations

import base64
import hashlib
import io
import statistics
import zipfile
from collections import Counter
from pathlib import Path

from v17_r03 import score_case
from v17_revision import REPO, require

from llm_pathway_curator.contract_v17.atomic import aggregate, make_request, parse_response
from llm_pathway_curator.contract_v161.checks import prepare
from llm_pathway_curator.contract_v161.models import ASPECTS, ModelConfig
from llm_pathway_curator.contract_v161.prompt import digest, strict_json
from llm_pathway_curator.contract_v161.runtime import TechnicalError


def required_source_files():
    files = [
        REPO / "paper/revision/CRM_R1/scripts" / name
        for name in ("v17_r03.py", "62_revision_r03.py", "v17_revision.py")
    ]
    files += [
        REPO / "paper/revision/CRM_R1/config/r03_development_plan.json",
        REPO / "tests/fixtures/crm_r1_v16_1/cases_v161.json",
    ]
    for folder in ("contract_v161", "contract_v17"):
        files.extend(sorted((REPO / "src/llm_pathway_curator" / folder).glob("*.py")))
    return files


def diagnose_r03(raw, cases):
    """Recompute the unchanged R03 parser, aggregation and scoring on every case."""
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        names = archive.namelist()
        require(len(names) == len(set(names)), "Duplicate returned ZIP paths")
        require(
            all(not Path(n).is_absolute() and ".." not in Path(n).parts for n in names),
            "Unsafe returned ZIP path",
        )
        manifest = strict_json(archive.read("EXPORT_SHA256.json"))
        require(
            set(manifest) == set(names) - {"EXPORT_SHA256.json"},
            "Returned ZIP has unlisted or missing files",
        )
        for name, expected in manifest.items():
            require(
                hashlib.sha256(archive.read(name)).hexdigest() == expected,
                f"Returned R03 member hash mismatch: {name}",
            )
        output_manifest = strict_json(archive.read("results/OUTPUT_MANIFEST.json"))
        result_names = {n.removeprefix("results/") for n in names if n.startswith("results/")}
        require(
            set(output_manifest) == result_names - {"OUTPUT_MANIFEST.json"},
            "Returned R03 output manifest coverage differs",
        )
        for name, expected in output_manifest.items():
            require(manifest["results/" + name] == expected, "R03 output manifest differs")
        census = strict_json(archive.read("results/canonical_census.private.json"))
        require(
            census
            == [
                {"case_id": c["case_id"], "evidence": c["evidence"], "claim": c["claim"]}
                for c in cases
            ],
            "Returned R03 case census differs from the original fixed controls",
        )
        source_rows = strict_json(archive.read("results/INPUT_MANIFEST.private.json"))
        recorded_sources = {}
        for row in source_rows:
            marker = "/LLM-PathwayCurator/"
            if marker in row["path"]:
                relative = row["path"].split(marker, 1)[1]
                require(relative not in recorded_sources, "Duplicate recorded source path")
                recorded_sources[relative] = row["sha256"]
        for source in required_source_files():
            relative = str(source.relative_to(REPO))
            require(
                recorded_sources.get(relative) == hashlib.sha256(source.read_bytes()).hexdigest(),
                f"R03 replay source changed: {relative}",
            )
        model = ModelConfig.model_validate(
            strict_json(archive.read("results/model_config.private.json"))
        )
        plan = strict_json(archive.read("results/development_plan.json"))
        require(
            plan
            == strict_json(
                (REPO / "paper/revision/CRM_R1/config/r03_development_plan.json").read_bytes()
            ),
            "Returned R03 scoring protocol differs",
        )
        results = strict_json(archive.read("results/audit_results.private.json"))
        scores = strict_json(archive.read("results/scores.private.json"))
        require(
            [r["case_id"] for r in results] == [c["case_id"] for c in cases],
            "Returned R03 results reordered or incomplete",
        )
        require(
            [r["case_id"] for r in scores] == [c["case_id"] for c in cases],
            "Returned R03 scores reordered or incomplete",
        )
        index = strict_json(archive.read("results/request_index.private.json"))
        expected_index, rows, replayed_scores = [], [], []
        for case, saved in zip(cases, results, strict=True):
            prepared, _, _ = prepare(case["evidence"], case["claim"])
            require(
                saved["deterministic_result"] == prepared, "Returned deterministic result differs"
            )
            slots = saved["aspect_results"]
            require(
                set(slots)
                == (set() if prepared["contract_status"] == "VIOLATION" else set(ASPECTS)),
                "Returned aspect census differs",
            )
            for aspect in ASPECTS if slots else ():
                request = make_request(case["evidence"], case["claim"], model, aspect)
                key = request["key"]
                expected_index.append(
                    {"case_id": case["case_id"], "aspect": aspect, "request_key": key}
                )
                prefix = "cache/" + key + "/"
                require(
                    strict_json(archive.read(prefix + "request.json")) == request,
                    "Returned request differs from original source and inputs",
                )
                require(
                    strict_json(archive.read(prefix + "STARTED.json"))["request_key"] == key,
                    "Returned request-start key differs",
                )
                record = strict_json(archive.read(prefix + "FIRST_RESULT.json"))
                require(record["request_key"] == key, "Returned first outcome key differs")
                require(
                    strict_json(archive.read(prefix + "FIRST_RESULT_SHA256.json"))
                    == {"sha256": digest(record)},
                    "Returned first outcome seal differs",
                )
                require(
                    all(slots[aspect].get(k) == value for k, value in record.items()),
                    "Returned aspect result detached from first outcome",
                )
                response = base64.b64decode(record["raw_response_base64"], validate=True)
                require(
                    hashlib.sha256(response).hexdigest() == record["raw_response_sha256"],
                    "Returned raw response hash differs",
                )
                identity = {
                    "model": model.model,
                    "digest": model.model_digest,
                    "version": model.server_version,
                }
                require(
                    all(
                        record["observations"].get(k) == identity
                        for k in ("identity_before", "identity_after")
                    ),
                    "Returned backend identity differs",
                )
                try:
                    parsed = parse_response(response, request)
                    require(
                        record["status"] == "SUCCEEDED"
                        and record["parsed_review"] == parsed.model_dump(),
                        "Returned validated outcome differs from replay",
                    )
                    valid, error_code = True, None
                except TechnicalError as error:
                    require(
                        record["status"] == "TECHNICAL_ERROR"
                        and record["parsed_review"] is None
                        and record["error_code"] == error.code,
                        "Returned invalid outcome was changed or repaired",
                    )
                    valid, error_code = False, error.code
                wire = strict_json(response)
                unvalidated = strict_json(wire["response"])
                rows.append(
                    {
                        "case_id": case["case_id"],
                        "aspect": aspect,
                        "request_key": key,
                        "raw_response_sha256": record["raw_response_sha256"],
                        "validated": valid,
                        "original_status": record["status"],
                        "error_code": error_code,
                        "done_reason": wire.get("done_reason"),
                        "eval_count": wire.get("eval_count"),
                        "total_duration_ns": wire.get("total_duration"),
                        "claim_text": case["claim"]["text"],
                        "raw_unvalidated_review": unvalidated,
                        "resolved_sentences": [
                            s
                            for s in request["envelope"]["payload"]["sentences"]
                            if s["id"] in unvalidated.get("sentence_ids", [])
                        ],
                        "supplied_facts": request["envelope"]["payload"]["facts"],
                    }
                )
            outcome = aggregate(prepared, slots)
            require(saved["outcome"] == outcome, "Returned R03 aggregate differs")
            require(
                strict_json(archive.read(f"results/cases/{case['case_id']}.private.json")) == saved,
                "Returned per-case checkpoint differs",
            )
            replayed_scores.append(score_case(case, prepared, outcome, plan, slots))
        require(index == expected_index, "Returned R03 request census/order differs")
        require(len({r["request_key"] for r in rows}) == len(rows), "Duplicate returned requests")
        require(scores == replayed_scores, "Returned R03 scoring differs from unchanged replay")
        faithful = [s for s in scores if s["faithful_control"]]
        incorrect = [s for s in scores if not s["faithful_control"]]
        counts = {
            "cases": len(cases),
            "candidate_retained_count": sum(s["candidate_retained"] for s in scores),
            "planned_semantic_requests": len(rows),
            "known_control_matches": sum(s["expected_match"] for s in scores),
            "execution_status_counts": dict(Counter(s["execution_status"] for s in scores)),
            "forbidden_concern_case_count": sum(bool(s["forbidden_concerns"]) for s in scores),
            "polarity_mismatch_case_count": sum(bool(s["polarity_mismatches"]) for s in scores),
            "faithful_controls": {
                "total": len(faithful),
                "incomplete": sum(s["execution_status"] == "INCOMPLETE" for s in faithful),
                "not_run": sum(s["execution_status"] == "NOT_RUN" for s in faithful),
                "eligible": sum(s["interpretation_eligible"] for s in faithful),
                "complete_but_withheld": sum(
                    s["complete"] and not s["interpretation_eligible"] for s in faithful
                ),
            },
            "incorrect_controls": {
                "total": len(incorrect),
                "incomplete": sum(s["execution_status"] == "INCOMPLETE" for s in incorrect),
                "not_run": sum(s["execution_status"] == "NOT_RUN" for s in incorrect),
                "eligible": sum(s["interpretation_eligible"] for s in incorrect),
            },
        }
        original_summary = strict_json(archive.read("results/summary.json"))
        require(
            all(original_summary[k] == v for k, v in counts.items()),
            "Returned R03 summary disagrees with unchanged replay",
        )
    invalid = [r for r in rows if not r["validated"]]
    deterministic = [s for s in scores if s["semantic_status"].startswith("NOT_RUN_")]
    semantic_scores = [s for s in scores if s not in deterministic]
    tokens = [r["eval_count"] for r in rows if isinstance(r["eval_count"], int)]
    summary = {
        "schema": "CRM_R1_RETURNED_R03_DIAGNOSTIC",
        "scope": original_summary["scope"],
        "archive_sha256": hashlib.sha256(raw).hexdigest(),
        "exported_files_hash_checked": len(manifest),
        "replay_source_files_checked": len(required_source_files()),
        "all_original_scores_reproduced": True,
        "original_records_modified": False,
        "new_model_calls": 0,
        "backend_receipts_authenticated": len(rows),
        "validated_atomic_responses": sum(r["validated"] for r in rows),
        "invalid_atomic_responses": len(invalid),
        "done_reason_counts": dict(Counter(r["done_reason"] for r in rows)),
        "completion_tokens": {
            "min": min(tokens),
            "max": max(tokens),
            "limit": model.options.num_predict,
        },
        "summed_model_duration_seconds": sum(r["total_duration_ns"] or 0 for r in rows) / 1e9,
        "median_model_duration_seconds": statistics.median(
            r["total_duration_ns"] or 0 for r in rows
        )
        / 1e9,
        "deterministic_cases": len(deterministic),
        "deterministic_matches": sum(s["expected_match"] for s in deterministic),
        "semantic_cases": len(semantic_scores),
        "semantic_cases_complete": sum(s["complete"] for s in semantic_scores),
        "semantic_cases_meeting_original_rules": sum(s["expected_match"] for s in semantic_scores),
        "invalid_raw_mode_patterns": dict(
            Counter(
                r["raw_unvalidated_review"]["assertion_mode"]
                + "/"
                + (
                    "NONEMPTY_SENTENCE_IDS"
                    if r["raw_unvalidated_review"]["sentence_ids"]
                    else "EMPTY_SENTENCE_IDS"
                )
                for r in invalid
            )
        ),
        "r03_original_summary_counts": counts,
        "biological_or_independent_semantic_accuracy_estimated": False,
        "duration_is_summed_model_time_not_end_to_end_wall_time": True,
    }
    return summary, rows, scores, model
