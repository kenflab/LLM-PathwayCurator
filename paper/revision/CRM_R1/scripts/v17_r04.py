"""Diagnose saved R03 data, then optionally test source-only reading on known cases.

No new audit verdicts, enrichment, external validation, ratings or expert contact.
Source location accuracy is a different task from R03 audit accuracy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from v17_r03 import export_live_results, pin_model
from v17_r04_diagnostics import diagnose_r03, required_source_files
from v17_revision import (
    REPO,
    Inputs,
    data_path,
    git_snapshot,
    output_directory,
    require,
    write_json,
)

from llm_pathway_curator.claim_locator_r04 import METHOD_ID, PROMPT_VERSION
from llm_pathway_curator.claim_locator_r04.locator import (
    LocatorResponse,
    implementation_digest,
    make_request,
    resolve,
    sentence_spans,
)
from llm_pathway_curator.claim_locator_r04.runtime import CallBudget, FirstOutcomeStore
from llm_pathway_curator.contract_v161.checks import prepare
from llm_pathway_curator.contract_v161.models import ASPECTS, GenerationOptions
from llm_pathway_curator.contract_v161.prompt import strict_json
from llm_pathway_curator.contract_v161.runtime import TechnicalError

PLAN = REPO / "paper/revision/CRM_R1/config/r04_development_plan.json"
VERSION = "CRM_R1_REVISION_v17_R04"


def validate_plan(plan, cases):
    require(plan["schema"] == "CRM_R1_R04_DEVELOPMENT_PLAN", "Wrong R04 protocol")
    require([c["case_id"] for c in cases] == plan["case_order"], "R04 control census/order changed")
    ids, deterministic = [], []
    for case in cases:
        result, _, _ = prepare(case["evidence"], case["claim"])
        require(result["canonical_status"] == "VALID", "Invalid canonical development control")
        (deterministic if result["contract_status"] == "VIOLATION" else ids).append(case["case_id"])
    require(
        ids == plan["semantic_case_ids"] and deterministic == plan["deterministic_case_ids"],
        "R04 semantic/deterministic census differs",
    )
    require(set(ids) == set(plan["expected_source_locations"]), "R04 annotation census differs")
    for case in cases:
        if case["case_id"] not in ids:
            continue
        expected = LocatorResponse.model_validate(
            plan["expected_source_locations"][case["case_id"]]
        )
        # Validate all expected references without constructing a model prompt.
        supplied = {s["id"] for s in sentence_spans(case["claim"]["text"])}
        for aspect in ASPECTS:
            for selected in getattr(expected, aspect).model_dump().values():
                require(set(selected) <= supplied, "R04 annotation refers outside source text")


def load_sources(bundle, inputs):
    plan = strict_json(PLAN.read_bytes())
    inputs.add(PLAN)
    fixture = REPO / plan["fixture"]
    require(inputs.add(fixture) == plan["fixture_sha256"], "R04 fixture hash mismatch")
    cases = strict_json(fixture.read_bytes())["cases"]
    validate_plan(plan, cases)
    archive = bundle / plan["returned_r03_snapshot"]
    require(archive.resolve().is_relative_to(bundle), "R03 snapshot escapes bundle")
    require(inputs.add(archive) == plan["returned_r03_sha256"], "Returned R03 ZIP hash mismatch")
    for source in required_source_files():
        inputs.add(source)
    diagnosis, raw_rows, original_scores, model = diagnose_r03(archive.read_bytes(), cases)
    require(
        diagnosis["scope"] == "known_development_controls_not_independent_validation",
        "R04 requires the authenticated original live R03 run",
    )
    return plan, cases, diagnosis, raw_rows, original_scores, model


def score_locations(case_id, record, expected):
    complete = record["status"] == "SUCCEEDED"
    mismatches = {}
    if complete:
        actual = LocatorResponse.model_validate(record["parsed_locations"]).model_dump()
        for aspect in ASPECTS:
            for kind in ("asserted", "limited", "uncertain"):
                observed, required = set(actual[aspect][kind]), set(expected[aspect][kind])
                if observed != required:
                    mismatches[aspect + "/" + kind] = {
                        "expected": sorted(required),
                        "observed": sorted(observed),
                        "missing": sorted(required - observed),
                        "extra": sorted(observed - required),
                    }
    return {
        "case_id": case_id,
        "complete": complete,
        "source_locations_match": complete and not mismatches,
        "mismatches": mismatches,
        "execution_status": record["status"],
        "interpretation_eligibility_assessed": False,
    }


def run_r04(
    root,
    source_bundle,
    *,
    live=False,
    output=None,
    calls=16,
    seconds=600,
    model=None,
    transport=None,
):
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    output = output or Path(root) / "output/revision_v17" / f"r04_{stamp}"
    root, out = output_directory(root, output)
    bundle = Path(source_bundle).expanduser().resolve(strict=True)
    require(bundle.is_relative_to(root / "input"), "Place R04 bundle under CRM_R1/input/")
    require(live or (model is None and transport is None), "Model transport requires live mode")
    injected = transport is not None
    budget, inputs = CallBudget(calls, seconds), Inputs()
    for source in [
        Path(__file__),
        Path(__file__).with_name("63_revision_r04.py"),
        Path(__file__).with_name("v17_r04_diagnostics.py"),
    ]:
        inputs.add(source)
    for file in sorted((REPO / "src/llm_pathway_curator/claim_locator_r04").glob("*.py")):
        inputs.add(file)
    plan, cases, diagnosis, raw_rows, old_scores, historical = load_sources(bundle, inputs)
    out.mkdir(parents=True)
    write_json(
        out / "STARTED.json",
        {
            "schema": VERSION,
            "method_id": METHOD_ID,
            "git": git_snapshot(),
            "mode": "software_transport_test"
            if injected
            else "live_source_only_probe"
            if live
            else "offline_diagnosis_and_preparation",
            "source_bundle": str(bundle),
            "implementation_sha256": implementation_digest(),
            "new_request_limit": budget.limit,
            "wall_budget_seconds": budget.seconds,
        },
    )
    print(f"[R04] output: {out}", flush=True)
    try:
        write_json(out / "r03_diagnostic_summary.json", diagnosis)
        write_json(out / "r03_raw_response_diagnostics.private.json", raw_rows)
        write_json(out / "r03_replayed_scores.private.json", old_scores)
        write_json(out / "development_plan.json", plan)
        write_json(
            out / "canonical_census.private.json",
            [
                {"case_id": c["case_id"], "evidence": c["evidence"], "claim": c["claim"]}
                for c in cases
            ],
        )
        if live:
            if transport is None:
                require(model is None, "Use the pinned backend, not an unverified supplied model")
                model, transport = pin_model(historical, plan)
            else:
                require(model is not None, "Injected transport needs a model; software tests only")
            write_json(out / "model_config.private.json", model.model_dump())
        else:
            model = historical.model_copy(deep=True)
            model.options = GenerationOptions.model_validate(plan["generation_options"])
            model.timeout_seconds = plan["timeout_seconds"]
            write_json(
                out / "model_config.planned.json",
                {"identity_is_historical_not_currently_verified": True, **model.model_dump()},
            )
        store = (
            FirstOutcomeStore(data_path(root, "output/revision_v17/r04_locator_cache"))
            if live
            else None
        )
        results, scores, request_index = [], [], []
        for index, case in enumerate(cases, 1):
            prepared, _, _ = prepare(case["evidence"], case["claim"])
            if case["case_id"] in plan["deterministic_case_ids"]:
                record, resolved = {"status": "NOT_REQUESTED_DETERMINISTIC_CONTROL"}, None
            else:
                request = make_request(case["claim"]["text"], model)
                request_index.append(
                    {
                        "case_id": case["case_id"],
                        "stage": "source_only_locator",
                        "request_key": request["key"],
                    }
                )
                if live:
                    print(
                        f"[R04] {index}/20 {case['case_id']} source-only; new requests "
                        f"{budget.calls}/{budget.limit}",
                        flush=True,
                    )
                    record, hit, called = store.run(
                        request, transport, allow_new=budget.available()
                    )
                    budget.calls += called
                    record = record | {"cache_hit": hit}
                else:
                    record = {
                        "status": "NOT_RUN",
                        "request_key": request["key"],
                        "parsed_locations": None,
                    }
                    folder = out / "planned_requests"
                    folder.mkdir(exist_ok=True)
                    write_json(folder / f"{case['case_id']}.json", request)
                resolved = (
                    resolve(LocatorResponse.model_validate(record["parsed_locations"]), request)
                    if record["status"] == "SUCCEEDED"
                    else None
                )
                scores.append(
                    score_locations(
                        case["case_id"], record, plan["expected_source_locations"][case["case_id"]]
                    )
                )
            result = {
                "case_id": case["case_id"],
                "deterministic_result_preserved": prepared,
                "statistical_candidate_retained": prepared["statistical_candidate_retained"],
                "source_locator": record,
                "resolved_source_sentences": resolved,
                "interpretation_eligibility_assessed": False,
                "audit_verdict_produced": False,
                "biological_truth": "NOT_ESTABLISHED",
            }
            results.append(result)
            folder = out / "cases"
            folder.mkdir(exist_ok=True)
            write_json(folder / f"{case['case_id']}.private.json", result)
        inputs.verify()
        matched = sum(s["source_locations_match"] for s in scores)
        passed = live and matched == len(plan["semantic_case_ids"])
        summary = {
            "schema": VERSION,
            "scope": "injected_software_transport_not_real_llm" if injected else plan["scope"],
            "status": "SOURCE_PROBE_DEVELOPMENT_MATCH"
            if passed
            else "SOURCE_PROBE_COMPLETE_WITH_FINDINGS"
            if live
            else "OFFLINE_READY_NOT_LIVE_TESTED",
            "outdir": str(out),
            "results_archive": str(out.with_suffix(".zip")) if live else None,
            "cases": len(results),
            "candidate_retained_count": sum(r["statistical_candidate_retained"] for r in results),
            "deterministic_controls_preserved": len(plan["deterministic_case_ids"]),
            "source_only_cases": len(scores),
            "planned_model_requests": len(request_index),
            "new_model_requests": budget.calls if not injected else 0,
            "new_transport_attempts": budget.calls,
            "model_backend_authenticated": bool(live and not injected),
            "source_only_responses_valid": sum(s["complete"] for s in scores),
            "source_location_matches": matched,
            "source_location_mismatch_cases": sum(
                s["complete"] and not s["source_locations_match"] for s in scores
            ),
            "source_only_execution_status_counts": dict(
                Counter(s["execution_status"] for s in scores)
            ),
            "r03_original_scores_reproduced": diagnosis["all_original_scores_reproduced"],
            "r03_exported_files_checked": diagnosis["exported_files_hash_checked"],
            "original_r03_records_modified": False,
            "input_hashes_unchanged": True,
            "audit_verdicts_produced": False,
            "interpretation_eligibility_assessed": False,
            "biological_or_independent_semantic_accuracy_estimated": False,
            "expert_ratings_requested": False,
            "external_expression_outcomes_loaded": False,
            "natural_text_comparison_frozen": False,
            "prompt_version": PROMPT_VERSION,
            "next_step": "separate_evidence_comparison_development"
            if passed
            else "review_all_saved_source_probe_outcomes"
            if live
            else "run_one_bounded_source_probe",
        }
        write_json(out / "source_probe_results.private.json", results)
        write_json(out / "source_probe_scores.private.json", scores)
        write_json(out / "request_index.private.json", request_index)
        write_json(out / "INPUT_MANIFEST.private.json", list(inputs.files.values()))
        write_json(out / "summary.json", summary)
        write_json(
            out / "OUTPUT_MANIFEST.json",
            {
                str(p.relative_to(out)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted(out.rglob("*"))
                if p.is_file()
            },
        )
        if live:
            export_live_results(out, store.root, request_index)
        inputs.verify()
        print(json.dumps(summary, indent=2), flush=True)
        return summary
    except Exception as error:
        write_json(
            out / "FAILED.json",
            {
                "schema": VERSION,
                "error": str(error),
                "new_model_requests": budget.calls if not injected else 0,
                "new_transport_attempts": budget.calls,
                "retained_case_checkpoints": len(list(out.glob("cases/*.json"))),
            },
        )
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=os.environ.get("CRM_R1_DATA_ROOT"))
    parser.add_argument("--source-bundle", type=Path, required=True)
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--outdir", type=Path)
    parser.add_argument("--max-new-requests", type=int, default=16)
    parser.add_argument("--wall-budget-seconds", type=float, default=600)
    args = parser.parse_args()
    if args.data_root is None:
        parser.error("Set CRM_R1_DATA_ROOT or use --data-root")
    try:
        run_r04(
            args.data_root,
            args.source_bundle,
            live=args.live,
            output=args.outdir,
            calls=args.max_new_requests,
            seconds=args.wall_budget_seconds,
        )
    except (OSError, ValueError, KeyError, TechnicalError) as error:
        print(f"[R04 STOP] {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
