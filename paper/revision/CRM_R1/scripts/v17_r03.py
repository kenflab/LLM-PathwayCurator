"""Prepare/replay known controls, then optionally run the fixed atomic pilot.

No enrichment, external expression analysis, human grading, or expert contact.
No successful historical outcome is fabricated from a failed response.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import os
import sys
import zipfile
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from v17_revision import (
    REPO,
    Inputs,
    data_path,
    git_snapshot,
    output_directory,
    require,
    write_json,
)

from llm_pathway_curator.contract_v17 import METHOD_ID, PROMPT_VERSION
from llm_pathway_curator.contract_v17.atomic import (
    CallBudget,
    FirstOutcomeStore,
    aggregate,
    implementation_digest,
    make_request,
)
from llm_pathway_curator.contract_v161.checks import prepare
from llm_pathway_curator.contract_v161.models import ASPECTS, GenerationOptions, ModelConfig
from llm_pathway_curator.contract_v161.prompt import digest, strict_json
from llm_pathway_curator.contract_v161.runtime import (
    OllamaTransport,
    TechnicalError,
)
from llm_pathway_curator.contract_v161.runtime import (
    parse_response as parse_legacy,
)

PLAN = REPO / "paper/revision/CRM_R1/config/r03_development_plan.json"
VERSION = "CRM_R1_REVISION_v17_R03"


def read_baseline(raw, cases):
    """Authenticate every exported member and replay the original strict parser."""
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        names = archive.namelist()
        require(len(names) == len(set(names)), "Duplicate baseline ZIP paths")
        require(
            all(not Path(n).is_absolute() and ".." not in Path(n).parts for n in names),
            "Unsafe baseline ZIP path",
        )
        expected = strict_json(archive.read("EXPORT_SHA256.json"))
        checked = 0
        for name, sha256 in expected.items():
            require(
                hashlib.sha256(archive.read(name)).hexdigest() == sha256,
                f"Historical baseline hash mismatch: {name}",
            )
            checked += 1
        original_cases = strict_json(archive.read("cases.json"))["cases"]
        require(original_cases == cases, "Fixed cases differ from the historical live inputs")
        run_files = [n for n in names if n.endswith("/audit_results.private.jsonl")]
        require(len(run_files) == 1, "Require exactly one historical baseline run")
        old = [strict_json(line) for line in archive.read(run_files[0]).splitlines() if line]
        require(len(old) == len(cases), "Historical run has an incomplete case census")
        rows = []
        for case, result in zip(cases, old, strict=True):
            require(
                result["raw_evidence_preserved"] == case["evidence"]
                and result["submitted_claim_preserved"] == case["claim"],
                "Historical results reordered or detached from their inputs",
            )
            row = {
                "case_id": case["case_id"],
                "original_execution_status": result["execution_status"],
                "original_semantic_status": result["semantic_status"],
                "original_issues": result["issue_codes"],
                "original_error": result["technical_error"],
            }
            if result.get("request_key"):
                folder = "cache/" + result["request_key"] + "/"
                request = strict_json(archive.read(folder + "request.json"))
                require(
                    digest(request["envelope"]) == request["key"] == result["request_key"],
                    "Historical request key mismatch",
                )
                require(
                    request["envelope"]["canonical_evidence"] == case["evidence"]
                    and request["envelope"]["submitted_claim"] == case["claim"],
                    "Historical request/input mismatch",
                )
                records = [n for n in names if n.startswith(folder) and n.endswith(".result.json")]
                require(len(records) == 1, "Require the unchanged first historical response")
                record = strict_json(archive.read(records[0]))
                response = base64.b64decode(record["raw_response_base64"], validate=True)
                require(
                    hashlib.sha256(response).hexdigest() == record["raw_response_sha256"],
                    "Historical raw response hash mismatch",
                )
                model = request["envelope"]["model_config"]
                for identity in ("identity_before", "identity_after"):
                    observed = record["observations"][identity]
                    require(
                        observed
                        == {
                            "model": model["model"],
                            "digest": model["model_digest"],
                            "version": model["server_version"],
                        },
                        "Historical backend identity mismatch",
                    )
                try:
                    parsed = parse_legacy(response, request)
                    row["replayed_response_valid"] = True
                    row["raw_aspect_verdicts"] = {n: getattr(parsed, n).verdict for n in ASPECTS}
                except TechnicalError as error:
                    row["replayed_response_valid"] = False
                    row["replayed_error"] = {"code": error.code, "detail": str(error)}
                    # Raw model decisions remain visible; never promote them to validated outcomes.
                    row["raw_aspect_verdicts"] = {
                        n: strict_json(strict_json(response)["response"])[n]["verdict"]
                        for n in ASPECTS
                    }
                require(
                    row["replayed_response_valid"] == (result["execution_status"] == "COMPLETE"),
                    "Replay differs from the historical technical status",
                )
            rows.append(row)
        model = ModelConfig.model_validate(strict_json(archive.read("model_config.json")))
    response_rows = [r for r in rows if "replayed_response_valid" in r]
    summary = {
        "exported_files_hash_checked": checked,
        "cases": len(cases),
        "historical_live_responses": len(response_rows),
        "historical_responses_valid": sum(r["replayed_response_valid"] for r in response_rows),
        "historical_responses_incomplete": sum(
            not r["replayed_response_valid"] for r in response_rows
        ),
        "historical_records_modified": False,
        "new_model_calls": 0,
    }
    return rows, summary, model


def load_sources(source_bundle, inputs):
    plan = strict_json(PLAN.read_bytes())
    inputs.add(PLAN)
    fixture = REPO / plan["fixture"]
    require(inputs.add(fixture) == plan["fixture_sha256"], "Development fixture hash mismatch")
    cases = strict_json(fixture.read_bytes())["cases"]
    require([c["case_id"] for c in cases] == plan["case_order"], "Case order/census changed")
    baseline = source_bundle / plan["baseline_snapshot"]
    require(
        inputs.add(baseline) == plan["baseline_sha256"], "Historical baseline ZIP hash mismatch"
    )
    rows, summary, historical_model = read_baseline(baseline.read_bytes(), cases)
    return plan, cases, rows, summary, historical_model


def score_case(case, prepared, outcome, plan, aspects=None):
    expected = case["expected_v161"]
    observed = set(outcome["concern_aspects"])
    required = set(plan["required_concerns"].get(case["case_id"], []))
    allowed = set(plan["allowed_concerns"].get(case["case_id"], []))
    forbidden = observed - allowed
    modes = plan.get("required_assertion_modes", {}).get(case["case_id"], {})
    polarity_mismatches, polarity_unassessed = {}, []
    for aspect, allowed_modes in modes.items():
        record = (aspects or {}).get(aspect, {})
        if record.get("status") != "SUCCEEDED":
            polarity_unassessed.append(aspect)
        elif record["parsed_review"]["assertion_mode"] not in allowed_modes:
            polarity_mismatches[aspect] = {
                "expected": allowed_modes,
                "observed": record["parsed_review"]["assertion_mode"],
            }
    complete = outcome["execution_status"] == "COMPLETE"
    match = (
        complete
        and outcome["semantic_status"] == expected["semantic_status"]
        and outcome["interpretation_eligible"] == expected["interpretation_eligible"]
        and required <= observed
        and not forbidden
        and not polarity_mismatches
        and not polarity_unassessed
        and prepared["deterministic_reason_codes"] == expected["deterministic_reason_codes"]
    )
    return {
        "case_id": case["case_id"],
        "complete": complete,
        "expected_match": match,
        "required_concerns": sorted(required),
        "allowed_concerns": sorted(allowed),
        "missing_required_concerns": sorted(required - observed),
        "forbidden_concerns": sorted(forbidden),
        "polarity_mismatches": polarity_mismatches,
        "polarity_unassessed": polarity_unassessed,
        "faithful_control": case["case_id"] in plan["faithful_case_ids"],
        "candidate_retained": prepared["statistical_candidate_retained"],
        **outcome,
    }


def pin_model(historical, plan):
    """Same model weights; capture current server version before constructing requests."""
    provisional = historical.model_copy(deep=True)
    provisional.options = GenerationOptions.model_validate(plan["generation_options"])
    provisional.timeout_seconds = plan["timeout_seconds"]
    connection = OllamaTransport(provisional)
    tags = strict_json(connection._http("/api/tags"))
    version = strict_json(connection._http("/api/version"))["version"]
    matches = [m for m in tags["models"] if m.get("name") == historical.model]
    require(
        len(matches) == 1 and matches[0].get("digest") == historical.model_digest,
        "Local model weights differ from historical llama3.1:8b; do not switch automatically",
    )
    provisional.server_version = version
    transport = OllamaTransport(provisional)
    transport.identity()
    return provisional, transport


def export_live_results(out, cache, request_index):
    """Return this run and its exact first-response records; exclude unrelated cache."""
    path = out.with_suffix(".zip")
    files = {"results/" + str(p.relative_to(out)): p for p in sorted(out.rglob("*")) if p.is_file()}
    for row in request_index:
        key = row["request_key"]
        for file in sorted((cache / key).glob("*.json")):
            files["cache/" + key + "/" + file.name] = file
    manifest = {name: hashlib.sha256(p.read_bytes()).hexdigest() for name, p in files.items()}
    with zipfile.ZipFile(path, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, file in files.items():
            raw = file.read_bytes()
            require(
                hashlib.sha256(raw).hexdigest() == manifest[name], "Result changed during export"
            )
            archive.writestr(name, raw)
        archive.writestr("EXPORT_SHA256.json", json.dumps(manifest, indent=2) + "\n")
    return path


def run_r03(
    root,
    source_bundle,
    *,
    live=False,
    output=None,
    calls=96,
    seconds=1800,
    model=None,
    transport=None,
):
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    output = output or Path(root) / "output/revision_v17" / f"r03_{stamp}"
    root, out = output_directory(root, output)
    source_bundle = Path(source_bundle).expanduser().resolve(strict=True)
    require(source_bundle.is_relative_to(root / "input"), "Place R03 bundle under CRM_R1/input/")
    require(live or (model is None and transport is None), "Model transport requires live mode")
    injected = transport is not None
    budget = CallBudget(calls, seconds)
    inputs = Inputs()
    for file in [Path(__file__), Path(__file__).with_name("62_revision_r03.py")]:
        inputs.add(file)
    for folder in ("contract_v161", "contract_v17"):
        for file in sorted((REPO / "src/llm_pathway_curator" / folder).glob("*.py")):
            inputs.add(file)
    inputs.add(Path(__file__).with_name("v17_revision.py"))
    plan, cases, baseline_rows, baseline_summary, historical = load_sources(source_bundle, inputs)
    out.mkdir(parents=True)
    write_json(
        out / "STARTED.json",
        {
            "schema": VERSION,
            "method_id": METHOD_ID,
            "mode": "software_transport_test"
            if injected
            else "live_development"
            if live
            else "offline_preparation",
            "git": git_snapshot(),
            "outdir": str(out),
            "source_bundle": str(source_bundle),
            "implementation_sha256": implementation_digest(),
            "new_request_limit": budget.limit,
            "wall_budget_seconds": budget.seconds,
        },
    )
    print(f"[R03] output: {out}", flush=True)
    try:
        write_json(out / "baseline_replay.private.json", baseline_rows)
        write_json(out / "baseline_summary.json", baseline_summary)
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
                require(model is None, "Use the pinned local backend; no supplied unverified model")
                model, transport = pin_model(historical, plan)
            else:
                require(model is not None, "Injected transport needs a model (software tests only)")
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
            FirstOutcomeStore(data_path(root, "output/revision_v17/r03_atomic_cache"))
            if live
            else None
        )
        results, scores, request_index = [], [], []
        for index, case in enumerate(cases, 1):
            prepared, _, _ = prepare(case["evidence"], case["claim"])
            require(prepared["canonical_status"] == "VALID", "A fixed canonical control is invalid")
            slots = {}
            if prepared["contract_status"] != "VIOLATION":
                for aspect in ASPECTS:
                    request = make_request(case["evidence"], case["claim"], model, aspect)
                    request_index.append(
                        {
                            "case_id": case["case_id"],
                            "aspect": aspect,
                            "request_key": request["key"],
                        }
                    )
                    if live:
                        print(
                            f"[R03] {index}/20 {case['case_id']} {aspect}; new requests "
                            f"{budget.calls}/{budget.limit}",
                            flush=True,
                        )
                        record, cache_hit, called = store.run(
                            request, transport, allow_new=budget.available()
                        )
                        budget.calls += called
                        payload = request["envelope"]["payload"]
                        parsed = record.get("parsed_review")
                        slots[aspect] = record | {
                            "cache_hit": cache_hit,
                            "resolved_sentences": [
                                s
                                for s in payload["sentences"]
                                if parsed and s["id"] in parsed["sentence_ids"]
                            ],
                            "resolved_facts": {
                                f: payload["facts"][f]
                                for f in (parsed["fact_ids"] if parsed else [])
                            },
                        }
                    else:
                        slots[aspect] = {
                            "status": "NOT_RUN",
                            "parsed_review": None,
                            "request_key": request["key"],
                        }
                        folder = out / "planned_requests" / case["case_id"]
                        folder.mkdir(parents=True, exist_ok=True)
                        write_json(folder / f"{aspect}.json", request)
            outcome = aggregate(prepared, slots)
            result = {
                "schema": "CRM_R1_ATOMIC_AUDIT_v17_R03",
                "method_id": METHOD_ID,
                "case_id": case["case_id"],
                "deterministic_result": prepared,
                "aspect_results": slots,
                "outcome": outcome,
                "statistical_candidate_retained": prepared["statistical_candidate_retained"],
                "biological_truth": "NOT_ESTABLISHED",
                "automatic_free_text_publication_allowed": False,
            }
            results.append(result)
            scores.append(score_case(case, prepared, outcome, plan, slots))
            # Each completed case is a checkpoint, even if the process stops later.
            folder = out / "cases"
            folder.mkdir(exist_ok=True)
            write_json(folder / f"{case['case_id']}.private.json", result)
        inputs.verify()
        faithful = [s for s in scores if s["faithful_control"]]
        incorrect = [s for s in scores if not s["faithful_control"]]
        summary = {
            "schema": VERSION,
            "scope": "injected_software_transport_not_real_llm"
            if injected
            else "known_development_controls_not_independent_validation",
            "status": "COMPLETE_DEVELOPMENT_PASS"
            if live and all(s["expected_match"] for s in scores)
            else "COMPLETE_WITH_DEVELOPMENT_FINDINGS"
            if live
            else "PREPARED_NOT_LIVE_TESTED",
            "outdir": str(out),
            "results_archive": str(out.with_suffix(".zip")) if live else None,
            "cases": len(results),
            "candidate_retained_count": sum(s["candidate_retained"] for s in scores),
            "planned_semantic_requests": len(request_index),
            "new_model_requests": budget.calls if not injected else 0,
            "new_transport_attempts": budget.calls,
            "model_backend_authenticated": bool(live and not injected),
            "prompt_version": PROMPT_VERSION,
            "known_control_matches": sum(s["expected_match"] for s in scores),
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
            "execution_status_counts": dict(Counter(s["execution_status"] for s in scores)),
            "forbidden_concern_case_count": sum(bool(s["forbidden_concerns"]) for s in scores),
            "polarity_mismatch_case_count": sum(bool(s["polarity_mismatches"]) for s in scores),
            "historical_baseline": baseline_summary,
            "input_hashes_unchanged": True,
            "natural_text_comparison_frozen": False,
            "external_expression_outcomes_loaded": False,
            "biological_or_independent_semantic_accuracy_estimated": False,
            "expert_ratings_requested": False,
            "next_step": "separate_natural_text_protocol_freeze"
            if live and all(s["expected_match"] for s in scores)
            else "review_atomic_development_failures"
            if live
            else "run_bounded_local_atomic_pilot",
        }
        write_json(out / "audit_results.private.json", results)
        write_json(out / "scores.private.json", scores)
        write_json(
            out / "baseline_vs_r03.private.json",
            [
                {"case_id": old["case_id"], "historical": old, "r03": new}
                for old, new in zip(baseline_rows, scores, strict=True)
            ],
        )
        write_json(out / "request_index.private.json", request_index)
        write_json(
            out / "candidate_ledger.private.json",
            [
                {
                    "case_id": r["case_id"],
                    "evidence": r["deterministic_result"]["raw_evidence_preserved"],
                    "retained": r["statistical_candidate_retained"],
                }
                for r in results
            ],
        )
        write_json(
            out / "interpretation_ledger.private.json",
            [{"case_id": r["case_id"], **r["outcome"]} for r in results],
        )
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
                "new_model_requests": budget.calls,
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
    parser.add_argument("--max-new-requests", type=int, default=96)
    parser.add_argument("--wall-budget-seconds", type=float, default=1800)
    args = parser.parse_args()
    if args.data_root is None:
        parser.error("Set CRM_R1_DATA_ROOT or use --data-root")
    try:
        run_r03(
            args.data_root,
            args.source_bundle,
            live=args.live,
            output=args.outdir,
            calls=args.max_new_requests,
            seconds=args.wall_budget_seconds,
        )
    except (OSError, ValueError, TechnicalError, KeyError) as error:
        print(f"[R03 STOP] {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
