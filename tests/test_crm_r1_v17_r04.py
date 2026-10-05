"""Source isolation, unchanged replay, immutable outcomes, and honest probe scope.

Generated responses in these tests are software fixtures, not real-model evidence.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import sys
import zipfile
from copy import deepcopy
from pathlib import Path

import pytest

from llm_pathway_curator.claim_locator_r04.locator import (
    make_request,
    parse_response,
    resolve,
    sentence_spans,
)
from llm_pathway_curator.claim_locator_r04.runtime import CallBudget, FirstOutcomeStore
from llm_pathway_curator.contract_v17.atomic import aggregate
from llm_pathway_curator.contract_v17.atomic import make_request as r03_request
from llm_pathway_curator.contract_v161.checks import prepare
from llm_pathway_curator.contract_v161.models import ASPECTS, ModelConfig
from llm_pathway_curator.contract_v161.prompt import canonical_json, digest
from llm_pathway_curator.contract_v161.runtime import TechnicalError, WireResponse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "paper/revision/CRM_R1/scripts"))
import v17_r03 as r03  # noqa: E402
import v17_r04 as runner  # noqa: E402
from v17_r04_diagnostics import diagnose_r03, required_source_files  # noqa: E402

CASES = json.loads((ROOT / "tests/fixtures/crm_r1_v16_1/cases_v161.json").read_text())["cases"]
PLAN = json.loads(runner.PLAN.read_text())


@pytest.fixture
def model():
    return ModelConfig(
        host="http://127.0.0.1:11434",
        model="software-test",
        model_digest="a" * 64,
        server_version="synthetic",
    )


def wire(request, locations, *, done_reason="stop", identity=True):
    model = request["envelope"]["model_config"]
    observed = {
        "model": model["model"],
        "digest": model["model_digest"],
        "version": model["server_version"],
    }
    raw = json.dumps(
        {
            "model": model["model"],
            "done": True,
            "done_reason": done_reason,
            "response": json.dumps(locations),
            "eval_count": 100,
            "total_duration": 1000000000,
        }
    ).encode()
    return WireResponse(
        raw, {"identity_before": observed, "identity_after": observed} if identity else {}
    )


def locations(case_id="P03"):
    return deepcopy(PLAN["expected_source_locations"][case_id])


def test_only_source_text_and_schema_reach_model_no_evidence_or_gold_labels(model):
    text = "An observational comparison has NES=1.75. Causality is not established."
    request = make_request(text, model)
    payload = request["envelope"]["payload"]
    assert set(payload) == {"SOURCE_TEXT", "sentences"}
    assert payload["SOURCE_TEXT"] == text
    assert all(text[s["start"] : s["end"]] == s["text"] for s in payload["sentences"])
    body = canonical_json(request["envelope"]["body"])
    for forbidden in (
        "canonical_evidence",
        "expected_source_locations",
        "expected_v161",
        "ENRICHMENT_LIMIT",
        "METADATA_MISMATCH",
        "evidence_id",
        "case_id",
    ):
        assert forbidden not in body
    assert make_request(text, model) == request  # Evidence-independent exact caching.


def test_full_source_spans_keep_negation_and_dont_split_decimal_or_scientific_notation(model):
    text = "NES=-1.75; q=5e-2. This does not establish causality."
    assert len(sentence_spans(text)) == 2
    req = make_request(text, model)
    parsed = parse_response(wire(req, locations()).raw, req)
    resolved = resolve(parsed, req)
    assert resolved["causality"]["limited"][0]["text"] == "This does not establish causality."
    assert resolved["numerics_and_direction"]["asserted"][0]["text"] == "NES=-1.75; q=5e-2."


@pytest.mark.parametrize(
    "change", ["unknown", "duplicate", "verdict", "missing_aspect", "truncated"]
)
def test_invalid_or_truncated_locations_stay_incomplete(model, change):
    req = make_request("Statement one. Statement two.", model)
    value = locations()
    reason = "stop"
    if change == "unknown":
        value["causality"]["limited"] = ["S999"]
    elif change == "duplicate":
        value["causality"]["limited"] = ["S002", "S002"]
    elif change == "verdict":
        value["causality"]["verdict"] = "CLEAR"
    elif change == "missing_aspect":
        value.pop("metadata")
    else:
        reason = "length"
    with pytest.raises(TechnicalError):
        parse_response(wire(req, value, done_reason=reason).raw, req)


def test_valid_locations_do_not_imply_correct_aspect_or_polarity(model):
    req = make_request("A numeric result. Causality is not established.", model)
    wrong = locations()
    wrong["numerics_and_direction"] = {"asserted": [], "limited": ["S002"], "uncertain": []}
    parsed = parse_response(wire(req, wrong).raw, req)
    score = runner.score_locations(
        "P03", {"status": "SUCCEEDED", "parsed_locations": parsed.model_dump()}, locations()
    )
    assert score["complete"] and not score["source_locations_match"]
    assert "numerics_and_direction/asserted" in score["mismatches"]
    assert "numerics_and_direction/limited" in score["mismatches"]
    assert not score["interpretation_eligibility_assessed"]


def test_empty_claim_coverage_cannot_pass_and_list_order_is_not_semantic(model):
    empty = {aspect: {"asserted": [], "limited": [], "uncertain": []} for aspect in ASPECTS}
    score = runner.score_locations(
        "P03", {"status": "SUCCEEDED", "parsed_locations": empty}, locations()
    )
    assert not score["source_locations_match"]
    expected = locations("P02")
    reversed_ids = deepcopy(expected)
    reversed_ids["numerics_and_direction"]["asserted"].reverse()
    score = runner.score_locations(
        "P02", {"status": "SUCCEEDED", "parsed_locations": reversed_ids}, expected
    )
    assert score["source_locations_match"]


@pytest.mark.parametrize("bad", [False, True])
def test_first_returned_outcome_is_reused_without_retry_even_when_invalid(tmp_path, model, bad):
    req = make_request("Statement one. Statement two.", model)
    value = locations()
    if bad:
        value["causality"]["limited"] = ["S999"]
    calls = []

    def transport(request):
        calls.append(request)
        return wire(request, value)

    store = FirstOutcomeStore(tmp_path)
    first, hit, called = store.run(req, transport)
    before = {p: p.read_bytes() for p in tmp_path.glob("*/*.json")}
    again, hit, called = store.run(req, transport)
    assert hit and not called and again == first and len(calls) == 1
    assert all(p.read_bytes() == raw for p, raw in before.items())
    assert first["status"] == ("TECHNICAL_ERROR" if bad else "SUCCEEDED")


def test_backend_receipt_is_required_for_first_success_and_tampering_is_rejected(tmp_path, model):
    req = make_request("Statement one. Statement two.", model)
    store = FirstOutcomeStore(tmp_path)
    first, _, _ = store.run(req, lambda r: wire(r, locations(), identity=False))
    assert (
        first["status"] == "TECHNICAL_ERROR" and first["error_code"] == "BACKEND_IDENTITY_MISMATCH"
    )
    path = tmp_path / req["key"] / "FIRST_RESULT.json"
    changed = json.loads(path.read_text())
    changed["error_message"] = "altered"
    path.write_text(json.dumps(changed))
    with pytest.raises(TechnicalError, match="digest mismatch"):
        store.run(req, lambda _: pytest.fail("tampered outcome was resampled"))


def test_budget_skip_can_resume_but_interrupted_request_cannot(tmp_path, model):
    store = FirstOutcomeStore(tmp_path)
    req = make_request("Statement one. Statement two.", model)
    skipped, _, called = store.run(req, lambda _: pytest.fail("budget exceeded"), allow_new=False)
    assert not called and skipped["status"] == "NOT_RUN_BUDGET"
    (tmp_path / req["key"] / "STARTED.json").write_text(json.dumps({"request_key": req["key"]}))
    interrupted, hit, called = store.run(req, lambda _: pytest.fail("interruption retried"))
    assert hit and not called and interrupted["status"] == "INTERRUPTED"
    with pytest.raises(ValueError):
        CallBudget(17)
    assert not CallBudget(0).available()


def synthetic_r03_archive(model):
    """Full fake returned run for integrity tests; never a real-model evaluation."""
    files = {}

    def put(name, value):
        files[name] = (json.dumps(value, indent=2) + "\n").encode()

    original_plan = json.loads(r03.PLAN.read_text())
    results, scores, index = [], [], []
    for case in CASES:
        prepared, _, _ = prepare(case["evidence"], case["claim"])
        slots = {}
        if prepared["contract_status"] != "VIOLATION":
            for aspect in ASPECTS:
                req = r03_request(case["evidence"], case["claim"], model, aspect)
                parsed = {
                    "assertion_mode": "NOT_MENTIONED",
                    "verdict": "CLEAR",
                    "sentence_ids": [],
                    "fact_ids": [],
                    "reason": "Software fixture, not semantic evidence.",
                }
                response = wire(req, parsed)
                record = {
                    "request_key": req["key"],
                    "status": "SUCCEEDED",
                    "ended_utc": "synthetic",
                    "error_code": None,
                    "error_message": None,
                    "raw_response_base64": base64.b64encode(response.raw).decode(),
                    "raw_response_sha256": hashlib.sha256(response.raw).hexdigest(),
                    "parsed_review": parsed,
                    "observations": response.observations,
                }
                slots[aspect] = record
                index.append(
                    {"case_id": case["case_id"], "aspect": aspect, "request_key": req["key"]}
                )
                prefix = "cache/" + req["key"] + "/"
                put(prefix + "request.json", req)
                put(prefix + "STARTED.json", {"request_key": req["key"]})
                put(prefix + "FIRST_RESULT.json", record)
                put(prefix + "FIRST_RESULT_SHA256.json", {"sha256": digest(record)})
        outcome = aggregate(prepared, slots)
        result = {
            "case_id": case["case_id"],
            "deterministic_result": prepared,
            "aspect_results": slots,
            "outcome": outcome,
        }
        results.append(result)
        scores.append(r03.score_case(case, prepared, outcome, original_plan, slots))
        put(f"results/cases/{case['case_id']}.private.json", result)
    incorrect = [s for s in scores if not s["faithful_control"]]
    summary = {
        "scope": "injected_software_transport_not_real_llm",
        "cases": 20,
        "candidate_retained_count": 20,
        "planned_semantic_requests": 96,
        "known_control_matches": sum(s["expected_match"] for s in scores),
        "execution_status_counts": {"COMPLETE": 20},
        "forbidden_concern_case_count": 0,
        "polarity_mismatch_case_count": sum(bool(s["polarity_mismatches"]) for s in scores),
        "faithful_controls": {
            "total": 8,
            "incomplete": 0,
            "not_run": 0,
            "eligible": 8,
            "complete_but_withheld": 0,
        },
        "incorrect_controls": {
            "total": 12,
            "incomplete": 0,
            "not_run": 0,
            "eligible": sum(s["interpretation_eligible"] for s in incorrect),
        },
    }
    put("results/summary.json", summary)
    put("results/model_config.private.json", model.model_dump())
    put("results/development_plan.json", original_plan)
    put(
        "results/canonical_census.private.json",
        [{"case_id": c["case_id"], "evidence": c["evidence"], "claim": c["claim"]} for c in CASES],
    )
    put(
        "results/INPUT_MANIFEST.private.json",
        [
            {
                "path": "/synthetic/LLM-PathwayCurator/" + str(p.relative_to(ROOT)),
                "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
            }
            for p in required_source_files()
        ],
    )
    put("results/audit_results.private.json", results)
    put("results/scores.private.json", scores)
    put("results/request_index.private.json", index)
    put(
        "results/OUTPUT_MANIFEST.json",
        {
            n.removeprefix("results/"): hashlib.sha256(raw).hexdigest()
            for n, raw in files.items()
            if n.startswith("results/")
        },
    )
    put("EXPORT_SHA256.json", {n: hashlib.sha256(raw).hexdigest() for n, raw in files.items()})
    target = io.BytesIO()
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, raw in files.items():
            archive.writestr(name, raw)
    return target.getvalue()


def test_all_returned_requests_and_scores_are_replayed_without_promoting_fake_semantics(model):
    raw = synthetic_r03_archive(model)
    summary, rows, scores, returned_model = diagnose_r03(raw, CASES)
    assert summary["all_original_scores_reproduced"] and not summary["original_records_modified"]
    assert summary["backend_receipts_authenticated"] == len(rows) == 96
    assert summary["scope"] == "injected_software_transport_not_real_llm"
    assert summary["new_model_calls"] == 0 and len(scores) == 20 and returned_model == model
    assert (
        summary["semantic_cases_meeting_original_rules"] < 16
    )  # Empty claims are not correct audits.


def test_modified_returned_file_rejected_even_when_outer_zip_is_readable(model):
    raw = synthetic_r03_archive(model)
    output = io.BytesIO()
    with zipfile.ZipFile(io.BytesIO(raw)) as original, zipfile.ZipFile(output, "w") as changed:
        for name in original.namelist():
            data = original.read(name)
            changed.writestr(name, data + b" " if name == "results/summary.json" else data)
    with pytest.raises(ValueError, match="member hash mismatch"):
        diagnose_r03(output.getvalue(), CASES)


def test_failed_probe_keeps_all_candidates_and_never_produces_audit_eligibility(
    tmp_path, monkeypatch, model
):
    data = tmp_path / "CRM_R1"
    (data / "input/bundle").mkdir(parents=True)
    (data / "output").mkdir()
    monkeypatch.setattr(
        runner,
        "load_sources",
        lambda bundle, inputs: (
            PLAN,
            CASES,
            {"all_original_scores_reproduced": True, "exported_files_hash_checked": 419},
            [],
            [],
            model,
        ),
    )
    summary = runner.run_r04(
        data,
        data / "input/bundle",
        live=True,
        model=model,
        transport=lambda request: wire(
            request,
            {aspect: {"asserted": [], "limited": [], "uncertain": []} for aspect in ASPECTS},
        ),
        calls=1,
    )
    assert summary["candidate_retained_count"] == summary["cases"] == 20
    assert summary["planned_model_requests"] == 16 and summary["new_transport_attempts"] == 1
    assert summary["new_model_requests"] == 0 and summary["source_location_matches"] == 0
    assert summary["scope"] == "injected_software_transport_not_real_llm"
    assert summary["source_only_execution_status_counts"] == {"SUCCEEDED": 1, "NOT_RUN_BUDGET": 15}
    assert (
        not summary["audit_verdicts_produced"]
        and not summary["interpretation_eligibility_assessed"]
    )
    assert (
        not summary["expert_ratings_requested"]
        and not summary["external_expression_outcomes_loaded"]
    )
    assert Path(summary["results_archive"]).is_file()


def test_complete_software_probe_is_labelled_as_software_and_same_failed_outcomes_are_reused(
    tmp_path, monkeypatch, model
):
    data = tmp_path / "CRM_R1"
    (data / "input/bundle").mkdir(parents=True)
    (data / "output").mkdir()
    monkeypatch.setattr(
        runner,
        "load_sources",
        lambda bundle, inputs: (
            PLAN,
            CASES,
            {"all_original_scores_reproduced": True, "exported_files_hash_checked": 419},
            [],
            [],
            model,
        ),
    )
    expected_by_text = {
        c["claim"]["text"]: PLAN["expected_source_locations"][c["case_id"]]
        for c in CASES
        if c["case_id"] in PLAN["semantic_case_ids"]
    }
    summary = runner.run_r04(
        data,
        data / "input/bundle",
        live=True,
        model=model,
        transport=lambda request: wire(
            request, expected_by_text[request["envelope"]["payload"]["SOURCE_TEXT"]]
        ),
    )
    assert summary["source_location_matches"] == 16 and summary["new_model_requests"] == 0
    assert summary["scope"] == "injected_software_transport_not_real_llm"
    again = runner.run_r04(
        data,
        data / "input/bundle",
        live=True,
        model=model,
        transport=lambda _: pytest.fail("completed outcome resampled"),
    )
    assert again["new_transport_attempts"] == 0 and again["source_location_matches"] == 16
