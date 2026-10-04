"""Reference integrity, denied inference, failure accounting and immutable outcomes.

All generated wire responses here are software fixtures, not real-model evidence.
"""

import importlib.util
import json
import sys
from copy import deepcopy
from pathlib import Path

import pytest

from llm_pathway_curator.contract_v17.atomic import (
    AtomicReview,
    CallBudget,
    FirstOutcomeStore,
    aggregate,
    make_request,
    parse_response,
    sentence_spans,
)
from llm_pathway_curator.contract_v161.checks import prepare
from llm_pathway_curator.contract_v161.models import ASPECTS, ModelConfig
from llm_pathway_curator.contract_v161.runtime import TechnicalError, WireResponse

ROOT = Path(__file__).resolve().parents[1]
CASES = json.loads((ROOT / "tests/fixtures/crm_r1_v16_1/cases_v161.json").read_text())["cases"]
SCRIPT_DIR = ROOT / "paper/revision/CRM_R1/scripts"
sys.path.insert(0, str(SCRIPT_DIR))
SPEC = importlib.util.spec_from_file_location("crm_r03_test", SCRIPT_DIR / "v17_r03.py")
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


@pytest.fixture
def model():
    return ModelConfig(
        host="http://127.0.0.1:11434",
        model="software-test",
        model_digest="a" * 64,
        server_version="synthetic",
    )


def response(request, review):
    return WireResponse(
        json.dumps(
            {
                "model": request["envelope"]["model_config"]["model"],
                "done": True,
                "done_reason": "stop",
                "response": json.dumps(review),
            }
        ).encode()
    )


def clear(mode="NOT_MENTIONED", **updates):
    return {
        "assertion_mode": mode,
        "verdict": "CLEAR",
        "sentence_ids": [],
        "fact_ids": [],
        "reason": "Software test only.",
        **updates,
    }


def case(case_id):
    return deepcopy(next(c for c in CASES if c["case_id"] == case_id))


def request_for(model, case_id="P03", aspect="causality"):
    c = case(case_id)
    return make_request(c["evidence"], c["claim"], model, aspect)


def test_sentence_offsets_preserve_negation_and_scientific_numbers(model):
    text = "NES=-1.75; q=1.2e-3. This does not establish a causal effect."
    spans = sentence_spans(text)
    assert len(spans) == 2
    assert all(text[s["start"] : s["end"]] == s["text"] for s in spans)
    r = request_for(model)
    payload = r["envelope"]["payload"]
    assert payload["claim_text"] == case("P03")["claim"]["text"]
    assert "does not establish" in payload["sentences"][1]["text"]
    assert "expected_v161" not in json.dumps(r["envelope"]["body"])
    assert set(payload["facts"]) == {"CONTRAST", "ENRICHMENT_LIMIT"}
    assert payload["response_schema"]["properties"]["sentence_ids"]["items"]["enum"] == [
        "S001",
        "S002",
    ]


def test_denied_inference_is_valid_but_cannot_be_a_concern(model):
    r = request_for(model)
    value = clear("DENIED", sentence_ids=["S002"], fact_ids=["CONTRAST"])
    parsed = parse_response(response(r, value).raw, r)
    assert parsed.verdict == "CLEAR"
    value["verdict"] = "CONCERN"
    with pytest.raises(TechnicalError, match="affirmed assertion"):
        parse_response(response(r, value).raw, r)


@pytest.mark.parametrize(
    "updates",
    [
        {"sentence_ids": ["S999"]},
        {"fact_ids": ["/evidence/metadata"]},
        {"fact_ids": ["COHORT"]},
        {"sentence_ids": ["S001", "S001"]},
        {"assertion_mode": "NOT_MENTIONED"},
        {"reason": " "},
        {"confidence": 0.95},
    ],
)
def test_bad_reference_or_logical_schema_stays_invalid(model, updates):
    r = request_for(model, "N03")
    value = {
        "assertion_mode": "AFFIRMED",
        "verdict": "CONCERN",
        "sentence_ids": ["S001"],
        "fact_ids": ["CONTRAST"],
        "reason": "Software test.",
        **updates,
    }
    with pytest.raises(TechnicalError):
        parse_response(response(r, value).raw, r)


def test_mixed_assertion_is_not_cleared_by_a_disclaimer(model):
    c = case("N03")
    c["claim"]["text"] += " These data do not establish a causal effect."
    r = make_request(c["evidence"], c["claim"], model, "causality")
    value = {
        "assertion_mode": "MIXED",
        "verdict": "CONCERN",
        "sentence_ids": ["S001"],
        "fact_ids": ["CONTRAST"],
        "reason": "An affirmative causal claim remains.",
    }
    assert parse_response(response(r, value).raw, r).verdict == "CONCERN"


@pytest.mark.parametrize("fail", [False, True])
def test_first_outcome_is_reused_even_if_invalid_without_resampling(tmp_path, model, fail):
    r = request_for(model)
    calls = []

    def transport(request):
        calls.append(request)
        v = clear() if not fail else clear("DENIED", verdict="CONCERN", sentence_ids=["S002"])
        return response(request, v)

    store = FirstOutcomeStore(tmp_path)
    first, hit, called = store.run(r, transport)
    assert called and not hit
    before = {p: p.read_bytes() for p in tmp_path.glob("*/*.json")}
    repeated, hit, called = store.run(r, transport)
    assert repeated == first and hit and not called and len(calls) == 1
    assert all(p.read_bytes() == raw for p, raw in before.items())
    assert first["status"] == ("TECHNICAL_ERROR" if fail else "SUCCEEDED")


def test_budget_skips_can_resume_but_interrupted_attempts_cannot(tmp_path, model):
    store = FirstOutcomeStore(tmp_path)
    r = request_for(model)
    skipped, _, called = store.run(r, lambda _: pytest.fail("budget exceeded"), allow_new=False)
    assert skipped["status"] == "NOT_RUN_BUDGET" and not called
    folder = tmp_path / r["key"]
    (folder / "STARTED.json").write_text(json.dumps({"request_key": r["key"]}))

    interrupted, hit, called = store.run(r, lambda _: pytest.fail("resampled interruption"))
    assert interrupted["status"] == "INTERRUPTED" and hit and not called


def test_cache_tampering_and_model_identity_fail_closed(tmp_path, model):
    r = request_for(model)
    store = FirstOutcomeStore(tmp_path)
    store.run(r, lambda req: response(req, clear()))
    path = tmp_path / r["key"] / "FIRST_RESULT.json"
    altered = json.loads(path.read_text())
    altered["observations"] = {"altered": True}
    path.write_text(json.dumps(altered))
    with pytest.raises(TechnicalError, match="digest mismatch"):
        store.run(r, lambda _: pytest.fail("tampered cache resampled"))
    raw = response(r, clear()).raw.replace(b'"software-test"', b'"other-model"')
    with pytest.raises(TechnicalError, match="response model"):
        parse_response(raw, r)


def test_incomplete_does_not_disappear_and_does_not_erase_other_concerns():
    c = case("N03")
    prepared, _, _ = prepare(c["evidence"], c["claim"])
    slots = {n: {"status": "SUCCEEDED", "parsed_review": clear()} for n in ASPECTS}
    slots["metadata"] = {"status": "TECHNICAL_ERROR", "parsed_review": None}
    slots["causality"] = {
        "status": "SUCCEEDED",
        "parsed_review": {
            "assertion_mode": "AFFIRMED",
            "verdict": "CONCERN",
            "sentence_ids": ["S001"],
            "fact_ids": ["CONTRAST"],
            "reason": "Software fixture concern.",
        },
    }
    result = aggregate(prepared, slots)
    assert result["execution_status"] == "INCOMPLETE"
    assert result["concern_aspects"] == ["causality"]
    assert not result["interpretation_eligible"] and prepared["statistical_candidate_retained"]
    assert result["incomplete_aspects"] == ["metadata"]


def test_scoring_rejects_extra_concerns_even_when_required_issue_is_present():
    c = case("N03")
    prepared, _, _ = prepare(c["evidence"], c["claim"])
    outcome = {
        "execution_status": "COMPLETE",
        "semantic_status": "CONCERN_REQUIRES_REVIEW",
        "interpretation_eligible": False,
        "concern_aspects": ["causality", "metadata"],
    }
    plan = json.loads(runner.PLAN.read_text())
    scored = runner.score_case(c, prepared, outcome, plan)
    assert scored["missing_required_concerns"] == []
    assert scored["forbidden_concerns"] == ["metadata"] and not scored["expected_match"]


def test_correct_verdict_alone_cannot_hide_wrong_negation_classification():
    c = case("P03")
    prepared, _, _ = prepare(c["evidence"], c["claim"])
    slots = {n: {"status": "SUCCEEDED", "parsed_review": clear()} for n in ASPECTS}
    slots["causality"]["parsed_review"] = clear(
        "AFFIRMED", sentence_ids=["S002"], fact_ids=["CONTRAST"]
    )
    outcome = aggregate(prepared, slots)
    assert outcome["interpretation_eligible"]
    scored = runner.score_case(c, prepared, outcome, json.loads(runner.PLAN.read_text()), slots)
    assert not scored["expected_match"]
    assert scored["polarity_mismatches"]["causality"]["expected"] == ["DENIED"]


def test_unrun_slots_are_not_technical_failures_and_budget_is_bounded():
    c = case("P01")
    prepared, _, _ = prepare(c["evidence"], c["claim"])
    slots = {n: {"status": "NOT_RUN", "parsed_review": None} for n in ASPECTS}
    assert aggregate(prepared, slots)["execution_status"] == "NOT_RUN"
    assert not CallBudget(0).available()
    with pytest.raises(ValueError):
        CallBudget(97)
    with pytest.raises(ValueError):
        CallBudget(1, float("nan"))


def test_reversed_direction_is_in_prompt_from_canonical_evidence(model):
    r = request_for(model, "A05", "numerics_and_direction")
    payload = r["envelope"]["payload"]
    assert payload["facts"]["DIRECTION"]["enrichment_toward"] == "TP53_WILD_TYPE"
    assert "toward the TP53-mutant group" in payload["claim_text"]
    assert AtomicReview.model_validate(clear()).verdict == "CLEAR"


def test_identity_pin_rejects_weight_changes_and_captures_current_server(model, monkeypatch):
    plan = json.loads(runner.PLAN.read_text())

    def metadata(self, suffix, body=None):
        value = (
            {"version": "current-test-version"}
            if suffix == "/api/version"
            else {"models": [{"name": model.model, "digest": model.model_digest}]}
        )
        return json.dumps(value).encode()

    monkeypatch.setattr(runner.OllamaTransport, "_http", metadata)
    pinned, _ = runner.pin_model(model, plan)
    assert pinned.server_version == "current-test-version"
    assert model.server_version == "synthetic"
    assert pinned.options.num_predict == 512

    def wrong_weights(self, suffix, body=None):
        if suffix == "/api/tags":
            return json.dumps({"models": [{"name": model.model, "digest": "b" * 64}]}).encode()
        return metadata(self, suffix, body)

    monkeypatch.setattr(runner.OllamaTransport, "_http", wrong_weights)
    with pytest.raises(ValueError, match="weights differ"):
        runner.pin_model(model, plan)


def test_live_software_transport_retains_failures_denominator_and_export(
    tmp_path, model, monkeypatch, capsys
):
    import zipfile

    root = tmp_path / "CRM_R1"
    source = root / "input/revision_code/r03"
    source.mkdir(parents=True)
    (root / "output").mkdir()
    plan = json.loads(runner.PLAN.read_text())
    baseline_rows = [{"case_id": c["case_id"], "synthetic": True} for c in CASES]
    monkeypatch.setattr(
        runner,
        "load_sources",
        lambda source, inputs: (plan, CASES, baseline_rows, {"synthetic": True}, model),
    )
    calls = []

    def invalid_wire(request):
        calls.append(request["key"])
        return response(request, {"missing": "required fields"})

    first = runner.run_r03(root, source, live=True, model=model, transport=invalid_wire, calls=5)
    assert first["cases"] == first["candidate_retained_count"] == 20
    assert first["new_transport_attempts"] == 5 and first["new_model_requests"] == 0
    assert first["scope"] == "injected_software_transport_not_real_llm"
    assert first["faithful_controls"]["incomplete"] == 1
    assert first["faithful_controls"]["not_run"] == 7
    assert first["known_control_matches"] == 4
    path = Path(first["results_archive"])
    with zipfile.ZipFile(path) as archive:
        ledger = json.loads(archive.read("results/candidate_ledger.private.json"))
        assert len(ledger) == 20
        records = [n for n in archive.namelist() if n.endswith("FIRST_RESULT.json")]
        assert len(records) == 5
        assert all(json.loads(archive.read(n))["status"] == "TECHNICAL_ERROR" for n in records)
    repeated = runner.run_r03(root, source, live=True, model=model, transport=invalid_wire, calls=1)
    assert repeated["new_transport_attempts"] == 1
    assert len(calls) == 6 and len(set(calls)) == 6
    assert repeated["candidate_retained_count"] == 20
    assert repeated["status"] == "COMPLETE_WITH_DEVELOPMENT_FINDINGS"
    capsys.readouterr()
