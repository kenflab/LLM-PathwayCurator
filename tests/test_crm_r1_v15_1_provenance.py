"""Scientific failure modes: proxy aliases, incomplete caches, K=0, and report drift."""

from __future__ import annotations

import copy
import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "paper/revision/CRM_R1/scripts"
sys.path.insert(0, str(SCRIPTS))


def module(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / (name + ".py"))
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


common = module("v15_1_provenance_common")
legacy = module("95_reconstruct_legacy_proxy_v15_1")
verify = module("96_verify_llm_audit_v15_1")
contract = module("98_check_structured_report_v15_1")


def fixture():
    audit = pd.DataFrame(
        {
            "claim_id": ["a", "b"],
            "term_uid": ["H:T1", "H:T2"],
            "context_signature": ["123456abcdef"] * 2,
            "context_method": ["llm"] * 2,
            "context_review_mode": ["llm"] * 2,
            "context_evaluated": ["true"] * 2,
            "context_status": ["PASS", "FAIL"],
            "context_confidence": ["0.7", "0.8"],
            "context_reason": ["Consistent", "Mismatch"],
            "status": ["PASS", "FAIL"],
        }
    )
    evidence = audit[["term_uid"]].copy()
    meta = {
        "status": "ok",
        "inputs": {
            "claims": {
                "context_review_mode": "llm",
                "context_review_mode_effective_for_select": "llm",
            },
            "llm": {
                "review": {"enabled": True, "backend_enabled": True},
                "select_entrypoint": {"backend_attached": True},
                "backend_identity": {
                    "class": "OllamaBackend",
                    "model_name": "test-model",
                    "host": "http://127.0.0.1:11434",
                },
            },
        },
    }
    cache = {
        "123456abcdef::" + r.term_uid: {
            "status": r.context_status,
            "confidence": float(r.context_confidence),
            "reason": r.context_reason,
        }
        for r in audit.itertuples()
    }
    return audit, evidence, meta, cache


def test_all_valid_and_real_zero_k_are_distinct_from_technical_failure():
    audit, evidence, meta, cache = fixture()
    _, summary = verify.validate(audit, evidence, meta, cache, "test-model")
    assert summary["technical_status"] == "VALID_NONZERO_K"
    audit["status"] = "ABSTAIN"
    _, summary = verify.validate(audit, evidence, meta, cache, "test-model")
    assert summary["technical_status"] == "VALID_ZERO_K"
    assert summary["zero_k_retry_authorized"] is False


@pytest.mark.parametrize(
    "change",
    [
        "proxy",
        "backend",
        "missing_cache",
        "extra_cache",
        "error",
        "blank",
        "infinite",
        "bool",
        "mismatch",
        "url",
        "model",
        "pool",
        "mode",
        "duplicate",
    ],
)
def test_invalid_runtime_cannot_receive_a_receipt(change):
    audit, evidence, meta, cache = fixture()
    key = next(iter(cache))
    if change == "proxy":
        audit.loc[0, "context_method"] = "proxy"
    if change == "backend":
        meta["inputs"]["llm"]["review"]["enabled"] = False
    if change == "missing_cache":
        del cache[key]
    if change == "extra_cache":
        cache["foreign::T3"] = {}
    if change == "error":
        cache[key]["error"] = {"message": "Connection refused"}
    if change == "blank":
        cache[key]["reason"] = " "
    if change == "infinite":
        cache[key]["confidence"] = float("inf")
    if change == "bool":
        cache[key]["confidence"] = True
    if change == "mismatch":
        cache[key]["status"] = "FAIL"
    if change == "url":
        meta["inputs"]["llm"]["backend_identity"]["host"] = "[http://x](http://x)"
    if change == "model":
        meta["inputs"]["llm"]["backend_identity"]["model_name"] = "wrong"
    if change == "pool":
        evidence = evidence.iloc[:1]
    if change == "mode":
        meta["inputs"]["claims"]["context_review_mode"] = "proxy"
    if change == "duplicate":
        audit.loc[1, "term_uid"] = audit.loc[0, "term_uid"]
    with pytest.raises(ValueError):
        verify.validate(audit, evidence, meta, cache, "test-model")


def test_full_reason_is_preserved_but_arbitrary_prefix_is_not_allowed():
    audit, evidence, meta, cache = fixture()
    key = next(iter(cache))
    cache[key]["reason"] = "a" * 300
    audit.loc[0, "context_reason"] = "a" * 160
    result, _ = verify.validate(audit, evidence, meta, cache, "test-model")
    assert len(result.iloc[0].context_reason_full) == 300
    audit.loc[0, "context_reason"] = "a" * 159
    with pytest.raises(ValueError):
        verify.validate(audit, evidence, meta, cache, "test-model")


def test_hash_renamed_confidence_remains_proxy_and_cannot_pass_guard():
    u = common.proxy_v2("test", ["condition"], "H:T1")
    audit = pd.DataFrame(
        {
            "claim_id": ["a"],
            "term_uid": ["H:T1"],
            "context_ctx_id": ["test"],
            "context_keys": ["condition"],
            "context_confidence": [str(u)],
        }
    )
    ranked = pd.DataFrame(
        {
            "claim_id": ["a"],
            "term_uid": ["H:T1"],
            "context_fit": [str(u)],
            "evidence_strength": ["2"],
            "stability": ["0.5"],
            "utility_score": [str(u)],
        }
    )
    _, aliases, result = legacy.reconstruct(audit, ranked, {})
    assert result["hash_derivation_confirmed"]
    assert aliases.iloc[0].hash_matches == 1
    with pytest.raises(ValueError):
        verify.validate(audit, audit, {}, {}, "test-model")


def es_frame():
    return pd.DataFrame(
        {
            "claim_id": ["b", "a"],
            "term_uid": ["H:B", "H:A"],
            "E": [2, 1],
            "S": [0.5, 1],
            "decision": ["FAIL", "PASS"],
            "context_score": [0, 1],
        }
    )


def test_es_keeps_fail_rows_ignores_context_and_breaks_ties_by_identity():
    table = es_frame()
    first = common.stored_es(table, "E", "S")
    table["context_score"] = [float("nan"), -5]
    table["decision"] = ["ABSTAIN", "FAIL"]
    second = common.stored_es(table.iloc[::-1], "E", "S")
    pd.testing.assert_frame_equal(first.reset_index(drop=True), second.reset_index(drop=True))
    assert first.claim_id.tolist() == ["a", "b"]


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -1, 1.1])
def test_invalid_s_never_defaults_to_one(bad):
    table = es_frame()
    table.loc[0, "S"] = bad
    with pytest.raises(ValueError):
        common.stored_es(table, "E", "S")


def test_context_alias_cannot_be_an_es_component():
    with pytest.raises(ValueError):
        common.stored_es(es_frame(), "E", "context_score")


def test_fresh_outputs_refuse_existing_paths_and_input_mutation(tmp_path):
    source = tmp_path / "source.txt"
    source.write_text("before")
    inputs = [common.record(source)]
    out = tmp_path / "result"
    with pytest.raises(ValueError), common.fresh_output(out):
        source.write_text("after")
        common.finish(out, inputs, {})
    assert not (out / "manifest.json").exists()
    assert (out / "FAILED.json").exists()
    with pytest.raises(FileExistsError), common.fresh_output(out):
        pass


def report():
    evidence = {
        "evidence_id": "E1",
        "cohort_id": "ACC",
        "comparison": "mut_vs_wt",
        "term_uid": "H:T1",
        "direction": "UP",
        "q_value": 0.2,
        "evidence_genes": ["TP53", "MDM2"],
    }
    claim = {
        key: evidence[key]
        for key in ("evidence_id", "cohort_id", "comparison", "term_uid", "direction")
    }
    claim.update(reported_q_value=0.2, supporting_genes=["MDM2"], asserts_fdr_significance=False)
    return {"case_id": "valid", "evidence": evidence, "claim": claim}


@pytest.mark.parametrize(
    "field,value,code",
    [
        ("cohort_id", "LUAD", "MISMATCH_COHORT_ID"),
        ("direction", "DOWN", "MISMATCH_DIRECTION"),
        ("reported_q_value", 0.02, "Q_VALUE_TRANSCRIPTION"),
        ("supporting_genes", ["WRONG"], "GENE_NOT_IN_SUPPLIED_EVIDENCE"),
        ("asserts_fdr_significance", True, "UNSUPPORTED_FDR_SIGNIFICANCE_AT_0_05"),
    ],
)
def test_structured_violations_use_supplied_evidence_not_pathway_plausibility(field, value, code):
    item = report()
    before = copy.deepcopy(item)
    result = contract.check_record(item)
    assert result["contract_status"] == "NO_STRUCTURED_VIOLATION_DETECTED"
    assert result["biological_correctness"] == "NOT_ASSESSED"
    assert item == before
    item["claim"][field] = value
    assert code in contract.check_record(item)["reason_codes"]


def test_q_boundary_and_gene_order():
    item = report()
    item["evidence"]["q_value"] = item["claim"]["reported_q_value"] = 0.05
    item["claim"]["asserts_fdr_significance"] = True
    item["claim"]["supporting_genes"] = ["MDM2", "TP53"]
    assert contract.check_record(item)["reason_codes"] == ""


def test_duplicate_json_keys_cannot_hide_a_cache_error(tmp_path):
    path = tmp_path / "cache.json"
    path.write_text('{"error": "connection failed", "error": null}')
    with pytest.raises(ValueError, match="Duplicate JSON key"):
        common.read_json(path)


def test_complete_disk_bundle_verifies_hashes_and_refuses_changed_evidence(tmp_path):
    audit, evidence, meta, cache = fixture()
    audit.to_csv(tmp_path / "audit_log.tsv", sep="\t", index=False)
    evidence.to_csv(tmp_path / "evidence.normalized.tsv", sep="\t", index=False)
    common.write_json(tmp_path / "sample_card.normalized.json", {"condition": "SYNTHETIC"})
    meta["inputs"]["evidence_normalized_sha256"] = common.sha(tmp_path / "evidence.normalized.tsv")
    meta["inputs"]["sample_card_normalized_sha256"] = common.sha(
        tmp_path / "sample_card.normalized.json"
    )
    common.write_json(tmp_path / "run_meta.json", meta)
    common.write_json(tmp_path / "context_review_cache.synthetic.json", cache)
    annotations, summary, inputs = verify.check_bundle(tmp_path, "test-model")
    assert len(annotations) == 2 and len(inputs) == 5
    assert summary["technical_status"] == "VALID_NONZERO_K"
    with (tmp_path / "evidence.normalized.tsv").open("a") as stream:
        stream.write("H:T3\n")
    with pytest.raises(ValueError, match="Evidence hash mismatch"):
        verify.check_bundle(tmp_path, "test-model")
