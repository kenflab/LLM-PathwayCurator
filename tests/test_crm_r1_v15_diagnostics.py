"""Regression tests for diagnostic denominators, lineage ambiguity, and explicit utility."""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "paper/revision/CRM_R1/scripts"
sys.path.insert(0, str(SCRIPTS))


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


selection = load("v15_selection_test", "92_diagnose_selection_v15.py")
utility = load("v15_trace_test", "93_trace_utility_v15.py")
explicit = load("v15_explicit_test", "94_build_explicit_utility_v15.py")
common = load("v15_common_test", "v15_diagnostic_common.py")


def test_zero_padding_and_equal_cohort_gate_contributions():
    rows = []
    for cohort in ["A", "B"]:
        for split in ["S1", "S2"]:
            for claim in ["x", "y", "z"]:
                discordant = cohort == "A" and split == "S1"
                full = claim == "x"
                comparator = claim == ("y" if discordant else "x")
                rows.append(
                    {
                        "cohort_id": cohort,
                        "split_id": split,
                        "claim_uid": claim,
                        "full_audit": full,
                        "q_value_matched": comparator,
                        "stability_matched": comparator,
                        "replicated": claim == "y",
                        "gate": "PASS" if full else "FAIL:context_fail",
                        "context_status": "PASS" if full else "FAIL",
                        "context_reason": "No clear evidence",
                        "reason_at_160_char_limit": False,
                    }
                )
    policy = {"lexical_flags": {"mentions_evidence": "evidence"}, "examples_per_cohort_outcome": 2}
    tables = selection.diagnostic_tables(pd.DataFrame(rows), policy)
    contrasts = tables["cohort_contrasts_reproduced.tsv"]
    assert contrasts.groupby("comparator").full_minus_comparator.mean().eq(-0.25).all()
    contributions = tables["gate_contribution_summary.tsv"]
    assert contributions.groupby("comparator").signed_replication_contribution.sum().eq(-0.25).all()
    # Absent gate in B and three other cohort/split combinations must contribute zero.
    assert len(tables["rationale_examples.private.tsv"]) == 1
    shuffled = selection.diagnostic_tables(
        pd.DataFrame(rows).sample(frac=1, random_state=9), policy
    )
    assert tables["rationale_examples.private.tsv"].example_order.tolist() == (
        shuffled["rationale_examples.private.tsv"].example_order.tolist()
    )


def test_strict_boolean_rejects_unknown():
    with pytest.raises(ValueError, match="Invalid boolean"):
        common.boolean(pd.Series(["true", "unknown"], name="selected"))


def test_frozen_hash_mismatch_stops(tmp_path):
    source = tmp_path / "claims.tsv"
    source.write_text("not a frozen source\n")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": "CRM_R1_P2B_FIGURE2_SOURCE_v14_1_3",
                "outputs": {"claim_source_private": {"sha256": "0" * 64}},
            }
        )
    )
    with pytest.raises(ValueError, match="SHA mismatch"):
        selection.load_claims(source, manifest)


def test_missing_llm_reason_is_a_technical_failure(tmp_path):
    path = tmp_path / "jobs/A/S1/audit"
    path.mkdir(parents=True)
    audit = pd.DataFrame(
        {
            "entity": [f"T{x}" for x in range(50)],
            "status": "PASS",
            "context_method": "llm",
            "context_review_mode": "llm",
            "context_evaluated": True,
            "context_reason": "",
            "context_confidence": 0.9,
            "context_status": "PASS",
            "term_survival_agg": 1,
            "direction": "up",
        }
    )
    audit.to_csv(path / "audit_log.tsv", sep="\t", index=False)
    base = pd.DataFrame({"cohort_id": "A", "split_id": "S1", "claim_uid": audit.entity})
    with pytest.raises(ValueError, match="Missing LLM context reason"):
        selection.load_audits(tmp_path, base)


def test_output_never_overwrites_or_writes_frozen_bundle(tmp_path):
    exists = tmp_path / "existing"
    exists.mkdir()
    with pytest.raises(ValueError, match="already exists"):
        with common.new_output(exists):
            pass
    with pytest.raises(ValueError, match="frozen bundle"):
        with common.new_output(tmp_path / "membership_lock_v14_1_3" / "new"):
            pass


def test_input_change_during_run_is_detected(tmp_path):
    source = tmp_path / "source"
    source.write_text("original")
    recorded = common.record(source)
    source.write_text("changed")
    with pytest.raises(ValueError, match="Input changed"):
        common.finish(tmp_path, [recorded], {})


def test_matching_values_are_not_declared_historical_provenance():
    ranked = pd.DataFrame(
        {"claim_id": ["x", "y"], "term_uid": ["a", "b"], "context_fit": [0.2, 0.8]}
    )
    audit = pd.DataFrame(
        {
            "claim_id": ["y", "x"],
            "term_uid": ["b", "a"],
            "context_score_proxy_u01_norm": [0.8, 0.2],
            "context_score": [0.1, 0.9],
        }
    )
    found = utility.compare_candidate(ranked, audit, common.CONTEXT_COLUMNS)
    assert found[0]["context_fit_matches"]
    assert found[0]["chosen_by_supplied_code"]
    assert not any(x["historical_input_identity_proven"] for x in found)
    audit.loc[0, "term_uid"] = "wrong"
    assert utility.compare_candidate(ranked, audit, common.CONTEXT_COLUMNS) == []


def test_precedence_is_read_from_code_not_assumed(tmp_path):
    path = tmp_path / "ranked.py"
    path.write_text("context_fit_col = _first_existing_col(audit, ['real', 'proxy']) or ''\n")
    assert utility.context_precedence(path) == ["real", "proxy"]


def contract():
    return {
        "status": "DEVELOPMENT_ONLY",
        "evidence_column": "evidence",
        "stability_column": "stability",
        "context": {
            "column": "context_score",
            "kind": "llm_derived_score",
            "provenance": "Synthetic test fixture only",
        },
    }


def test_explicit_context_ignores_higher_hash_proxy_and_deterministic_ties():
    frame = pd.DataFrame(
        {
            "claim_id": ["b", "a"],
            "evidence": [1, 1],
            "stability": [1, 1],
            "context_score": [0.2, 0.2],
            "context_score_proxy_u01_norm": [1, 0],
        }
    )
    result = explicit.explicit_utility(frame, contract())
    assert result.claim_id.tolist() == ["a", "b"]
    assert result.utility_score_v15_development.tolist() == [0.2, 0.2]


@pytest.mark.parametrize("value", [np.nan, np.inf, -0.1, 1.1])
def test_invalid_context_never_defaults_to_one(value):
    frame = pd.DataFrame(
        {"claim_id": ["a"], "evidence": [1], "stability": [1], "context_score": [value]}
    )
    with pytest.raises(ValueError):
        explicit.explicit_utility(frame, contract())


def test_explicit_provenance_is_required_and_hash_proxy_refused():
    frame = pd.DataFrame(
        {
            "claim_id": ["a"],
            "evidence": [1],
            "stability": [1],
            "context_score_proxy_u01_norm": [0.5],
        }
    )
    cfg = contract()
    cfg["context"]["column"] = "context_score_proxy_u01_norm"
    with pytest.raises(ValueError, match="Proxy/hash"):
        explicit.explicit_utility(frame, cfg)
    cfg["context"]["provenance"] = ""
    with pytest.raises(ValueError, match="provenance"):
        explicit.explicit_utility(frame, cfg)


def test_product_and_three_component_rank_sensitivity():
    frame = pd.DataFrame(
        {
            "claim_id": ["x", "y"],
            "evidence_strength_filled": [2, 1],
            "stability": [1, 1],
            "context_fit": [0.1, 1],
            "utility_score": [0.2, 1],
            "score_source": "-log10(qval)",
        }
    )
    assert utility.formula_check(frame)["three_component_product_matches"]
    variants = {x["variant"] for x in utility.sensitivity(frame, "fixture")}
    assert variants == {
        "E*S*C_stored_components",
        "E*S_omit_context",
        "E*C_omit_stability",
        "S*C_omit_evidence",
    }
