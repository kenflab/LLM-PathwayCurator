from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from llm_pathway_curator.audit import audit_claims
from llm_pathway_curator.claim_schema import Claim
from llm_pathway_curator.pipeline import (
    RunConfig,
    _apply_context_gate_to_audited,
    _apply_context_review,
    _proxy_context_review,
    _synthesize_claim_json_row,
    run_pipeline,
)
from llm_pathway_curator.sample_card import SampleCard


def _base_claim() -> tuple[Claim, str, list[str]]:
    term_uid = "fgsea:HALLMARK_P53_PATHWAY"
    genes = ["TP53", "CDKN1A", "MDM2", "BAX"]
    claim = Claim(
        entity=term_uid,
        direction="up",
        context_keys=["condition"],
        evidence_ref={"term_ids": [term_uid], "gene_ids": genes},
    )
    return claim, term_uid, genes


def test_proxy_warn_is_schema_valid_and_audit_consistent(monkeypatch):
    monkeypatch.setenv("LLMPATH_PROXY_CONTEXT_P_FAIL", "0")
    monkeypatch.setenv("LLMPATH_PROXY_CONTEXT_P_WARN", "1")

    claim, term_uid, genes = _base_claim()
    proposed = pd.DataFrame(
        [
            {
                "claim_id": claim.claim_id,
                "claim_json": claim.model_dump_json(by_alias=True),
                "entity": term_uid,
                "direction": "up",
                "term_uid": term_uid,
                "context_gate_hit": False,
            }
        ]
    )
    card = SampleCard(
        condition="ENDO",
        extra={
            "context_gate_mode": "hard",
            "context_review_mode": "proxy",
            "audit_min_union_genes": 1,
        },
    )

    reviewed, meta = _proxy_context_review(
        proposed,
        card,
        gate_mode="hard",
        review_mode="proxy",
        distilled_for_proxy=None,
    )

    assert reviewed.loc[0, "context_status"] == "WARN"
    assert reviewed.loc[0, "context_method"] == "proxy"
    assert meta["n_warn"] == 1

    payload = json.loads(reviewed.loc[0, "claim_json"])
    assert "context_confidence" not in payload
    assert "context_gate_mode" not in payload
    assert "context_review_mode" not in payload
    parsed = Claim.model_validate(payload)
    assert parsed.context_evaluated is True
    assert parsed.context_method == "proxy"
    assert parsed.context_status == "WARN"

    distilled = pd.DataFrame(
        [
            {
                "term_uid": term_uid,
                "term_id": "HALLMARK_P53_PATHWAY",
                "source": "fgsea",
                "evidence_genes": genes,
                "term_survival": 1.0,
            }
        ]
    )
    audited = audit_claims(reviewed, distilled, card, tau=0.8)

    row = audited.iloc[0]
    assert row["status"] == "ABSTAIN"
    assert row["abstain_reason"] == "context_nonspecific"
    assert row["context_status"] == "WARN"
    assert bool(row["context_gate_blocked"]) is True
    assert bool(row["context_gate_hit"]) is True


def test_synthesized_claim_json_contains_only_strict_claim_fields():
    card = SampleCard(condition="ENDO")
    row = pd.Series(
        {
            "claim_id": "c_synth",
            "entity": "HALLMARK_P53_PATHWAY",
            "direction": "up",
            "term_ids": "fgsea:HALLMARK_P53_PATHWAY",
            "gene_ids": "TP53;CDKN1A;MDM2;BAX",
            "context_evaluated": True,
            "context_method": "proxy_context_v2",
            "context_status": "ABSTAIN",
            "context_reason": "PROXY_U01_LOW_ABSTAIN",
            "context_notes": "proxy_context_v2",
            "context_confidence": 0.1,
            "context_gate_mode": "hard",
            "context_review_mode": "proxy",
        }
    )

    payload = json.loads(_synthesize_claim_json_row(row, card))
    parsed = Claim.model_validate(payload)

    assert parsed.context_evaluated is True
    assert parsed.context_method == "proxy"
    assert parsed.context_status == "WARN"
    assert "context" not in payload
    assert "context_confidence" not in payload
    assert "context_gate_mode" not in payload
    assert "context_review_mode" not in payload


def test_off_mode_writes_schema_valid_unevaluated_claim_and_does_not_reproxy():
    claim, term_uid, genes = _base_claim()
    proposed = pd.DataFrame(
        [
            {
                "claim_id": claim.claim_id,
                "claim_json": claim.model_dump_json(by_alias=True),
                "entity": term_uid,
                "direction": "up",
                "term_uid": term_uid,
                "context_gate_hit": False,
                # A ranking/provenance score must not reactivate review=off.
                "context_score": 0.01,
            }
        ]
    )
    card = SampleCard(
        condition="ENDO",
        extra={"context_gate_mode": "note", "context_review_mode": "off"},
    )

    reviewed, meta = _apply_context_review(
        proposed,
        card,
        gate_mode="note",
        review_mode="off",
        backend=None,
        seed=42,
        distilled_for_proxy=None,
    )

    assert meta["evaluated"] is False
    assert reviewed.loc[0, "context_method"] == "off"
    assert reviewed.loc[0, "context_status"] == "UNEVALUATED"

    parsed = Claim.model_validate_json(reviewed.loc[0, "claim_json"])
    assert parsed.context_evaluated is False
    assert parsed.context_method == "none"
    assert parsed.context_status is None
    assert parsed.context_reason is None
    assert parsed.context_notes is None

    distilled = pd.DataFrame(
        [
            {
                "term_uid": term_uid,
                "term_id": "HALLMARK_P53_PATHWAY",
                "source": "fgsea",
                "evidence_genes": genes,
                "term_survival": 1.0,
            }
        ]
    )
    audited = audit_claims(reviewed, distilled, card, tau=0.8)
    row = audited.iloc[0]
    assert row["status"] == "PASS"
    assert bool(row["context_evaluated"]) is False
    assert row["context_status"] == ""
    assert bool(row["context_gate_blocked"]) is False
    assert bool(row["context_gate_hit"]) is False


def test_pipeline_effective_context_modes_reach_audit(tmp_path, monkeypatch):
    monkeypatch.setenv("LLMPATH_CONTEXT_REVIEW_MODE", "off")
    monkeypatch.setenv("LLMPATH_CONTEXT_GATE_MODE", "note")
    repository = Path(__file__).resolve().parents[1]
    demo = repository / "examples" / "demo"
    outdir = tmp_path / "context_off_note"

    run_pipeline(
        RunConfig(
            evidence_table=str(demo / "evidence_table.tsv"),
            sample_card=str(demo / "sample_card.json"),
            outdir=str(outdir),
            force=True,
            seed=42,
            tau=0.8,
            k_claims=20,
        )
    )

    proposed = pd.read_csv(outdir / "claims.proposed.tsv", sep="\t")
    audit = pd.read_csv(outdir / "audit_log.tsv", sep="\t")
    effective_card = json.loads((outdir / "sample_card.effective.json").read_text())

    for value in proposed["claim_json"]:
        parsed = Claim.model_validate_json(value)
        assert parsed.context_evaluated is False
        assert parsed.context_method == "none"
        assert parsed.context_status is None

    assert effective_card["extra"]["context_review_mode"] == "off"
    assert effective_card["extra"]["context_gate_mode"] == "note"
    assert not audit["context_evaluated"].astype(bool).any()
    assert not audit["context_gate_blocked"].astype(bool).any()
    assert not audit["context_gate_hit"].astype(bool).any()
    assert not audit["abstain_reason"].fillna("").str.contains("context").any()
    assert not audit["fail_reason"].fillna("").str.contains("context").any()


@pytest.mark.parametrize(
    ("proposed_status", "evaluated", "expected_status", "expected_reason"),
    [
        ("WARN", True, "ABSTAIN", "context_nonspecific"),
        ("ABSTAIN", True, "ABSTAIN", "context_nonspecific"),
        ("FAIL", True, "FAIL", "context_fail"),
        ("", False, "ABSTAIN", "context_missing"),
    ],
)
def test_post_audit_safety_gate_synchronizes_decision_and_diagnostics(
    proposed_status, evaluated, expected_status, expected_reason
):
    audited = pd.DataFrame(
        [
            {
                "claim_id": "c_test",
                "status": "PASS",
                "abstain_reason": "",
                "fail_reason": "",
                "audit_notes": "",
                "context_status": "PASS",
                "context_evaluated": True,
                "context_gate_blocked": False,
                "context_gate_hit": False,
            }
        ]
    )
    proposed = pd.DataFrame(
        [
            {
                "claim_id": "c_test",
                "context_status": proposed_status,
                "context_evaluated": evaluated,
                "context_method": "proxy" if evaluated else "none",
                "context_reason": "test",
            }
        ]
    )

    result = _apply_context_gate_to_audited(audited, proposed, gate_mode="hard")
    row = result.iloc[0]

    assert row["status"] == expected_status
    reason_column = "fail_reason" if expected_status == "FAIL" else "abstain_reason"
    assert row[reason_column] == expected_reason
    expected_context = "WARN" if proposed_status == "ABSTAIN" else proposed_status
    assert row["context_status"] == expected_context
    assert bool(row["context_gate_blocked"]) is True
    assert bool(row["context_gate_hit"]) is True
    assert bool(row["eligible_context"]) is False


def test_post_audit_safety_gate_keeps_pass_diagnostics_aligned():
    audited = pd.DataFrame(
        [
            {
                "claim_id": "c_test",
                "status": "PASS",
                "abstain_reason": "",
                "fail_reason": "",
                "audit_notes": "",
                "context_status": "WARN",
                "context_evaluated": True,
                "context_gate_blocked": True,
                "context_gate_hit": True,
            }
        ]
    )
    proposed = pd.DataFrame(
        [
            {
                "claim_id": "c_test",
                "context_status": "PASS",
                "context_evaluated": True,
                "context_method": "proxy",
                "context_reason": "PROXY_U01_OK",
            }
        ]
    )

    result = _apply_context_gate_to_audited(audited, proposed, gate_mode="hard")
    row = result.iloc[0]

    assert row["status"] == "PASS"
    assert row["context_status"] == "PASS"
    assert bool(row["context_gate_blocked"]) is False
    assert bool(row["context_gate_hit"]) is False
    assert bool(row["eligible_context"]) is True
