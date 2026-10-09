"""Regression checks for provenance diagnostics, independent of archived labels."""

import importlib.util
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "saved_labels", Path(__file__).parents[1] / "paper/scripts/audit_saved_label_provenance.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_reused_identity_does_not_certify_task_equivalence():
    label = {
        "claim_id": "x",
        "human_label": "ACCEPT",
        "rater_id": "R1",
        "entity": "A",
        "direction": "up",
        "gene_symbols_str": "G",
        "_src": "context_swap/audit_log.tsv",
    }
    audit = {**label, "status": "PASS"}
    result = MODULE.inspect_labels([label], [audit, {**audit, "claim_id": "y"}])
    assert result["matches"][0]["source_record_fields_match"]
    assert result["matches"][0]["task_equivalence"] == "NOT_ESTABLISHED_BY_CLAIM_ID"
    assert result["target_ids_without_labels"] == ["y"]
    assert result["performance_comparison"].startswith("NOT_ESTIMATED")


def test_duplicate_claims_do_not_silently_discard_a_rater():
    labels = [{"claim_id": "x", "human_label": "ACCEPT", "rater_id": r} for r in ["R1", "R2"]]
    with pytest.raises(ValueError, match="duplicate"):
        MODULE.inspect_labels(labels, [])


def test_changed_content_is_reported():
    label = {"claim_id": "x", "human_label": "REJECT", "entity": "A"}
    result = MODULE.inspect_labels([label], [{"claim_id": "x", "entity": "B", "status": "PASS"}])
    assert result["matched_field_mismatches"] == 1
