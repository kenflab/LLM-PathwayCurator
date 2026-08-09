from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CRM = ROOT / "paper" / "revision" / "CRM_R1"


def load_freeze_module():
    path = CRM / "scripts" / "20_freeze_claim_pool.py"
    spec = importlib.util.spec_from_file_location("crm_r1_p2_freeze", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def synthetic_audit(*, review_mode: str, gate_mode: str) -> pd.DataFrame:
    entities = [f"HALLMARK_TERM_{index}" for index in range(1, 7)]
    status = ["PASS", "PASS", "PASS", "ABSTAIN", "ABSTAIN", "FAIL"]
    return pd.DataFrame(
        {
            "claim_id": [f"claim_{review_mode}_{index}" for index in range(6)],
            "entity": entities,
            "direction": ["up", "down", "up", "down", "up", "down"],
            "status": status,
            "term_survival_agg": [0.91, 0.99, 0.93, 0.96, 0.90, 0.88],
            "tau_used": [0.9] * 6,
            "claim_mode": ["deterministic"] * 6,
            "context_review_mode": [review_mode] * 6,
            "context_gate_mode": [gate_mode] * 6,
            "context_evaluated": [review_mode == "llm"] * 6,
            "context_method": ["llm" if review_mode == "llm" else "none"] * 6,
            "context_status": ["PASS", "PASS", "PASS", "WARN", "WARN", "FAIL"],
            "context_confidence": [0.9] * 6,
            "abstain_reason": [pd.NA, pd.NA, pd.NA, "context_nonspecific", "unstable", pd.NA],
            "fail_reason": [pd.NA, pd.NA, pd.NA, pd.NA, pd.NA, "context_fail"],
        }
    )


def test_priority2_protocol_keeps_later_outcomes_unfrozen() -> None:
    protocol = json.loads((CRM / "config" / "priorities2_5_protocol.json").read_text())
    assert protocol["status"] == "DRAFT_NOT_FROZEN"
    assert protocol["priority2"]["review_sampling"]["mode"] == "census"
    assert protocol["priority2"]["proposal_mode"] == "deterministic"
    assert protocol["priority2"]["full_audit_run"]["context_review_mode"] == "llm"
    assert protocol["priority3"]["release_and_search_dates"] is None
    assert protocol["priority5"]["ontology_release_dates"] is None


def test_build_outputs_matches_all_selected_methods_to_full_audit_k() -> None:
    module = load_freeze_module()
    protocol = json.loads((CRM / "config" / "priorities2_5_protocol.json").read_text())
    protocol["priority2"]["candidate_count_expected"] = 6
    protocol["priority2"]["review_sampling"]["n"] = 6
    mechanical = synthetic_audit(review_mode="off", gate_mode="note")
    full_audit = synthetic_audit(review_mode="llm", gate_mode="hard")
    evidence = pd.DataFrame(
        {
            "term_id": mechanical["entity"],
            "direction": mechanical["direction"],
            "stat": [1.0, -5.0, 3.0, -2.0, 4.0, -0.5],
            "qval": [0.06, 0.001, 0.02, 0.03, 0.01, 0.5],
        }
    )
    claims, membership, metrics, sampling = module.build_outputs(
        mechanical=mechanical,
        full_audit=full_audit,
        evidence=evidence,
        protocol=protocol,
        freeze_label="TEST",
    )
    assert len(claims) == len(membership) == len(sampling) == 6
    assert membership["full_audit_selected"].sum() == 3
    assert membership["q_value_matched_selected"].sum() == 3
    assert membership["stability_matched_selected"].sum() == 3
    assert membership["raw_pool_selected"].sum() == 6
    assert set(metrics["outcome_status"]) == {"PENDING_P3_P4"}
    assert claims["claim_uid"].is_unique
    assert sampling["review_id"].is_unique


def test_validate_audit_requires_same_pool_proposal_mode_and_review_contract() -> None:
    module = load_freeze_module()
    mechanical = module.validate_audit(
        synthetic_audit(review_mode="off", gate_mode="note"),
        expected_n=6,
        expected_tau=0.9,
        review_mode="off",
        gate_mode="note",
    )
    full = module.validate_audit(
        synthetic_audit(review_mode="llm", gate_mode="hard"),
        expected_n=6,
        expected_tau=0.9,
        review_mode="llm",
        gate_mode="hard",
    )
    assert not mechanical["context_evaluated"].any()
    assert full["context_evaluated"].all()

    module.validate_run_metadata(
        {
            "inputs": {
                "llm": {
                    "review": {
                        "review_mode_effective_for_select": "llm",
                        "backend_enabled": True,
                        "backend_env": "ollama",
                    },
                    "backend_identity": {"model_name": "llama3.1:8b"},
                }
            }
        },
        review_mode="llm",
        expected_backend="ollama",
        expected_model="llama3.1:8b",
    )


def test_freeze_precedes_p3_p4_and_refuses_overwrite() -> None:
    source = (CRM / "scripts" / "20_freeze_claim_pool.py").read_text()
    assert "Priority 2 freeze outputs are immutable" in source
    assert 'for later_priority in ("priority3", "priority4", "priority5")' in source
    assert '"validation_outcomes_inspected": False' in source
    assert "require_clean_tracked_worktree()" in source


def test_pipeline_records_nonsecret_backend_identity_for_freeze() -> None:
    source = (ROOT / "src" / "llm_pathway_curator" / "pipeline.py").read_text()
    assert '"backend_identity": backend_identity' in source
    assert '"model_name": str(getattr(shared_backend, "model_name", "") or "")' in source
    assert "api_key" not in source[source.index("backend_identity =") : source.index('"claim": {')]
