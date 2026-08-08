from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CRM = ROOT / "paper" / "revision" / "CRM_R1"


def load_preview_module():
    path = CRM / "scripts" / "15_preview_empirical_membership.py"
    spec = importlib.util.spec_from_file_location("crm_r1_v4_preview", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_evidence_module():
    path = CRM / "scripts" / "14_build_empirical_evidence.py"
    spec = importlib.util.spec_from_file_location("crm_r1_v4_evidence", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_v5_protocol_preserves_empirical_design_and_freezes_tau() -> None:
    config = json.loads((CRM / "config" / "priority1_protocol.json").read_text())

    assert config["protocol_version"] == "CRM_R1_PRIORITY1_v5"
    assert config["status"] == "FROZEN"
    assert config["primary_analysis"]["context_review_mode"] == "off"
    assert config["primary_analysis"]["context_gate_mode"] == "note"
    assert config["empirical_stability"]["expected_resamples"] == 81
    assert config["empirical_stability"]["distill_mode"] == "replicates_proxy"
    assert config["empirical_stability"]["primary_tau"] == 0.8
    assert config["empirical_stability"]["calibration_tau_grid"] == [0.8, 0.9, 0.95, 0.98]
    assert config["freeze_decision"]["selected_k"] == 23
    assert config["freeze_decision"]["expected_empirical_q_value_overlap"] == 16
    assert config["freeze_decision"]["expected_empirical_size_matched_overlap"] == 17


def test_resampling_script_has_no_held_out_analysis_path() -> None:
    source = (CRM / "scripts" / "13_resample_discovery_48h.R").read_text()

    assert "expected_resamples), 81L" in source
    assert "discovery_48h_gene_universe.tsv" in source
    assert "hallmark_gene_sets.tsv" in source
    assert "cell_state == primary$cell_state" in source
    assert "time_h) == as.integer(primary$discovery_time_h)" in source
    assert "validation_time_h" not in source
    assert "msigdbr" not in source


def test_size_strata_and_q_order_are_deterministic() -> None:
    module = load_preview_module()
    table = pd.DataFrame(
        {
            "claim_id": [f"c{i}" for i in range(8)],
            "pathway": [f"p{i}" for i in range(8)],
            "leading_edge_n": [40, 10, 30, 20, 80, 60, 70, 50],
            "padj": [0.08, 0.01, 0.03, 0.02, 0.07, 0.05, 0.06, 0.04],
            "pval": [0.008, 0.001, 0.003, 0.002, 0.007, 0.005, 0.006, 0.004],
            "NES": [1.0] * 8,
        }
    )

    stratified = module.add_size_strata(table, n_strata=4)
    strata = stratified.set_index("claim_id")["leading_edge_size_stratum"].to_dict()
    assert strata == {
        "c0": 2,
        "c1": 1,
        "c2": 2,
        "c3": 1,
        "c4": 4,
        "c5": 3,
        "c6": 4,
        "c7": 3,
    }
    assert module.q_value_order(table)["claim_id"].tolist() == [
        "c1",
        "c3",
        "c2",
        "c7",
        "c5",
        "c6",
        "c4",
        "c0",
    ]


def test_empirical_evidence_builder_uses_production_adapter() -> None:
    module = load_evidence_module()
    raw = pd.DataFrame(
        {
            "pathway": ["HALLMARK_A", "HALLMARK_B"],
            "NES": [2.0, -1.5],
            "pval": [0.001, 0.01],
            "padj": [0.002, 0.02],
            "leadingEdge": ["1,2,3", "4,5,6"],
        }
    )

    adapted = module.adapt_one(raw, replicate_id="balanced_delete1_001", resample_index=1)

    assert adapted["replicate_id"].tolist() == ["balanced_delete1_001"] * 2
    assert adapted["resample_index"].tolist() == [1, 1]
    assert adapted["term_id"].tolist() == ["HALLMARK_A", "HALLMARK_B"]
    assert adapted["direction"].tolist() == ["up", "down"]
    assert adapted["evidence_genes"].tolist() == [["1", "2", "3"], ["4", "5", "6"]]
