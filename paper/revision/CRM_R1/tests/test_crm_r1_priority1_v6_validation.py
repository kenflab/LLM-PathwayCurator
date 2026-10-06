from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
CRM = ROOT / "paper" / "revision" / "CRM_R1"


def load_evaluation_module():
    path = CRM / "scripts" / "19_evaluate_replication.py"
    spec = importlib.util.spec_from_file_location("crm_r1_v6_evaluation", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_72h_r_script_releases_gate_before_reading_counts() -> None:
    source = (CRM / "scripts" / "18_validation_72h.R").read_text()

    gate_index = source.index("freeze_gate_output <- run_freeze_gate")
    count_read_index = source.index("counts <- fread(")
    assert gate_index < count_read_index
    assert "cell_state == primary$cell_state" in source
    assert "primary$validation_time_h" in source
    assert "discovery_48h_gene_universe.tsv" in source
    assert "hallmark_gene_sets.tsv" in source
    assert "fgseaMultilevel" in source
    assert "filterByExpr" not in source
    assert "msigdbr" not in source
    assert "--force" in source and "has no --force option" in source


def test_72h_r_script_uses_same_frozen_interaction() -> None:
    source = (CRM / "scripts" / "18_validation_72h.R").read_text()

    assert "(WT_MMS - WT_UT) - (TP53_KO_MMS - TP53_KO_UT)" in source
    assert 'calcNormFactors(dge, method = "TMM")' in source
    assert "voom(dge, design, plot = FALSE)" in source
    assert "pathway_statistics_72h.tsv" in source
    assert "discovery_expression_columns_loaded = FALSE" in source


def test_wilson_interval_and_auc() -> None:
    module = load_evaluation_module()

    low, high = module.wilson_interval(0, 10)
    assert low >= -1e-15
    assert math.isclose(high, 0.2775327998628892, rel_tol=1e-12)
    assert module.binary_auc(np.array([0.1, 0.2, 0.8, 0.9]), np.array([0, 0, 1, 1])) == 1.0
    assert module.binary_auc(np.array([0.8, 0.9, 0.1, 0.2]), np.array([0, 0, 1, 1])) == 0.0


def test_overlap_exact_randomization_is_exhaustive_and_overlap_aware() -> None:
    module = load_evaluation_module()
    pathway = pd.DataFrame(
        {
            "claim_id": ["common1", "common2", "e1", "e2", "q1", "q2"],
            "replicated_primary": [True, False, True, True, False, False],
        }
    )
    empirical = {"common1", "common2", "e1", "e2"}
    q_value = {"common1", "common2", "q1", "q2"}

    summary, null = module.overlap_exact_randomization(
        pathway,
        empirical_ids=empirical,
        q_value_ids=q_value,
    )

    assert summary["common_claims"] == 2
    assert summary["assignments"] == math.comb(4, 2) == 6
    assert summary["observed_replication_fraction_difference"] == 0.5
    assert summary["p_one_sided_empirical_greater"] == 1 / 6
    assert summary["p_two_sided"] == 2 / 6
    assert null["assignments"].sum() == 6
    assert math.isclose(null["probability"].sum(), 1.0)


def test_auc_resampling_is_seeded_and_handles_single_class() -> None:
    module = load_evaluation_module()
    scores = np.array([0.1, 0.2, 0.4, 0.6, 0.8, 0.9])
    outcomes = np.array([0, 0, 0, 1, 1, 1])
    first = module.auc_resampling(
        scores,
        outcomes,
        bootstrap_draws=100,
        bootstrap_seed=11,
        permutation_draws=100,
        permutation_seed=12,
    )
    second = module.auc_resampling(
        scores,
        outcomes,
        bootstrap_draws=100,
        bootstrap_seed=11,
        permutation_draws=100,
        permutation_seed=12,
    )
    assert first == second
    assert first["status"] == "ESTIMABLE"
    assert first["auroc"] == 1.0

    single = module.auc_resampling(
        scores,
        np.ones(len(scores), dtype=int),
        bootstrap_draws=10,
        bootstrap_seed=11,
        permutation_draws=10,
        permutation_seed=12,
    )
    assert single["status"] == "NON_ESTIMABLE_SINGLE_OUTCOME_CLASS"
    assert math.isnan(single["auroc"])


def test_stop_gate_inputs_are_equal_coverage_method_summaries() -> None:
    module = load_evaluation_module()
    pathway = pd.DataFrame(
        {
            "replicated_primary": [True, True, False, False, True, False],
            "same_direction": [True, True, True, False, True, False],
            "empirical_selected": [True, True, True, False, False, False],
            "q_value_matched_selected": [False, False, True, True, True, False],
            "q_value_size_matched_selected": [True, False, False, True, True, False],
        }
    )
    summary = module.method_summary(pathway).set_index("method")
    assert summary.loc["empirical_stability_audit", "n_selected"] == 3
    assert summary.loc["q_value_matched", "n_selected"] == 3
    assert summary.loc["q_value_and_leading_edge_size_matched", "n_selected"] == 3
    assert summary.loc["empirical_stability_audit", "replication_fraction"] == 2 / 3
    assert summary.loc["q_value_matched", "replication_fraction"] == 1 / 3


def test_evaluation_refuses_output_collision_before_reading_72h_statistics() -> None:
    source = (CRM / "scripts" / "19_evaluate_replication.py").read_text()

    collision_index = source.index("Replication outputs are immutable")
    validation_read_index = source.index('validation = pd.read_csv(validation_path, sep="\\t")')
    assert collision_index < validation_read_index
    assert "--allow-post-validation" in source


def test_v6_does_not_modify_the_frozen_protocol() -> None:
    source = (CRM / "scripts" / "19_evaluate_replication.py").read_text()
    config = (CRM / "config" / "priority1_protocol.json").read_text()

    assert '"protocol_version": "CRM_R1_PRIORITY1_v5"' in config
    assert '"primary_tau": 0.8' in config
    assert (
        "write_text" not in source[source.index("config_path =") : source.index("benchmark_id =")]
    )
    assert "STOP_NO_POINT_ESTIMATE_IMPROVEMENT" in source
    assert "primary_difference > 0" in source
    assert "pathway table hash does not match its run metadata" in source
