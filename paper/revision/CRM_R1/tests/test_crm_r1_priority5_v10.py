from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
SCRIPTS = ROOT / "paper/revision/CRM_R1/scripts"
CONFIG = ROOT / "paper/revision/CRM_R1/config/priority5_protocol.json"
UMBRELLA_CONFIG = ROOT / "paper/revision/CRM_R1/config/priorities2_5_protocol.json"


def load_script(filename: str):
    path = SCRIPTS / filename
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_priority5_protocol_has_safe_parallel_boundary() -> None:
    protocol = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert protocol["protocol_version"] == "CRM_R1_PRIORITY5_v10_3"
    assert protocol["status"] == "PRESPECIFIED_AWAITING_INPUT_FREEZE"
    assert protocol["parallel_work_boundary"][
        "ontology_evaluation_may_run_before_priority3_priority4_lock"
    ]
    assert not protocol["parallel_work_boundary"]["priority3_priority4_outcomes_may_be_read"]
    go = protocol["ontology_sources"]["go"]
    assert go["allowed_relations"] == ["is_a", "part_of"]
    assert {"has_part", "regulates"}.issubset(go["excluded_relations"])
    assert protocol["hierarchy_estimands"]["primary_scope"] == "direct_parent_child"
    assert (
        protocol["hierarchy_estimands"]["sensitivity_scope"]
        == "safe_ancestor_descendant_including_direct_edges"
    )
    assert protocol["hierarchy_estimands"]["minimum_pairs_for_primary_estimate"] == 10
    assert protocol["audit_inputs"]["k_claims_per_collection"] == 500
    census = protocol["audit_inputs"]["candidate_census"]
    assert census["membership_columns_read"] == ["claim_id", "entity", "direction"]
    assert census["audit_status_columns_read"] is False
    assert census["status_dependent_reselection"] is False
    assert census["hierarchy_outcomes_read"] is False


def test_candidate_census_uses_membership_only_and_preserves_order(
    tmp_path: Path,
) -> None:
    module = load_script("49_lock_priority5_candidate_census.py")
    module.EXPECTED_K = 2
    audit_path = tmp_path / "audit_log.tsv"
    pd.DataFrame(
        {
            "claim_id": ["claim-b", "claim-a"],
            "entity": ["TERM_B", "TERM_A"],
            "direction": ["DOWN", "UP"],
            "status": ["ABSTAIN", "PASS"],
            "context_evaluated": [False, True],
        }
    ).to_csv(audit_path, sep="\t", index=False)
    membership = module.load_membership_only(audit_path, "C5_GO_BP")
    assert membership["entity"].tolist() == ["TERM_B", "TERM_A"]
    assert membership["direction"].tolist() == ["down", "up"]
    assert "status" not in membership.columns
    assert "context_evaluated" not in membership.columns

    evidence_path = tmp_path / "evidence.tsv"
    pd.DataFrame(
        {
            "term_id": ["TERM_A", "TERM_B", "TERM_C"],
            "direction": ["up", "down", "up"],
            "stat": [1.0, -2.0, 3.0],
        }
    ).to_csv(evidence_path, sep="\t", index=False)
    census = module.build_census_evidence(membership, evidence_path, "C5_GO_BP")
    assert census["term_id"].tolist() == ["TERM_B", "TERM_A"]


def test_priority5_freeze_requires_complete_llm_review_and_locked_membership(
    tmp_path: Path,
) -> None:
    module = load_script("50_freeze_priority5_inputs.py")
    module.EXPECTED_K = 2
    expected = pd.DataFrame({"entity": ["TERM_A", "TERM_B"], "direction": ["up", "down"]})
    audit_path = tmp_path / "audit_log.tsv"
    audit = pd.DataFrame(
        {
            "claim_id": ["a", "b"],
            "entity": ["TERM_A", "TERM_B"],
            "direction": ["up", "down"],
            "status": ["PASS", "ABSTAIN"],
            "gene_ids": ["1;2", "3;4"],
            "tau_used": [0.9, 0.9],
            "context_review_mode": ["llm", "llm"],
            "context_evaluated": [True, False],
            "context_status": ["PASS", ""],
            "context_method": ["llm", "none"],
        }
    )
    audit.to_csv(audit_path, sep="\t", index=False)
    try:
        module.validate_audit_log(audit_path, "C5_GO_BP", expected)
    except ValueError as error:
        assert "missing context evaluations" in str(error)
    else:
        raise AssertionError("Incomplete context review was accepted")

    audit.loc[1, "context_evaluated"] = True
    audit.loc[1, "context_status"] = "WARN"
    audit.loc[1, "context_method"] = "llm"
    audit.to_csv(audit_path, sep="\t", index=False)
    summary = module.validate_audit_log(audit_path, "C5_GO_BP", expected)
    assert summary == {"rows": 2, "pass": 1, "abstain": 1, "fail": 0}

    wrong = expected.copy()
    wrong.loc[1, "direction"] = "up"
    try:
        module.validate_audit_log(audit_path, "C5_GO_BP", wrong)
    except ValueError as error:
        assert "differs from locked census" in str(error)
    else:
        raise AssertionError("Membership drift was accepted")


def test_v10_preserves_v8_protocol_contract_and_strict_zip_calls() -> None:
    assert (
        hashlib.sha256(UMBRELLA_CONFIG.read_bytes()).hexdigest()
        == "9e84af9c6f5282f08e4034d512efeb7efbcccf4183ee5c27319809f9a13a5086"
    )
    umbrella = json.loads(UMBRELLA_CONFIG.read_text(encoding="utf-8"))
    assert umbrella["priority5"]["ontology_release_dates"] is None
    plot_text = (SCRIPTS / "91_plot_priority5_figure3.py").read_text(encoding="utf-8")
    assert plot_text.count("strict=True") == 4


def test_go_parser_uses_only_is_a_and_part_of(tmp_path: Path) -> None:
    module = load_script("52_evaluate_ontology_hierarchy.py")
    obo = tmp_path / "go-basic.obo"
    obo.write_text(
        """format-version: 1.2
data-version: releases/2026-08-01

[Term]
id: GO:0008150
name: biological process
namespace: biological_process

[Term]
id: GO:1000001
name: cell cycle
namespace: biological_process
is_a: GO:0008150 ! biological process

[Term]
id: GO:1000002
name: mitotic cell cycle
namespace: biological_process
is_a: GO:1000001 ! cell cycle
relationship: regulates GO:0008150 ! biological process

[Term]
id: GO:1000003
name: DNA replication
namespace: biological_process
relationship: part_of GO:1000002 ! mitotic cell cycle
relationship: has_part GO:0008150 ! biological process
""",
        encoding="utf-8",
    )
    names, parents = module.parse_go_obo(obo)
    assert set(names) == {"GO:0008150", "GO:1000001", "GO:1000002", "GO:1000003"}
    assert parents["GO:1000002"] == {"GO:1000001"}
    assert parents["GO:1000003"] == {"GO:1000002"}
    depths = module.minimum_depths(names, parents)
    assert depths["GO:1000003"] == 3


def test_reactome_parser_filters_species_and_preserves_edge_direction(
    tmp_path: Path,
) -> None:
    module = load_script("52_evaluate_ontology_hierarchy.py")
    pathways = tmp_path / "ReactomePathways.txt"
    relations = tmp_path / "ReactomePathwaysRelation.txt"
    pathways.write_text(
        "R-HSA-1\tParent pathway\tHomo sapiens\n"
        "R-HSA-2\tChild pathway\tHomo sapiens\n"
        "R-MMU-1\tMouse pathway\tMus musculus\n",
        encoding="utf-8",
    )
    relations.write_text("R-HSA-1\tR-HSA-2\nR-MMU-1\tR-HSA-2\n", encoding="utf-8")
    names, parents = module.parse_reactome(pathways, relations)
    assert names == {"R-HSA-1": "Parent pathway", "R-HSA-2": "Child pathway"}
    assert parents == {"R-HSA-2": {"R-HSA-1"}}


def test_mapping_and_pair_metrics_are_directional() -> None:
    module = load_script("52_evaluate_ontology_hierarchy.py")
    names = {
        "GO:0008150": "biological process",
        "GO:1000001": "cell cycle",
        "GO:1000002": "mitotic cell cycle",
    }
    parents = {
        "GO:1000001": {"GO:0008150"},
        "GO:1000002": {"GO:1000001"},
    }
    audit = pd.DataFrame(
        {
            "claim_id": ["a", "b"],
            "entity": ["GOBP_CELL_CYCLE", "GOBP_MITOTIC_CELL_CYCLE"],
            "status": ["PASS", "ABSTAIN"],
            "direction": ["up", "down"],
            "gene_ids": ["'1;2;3", "'2;3;4"],
        }
    )
    depths = module.minimum_depths(names, parents)
    mapped, qc = module.map_audit_terms(audit, collection="C5_GO_BP", names=names, depths=depths)
    assert qc["mapping_status"].eq("MAPPED_UNIQUE").all()
    pairs = module.build_pairs(mapped, parents, module.ancestor_sets(names, parents))
    direct = pairs.loc[pairs["relation_scope"].eq("direct_parent_child")].iloc[0]
    assert direct["parent_name"] == "cell cycle"
    assert direct["child_name"] == "mitotic cell cycle"
    assert direct["directional_contradiction"] == 1
    assert direct["intersection_n"] == 2
    assert direct["child_covered_by_parent"] == 2 / 3


def test_matched_nonedge_reference_is_seeded_and_vectorized() -> None:
    module = load_script("52_evaluate_ontology_hierarchy.py")
    mapped = pd.DataFrame(
        {
            "collection": ["C5_GO_BP"] * 4,
            "claim_id": ["a", "b", "c", "d"],
            "entity": ["A", "B", "C", "D"],
            "ontology_id": ["A", "B", "C", "D"],
            "ontology_name": ["A", "B", "C", "D"],
            "depth": [1, 2, 1, 2],
            "status": ["PASS"] * 4,
            "direction": ["up", "up", "down", "down"],
            "gene_ids": [
                frozenset({"1", "2"}),
                frozenset({"2", "3"}),
                frozenset({"4", "5"}),
                frozenset({"5", "6"}),
            ],
            "gene_n": [2, 2, 2, 2],
        }
    )
    parents = {"B": {"A"}, "D": {"C"}}
    ancestors = {"A": set(), "B": {"A"}, "C": set(), "D": {"C"}}
    observed = module.build_pairs(mapped, parents, ancestors)
    null_a, summary_a = module.matched_nonedge_reference(
        mapped, observed, ancestors, draws=100, seed=7
    )
    null_b, summary_b = module.matched_nonedge_reference(
        mapped, observed, ancestors, draws=100, seed=7
    )
    pd.testing.assert_frame_equal(null_a, null_b)
    assert summary_a == summary_b
    assert len(null_a) == 100


def test_utility_grid_and_rank_sensitivity_contract() -> None:
    module = load_script("53_evaluate_utility_sensitivity.py")
    assert len(module.weight_grid(0.25)) == 35
    table = pd.DataFrame(
        {
            "claim_id": [f"C{i:02d}" for i in range(1, 51)],
            "statistical_support": [i / 50 for i in range(1, 51)],
            "stability_support": [1 - (i - 1) / 50 for i in range(1, 51)],
            "independent_evidence": [0.8] * 50,
            "wording_safety": [0.9] * 50,
        }
    )
    ranks = module.utility_scores(table)
    assert ranks["aggregation"].nunique() == 38
    assert len(ranks) == 50 * 38
    summary = module.summarize_sensitivity(ranks, top_k=25, material_shift=10)
    assert len(summary) == 38
    primary = summary.loc[summary["aggregation"].eq("multiplicative")].iloc[0]
    assert primary["top_k_overlap_n"] == 25
    assert primary["maximum_absolute_rank_shift"] == 0


def test_ontology_script_has_no_priority3_or_priority4_input_path() -> None:
    text = (SCRIPTS / "52_evaluate_ontology_hierarchy.py").read_text(encoding="utf-8")
    assert "/priority3/" not in text
    assert "/priority4/" not in text
    plot_text = (SCRIPTS / "91_plot_priority5_figure3.py").read_text(encoding="utf-8")
    for forbidden in (
        "audit_log.tsv",
        "evidence_table.tsv",
        "go-basic.obo",
        "expression",
    ):
        assert forbidden not in plot_text
