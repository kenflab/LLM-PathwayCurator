from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
CRM = ROOT / "paper" / "revision" / "CRM_R1"


def load_script(name: str, module_name: str):
    path = CRM / "scripts" / name
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_freeze_order_and_size_matching_are_deterministic() -> None:
    module = load_script("16_freeze_priority1_membership.py", "crm_r1_v5_freeze")
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
    strata = (
        module.add_size_strata(table, n_strata=4)
        .set_index("claim_id")["leading_edge_size_stratum"]
        .to_dict()
    )
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


def test_freeze_table_serialization_is_stable() -> None:
    module = load_script("16_freeze_priority1_membership.py", "crm_r1_v5_serialize")
    table = pd.DataFrame({"claim_id": ["b", "a"], "score": [0.8, 1.0], "selected": [False, True]})
    observed = module.table_text(table)
    assert observed == "claim_id\tscore\tselected\nb\t0.80000000000000004\tFalse\na\t1\tTrue\n"


def test_freeze_checker_parses_sidecar_and_boolean_membership(tmp_path: Path) -> None:
    module = load_script("17_check_priority1_freeze.py", "crm_r1_v5_check")
    sidecar = tmp_path / "priority1_freeze_manifest.sha256"
    digest = "a" * 64
    sidecar.write_text(f"{digest}  priority1_freeze_manifest.json\n")
    assert module.parse_sidecar(sidecar, expected_name="priority1_freeze_manifest.json") == digest

    table = pd.DataFrame({"claim_id": ["c1", "c2", "c3"], "selected": ["True", "False", "true"]})
    assert module.selected_ids(table, "selected") == {"c1", "c3"}


def test_freeze_scripts_do_not_calculate_72h_statistics() -> None:
    freeze_source = (CRM / "scripts" / "16_freeze_priority1_membership.py").read_text()
    check_source = (CRM / "scripts" / "17_check_priority1_freeze.py").read_text()

    assert "pathway_statistics_72h" not in freeze_source
    assert "validation_time_h" not in freeze_source
    assert "validation_time_h" not in check_source
    assert "validation_output_absent_at_freeze" in freeze_source
    assert "--allow-post-validation" in check_source
