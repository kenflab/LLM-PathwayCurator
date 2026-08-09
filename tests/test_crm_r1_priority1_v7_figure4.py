from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
CRM = ROOT / "paper" / "revision" / "CRM_R1"


def load_figure_module():
    path = CRM / "scripts" / "90_plot_priority1_figure4.py"
    spec = importlib.util.spec_from_file_location("crm_r1_v7_figure4", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def synthetic_summary() -> dict:
    return {
        "stop_gate_p1": "PASS_EMPIRICAL_POINT_ESTIMATE_IMPROVED",
        "primary_replication_fraction_difference_empirical_minus_q_value": 1 / 23,
        "overlap_exact_reference": {
            "p_one_sided_empirical_greater": 0.5,
            "p_two_sided": 1.0,
        },
        "continuous_secondary": {
            "auroc": 0.713073,
            "bootstrap_ci_low": 0.557725,
            "bootstrap_ci_high": 0.857385,
            "permutation_p_one_sided": 0.005499,
        },
    }


def synthetic_source() -> pd.DataFrame:
    workflow = pd.DataFrame(
        {
            "panel": "A",
            "record_type": "workflow",
            "metric": [
                "discovery_samples",
                "balanced_resamples",
                "primary_tau",
                "frozen_k",
                "validation_samples",
            ],
            "value": [12, 81, 0.8, 23, 12],
        }
    )
    replicated = np.array([True] * 31 + [False] * 19)
    survival = np.concatenate([np.linspace(0.60, 1.0, 31), np.linspace(0.0, 0.75, 19)])
    pathway = pd.DataFrame(
        {
            "panel": "B",
            "record_type": "pathway",
            "claim_id": [f"claim-{index:02d}" for index in range(50)],
            "pathway": [f"HALLMARK_PATHWAY_{index:02d}" for index in range(50)],
            "empirical_survival_48h": survival,
            "replicated_primary": replicated,
        }
    )
    methods = pd.DataFrame(
        {
            "panel": "C",
            "record_type": "method",
            "method": [
                "raw_pool",
                "empirical_stability_audit",
                "q_value_matched",
                "q_value_and_leading_edge_size_matched",
                "random_matched_secondary",
            ],
            "n_selected": [50, 23, 23, 23, 23],
            "n_replicated": [31, 18, 17, 16, np.nan],
            "replication_fraction": [0.62, 18 / 23, 17 / 23, 16 / 23, 0.619048],
            "replication_ci_low": [0.481504, 0.580965, 0.535300, 0.491342, 0.478261],
            "replication_ci_high": [0.741372, 0.903360, 0.874514, 0.843960, 0.782609],
        }
    )
    tau = pd.DataFrame(
        {
            "panel": "D",
            "record_type": "tau_grid",
            "tau": [0.80, 0.90, 0.95, 0.98],
            "coverage": [0.46, 0.30, 0.20, 0.10],
            "replication_fraction": [18 / 23, 13 / 15, 0.8, 0.8],
            "replication_ci_low": [0.580965, 0.621180, 0.490162, 0.375535],
            "replication_ci_high": [0.903360, 0.962639, 0.943318, 0.963776],
            "nonreplication_risk": [5 / 23, 2 / 15, 0.2, 0.2],
        }
    )
    return pd.concat([workflow, pathway, methods, tau], ignore_index=True, sort=False)


def test_plot_script_is_render_only() -> None:
    source = (CRM / "scripts" / "90_plot_priority1_figure4.py").read_text()

    assert 'source_data" / "figure4.tsv' in source
    assert 'analysis_recomputed": False' in source
    assert "fgseaMultilevel" not in source
    assert "filterByExpr" not in source
    assert "voom(" not in source
    assert "llm-pathway-curator" not in source
    assert "18_validation_72h.R" not in source
    assert "19_evaluate_replication.py" not in source


def test_frozen_source_contract_and_result_invariants() -> None:
    module = load_figure_module()
    panels = module.validate_source(synthetic_source(), synthetic_summary())

    assert set(panels) == {"A", "B", "C", "D"}
    assert len(panels["B"]) == 50
    assert int(panels["B"]["replicated_primary"].sum()) == 31
    assert set(np.round(panels["D"]["tau"], 2)) == {0.8, 0.9, 0.95, 0.98}


def test_rendered_figure_has_four_readable_panels(tmp_path: Path) -> None:
    module = load_figure_module()
    figure = module.render_figure(synthetic_source(), synthetic_summary(), fontsize=12)

    assert np.allclose(figure.get_size_inches(), [8.0, 8.0])
    assert len(figure.axes) == 4
    titles = [axis.get_title(loc="left") for axis in figure.axes]
    assert titles == [
        "Frozen discovery-to-validation\ndesign",
        "Empirical stability stratifies\n72 h replication",
        "Matched-coverage replication",
        "Frozen risk–coverage sensitivity",
    ]
    all_text = "\n".join(text.get_text() for axis in figure.axes for text in axis.texts)
    assert "exact one-sided P = 0.50" in all_text
    assert "AUROC = 0.713" in all_text
    assert "Membership fixed before 72 h" in all_text

    output = tmp_path / "figure4.pdf"
    figure.savefig(output, bbox_inches="tight")
    plt.close(figure)
    assert output.stat().st_size > 10_000


def test_render_provenance_hashes_are_enforced(tmp_path: Path) -> None:
    module = load_figure_module()
    source_path = tmp_path / "figure4.tsv"
    summary_path = tmp_path / "summary.json"
    source_path.write_text("panel\trecord_type\n", encoding="utf-8")
    summary_path.write_text(json.dumps(synthetic_summary()) + "\n", encoding="utf-8")

    source_hash = hashlib.sha256(source_path.read_bytes()).hexdigest()
    summary_hash = hashlib.sha256(summary_path.read_bytes()).hexdigest()
    run_meta = {
        "outputs": {
            "figure4": {"sha256": source_hash},
            "summary": {"sha256": summary_hash},
        },
        "stop_gate_p1": "PASS_EMPIRICAL_POINT_ESTIMATE_IMPROVED",
    }
    module.verify_provenance(
        source_path=source_path,
        summary_path=summary_path,
        run_meta=run_meta,
    )

    source_path.write_text("panel\trecord_type\nA\tworkflow\n", encoding="utf-8")
    with pytest.raises(ValueError, match="source hash differs"):
        module.verify_provenance(
            source_path=source_path,
            summary_path=summary_path,
            run_meta=run_meta,
        )


def test_font_and_resolution_guards_are_publication_oriented() -> None:
    source = (CRM / "scripts" / "90_plot_priority1_figure4.py").read_text()

    assert "default=12.0" in source
    assert "default=600" in source
    assert "fontsize >= 10.0" in source
    assert "dpi >= 300" in source
    assert '"pdf.fonttype": 42' in source
    assert '"#0072B2"' in source
    assert '"#D55E00"' in source
    assert '"#CC79A7"' in source
