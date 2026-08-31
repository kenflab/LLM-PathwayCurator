from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PACKAGE = Path(__file__).resolve().parents[1]
SCRIPTS = PACKAGE / "paper" / "revision" / "CRM_R1" / "scripts"
sys.path.insert(0, str(SCRIPTS))


def load_script(name: str):
    path = SCRIPTS / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


common = load_script("v11_lock_common.py")
p3 = load_script("32_lock_priority3_grades.py")
p4 = load_script("42_lock_priority4_ratings.py")


def p3_protocol() -> dict:
    return {
        "record_grading_fields": {
            "eligible": ["YES", "NO", "UNCERTAIN"],
            "exclusion_reason": ["NA", "OTHER"],
            "evidence_grade": ["EXCLUDE", "E1", "E2", "E3", "E4"],
            "direction_match": ["MATCH", "MISMATCH", "NOT_TESTED", "UNCERTAIN"],
            "context_match": ["HNSC_DIRECT", "GENERAL", "UNCERTAIN"],
            "study_design": ["TP53_PERTURBATION", "OTHER", "UNCERTAIN"],
            "data_overlap": ["INDEPENDENT", "NOT_APPLICABLE", "UNCERTAIN"],
            "contradiction": ["YES", "NO", "UNCERTAIN"],
        }
    }


def p4_protocol() -> dict:
    return {
        "minimum_independent_raters": 3,
        "questions": {
            "q1_statistical_support": [
                "SUPPORTED",
                "PARTIALLY_SUPPORTED",
                "NOT_SUPPORTED",
                "UNCERTAIN",
            ],
            "q2_external_evidence": [
                "DIRECT",
                "INDIRECT",
                "NO_SUPPORT",
                "CONTRADICTED",
                "UNCERTAIN",
            ],
            "q3_overstatement": [
                "NO_OVERSTATEMENT",
                "MINOR_OVERSTATEMENT",
                "MAJOR_OVERSTATEMENT",
                "UNCERTAIN",
            ],
        },
        "confidence_scale": [1, 2, 3, 4, 5],
    }


def test_wilson_and_overlap_exact() -> None:
    low, high = common.wilson_interval(18, 23)
    assert np.isclose(low, 0.580965, atol=1e-6)
    assert np.isclose(high, 0.903360, atol=1e-6)
    outcomes = pd.Series([True, False, True, False, True, False])
    a = pd.Series([True, True, True, False, False, False])
    b = pd.Series([True, False, False, True, True, False])
    result = common.overlap_aware_exact(outcomes, a, b, alternative="greater")
    assert result["common_claims"] == 1
    assert result["a_only_claims"] == result["b_only_claims"] == 2
    assert 0 <= result["p_one_sided"] <= 1


def test_fleiss_kappa_complete_agreement() -> None:
    ratings = pd.DataFrame({"R1": ["A", "B"], "R2": ["A", "B"], "R3": ["A", "B"]})
    assert common.fleiss_kappa(ratings, ["A", "B"]) == 1.0


def test_p3_validation_and_claim_aggregation() -> None:
    rows = []
    for index in range(50):
        rows.append(
            {
                "screening_id": f"S{index:03d}",
                "review_id": f"R{index:03d}",
                "pmid": str(1000 + index),
                "title": f"Record {index}",
                "eligible": "YES",
                "exclusion_reason": "NA",
                "evidence_grade": "E3" if index % 2 == 0 else "E2",
                "direction_match": "MATCH",
                "context_match": "HNSC_DIRECT",
                "study_design": "TP53_PERTURBATION",
                "data_overlap": "INDEPENDENT",
                "contradiction": "NO",
                "supporting_note": "Coded evidence note.",
                "curator_id": "P3_C1",
            }
        )
    completed = pd.DataFrame(rows)
    blank = completed.copy()
    for column in p3.GRADING_COLUMNS:
        blank[column] = ""
    validated = p3.validate_completed(completed, blank, p3_protocol())
    claims = p3.aggregate_claim_evidence(validated, completed["review_id"])
    assert len(claims) == 50
    assert int(claims["primary_independent_support"].sum()) == 25


def test_p4_validation_consensus_and_agreement(tmp_path: Path) -> None:
    protocol = p4_protocol()
    templates = {}
    returned = []
    for rater_index in range(1, 4):
        rater_id = f"P4_R{rater_index}"
        frame = pd.DataFrame(
            {
                "rater_id": rater_id,
                "review_id": [f"R{index:03d}" for index in range(50)],
                "packet_order": range(1, 51),
                "q1_statistical_support": "SUPPORTED",
                "q2_external_evidence": "DIRECT" if rater_index < 3 else "INDIRECT",
                "q3_overstatement": "NO_OVERSTATEMENT",
                "confidence_1_to_5": 4,
                "concise_rationale": "The fixed evidence supports this coded rating.",
            }
        )
        template = frame.copy()
        for column in p4.RATING_FIELDS:
            template[column] = ""
        templates[rater_id] = template
        path = tmp_path / f"returned_{rater_id}.tsv"
        frame.to_csv(path, sep="\t", index=False)
        returned.append(path)
    long = p4.validate_returned_ratings(returned, templates, protocol)
    consensus, agreement = p4.build_consensus(long, protocol)
    assert len(long) == 150
    assert consensus["q2_majority"].eq("DIRECT").all()
    assert not consensus["major_overstatement_majority"].any()
    assert len(agreement) == 3


def test_layout_preview_is_explicitly_result_free(tmp_path: Path) -> None:
    script = SCRIPTS / "90_plot_priority2_figure2.py"
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--layout-preview",
            "--preview-outdir",
            str(tmp_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "layout preview" in result.stdout
    meta = json.loads((tmp_path / "Fig2_layout_only_v11.run_meta.json").read_text())
    assert meta["layout_only"] is True
    assert meta["analytical_endpoint_recomputed"] is False
    assert (tmp_path / "Fig2_layout_only_v11.pdf").is_file()


def test_figure3_renderer_is_render_only_v2() -> None:
    text = (SCRIPTS / "91_plot_priority5_figure3.py").read_text()
    assert "Fig3_priority5_ontology_v2.pdf" in text
    assert '"render_only_revision": True' in text
    assert "Pdesc" in text
    assert "Fig3_priority5_ontology_v1.pdf" not in text
