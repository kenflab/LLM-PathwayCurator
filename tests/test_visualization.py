"""Protect source/wording boundaries and historical plotting inputs."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from llm_pathway_curator import viz_ranked
from llm_pathway_curator.cli import build_parser
from llm_pathway_curator.review import ReviewConfig, review_enrichment
from llm_pathway_curator.viz_ranked import (
    PlotRankedConfig,
    _find_input_tsv,
    _load_tbl,
    _normalize_audit_log,
    _normalize_claims_ranked,
    _size_transform,
    plot_ranked,
)
from llm_pathway_curator.viz_source import load_source_plot, plot_source


def make_report(tmp_path, *, k=None, drafts=True):
    evidence = tmp_path / "evidence.tsv"
    evidence.write_text(
        "term_id\tterm_name\tsource\tstat\tqval\tdirection\tevidence_genes\n"
        "A\tAlpha\tfgsea\t-2.5\t0.001\tdown\tG1;G2;G3\n"
        "B\tBeta\tfgsea\t1.2\t0.2\tup\tG2;G3;G4\n"
        "C\tMissing\tfgsea\t\t\tna\t\n"
        "D\tZero\tfgsea\t2\t0\tup\tG5\n"
    )
    card = tmp_path / "card.json"
    card.write_text('{"comparison":"treatment (positive) versus vehicle (reference)"}')
    text = tmp_path / "drafts.tsv"
    text.write_text(
        "term_id\ttext\nB\tBeta is statistically significant (NES=1.2; q=0.2).\n"
        "B\tBeta has NES=1.2 and q=0.2.\n"
    )
    result = review_enrichment(
        ReviewConfig(
            str(evidence),
            str(card),
            str(tmp_path / "report"),
            claims_file=str(text) if drafts else None,
            k_claims=k,
            modules=True,
        )
    )
    return Path(result.outdir)


def test_source_values_missingness_and_checks_remain_separate(tmp_path):
    run = make_report(tmp_path, k=1)
    before = {p.name: p.read_bytes() for p in run.iterdir()}
    rows, meta = load_source_plot(run)
    by_id = {r["term_uid"]: r for r in rows}
    assert [r["term_uid"] for r in rows] == ["fgsea:D", "fgsea:A", "fgsea:B", "fgsea:C"]
    assert by_id["fgsea:A"]["stat"] == -2.5
    assert by_id["fgsea:A"]["reason_code"] == "OUTSIDE_DECLARED_TOP_K"
    assert by_id["fgsea:B"]["VIOLATION_DETECTED"] == 1
    assert by_id["fgsea:B"]["NO_LIMITED_FLAG"] == 1
    assert by_id["fgsea:B"]["draft_count"] == 2
    assert by_id["fgsea:C"]["qval"] is None
    assert by_id["fgsea:C"]["stat"] is None
    assert by_id["fgsea:C"]["draft_count"] == 0
    assert meta["selected_count"] == 1
    out = tmp_path / "plot"
    plot_source(run, out, top_n=0)
    assert before == {p.name: p.read_bytes() for p in run.iterdir()}
    manifest = json.loads((out / "plot_manifest.json").read_text())
    assert manifest["shown_count"] == 4
    assert manifest["model_calls"] == 0
    assert not manifest["selection_recomputed"]
    for name, expected in manifest["outputs_sha256"].items():
        assert hashlib.sha256((out / name).read_bytes()).hexdigest() == expected
    assert (out / "source_plot.pdf").read_bytes().startswith(b"%PDF")
    assert "Zero adjusted values" in (out / "source_plot.svg").read_text()
    with pytest.raises(ValueError, match="new plot output"):
        plot_source(run, out)


def test_no_draft_is_distinct_from_no_flag_and_truncation_is_recorded(tmp_path):
    run = make_report(tmp_path, drafts=False)
    rows, _ = load_source_plot(run)
    assert all(r["draft_count"] == r["NO_LIMITED_FLAG"] == 0 for r in rows)
    plot_source(run, tmp_path / "plot", top_n=2)
    manifest = json.loads((tmp_path / "plot/plot_manifest.json").read_text())
    assert (manifest["candidate_count"], manifest["shown_count"]) == (4, 2)


def test_changed_source_receipt_is_rejected(tmp_path):
    run = make_report(tmp_path)
    with (run / "report.jsonl").open("a") as handle:
        handle.write("\n")
    with pytest.raises(ValueError, match="receipt"):
        plot_source(run, tmp_path / "plot")
    assert not (tmp_path / "plot").exists()


def test_source_audit_never_gets_a_synthetic_rank_score(tmp_path):
    run = make_report(tmp_path)
    with pytest.raises(SystemExit, match="plot-source"):
        _load_tbl(run / "audit_log.tsv", True)


@pytest.mark.parametrize(
    "row",
    [
        {"term_id": "A", "status": "PASS"},
        {"term_id": "A", "stability": 0.8, "status": "PASS"},
        {"term_id": "A", "stability": None, "context_score": 0.9, "status": "PASS"},
        {"term_id": "A", "stability": 0.8, "context_score": float("inf"), "status": "PASS"},
    ],
)
def test_missing_audit_factors_are_not_imputed(row):
    with pytest.raises(SystemExit):
        _normalize_audit_log(pd.DataFrame([row]), False)


@pytest.mark.parametrize("score", [None, "not-a-number", float("inf")])
def test_ranked_missing_scores_are_not_zero(score):
    with pytest.raises(SystemExit, match="finite"):
        _normalize_claims_ranked(pd.DataFrame([{"utility_score": score}]), False)


def test_nested_runs_require_explicit_choice(tmp_path):
    for run in ("a", "b"):
        (tmp_path / run).mkdir()
        (tmp_path / run / "audit_log.tsv").write_text("sample\n")
    with pytest.raises(SystemExit, match="Multiple nested runs"):
        _find_input_tsv(tmp_path)
    (tmp_path / "b/audit_log.tsv").unlink()
    (tmp_path / "a/claims_ranked.tsv").write_text("sample\n")
    assert _find_input_tsv(tmp_path) == tmp_path / "a/claims_ranked.tsv"


@pytest.mark.parametrize("mode", ["bars", "packed"])
def test_complete_historical_inputs_still_render(tmp_path, mode):
    tsv = tmp_path / "claims_ranked.tsv"
    pd.DataFrame(
        [
            {"term_name": "Alpha", "module_id": "m1", "utility_score": 0.6, "decision": "PASS"},
            {"term_name": "Beta", "module_id": "m1", "utility_score": 0.3, "decision": "PASS"},
        ]
    ).to_csv(tsv, sep="\t", index=False)
    out = plot_ranked(
        PlotRankedConfig(mode=mode, in_tsv=str(tsv), out_png=str(tmp_path / "out.png"))
    )
    assert out.read_bytes().startswith(b"\x89PNG")
    table, label = _normalize_audit_log(
        pd.DataFrame([{"term_id": "A", "stability": 0.8, "context_score": 0.5, "status": "PASS"}]),
        False,
    )
    assert table.iloc[0]["score"] == 0.4
    assert label == "stability×context"


def test_both_cli_visualizations_are_available():
    parser = build_parser()
    args = parser.parse_args(["plot-source", "--run-dir", "report", "--outdir", "plot"])
    assert args.top_n == 50
    old = parser.parse_args(["plot-ranked", "--out-png", "plot.svg", "--hide-module-labels"])
    assert old.hide_module_labels


@pytest.mark.parametrize("writeable", [True, False])
def test_size_transform_preserves_input(writeable):
    scores = np.array([-0.5, 0.5, 1.0], dtype=float)
    before = scores.copy()
    scores.setflags(write=writeable)
    result = _size_transform(scores, gamma=2.0, min_norm=0.03)
    np.testing.assert_allclose(result, [0.03, 0.25, 1.0])
    np.testing.assert_array_equal(scores, before)
    assert scores.flags.writeable == writeable


def test_packed_render_with_read_only_scores(tmp_path, monkeypatch):
    original = viz_ranked._size_transform
    calls = []

    def read_only_scores(scores, *args, **kwargs):
        scores.setflags(write=False)
        calls.append(scores.flags.writeable)
        return original(scores, *args, **kwargs)

    monkeypatch.setattr(viz_ranked, "_size_transform", read_only_scores)
    test_complete_historical_inputs_still_render(tmp_path, "packed")
    assert calls and not any(calls)
