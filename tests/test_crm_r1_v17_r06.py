"""Synthetic saved-file layouts; no real ratings, model calls or remote writes."""

from __future__ import annotations

import csv
import importlib.util
import io
import json
import shutil
import socket
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = REPO / "paper/revision/CRM_R1/scripts"
SPEC = importlib.util.spec_from_file_location("crm_r06_test", SCRIPTS / "v17_r06.py")
r06 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(r06)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def record(path):
    return {"path": str(path), "sha256": r06.sha(path)}


def lock(path, value):
    write_json(path, value)
    path.with_suffix(".sha256").write_text(f"{r06.sha(path)}  {path.name}\n")


def save(path, frame):
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, sep="\t", index=False)
    return path


@pytest.fixture
def frozen(tmp_path):
    root, repo = tmp_path / "CRM_R1", tmp_path / "LLM-PathwayCurator"
    (root / "input").mkdir(parents=True)
    (root / "output").mkdir()
    scripts = repo / "paper/revision/CRM_R1/scripts"
    scripts.mkdir(parents=True)
    for name in ("v17_r06.py", "65_revision_r06.py"):
        shutil.copy2(SCRIPTS / name, scripts / name)
    policy = json.loads((SCRIPTS.parent / "config/r06_p4_reuse_policy.json").read_text())
    policy["bootstrap_replicates"] = 256
    policy_path = repo / "paper/revision/CRM_R1/config/r06_p4_reuse_policy.json"
    write_json(policy_path, policy)
    protocol = repo / "paper/revision/CRM_R1/config/priority4_review_protocol.json"
    write_json(protocol, {"minimum_independent_raters": 3, "questions": policy["questions"]})
    benchmark = r06.DEFAULT_BENCHMARK
    p2 = root / "output/priority2" / benchmark
    p4 = root / "output/priority4" / benchmark
    packet_root, rating_root = p4 / "packet_v1", p4 / "ratings_lock_v1"
    n = 50
    uid = [f"synthetic_uid_{i:03d}" for i in range(n)]
    reviews = [f"synthetic_review_{i:03d}" for i in range(n)]
    claims = pd.DataFrame(
        {
            "claim_uid": uid,
            "claim_text": [f"Synthetic fixed claim {i}." for i in range(n)],
            "pathway_label": [f"Synthetic pathway {i}" for i in range(n)],
            "direction": ["up" if i % 2 == 0 else "down" for i in range(n)],
            "statistic": [1.5 if i % 2 == 0 else -1.5 for i in range(n)],
            "q_value": [0.01 if i < 25 else 0.20 for i in range(n)],
            "status_full": ["PASS" if i % 2 == 0 else "ABSTAIN" for i in range(n)],
        }
    )
    membership = pd.DataFrame(
        {
            "claim_uid": uid,
            "raw_pool_selected": True,
            "q_value_matched_selected": [i < 25 for i in range(n)],
            "stability_matched_selected": [i >= 25 for i in range(n)],
            "full_audit_selected": [i % 2 == 0 for i in range(n)],
        }
    )
    sampling = pd.DataFrame(
        {"claim_uid": uid, "review_id": reviews, "packet_order": np.arange(1, n + 1)}
    )
    packet = sampling.merge(claims, on="claim_uid").drop(columns=["claim_uid", "status_full"])
    claims_path = save(p2 / "pool/claims.tsv", claims)
    membership_path = save(p2 / "membership/selection_membership.tsv", membership)
    sampling_path = save(p2 / "review/sampling_frame.locked.tsv", sampling)
    packet_path = save(packet_root / "blinded_claims.tsv", packet)
    literature_path = save(
        packet_root / "blinded_literature.private.tsv",
        pd.DataFrame({"review_id": reviews, "pmid": "synthetic"}),
    )
    p2_manifest_path = p2 / "metrics/priority2_freeze_manifest.json"
    packet_manifest_path = packet_root / "priority4_packet_manifest.json"
    rating_manifest_path = rating_root / "priority4_ratings_lock_manifest.json"
    lock(
        p2_manifest_path,
        {
            "status": "FROZEN",
            "benchmark_id": benchmark,
            "inputs": {},
            "outputs": {
                "claims": record(claims_path),
                "membership": record(membership_path),
                "sampling_frame": record(sampling_path),
            },
        },
    )
    outputs = {"claims_packet": record(packet_path), "literature_packet": record(literature_path)}
    inputs = {
        "packet_manifest": record(packet_manifest_path) if packet_manifest_path.exists() else {},
        "protocol": record(protocol),
    }
    all_ratings, raw_paths = [], []
    for rater_index in range(1, 4):
        rater = f"P4_R{rater_index}"
        template = pd.DataFrame(
            {"rater_id": rater, "review_id": reviews, "packet_order": np.arange(1, n + 1)}
        )
        for field in r06.RATING_FIELDS:
            template[field] = ""
        template_path = save(packet_root / f"ratings_template_R{rater_index}.tsv", template)
        ratings = template.copy()
        ratings["q1_statistical_support"] = [
            "UNCERTAIN"
            if rater_index == 3 and i % 4 == 0
            else "NOT_SUPPORTED"
            if i % 3 == 0
            else "PARTIALLY_SUPPORTED"
            for i in range(n)
        ]
        ratings["q2_external_evidence"] = ["DIRECT" if i % 2 == 0 else "INDIRECT" for i in range(n)]
        ratings["q3_overstatement"] = [
            "NO_OVERSTATEMENT"
            if rater_index == 2
            else "MAJOR_OVERSTATEMENT"
            if rater_index == 1 and i % 2 == 0
            else "MINOR_OVERSTATEMENT"
            if rater_index == 1 or i % 4 == 1
            else "UNCERTAIN"
            if i % 4 == 0
            else "NO_OVERSTATEMENT"
            for i in range(n)
        ]
        ratings["confidence_1_to_5"] = 3
        ratings["concise_rationale"] = "Synthetic software fixture; not expert evidence."
        raw_path = save(root / f"input/returned_ratings/rater{rater_index}/ratings.tsv", ratings)
        raw_paths.append(raw_path)
        outputs[f"ratings_R{rater_index}"] = record(template_path)
        inputs[f"blank_template_{rater}"] = record(template_path)
        inputs[f"returned_ratings_{rater_index}"] = record(raw_path)
        all_ratings.append(ratings)
    lock(
        packet_manifest_path,
        {
            "status": "PACKETS_FROZEN_RATINGS_NOT_STARTED",
            "benchmark_id": benchmark,
            "method_membership_disclosed": False,
            "ratings_inspected": False,
            "inputs": {
                "p2_manifest": record(p2_manifest_path),
                "p2_claims": record(claims_path),
                "p2_sampling_frame": record(sampling_path),
            },
            "outputs": outputs,
        },
    )
    inputs["packet_manifest"] = record(packet_manifest_path)
    long_path = save(rating_root / "ratings_long.private.tsv", pd.concat(all_ratings))
    # A deliberately unrelated legacy consensus demonstrates that it is not reused as truth.
    consensus_path = save(
        rating_root / "rating_consensus.tsv",
        pd.DataFrame({"review_id": reviews, "q3_majority": "NO_OVERSTATEMENT"}),
    )
    lock(
        rating_manifest_path,
        {
            "status": "P4_RATINGS_LOCKED_BEFORE_METHOD_UNBLINDING",
            "method_membership_read": False,
            "benchmark_id": benchmark,
            "inputs": inputs,
            "outputs": {
                "ratings_long_private": record(long_path),
                "rating_consensus": record(consensus_path),
            },
        },
    )
    return {
        "root": root,
        "repo": repo,
        "policy": policy,
        "policy_path": policy_path,
        "p2_manifest": p2_manifest_path,
        "packet_manifest": packet_manifest_path,
        "rating_manifest": rating_manifest_path,
        "claims": claims_path,
        "membership": membership_path,
        "packet": packet_path,
        "long": long_path,
        "raw_paths": raw_paths,
    }


def refresh_manifests(frozen):
    """Simulate consistent rehashed records to test content checks beyond digest checks."""
    for name in ("p2_manifest", "packet_manifest", "rating_manifest"):
        path = frozen[name]
        value = json.loads(path.read_text())
        for section in ("inputs", "outputs"):
            for item in value.get(section, {}).values():
                item["sha256"] = r06.sha(item["path"])
        lock(path, value)


def execute(frozen, **kwargs):
    return r06.run(
        frozen["root"],
        repo=frozen["repo"],
        policy_path=frozen["policy_path"],
        skip_plot=True,
        **kwargs,
    )


def test_original_linkage_and_individual_raters_without_p3_or_network(frozen, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Network access is forbidden")

    monkeypatch.setattr(socket, "create_connection", forbidden)
    before = {path: r06.sha(path) for path in frozen["root"].rglob("*") if path.is_file()}
    summary = execute(frozen)
    assert summary["status"] == "COMPLETE_EXPLORATORY_P4_COMPARISON"
    assert summary["ratings"] == 150 and summary["candidate_claims"] == 50
    assert summary["model_calls"] == 0 and summary["p3_grading_required"] is False
    assert summary["naive_llm_prose_comparison_performed"] is False
    assert all(r06.sha(path) == value for path, value in before.items())
    table = pd.read_csv(Path(summary["outdir"]) / "rater_method_endpoints.tsv", sep="\t")
    first = table.query(
        "rater_id == 'P4_R1' and method == 'legacy_full_audit' "
        "and endpoint == 'confirmed_major_overstatement'"
    ).iloc[0]
    assert first.confirmed_fraction == 1.0  # Unfavorable results are retained.
    assert set(table.rater_id) == {"P4_R1", "P4_R2", "P4_R3", "FIXED_RATER_MEAN"}
    assert not any("majority" in column for column in table)
    assert Path(summary["results_archive"]).is_file()


def test_uncertain_not_removed_or_called_safe_and_constant_bootstrap_not_zero_ci(frozen):
    summary = execute(frozen)
    output = Path(summary["outdir"])
    table = pd.read_csv(output / "rater_method_endpoints.tsv", sep="\t")
    row = table.query(
        "rater_id == 'P4_R3' and method == 'legacy_full_audit' "
        "and endpoint == 'confirmed_major_overstatement'"
    ).iloc[0]
    assert row.n_selected_claims == 25 and row.n_uncertain_ratings == 13
    assert row.confirmed_fraction == 0.0 and row.uncertainty_bound_high == 13 / 25
    assert row.ci_high > 0  # Wilson remains nonzero for zero observed events.
    delta = pd.read_csv(output / "paired_method_differences.tsv", sep="\t")
    constant = delta.query("rater_id == 'P4_R2' and endpoint == 'confirmed_major_overstatement'")
    assert constant.ci_status.eq("DEGENERATE_EMPIRICAL_BOOTSTRAP").all()
    assert constant.ci_low.isna().all() and constant.ci_high.isna().all()


@pytest.mark.parametrize("change", ["hash", "wording", "return", "category", "duplicate"])
def test_mismatches_block_all_performance_estimates(frozen, change):
    if change == "hash":
        frozen["packet"].write_text(frozen["packet"].read_text() + "\n")
    elif change == "wording":
        frame = pd.read_csv(frozen["packet"], sep="\t")
        frame.loc[0, "claim_text"] = "Changed wording that was not rated."
        save(frozen["packet"], frame)
        refresh_manifests(frozen)
    elif change == "return":
        path = frozen["raw_paths"][0]
        frame = pd.read_csv(path, sep="\t")
        frame.loc[0, "q3_overstatement"] = "NO_OVERSTATEMENT"
        save(path, frame)
        refresh_manifests(frozen)
    elif change == "category":
        for path in (frozen["raw_paths"][0], frozen["long"]):
            frame = pd.read_csv(path, sep="\t")
            frame.loc[0, "q3_overstatement"] = "FAVORABLE_NEW_LABEL"
            save(path, frame)
        refresh_manifests(frozen)
    else:
        frame = pd.read_csv(frozen["membership"], sep="\t")
        frame.loc[1, "claim_uid"] = frame.loc[0, "claim_uid"]
        save(frozen["membership"], frame)
        refresh_manifests(frozen)
    summary = execute(frozen)
    assert summary["status"] == "P4_REUSE_BLOCKED"
    assert summary["performance_estimated"] is False
    output = Path(summary["outdir"])
    assert not (output / "rater_method_endpoints.tsv").exists()
    assert not list(output.glob("Fig*.pdf"))


def test_matched_size_failure_is_not_silently_reselected(frozen):
    frame = pd.read_csv(frozen["membership"], sep="\t")
    frame.loc[0, "q_value_matched_selected"] = False
    save(frozen["membership"], frame)
    refresh_manifests(frozen)
    summary = execute(frozen)
    assert summary["blocker_code"] == "MATCHED_COVERAGE_CHANGED"


def test_k_zero_has_no_performance_claim(frozen):
    membership = pd.read_csv(frozen["membership"], sep="\t")
    for column in frozen["policy"]["methods"].values():
        if column != "raw_pool_selected":
            membership[column] = False
    save(frozen["membership"], membership)
    claims = pd.read_csv(frozen["claims"], sep="\t")
    claims["status_full"] = "ABSTAIN"
    save(frozen["claims"], claims)
    refresh_manifests(frozen)
    summary = execute(frozen)
    table = pd.read_csv(Path(summary["outdir"]) / "rater_method_endpoints.tsv", sep="\t")
    full = table[table.method.eq("legacy_full_audit")]
    assert summary["legacy_full_audit_selected"] == 0
    assert full.confirmed_fraction.isna().all() and full.ci_status.eq("UNDEFINED").all()
    assert full.coverage.eq(0).all()


def test_identical_memberships_preserve_overlap_in_paired_bootstrap(frozen):
    membership = pd.read_csv(frozen["membership"], sep="\t")
    membership["q_value_matched_selected"] = membership.full_audit_selected
    save(frozen["membership"], membership)
    refresh_manifests(frozen)
    summary = execute(frozen)
    output = Path(summary["outdir"])
    table = pd.read_csv(output / "paired_method_differences.tsv", sep="\t")
    same = table[table.method_b.eq("q_value_matched")]
    assert same.difference_confirmed_fraction.eq(0).all()
    assert same.ci_status.eq("DEGENERATE_EMPIRICAL_BOOTSTRAP").all()
    overlap = pd.read_csv(output / "method_overlap.tsv", sep="\t")
    same_overlap = overlap[overlap.method_b.eq("q_value_matched")].iloc[0]
    assert same_overlap.n_common == 25 and same_overlap.n_a_only == same_overlap.n_b_only == 0


def test_no_live_flag_and_no_overwrite(frozen):
    result = subprocess.run(
        [sys.executable, str(SCRIPTS / "65_revision_r06.py"), "--live"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2 and "unrecognized arguments" in result.stderr
    output = frozen["root"] / "output/revision_v17/fixed_test_output"
    execute(frozen, outdir=output)
    with pytest.raises(r06.ReuseBlocked, match="immutable"):
        execute(frozen, outdir=output)


def test_historical_paths_rebase_using_full_path_and_reject_escape(frozen):
    inputs = r06.Inputs(frozen["root"], frozen["repo"])
    relative = frozen["claims"].relative_to(frozen["root"])
    assert inputs.resolve("/Users/old/OneDrive/CRM_R1/" + str(relative)) == frozen["claims"]
    protocol = frozen["repo"] / "paper/revision/CRM_R1/config/priority4_review_protocol.json"
    assert (
        inputs.resolve(
            "/Users/old/projects/LLM-PathwayCurator/paper/revision/CRM_R1/config/"
            "priority4_review_protocol.json"
        )
        == protocol
    )
    with pytest.raises(r06.ReuseBlocked):
        inputs.resolve("output/../input/private.tsv")


def test_wilson_and_degenerate_agreement():
    assert np.allclose(r06.wilson(18, 23), [0.580965, 0.903360], atol=1e-6)
    assert r06.wilson(0, 0) == (None, None)
    assert r06.fleiss(np.full((50, 3), "ONE_CATEGORY"))[0] is None
    assert r06.fleiss(np.array([["A"] * 3, ["B"] * 3]))[0] == 1.0


def test_rendering_uses_saved_endpoint_tables(frozen):
    summary = r06.run(
        frozen["root"],
        repo=frozen["repo"],
        policy_path=frozen["policy_path"],
        skip_plot=False,
    )
    output = Path(summary["outdir"])
    assert (output / "Fig_R06_P4_major_overstatement_by_rater.pdf").stat().st_size > 1000
    assert (output / "Fig_R06_P4_major_overstatement_by_rater.png").stat().st_size > 1000


def test_changed_input_during_estimation_hard_fails_before_performance_export(frozen, monkeypatch):
    real_evaluate = r06.evaluate

    def changed(*args):
        tables = real_evaluate(*args)
        frozen["long"].write_text(frozen["long"].read_text() + "\n")
        return tables

    monkeypatch.setattr(r06, "evaluate", changed)
    with pytest.raises(r06.ReuseBlocked, match="changed"):
        execute(frozen)
    outputs = list((frozen["root"] / "output/revision_v17").glob("r06_*"))
    assert len(outputs) == 1
    assert (outputs[0] / "FAILED.json").is_file()
    assert not (outputs[0] / "rater_method_endpoints.tsv").exists()
    assert not (outputs[0] / "summary.json").exists()


def test_missing_original_return_is_reported_without_requesting_new_ratings(frozen):
    frozen["raw_paths"][1].unlink()
    summary = execute(frozen)
    assert summary["status"] == "P4_REUSE_BLOCKED"
    assert summary["blocker_code"] == "MISSING_INPUT"
    assert summary["new_expert_ratings"] == summary["model_calls"] == 0


def add_spreadsheet_supplement(path):
    """Synthetic Excel export with two blank headers and ungraded side notes."""
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle, delimiter="\t"))
    rows[0].extend(["", ""])
    for i, row in enumerate(rows[1:]):
        row.extend(["", "MAJOR_OVERSTATEMENT" if i < 2 else ""])
    output = io.StringIO(newline="")
    csv.writer(output, delimiter="\t", lineterminator="\n").writerows(rows)
    path.write_text(output.getvalue(), encoding="utf-8")


def test_unnamed_supplement_is_preserved_without_changing_any_rating_or_result(frozen):
    original = execute(frozen)
    original_output = Path(original["outdir"])
    raw = frozen["raw_paths"][1]
    add_spreadsheet_supplement(raw)
    refresh_manifests(frozen)
    before = r06.sha(raw)
    updated = execute(frozen)
    assert updated["status"] == "COMPLETE_EXPLORATORY_P4_COMPARISON"
    assert updated["model_calls"] == updated["new_expert_ratings"] == 0
    assert r06.sha(raw) == before
    output = Path(updated["outdir"])
    for name in ("rater_method_endpoints.tsv", "paired_method_differences.tsv"):
        pd.testing.assert_frame_equal(
            pd.read_csv(original_output / name, sep="\t"),
            pd.read_csv(output / name, sep="\t"),
        )
    checks = json.loads((output / "table_read_checks.private.json").read_text())
    check = next(row for row in checks["tables"] if row["path"] == str(raw))
    assert check["status"] == "PARSED_WITH_UNNAMED_COLUMNS"
    assert check["unnamed_header_positions_1_based"] == [9, 10]
    assert check["unnamed_column_nonempty_counts"] == {"Unnamed: 8": 0, "Unnamed: 9": 2}
    assert not check["input_bytes_rewritten"]
    reader = r06.Inputs(frozen["root"], frozen["repo"])
    parsed = reader.table(raw)
    assert parsed["Unnamed: 9"].iloc[0] == "MAJOR_OVERSTATEMENT"
    assert parsed.q3_overstatement.eq("NO_OVERSTATEMENT").all()


@pytest.mark.parametrize("prefix", ["\n", " \n\n", "\ufeff\n\n", "\ufeff \r\n"])
def test_leading_blank_lines_and_bom_use_original_pandas_field_resolution(frozen, prefix):
    raw = frozen["raw_paths"][1]
    raw.write_text(prefix + raw.read_text(), encoding="utf-8")
    refresh_manifests(frozen)
    before = r06.sha(raw)
    summary = execute(frozen)
    assert summary["status"] == "COMPLETE_EXPLORATORY_P4_COMPARISON"
    assert r06.sha(raw) == before
    checks = json.loads((Path(summary["outdir"]) / "table_read_checks.private.json").read_text())
    check = next(row for row in checks["tables"] if row["path"] == str(raw))
    assert check["leading_blank_lines"] >= 1
    assert check["pandas_columns"] == check["raw_header"]


@pytest.mark.parametrize("same_values", [True, False])
def test_duplicate_named_rating_column_is_never_chosen_automatically(frozen, same_values):
    raw = frozen["raw_paths"][1]
    rows = list(csv.reader(io.StringIO(raw.read_text()), delimiter="\t"))
    field = "q3_overstatement"
    index = rows[0].index(field)
    rows[0].append(field)
    for row in rows[1:]:
        row.append(row[index] if same_values else "MAJOR_OVERSTATEMENT")
    output = io.StringIO(newline="")
    csv.writer(output, delimiter="\t", lineterminator="\n").writerows(rows)
    raw.write_text(output.getvalue())
    refresh_manifests(frozen)
    summary = execute(frozen)
    assert summary["status"] == "P4_REUSE_BLOCKED"
    assert summary["blocker_code"] == "DUPLICATE_COLUMNS"
    assert not summary["descriptive_p4_comparisons_estimated"]
    checks = json.loads((Path(summary["outdir"]) / "table_read_checks.private.json").read_text())
    check = next(row for row in checks["tables"] if row["path"] == str(raw))
    assert check["duplicate_named_headers"] == [field]


def test_unnamed_supplements_do_not_relax_frozen_score_correspondence(frozen):
    raw = frozen["raw_paths"][1]
    add_spreadsheet_supplement(raw)
    frame = pd.read_csv(raw, sep="\t", dtype=str, keep_default_na=False)
    frame.loc[0, "q3_overstatement"] = "MAJOR_OVERSTATEMENT"
    output = io.StringIO(newline="")
    csv.writer(output, delimiter="\t", lineterminator="\n").writerows(
        [[*frame.columns[:8], "", ""], *frame.to_numpy().tolist()]
    )
    raw.write_text(output.getvalue())
    refresh_manifests(frozen)
    summary = execute(frozen)
    assert summary["blocker_code"] == "ORIGINAL_RETURN_DIFFERS"
    assert not summary["descriptive_p4_comparisons_estimated"]


def test_missing_rating_header_is_not_inferred_from_blank_column_position(frozen):
    raw = frozen["raw_paths"][1]
    text = raw.read_text()
    raw.write_text(text.replace("q1_statistical_support", "", 1))
    refresh_manifests(frozen)
    summary = execute(frozen)
    assert summary["blocker_code"] == "BAD_TABLE_SCHEMA"
    assert not summary["descriptive_p4_comparisons_estimated"]


def test_tab_only_row_is_not_skipped_or_treated_as_a_blank_line(frozen):
    raw = frozen["raw_paths"][1]
    raw.write_text("\t\t\n" + raw.read_text())
    refresh_manifests(frozen)
    summary = execute(frozen)
    assert summary["blocker_code"] == "EMPTY_TABLE"
    checks = json.loads((Path(summary["outdir"]) / "table_read_checks.private.json").read_text())
    check = next(row for row in checks["tables"] if row["path"] == str(raw))
    assert check["raw_header"] == ["", "", ""]
    assert check["leading_blank_lines"] == 0


def test_parse_failure_is_archived_as_a_block_with_headers_and_no_estimates(frozen):
    raw = frozen["raw_paths"][1]
    header = raw.read_text().splitlines()[0]
    raw.write_text(header + '\n"Unclosed field\n')
    refresh_manifests(frozen)
    summary = execute(frozen)
    assert summary["blocker_code"] == "TABLE_PARSE_FAILED"
    output = Path(summary["outdir"])
    checks = json.loads((output / "table_read_checks.private.json").read_text())
    assert checks["tables"][-1]["status"] == "TABLE_PARSE_FAILED"
    assert Path(summary["results_archive"]).is_file()
    assert not (output / "rater_method_endpoints.tsv").exists()
