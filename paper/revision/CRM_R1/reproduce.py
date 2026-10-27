#!/usr/bin/env python3
"""Reproduce manuscript summaries and figures from deposited source records.

No expression refits, model requests, new human labels, or study relocking.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import platform
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
GROUNDING_SHA256 = "a7b236e00c336e32f56a6bb374b79f44f4a1dc11605ff5b377622f3800799333"
METHODS = ["raw_pool", "q_value_matched", "stability_matched", "full_audit"]


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return pd.read_csv(path, sep="\t", keep_default_na=False, float_precision="round_trip")


def write(path, frame):
    frame.to_csv(path, sep="\t", index=False, float_format="%.17g")


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    loaded = importlib.util.module_from_spec(spec)
    sys.modules[name] = loaded
    spec.loader.exec_module(loaded)
    return loaded


def same(actual, expected, keys, label):
    require(set(actual.columns) == set(expected.columns), f"{label}: columns differ")
    require(not actual.duplicated(keys).any(), f"{label}: duplicate computed keys")
    require(not expected.duplicated(keys).any(), f"{label}: duplicate saved keys")
    a = actual.sort_values(keys).reset_index(drop=True)[expected.columns]
    b = expected.sort_values(keys).reset_index(drop=True)
    pd.testing.assert_frame_equal(
        a, b, check_dtype=False, check_exact=False, atol=1e-12, rtol=1e-10, obj=label
    )


def verify_inputs(data):
    manifest = json.loads((HERE / "data_manifest.json").read_text())
    for name, record in manifest["files"].items():
        path = (data / name).resolve()
        require(path.is_relative_to(data.resolve()), f"Invalid manifest path: {name}")
        require(path.is_file(), f"Missing source file: {name}")
        require(sha(path) == record["sha256"], f"Source checksum differs: {name}")
    return manifest


def agreement(a, b):
    require(len(a) == len(b) and len(a) > 0, "Paired ratings required")
    n, ca, cb = len(a), Counter(a), Counter(b)
    count = sum(x == y for x, y in zip(a, b, strict=True))
    observed = count / n
    expected = sum(ca[x] * cb[x] for x in set(a) | set(b)) / n**2
    return dict(
        n=n,
        agree=count,
        observed_agreement=observed,
        expected_marginal_agreement=expected,
        cohen_kappa=(observed - expected) / (1 - expected) if expected < 1 else None,
        constant_rater_A=len(ca) == 1,
        constant_rater_B=len(cb) == 1,
    )


def dex(data, out):
    source = data / "Supplementary_Data_2_DEX"
    code = REPO / "src/llm_pathway_curator/grounding.py"
    require(sha(code) == GROUNDING_SHA256, "Checker differs from the recorded evaluation version")
    checker = module(code, "manuscript_grounding")
    texts = read(source / "DEX_text_checks_50.tsv")
    require(len(texts) == 50 and texts.term_id.is_unique, "DEX text census differs")
    checked = []
    for row in texts.itertuples():
        require(
            hashlib.sha256(row.text.encode()).hexdigest() == row.text_sha256,
            f"Text checksum differs: {row.term_id}",
        )
        evidence = {
            "stat": row.NES,
            "stat_kind": "NES",
            "qval": row.BH_q,
            "direction": "up" if row.NES > 0 else "down" if row.NES < 0 else "na",
        }
        result = checker.inspect_text(row.text, evidence, 0.05)
        errors = ";".join(
            sorted({f["code"] for f in result["findings"] if f["severity"] == "ERROR"})
        )
        require(
            result["limited_checks_status"] == row.limited_checks_status
            and errors == row.error_codes,
            f"Check outcome differs: {row.term_id}",
        )
        checked.append(
            {**row._asdict(), "findings_json": json.dumps(result["findings"], ensure_ascii=False)}
        )
    write(out / "DEX_text_checks_50.tsv", pd.DataFrame(checked))
    returned = read(source / "DEX_returned_categories_48.tsv")
    cases = read(source / "DEX_case_ratings_24.tsv")
    require(
        len(returned) == 48 and len(cases) == 24 and cases.reference_id.is_unique,
        "DEX rating census differs",
    )
    require(not returned.duplicated(["judge_id", "reference_id"]).any(), "Duplicate DEX rating")
    require(set(returned.judge_id) == {"AUTHOR_J1", "AUTHOR_J2"}, "DEX evaluators differ")
    sub = [returned[returned.judge_id == f"AUTHOR_J{j}"].set_index("reference_id") for j in (1, 2)]
    for frame in sub:
        require(set(frame.index) == set(cases.reference_id), "DEX case identities differ")
    endpoints = []
    for field, positive, suffix, label in [
        (
            "material_source_status",
            "DEFINITE_MATERIAL_ERROR",
            "material",
            "material source contradiction",
        ),
        ("qualitative_support_status", "OVERSTATED", "strength", "qualitative overstatement"),
    ]:
        vals = [frame.loc[cases.reference_id, field].tolist() for frame in sub]
        for j, values in enumerate(vals, 1):
            require(values == cases[f"J{j}_{suffix}"].tolist(), "Exported DEX categories differ")
        endpoints.append(
            dict(
                endpoint=label,
                J1_positive=vals[0].count(positive),
                J1_n=24,
                J2_positive=vals[1].count(positive),
                J2_n=24,
                **agreement(*vals),
            )
        )
    frame = pd.DataFrame(endpoints)
    same(frame, read(source / "DEX_endpoint_summary.tsv"), ["endpoint"], "DEX endpoints")
    write(out / "DEX_endpoint_summary.tsv", frame)
    counts = []
    for j, frame in enumerate(sub, 1):
        for field in ("presentation", "support", "usefulness"):
            for choice, n in Counter(frame[field].replace("", "None")).items():
                counts.append(dict(judge=f"J{j}", field=field, choice=choice, n=n))
    frame = pd.DataFrame(counts)
    same(
        frame,
        read(source / "DEX_interpretation_counts.tsv"),
        ["judge", "field", "choice"],
        "DEX interpretation counts",
    )
    write(out / "DEX_interpretation_counts.tsv", frame)
    return {
        "counts_50": dict(Counter(texts.limited_checks_status)),
        "counts_24": dict(Counter(cases.tool_status)),
        "endpoints": endpoints,
        "accuracy_estimated": False,
        "checker_sha256": GROUNDING_SHA256,
    }


def hnsc(data, out):
    source = data / "Supplementary_Data_3_External_HNSC"
    historical = module(HERE / "scripts/v17_r06.py", "manuscript_hnsc")
    policy = json.loads((HERE / "config/r06_p4_reuse_policy.json").read_text())
    ratings = read(source / "statement_ratings.tsv")
    require(
        len(ratings) == 150 and not ratings.duplicated(["claim_uid", "rater_id"]).any(),
        "HNSC rating census differs",
    )
    require(ratings.groupby("claim_uid").claim_text.nunique().eq(1).all(), "HNSC text differs")
    # First-occurrence order retains the original review order. Do not sort claim_uid.
    ids, raters = ratings.claim_uid.drop_duplicates().tolist(), sorted(ratings.rater_id.unique())
    require(len(ids) == 50 and len(raters) == 3, "HNSC statement/evaluator census differs")
    for question, choices in policy["questions"].items():
        require(ratings[question].isin(choices).all(), f"Invalid category: {question}")
    rng = np.random.default_rng(policy["bootstrap_seed"])
    draws = rng.multinomial(
        len(ids), np.full(len(ids), 1 / len(ids)), size=policy["bootstrap_replicates"]
    )
    results = historical.agreement_tables(
        ratings.rename(columns={"claim_uid": "review_id"}), ids, raters, policy, draws
    )
    for name, keys in [
        ("interrater_agreement.tsv", ["question"]),
        ("rater_category_counts.tsv", ["rater_id", "question", "category"]),
    ]:
        same(results[name], read(source / name), keys, "HNSC " + name)
        write(out / ("HNSC_" + name), results[name])
    return {
        "statements": 50,
        "evaluators": 3,
        "seed": policy["bootstrap_seed"],
        "bootstrap_replicates": policy["bootstrap_replicates"],
        "claim_order": ids,
    }


def split_replication(data, out):
    source = data / "Supplementary_Data_4_Comparisons/split_replication"
    rows = read(source / "figure2_split_method_replication.tsv")
    require(
        len(rows) == 1760 and not rows.duplicated(["cohort_id", "split_id", "method_id"]).any(),
        "Split census differs",
    )
    require(set(rows.method_id) == set(METHODS), "Split methods differ")
    require(
        (
            (rows.n_replicated >= 0)
            & (rows.n_replicated <= rows.n_direction_concordant)
            & (rows.n_direction_concordant <= rows.n_selected)
            & (rows.n_selected > 0)
        ).all(),
        "Invalid split counts",
    )
    require(
        np.allclose(
            rows.n_replicated / rows.n_selected, rows.replication_fraction, atol=1e-12, rtol=0
        ),
        "Split replication fractions differ",
    )
    sizes = rows.pivot(index=["cohort_id", "split_id"], columns="method_id", values="n_selected")
    require(len(sizes) == 440 and not sizes.isna().any().any(), "Missing split methods")
    require(
        sizes.raw_pool.eq(50).all() and sizes[METHODS[1:]].nunique(axis=1).eq(1).all(),
        "Selected K is not matched",
    )
    cohort = (
        rows.groupby(["cohort_id", "method_id"], sort=True)
        .agg(
            mean_replication_fraction=("replication_fraction", "mean"),
            sd_replication_fraction_across_splits=("replication_fraction", "std"),
            n_splits=("split_id", "size"),
            mean_selected_k=("n_selected", "mean"),
            min_selected_k=("n_selected", "min"),
            max_selected_k=("n_selected", "max"),
        )
        .reset_index()
    )
    require(
        cohort.n_splits.eq(20).all() and cohort.cohort_id.nunique() == 22, "Cohort census differs"
    )
    same(
        cohort,
        read(source / "figure2_panelB_cohort_replication.tsv"),
        ["cohort_id", "method_id"],
        "Cohort aggregation",
    )
    write(out / "split_cohort_replication.tsv", cohort)
    pivot = cohort.pivot(
        index="cohort_id", columns="method_id", values="mean_replication_fraction"
    ).sort_index()
    intervals = []
    for filename, contrast in [
        ("figure2_panelB_method_summary.tsv", False),
        ("figure2_panelC_contrast_summary.tsv", True),
    ]:
        for row in read(source / filename).itertuples():
            values = (
                pivot[row.method_a] - pivot[row.method_b] if contrast else pivot[row.method_id]
            ).to_numpy()
            require(
                row.n_cohorts == 22 and row.bootstrap_draws == 10000, "Bootstrap design differs"
            )
            rng = np.random.default_rng(row.bootstrap_seed)
            draws = rng.choice(values, size=(row.bootstrap_draws, len(values)), replace=True)
            low, high = np.quantile(draws.mean(axis=1), [0.025, 0.975])
            expected = (
                row.mean_cohort_difference_a_minus_b
                if contrast
                else row.mean_cohort_replication_fraction
            )
            require(
                np.allclose(
                    [values.mean(), low, high],
                    [expected, row.cohort_bootstrap_ci_low, row.cohort_bootstrap_ci_high],
                    atol=1e-12,
                    rtol=0,
                ),
                "Cohort bootstrap differs",
            )
            intervals.append(
                dict(
                    table=filename,
                    method_or_comparator=row.method_b if contrast else row.method_id,
                    estimate=values.mean(),
                    ci_low=low,
                    ci_high=high,
                    bootstrap_seed=row.bootstrap_seed,
                )
            )
    write(out / "split_bootstrap_intervals.tsv", pd.DataFrame(intervals))
    return {
        "cohorts": 22,
        "splits_per_cohort": 20,
        "intervals_reproduced": len(intervals),
        "matched_K_verified": True,
        "raw_expression_refitted": False,
    }


def source_reports(data, out):
    import llm_pathway_curator.review as review

    require(
        Path(review.__file__).resolve() == REPO / "src/llm_pathway_curator/review.py",
        "Install this checkout with python -m pip install -e .",
    )
    source = data / "Supplementary_Data_1_TCGA"
    results = []
    for row in read(source / "q_baseline_comparison.tsv").itertuples():
        name = row.cancer
        result = review.review_enrichment(
            review.ReviewConfig(
                evidence_table=str(source / f"enrichment/{name}.evidence_table.tsv"),
                sample_card=str(source / f"sample_cards/{name}.json"),
                outdir=str(out / name),
                q_threshold=0.05,
                modules=True,
            )
        )
        expected = read(source / f"enrichment/{name}.evidence_table.tsv")
        log = read(Path(result.artifacts["audit_log"]))
        selected = set(log.loc[log.status == "PASS", "term_uid"])
        eligible = expected.loc[expected.qval <= 0.05]
        require(
            selected == set(eligible.source + ":" + eligible.term_id),
            f"{name}: selected pathway identities differ",
        )
        require(len(selected) == row.source_selected, f"{name}: selected count differs")
        results.append({"cohort": name, "candidates": len(expected), "selected": len(selected)})
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=HERE / "data")
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--figures-only", action="store_true")
    args = parser.parse_args(argv)
    data, out = args.data_dir.resolve(), args.outdir.resolve()
    require(not out.exists(), "Use a new output directory")
    manifest = verify_inputs(data)
    out.mkdir(parents=True)
    summary = {
        "schema": "manuscript-reproduction/1",
        "figures_only": args.figures_only,
        "verified_inputs": len(manifest["files"]),
        "data_manifest_sha256": sha(HERE / "data_manifest.json"),
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "model_calls": 0,
        "expression_models_refitted": False,
        "human_labels_changed": False,
    }
    try:
        if not args.figures_only:
            tables = out / "recomputed_tables"
            tables.mkdir()
            summary["dex"] = dex(data, tables)
            summary["hnsc"] = hnsc(data, tables)
            summary["split_replication"] = split_replication(data, tables)
            summary["source_reports"] = source_reports(data, out / "source_reports")
        for script, subdir, name in [
            ("plot_figure2.py", "Supplementary_Data_1_TCGA", "figure2"),
            ("plot_figure3.py", "Supplementary_Data_2_DEX", "figure3"),
            ("plot_supplement.py", "", "supplement"),
        ]:
            subprocess.run(
                [
                    sys.executable,
                    str(HERE / "figures" / script),
                    "--data-dir",
                    str(data / subdir),
                    "--outdir",
                    str(out / name),
                ],
                check=True,
            )
        verify_inputs(data)
        summary["status"] = "completed"
    except Exception as error:
        summary.update(status="failed", error=str(error))
        raise
    finally:
        code = [
            Path(__file__),
            HERE / "config/r06_p4_reuse_policy.json",
            HERE / "scripts/v17_r06.py",
            *sorted((HERE / "figures").glob("*.py")),
            *sorted((REPO / "src/llm_pathway_curator").rglob("*.py")),
        ]
        summary["code_sha256"] = {str(p.relative_to(REPO)): sha(p) for p in code}
        summary["outputs_sha256"] = {
            str(p.relative_to(out)): sha(p) for p in sorted(out.rglob("*")) if p.is_file()
        }
        (out / "reproduction_manifest.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"status": summary["status"], "outdir": str(out)}, indent=2))


if __name__ == "__main__":
    main()
