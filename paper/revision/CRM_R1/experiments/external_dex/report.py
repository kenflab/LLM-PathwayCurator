#!/usr/bin/env python3
"""Verify and report saved dex results. No downloads, R fitting or model calls."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import stat
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
METHODS = ("ALL_50_CANDIDATES", "Q_VALUE_0_05", "Q_PLUS_DONOR_LOO_3_OF_4", "MATCHED_Q_K")
LABELS = ("All 50 terms", "Discovery q ≤ 0.05", "q + donor stability", "Size-matched q")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def object_sha(value):
    return digest(json.dumps(value, sort_keys=True, separators=(",", ":")).encode())


def numeric(value):
    result = float(value)
    require(math.isfinite(result), "Non-finite saved numeric result")
    return result


def truth(value):
    require(value in ("True", "False", "TRUE", "FALSE"), "Unknown saved boolean")
    return value in ("True", "TRUE")


def json_bytes(raw):
    return json.loads(raw.decode("utf-8"))


def tsv_bytes(raw):
    return list(csv.DictReader(io.StringIO(raw.decode("utf-8")), delimiter="\t"))


def read_archive(path):
    """Verify bytes without extracting paths or modifying the source archive."""
    with zipfile.ZipFile(path) as archive:
        infos = archive.infolist()
        require(len(infos) == len({i.filename for i in infos}), "Duplicate ZIP entry")
        require(sum(i.file_size for i in infos) < 100 * 1024 * 1024, "Unexpectedly large result ZIP")
        for item in infos:
            name = Path(item.filename)
            require(not name.is_absolute() and len(name.parts) == 1 and ".." not in name.parts and
                    not item.is_dir() and not stat.S_ISLNK(item.external_attr >> 16), "Unsafe/nested result ZIP entry")
        require("EXPORT_SHA256.json" in archive.namelist(), "Result export manifest missing")
        export = json_bytes(archive.read("EXPORT_SHA256.json"))
        require(set(archive.namelist()) == set(export["files"]) | {"EXPORT_SHA256.json"}, "Unrecorded or missing result file")
        files = {}
        for name, expected in export["files"].items():
            raw = archive.read(name)
            require(digest(raw) == expected, f"Result export hash mismatch: {name}")
            files[name] = raw
    return files


def bh50(rows):
    good = [(numeric(r["pval"]), r["term_id"]) for r in rows if r["status"] == "ESTIMABLE"]
    good.sort()
    adjusted, running = {}, 1.0
    for rank in range(len(good), 0, -1):
        p, term = good[rank - 1]
        require(0 <= p <= 1, "Invalid enrichment p-value")
        running = min(running, 50 * p / rank)
        adjusted[term] = running
    return adjusted


def same_direction(a, b):
    return (a["status"] == b["status"] == "ESTIMABLE" and
            numeric(a["NES"]) * numeric(b["NES"]) > 0)


def endpoint(a, b):
    return same_direction(a, b) and numeric(b["q"]) <= 0.05


def comparisons(discovery, validation, folds):
    """Recompute selection from discovery/folds only; preserve denominators."""
    require(len(folds) == 4, "Four donor folds required")
    terms = sorted(discovery)
    require(set(validation) == set(terms) and all(set(f) == set(terms) for f in folds), "Term census differs across fits")
    hits = {t: sum(same_direction(discovery[t], f[t]) and numeric(f[t]["q"]) <= 0.05 for f in folds) for t in terms}
    q_pool = [t for t in terms if discovery[t]["status"] == "ESTIMABLE" and numeric(discovery[t]["q"]) <= 0.05]
    stable = [t for t in q_pool if hits[t] >= 3]
    ranked = sorted((t for t in terms if discovery[t]["status"] == "ESTIMABLE"), key=lambda t: (numeric(discovery[t]["q"]), t))
    selected = dict(zip(METHODS, (terms, q_pool, stable, ranked[:len(stable)])))
    rows = []
    for method, ids in selected.items():
        n, replicated = len(ids), sum(endpoint(discovery[t], validation[t]) for t in ids)
        rows.append({"method": method, "selected_count": n, "coverage": n / len(terms),
                     "validation_estimable_count": sum(validation[t]["status"] == "ESTIMABLE" for t in ids),
                     "same_direction_count": sum(same_direction(discovery[t], validation[t]) for t in ids),
                     "replicated_count": replicated, "replication_rate": replicated / n if n else None,
                     "selected_ids": ";".join(ids)})
    return rows, selected, hits


def verify_result(files):
    lock = json_bytes(files["DESIGN_LOCK.json"])
    protocol = json_bytes(files["protocol.json"])
    summary = json_bytes(files["SUMMARY.json"])
    job = json_bytes(files["JOB.private.json"])
    inputs = json_bytes(files["INPUT_MANIFEST.json"])
    marker = json_bytes(files["R_STATISTICS_COMPLETE.json"])
    amendment = json_bytes(files["TECHNICAL_AMENDMENT_R08_1.json"])
    spec = json_bytes(files["TECHNICAL_AMENDMENT_SPEC_R08_1.json"])
    require(summary["status"] == "COMPLETE_LIMITED_CROSS_STUDY_CASE" and summary["scope"] == protocol["scope"], "Result status/scope mismatch")
    require(summary["design_sha256"] == job["design_sha256"] == inputs["design_sha256"] == amendment["design_sha256"] == object_sha(lock), "Original design identity mismatch")
    require(summary["input_manifest_sha256"] == job["input_manifest_sha256"] == amendment["input_manifest_sha256"] == object_sha(inputs), "Recorded input-manifest identity mismatch")
    require(summary["execution_code_sha256"] == job["execution_code_sha256"] == amendment["amended_code_sha256"] == spec["amended_code_sha256"], "Recorded execution code identity mismatch")
    require(summary["technical_amendment_sha256"] == job["technical_amendment_sha256"] == digest(files["TECHNICAL_AMENDMENT_R08_1.json"]), "Amendment identity mismatch")
    require(amendment["spec_sha256"] == digest(files["TECHNICAL_AMENDMENT_SPEC_R08_1.json"]) and amendment["original_code_sha256"] == lock["code_sha256"] == spec["original_code_sha256"], "Amendment parent/spec mismatch")
    require(not amendment["scientific_protocol_bytes_changed"] and
            digest(files["protocol.json"]) == lock["code_sha256"]["protocol.json"] == job["execution_code_sha256"]["protocol.json"], "Scientific protocol changed")
    for name, expected in lock["frozen_file_sha256"].items():
        if name in files:
            require(digest(files[name]) == expected, f"Frozen snapshot mismatch: {name}")
    require(json_bytes(files["RUNTIME_IDENTITY.json"]) == job["R_runtime"], "R runtime record mismatch")
    require("R08_1_PROBE_FILTER_SELF_TEST_PASS" in files["R_PROBE_FILTER_SELF_TEST.txt"].decode(), "Base-R regression success marker missing")
    require(summary["model_calls"] == summary["new_expert_ratings"] == marker["model_calls"] == 0, "Unexpected model/rating activity")
    require(not summary["full_audit_performance_estimated"] and not summary["semantic_accuracy_estimated"] and not summary["cross_study_donor_disjointness_verified"], "Unsupported accuracy/donor-independence claim")
    require(lock["candidate_count"] == summary["candidate_count"] == summary["candidate_retained_count"] == 50, "Changed candidate count")
    members = tsv_bytes(files["hallmark_gene_symbols.tsv"])
    require(len({(r["term_id"], r["gene_symbol"]) for r in members}) == len(members), "Duplicate Hallmark membership")
    terms = sorted({r["term_id"] for r in members})
    require(len(terms) == 50 and all(t.startswith("HALLMARK_") for t in terms), "Wrong Hallmark census")
    genes = [r["gene_symbol"] for r in tsv_bytes(files["common_gene_universe.tsv"])]
    require(len(genes) >= 1000 and genes == sorted(set(genes)) and len(genes) == marker["common_genes"], "Common gene universe mismatch")
    gene_set = set(genes)
    discovery_genes = {r["gene_symbol"] for r in tsv_bytes(files["discovery_filtered_gene_universe.tsv"])}
    validation_genes = {r["gene_symbol"] for r in tsv_bytes(files["validation_measured_gene_universe.tsv"])}
    require(gene_set == discovery_genes & validation_genes and len(discovery_genes) == marker["discovery_filtered_genes"] and len(validation_genes) == marker["validation_measured_genes"], "Measured-universe intersection mismatch")
    pathways = {t: {r["gene_symbol"] for r in members if r["term_id"] == t} for t in terms}
    coverage = {r["term_id"]: r for r in tsv_bytes(files["pathway_measured_coverage.tsv"])}
    require(set(coverage) == set(terms), "Coverage census mismatch")
    donors = sorted({r["donor"] for r in protocol["discovery"]["samples"]})
    require(len(donors) == marker["donor_folds"] == 4, "Donor-fold census mismatch")
    fits = ["discovery", *("donor_loo_" + d for d in donors), "validation_24h", "validation_4h_secondary"]
    tables, max_q_difference = {}, 0.0
    for fit in fits:
        rows = tsv_bytes(files[fit + ".tsv"])
        require(len(rows) == 50 and {r["term_id"] for r in rows} == set(terms), f"Fit does not retain all 50 terms: {fit}")
        table = {r["term_id"]: r for r in rows}
        expected_q = bh50(rows)
        rank_rows = tsv_bytes(files[fit + ".gene_rank.tsv"])
        ranks = {r["gene_symbol"]: r for r in rank_rows}
        require(len(ranks) == len(rank_rows) and gene_set <= set(ranks), f"Rank census mismatch: {fit}")
        require(all(math.isfinite(float(ranks[g]["score"])) for g in genes), f"Non-finite fixed-universe rank: {fit}")
        for term, row in table.items():
            measured = pathways[term] & gene_set
            require(int(row["size"]) == len(measured) == int(coverage[term]["measured_set_size"]), "Pathway measured size mismatch")
            require(int(coverage[term]["original_set_size"]) == len(pathways[term]) and int(coverage[term]["common_universe_size"]) == len(genes), "Original set/universe coverage mismatch")
            leading = set(filter(None, row["leading_genes"].split(";")))
            require(leading <= measured, f"Leading-edge gene outside measured pathway: {fit}/{term}")
            require(row["status"] in ("ESTIMABLE", "OUTSIDE_SIZE_LIMITS", "NUMERICAL_FAILURE", "INCOMPLETE_FIXED_UNIVERSE"), "Unexpected saved fit status")
            if row["status"] == "ESTIMABLE":
                require(protocol["enrichment"]["minSize"] <= len(measured) <= protocol["enrichment"]["maxSize"], "Estimable term outside size limits")
                numeric(row["NES"])
                delta = abs(numeric(row["q"]) - expected_q[term])
                require(math.isclose(numeric(row["q"]), expected_q[term], rel_tol=1e-12, abs_tol=1e-15), f"BH family-50 mismatch: {fit}/{term}")
                max_q_difference = max(max_q_difference, delta)
        tables[fit] = table
    discovery_samples = tsv_bytes(files["discovery_samples.tsv"])
    expected_samples = protocol["discovery"]["samples"]
    require(len(discovery_samples) == len(expected_samples), "Discovery sample census mismatch")
    for recorded, expected_sample in zip(discovery_samples, expected_samples):
        require(set(recorded) == set(expected_sample), "Discovery sample columns mismatch")
        for field, value in expected_sample.items():
            require(numeric(recorded[field]) == value if isinstance(value, (int, float)) else recorded[field] == value,
                    f"Discovery sample mapping mismatch: {field}")
    require(tsv_bytes(files["validation_samples.tsv"]) == protocol["validation"]["samples"], "Validation sample mapping mismatch")
    primary, selected, hits = comparisons(tables["discovery"], tables["validation_24h"], [tables["donor_loo_" + d] for d in donors])
    secondary, selected4, _ = comparisons(tables["discovery"], tables["validation_4h_secondary"], [tables["donor_loo_" + d] for d in donors])
    require(selected == selected4, "Validation outcomes affected selection")
    saved = tsv_bytes(files["METHOD_COMPARISON.tsv"])
    require(len(saved) == 8, "Comparison row census mismatch")
    expected = {}
    for contrast, rows in (("24h_PRIMARY", primary), ("4h_SECONDARY_SHARED_CONTROLS", secondary)):
        for row in rows:
            expected[(contrast, row["method"])] = {**row, "contrast": contrast}
    for row in saved:
        key = (row["contrast"], row["method"])
        require(key in expected, "Unexpected/duplicate comparison row")
        calculated = expected.pop(key)
        for field in ("selected_count", "validation_estimable_count", "same_direction_count", "replicated_count"):
            require(int(row[field]) == calculated[field], f"Comparison count mismatch: {key}/{field}")
        for field in ("coverage", "replication_rate"):
            if calculated[field] is None:
                require(row[field] in ("", "NA", "None"), "Empty selection rate was assigned a value")
            else:
                require(math.isclose(numeric(row[field]), calculated[field], rel_tol=1e-12), "Comparison rate mismatch")
        require(set(filter(None, row["selected_ids"].split(";"))) == set(selected[row["method"]]), "Saved selection identities differ")
    require(not expected, "Comparison row missing")
    require(len(summary["comparisons"]) == 4, "Summary comparison census mismatch")
    for recorded, calculated in zip(summary["comparisons"], primary):
        for field in ("method", "selected_count", "replicated_count", "same_direction_count", "validation_estimable_count", "coverage", "replication_rate"):
            require(recorded[field] == calculated[field], "Printed summary differs from saved pathway records")
        require(set(recorded["selected_ids"].split(";")) == set(selected[calculated["method"]]), "Printed summary has different selected IDs")
    ledger = tsv_bytes(files["TERM_LEDGER.tsv"])
    require(len(ledger) == 50 and {r["term_id"] for r in ledger} == set(terms), "Term ledger census mismatch")
    for row in ledger:
        t = row["term_id"]
        require(int(row["donor_loo_hits_of_4"]) == hits[t], "Donor-fold ledger mismatch")
        require(truth(row["replicated_24h"]) == endpoint(tables["discovery"][t], tables["validation_24h"][t]) and truth(row["replicated_4h_secondary"]) == endpoint(tables["discovery"][t], tables["validation_4h_secondary"][t]), "Endpoint ledger mismatch")
        for method in METHODS:
            require(truth(row[method]) == (t in selected[method]), "Selection ledger mismatch")
    diagnostic = json_bytes(files["PROBE_FILTER_DIAGNOSTIC.json"])
    require(diagnostic == summary["probe_filter_diagnostic"] and diagnostic["frozen_protocol_retained_count"] == marker["validation_retained_probes"], "Probe-filter diagnostic mismatch")
    probe_rows = tsv_bytes(files["validation_probe_mapping.tsv"])
    require(len(probe_rows) == diagnostic["raw_probe_count"] and sum(truth(r["retained"]) for r in probe_rows) == diagnostic["frozen_protocol_retained_count"] and sum(truth(r["legacy_retained"]) for r in probe_rows) == diagnostic["legacy_R08_retained_count"], "Per-probe filter count mismatch")
    stable_ids, matched_ids = (set(selected[m]) for m in METHODS[2:])
    same_ids = stable_ids == matched_ids
    a, b = primary[2]["replication_rate"], primary[3]["replication_rate"]
    delta = a - b if a is not None and b is not None else None
    require(summary["descriptive_replication_rate_difference_stability_minus_matched_q"] == delta, "Saved comparator difference mismatch")
    return {"summary": summary, "protocol": protocol, "marker": marker, "terms": terms, "runtime": job["R_runtime"],
            "donors": donors, "fits": fits, "tables": tables, "primary": primary, "secondary": secondary,
            "selected": selected, "hits": hits, "same_selected_ids": same_ids,
            "selection_symmetric_difference": sorted(stable_ids ^ matched_ids),
            "comparison_difference": delta, "checked_export_files": len(files),
            "BH_checks": sum(len(bh50(list(tables[fit].values()))) for fit in fits), "max_BH_absolute_difference": max_q_difference,
            "probe_filter_diagnostic": diagnostic,
            "source_bytes_verified": True,
            "raw_input_payload_hashes_recomputed_from_this_compact_zip": False,
            "R_fitting_or_enrichment_rerun": False}


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def write_csv(path, rows):
    require(rows, "Cannot export an empty source-data table without a schema")
    with Path(path).open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def source_data(out, review, files):
    directory = out / "source_data"
    directory.mkdir()
    rows = [{"contrast": contrast, **r} for contrast, result in (("24h_PRIMARY", review["primary"]), ("4h_SECONDARY_SHARED_CONTROLS", review["secondary"])) for r in result]
    write_csv(directory / "method_comparison.csv", rows)
    records, long = [], []
    for term in review["terms"]:
        row = {"term_id": term, "donor_loo_hits_of_4": review["hits"][term]}
        for fit in review["fits"]:
            original = review["tables"][fit][term]
            row.update({fit + "_" + name: original[name] for name in ("NES", "pval", "q", "status")})
            long.append({"fit": fit, **original})
        row.update({method: term in review["selected"][method] for method in METHODS})
        row["validation_24h_endpoint_met"] = endpoint(review["tables"]["discovery"][term], review["tables"]["validation_24h"][term])
        row["validation_4h_endpoint_met_secondary"] = endpoint(review["tables"]["discovery"][term], review["tables"]["validation_4h_secondary"][term])
        records.append(row)
    write_csv(directory / "all_50_terms.csv", records)
    write_csv(directory / "all_350_fit_records.csv", long)
    snapshot = out / "statistical_source_snapshot"
    snapshot.mkdir()
    for name in [*(fit + ".tsv" for fit in review["fits"]), "protocol.json", "DESIGN_LOCK.json", "INPUT_MANIFEST.json", "TECHNICAL_AMENDMENT_R08_1.json", "RUNTIME_IDENTITY.json", "PROBE_FILTER_DIAGNOSTIC.json"]:
        (snapshot / name).write_bytes(files[name])


def plot(out, review):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "svg.fonttype": "none"})
    figure_dir = out / "figures"
    figure_dir.mkdir()
    colors = ["#A3ABB5", "#7A8DA3", "#D58B32", "#39749C"]
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.2), gridspec_kw={"width_ratios": [1.25, 1]})
    fig.subplots_adjust(left=0.20, right=0.98, top=0.85, bottom=0.20, wspace=0.75)
    ax = axes[0]
    ax.set_title("a  Validation at 24 h", loc="left", fontweight="bold", pad=12)
    for i, row in enumerate(review["primary"]):
        rate = row["replication_rate"]
        ax.barh(i, rate if rate is not None else 0, color=colors[i], height=0.57)
        label = (f"{row['replicated_count']}/{row['selected_count']} ({rate:.1%})" if rate is not None else "0 selected; rate undefined")
        ax.text((rate or 0) + 0.02, i, label, va="center", fontsize=8.3)
    ax.set_yticks(range(4), LABELS)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1], ["0", "25", "50", "75", "100"])
    ax.set_xlabel("Same NES sign and validation q ≤ 0.05 (%)")
    ax.tick_params(axis="y", length=0)
    ax = axes[1]
    ax.set_title("b  All 50 Hallmark terms", loc="left", fontweight="bold", pad=12)
    d, v = review["tables"]["discovery"], review["tables"]["validation_24h"]
    stable = set(review["selected"][METHODS[2]])
    groups = [("Other terms", [t for t in review["terms"] if t not in stable], "#BBC2CB", "o"),
              ("Selected; endpoint not met", [t for t in stable if not endpoint(d[t], v[t])], "#C87926", "x"),
              ("Selected; endpoint met", [t for t in stable if endpoint(d[t], v[t])], "#24765F", "o")]
    for label, ids, color, marker in groups:
        ids = [t for t in sorted(ids) if d[t]["status"] == v[t]["status"] == "ESTIMABLE"]
        ax.scatter([numeric(d[t]["NES"]) for t in ids], [numeric(v[t]["NES"]) for t in ids], c=color, marker=marker, s=27, label=label, linewidths=1)
    ax.axhline(0, color="#CED2D7", lw=0.7)
    ax.axvline(0, color="#CED2D7", lw=0.7)
    for term, label, offset in (("HALLMARK_ADIPOGENESIS", "Adipogenesis", (-60, -22)),
                                ("HALLMARK_E2F_TARGETS", "E2F targets", (8, 10)),
                                ("HALLMARK_P53_PATHWAY", "P53 pathway", (8, 12)),
                                ("HALLMARK_TNFA_SIGNALING_VIA_NFKB", "TNFA/NFKB", (-62, 30))):
        if term in d and d[term]["status"] == v[term]["status"] == "ESTIMABLE":
            ax.annotate(label, (numeric(d[term]["NES"]), numeric(v[term]["NES"])), xytext=offset, textcoords="offset points", fontsize=7.8,
                        arrowprops={"arrowstyle": "-", "color": "#707780", "lw": 0.5},
                        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 0.6})
    ax.set_xlabel("Discovery NES (18 h; 4 paired donors)")
    ax.set_ylabel("Validation NES (24 h; HASM1 cultures)")
    limit = max(3, math.ceil(max(abs(numeric(row["NES"])) for table in (d, v) for row in table.values() if row["status"] == "ESTIMABLE")))
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.legend(loc="upper left", fontsize=6.8, frameon=False)
    note = (f"Donor stability and size-matched q selected the same {len(stable)} terms." if review["same_selected_ids"] else "Donor stability and size-matched q selected different term sets.")
    fig.text(0.20, 0.055, note, fontsize=8.2)
    for ext in ("pdf", "svg", "png"):
        fig.savefig(figure_dir / ("Figure_Dex_primary." + ext), dpi=220, facecolor="white")
    plt.close(fig)
    # The supplementary view retains every candidate, including unselected
    # and non-replicated terms. Lexical order is unrelated to validation outcomes.
    fits = review["fits"]
    matrix = np.array([[float(review["tables"][fit][t]["NES"]) if review["tables"][fit][t]["status"] == "ESTIMABLE" else np.nan for fit in fits] for t in review["terms"]])
    fig, ax = plt.subplots(figsize=(8.6, 12.3))
    fig.subplots_adjust(left=0.46, right=0.91, bottom=0.15, top=0.94)
    limit = max(3, math.ceil(float(np.nanmax(np.abs(matrix))))) if np.isfinite(matrix).any() else 3
    im = ax.imshow(matrix, cmap="RdBu_r", vmin=-limit, vmax=limit, aspect="auto", interpolation="none")
    ax.set_yticks(range(50), [t.removeprefix("HALLMARK_").replace("_", " ") for t in review["terms"]], fontsize=7.3)
    fit_labels = ["Discovery 18 h", *("Leave out " + d for d in review["donors"]), "Validation 24 h", "Validation 4 h"]
    ax.set_xticks(range(len(fits)), fit_labels, rotation=50, ha="right", fontsize=8)
    for i, t in enumerate(review["terms"]):
        for j, fit in enumerate(fits):
            row = review["tables"][fit][t]
            if row["status"] == "ESTIMABLE" and numeric(row["q"]) <= 0.05:
                ax.text(j, i, "*", ha="center", va="center", fontsize=8, color="white" if abs(numeric(row["NES"])) >= 1.7 else "black")
    ax.set_title("All 50 terms across discovery, donor folds and validation", loc="left", pad=16, fontweight="bold", fontsize=9.5)
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.04)
    cb.set_label("NES", fontsize=9)
    fig.text(0.04, 0.032, "* BH q ≤ 0.05 within the fixed family of 50 terms for that fit.\n4 h is secondary and shares controls with 24 h. Colors show enrichment direction, not pathway activation.", fontsize=8.2)
    for ext in ("pdf", "svg", "png"):
        fig.savefig(figure_dir / ("Figure_Dex_all_terms." + ext), dpi=180, facecolor="white")
    plt.close(fig)


def manuscript_text(review):
    primary, secondary, marker = review["primary"], review["secondary"], review["marker"]
    selected_rep = [t for t in review["selected"][METHODS[2]] if endpoint(review["tables"]["discovery"][t], review["tables"]["validation_24h"][t])]
    names = ", ".join(t.removeprefix("HALLMARK_").replace("_", " ") for t in selected_rep) or "none"
    packages = review["runtime"]["packages"]
    estimable_counts = [sum(row["status"] == "ESTIMABLE" for row in review["tables"][fit].values()) for fit in review["fits"]]
    estimable_sentence = ("All 50 Hallmark terms were estimable in the full discovery analysis, four leave-one-donor-out analyses and both validation contrasts." if min(estimable_counts) == 50 else f"All 50 Hallmark candidates were retained across the seven analyses; estimable counts ranged from {min(estimable_counts)} to {max(estimable_counts)}.")
    equality = ("The donor-stability and size-matched q procedures selected exactly the same terms; therefore, this case provides no observed incremental selection or replication benefit from the stability requirement." if review["same_selected_ids"] else "The two size-matched procedures selected different term sets; their descriptive rates are reported without a superiority test.")
    return f'''DRAFT: Results (align figure numbering with the final manuscript)
To evaluate the donor-stability component on non-cancer data, we processed a dexamethasone contrast in human airway smooth muscle using RNA sequencing from four paired donor-derived cell lines (GSE52778; 18 h) and Agilent arrays from one validation cell line (GSE34313; HASM1). The primary validation contrast compared three 24-h treated cultures with four controls. Mapping and outcome-independent expression filters yielded {marker['common_genes']:,} shared measured genes. {estimable_sentence} These results describe a limited cross-study, culture-level application of the statistical component; cross-study donor overlap and sample pairing were not authenticated.

The validation endpoint required BH q ≤ 0.05 at 24 h and the same nonzero NES sign as discovery. It was met by {primary[0]['replicated_count']}/{primary[0]['selected_count']} candidates overall. Discovery q ≤ 0.05 selected {primary[1]['selected_count']} terms, of which {primary[1]['replicated_count']} met the endpoint. Requiring concordant significant enrichment in at least three of four donor folds retained {primary[2]['selected_count']} terms, with {primary[2]['replicated_count']}/{primary[2]['selected_count']} meeting the endpoint; a q-ranked comparator with the same number of terms also yielded {primary[3]['replicated_count']}/{primary[3]['selected_count']}. {equality} The selected terms meeting the 24-h endpoint were {names}. The secondary 4-h contrast, using the same controls, yielded {secondary[2]['replicated_count']}/{secondary[2]['selected_count']} for both size-matched sets and was not treated as independent confirmation. These endpoints do not evaluate semantic accuracy or full-audit performance.

DRAFT: STAR Methods, external dexamethasone application
We froze the sample census, original Hallmark snapshot, preprocessing, contrasts and reporting endpoints before computing the new pathway outcomes. Previously viewed published study results and metadata were disclosed in the protocol. GSE52778 integer counts were loaded from airway::airway (airway {packages['airway']}; STAR/Ensembl 75 re-quantification), retaining eight dexamethasone/control samples from four paired donors at 18 h and excluding albuterol-containing treatments. Single-symbol gene_name annotations were used, counts sharing a symbol were summed, and genes with count ≥10 in at least four of eight samples were retained. We used edgeR {packages['edgeR']} TMM normalization, limma {packages['limma']} voom, a donor-plus-treatment design and empirical Bayes moderation with trend=FALSE and robust=FALSE. For each donor fold, both samples from that donor were removed, using the fixed full-discovery filtered genes.

For GSE34313, all ten public Cy3 raw Agilent arrays were retained. The primary contrast used three 24-h treated cultures and four controls; the secondary contrast used three 4-h treated cultures and the same controls. We did not infer donor or pair identities from replicate suffixes. Median foreground/background signals were read with limma read.maimages(source='agilent', green.only=TRUE), corrected with normexp offset 50, and quantile normalized across all ten arrays. Raw non-control probes (ControlType=0), detected above background in at least three arrays, with finite normalized values and an unambiguous GPL6480 single-symbol mapping were retained. Multiple retained probes per symbol were aggregated by per-sample median. Positive-variance validation genes were intersected with the discovery filtered genes; the shared gene set was used for enrichment in every fit. Validation contrasts used unpaired treatment designs with limma eBayes(trend=TRUE, robust=FALSE).

Genes were ranked by moderated t-statistic, with exact ties ordered by gene symbol and no random jitter. fgseaMultilevel (fgsea {packages['fgsea']}) used minSize=15, maxSize=500, eps=0, sampleSize=101, nPermSimple=10000, nproc=1, gseaParam=1, scoreType='std' and seed=42. BH adjustment used a fixed family of 50 terms for each fit. Candidate selection used discovery data only: all terms, discovery q ≤ 0.05, q plus concordant q ≤ 0.05 in at least three of four donor folds, and the lowest discovery q values with the latter procedure's fixed K. Non-estimable terms would remain in selected denominators. We did not compute term-level binomial intervals or superiority tests because pathways overlap and the validation study used one cell line. The actual R and package versions, original input hashes and technical amendment are included in Source Data/provenance records.

A technical correction was documented after a preprocessing stop and before any fitted statistical output: the original implementation had added numeric coercion of the textual GPL CONTROL_TYPE annotation, absent from the frozen protocol. The corrected implementation follows raw ControlType=0. The original protocol, design SHA, downloaded files and failed run were preserved; no thresholds, sample assignments or enrichment parameters were changed.

DRAFT: Figure legend, primary dex application
(a) Fraction meeting the primary 24-h validation endpoint for all candidates, discovery q filtering, q plus donor stability and a size-matched q comparator. Labels give numerator and selected denominator. (b) Discovery versus 24-h validation NES for every Hallmark term; symbols distinguish the stability-selected terms that do or do not meet the endpoint. Labeled pathways are illustrative, and all 50 candidate values are provided in Source Data. Discovery comprises four paired donor units; validation comprises one HASM1 cell line with three treated cultures and four controls. Stability and the size-matched q comparator selected the same {primary[2]['selected_count']} terms. NES sign describes enrichment direction and does not establish pathway activation or a causal/clinical effect. No independent-term uncertainty intervals are shown.

DRAFT: Supplementary figure legend
NES for all 50 terms in the full discovery analysis, four donor folds and the primary 24-h and secondary 4-h validation contrasts. Terms are ordered lexically; no candidates were removed from the display based on validation outcomes. Asterisks denote BH q ≤ 0.05 within the fixed family of 50 terms for that fit. Each donor fold excludes both samples for the labeled donor. The 4-h and 24-h contrasts share controls and are not independent validation studies.

DRAFT: Response paragraph for the request for external biological validation
We added an analysis of the statistical donor-stability component on non-cancer, cross-study dexamethasone data, using a prospectively frozen computational protocol for the new pathway analyses. We processed four paired donor-derived cell lines at 18 h and an external Agilent study with a primary 24-h culture-level contrast, retaining all 50 Hallmark candidates and a fixed measured-gene universe. We report all candidate outcomes and size-matched q baselines, including the lack of observed improvement from donor stability in this case. We have limited the corresponding claim to cross-study pathway-enrichment replication in one worked example. Cross-study donor disjointness remains unverified, and the validation arrays derive from one cell line; this experiment does not establish independent-donor validation, semantic accuracy or superiority of the full audit. Figure/page references should be inserted after final manuscript assembly. This paragraph addresses the external-data component and must be combined with the separate existing-rater and unaudited-output responses.

Sources to cite in final STAR Methods (primary data/software records)
https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE52778
https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE34313
https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GPL6480
https://www.bioconductor.org/packages/release/data/experiment/vignettes/airway/inst/doc/airway.html
https://bioconductor.org/packages/release/bioc/vignettes/limma/inst/doc/usersguide.pdf
https://bioconductor.org/packages/release/bioc/vignettes/fgsea/inst/doc/fgsea-tutorial.html
'''


def build_report(archive, out):
    original_sha = file_sha(archive)
    files = read_archive(archive)
    review = verify_result(files)
    require(not out.exists(), "Report output already exists; original results and reports are never overwritten")
    # Check the existing project plotting dependency before writing output.
    import matplotlib
    out.mkdir(parents=True)
    try:
        source_data(out, review, files)
        plot(out, review)
        summary = {k: review[k] for k in ("primary", "secondary", "same_selected_ids", "selection_symmetric_difference", "comparison_difference", "checked_export_files", "BH_checks", "max_BH_absolute_difference", "probe_filter_diagnostic", "source_bytes_verified", "raw_input_payload_hashes_recomputed_from_this_compact_zip", "R_fitting_or_enrichment_rerun")}
        summary.update(schema="CRM_R1_EXTERNAL_DEX_REPORT_R09", status="COMPLETE_SAVED_RESULT_REPORT",
                       original_archive_sha256=original_sha, reporting_code_sha256=file_sha(Path(__file__)),
                       design_sha256=review["summary"]["design_sha256"], common_genes=review["marker"]["common_genes"],
                       matplotlib_version=matplotlib.__version__, model_calls=0, new_expert_ratings=0)
        write_json(out / "REPORT_SUMMARY.json", summary)
        text = ["外部dex実例：保存済み結果の確認と改訂方針", "",
                f"出力ファイル{review['checked_export_files']}件のhashを確認し、7 fits×50候補のBH調整と候補ごとの選択・分母・endpointを再計算しました。",
                f"共通測定遺伝子：{review['marker']['common_genes']:,}。元の設計と技術修正の来歴は保持されています。",
                "元のraw tarやairway counts自体はcompact ZIPに含まれず、この報告では原発現値からのR fitting/GSEAは再実行していません。", "", "24h primary"]
        for row in review["primary"]:
            rate = row["replication_rate"]
            text.append(f"{row['method']}: {row['replicated_count']}/{row['selected_count']} ({rate:.1%})" if rate is not None else f"{row['method']}: 選択0、率未定義")
        text.extend(["", "採択に向けた扱い", "統計的なドナー安定性成分を非がんデータへ適用した限定例として採用し、計算を固定して図・Source Data・Results・Methodsへ進みます。",
                     "安定性とmatched qが同じ候補を選ぶ場合、その比較から追加効果を主張しません。q-onlyとの選択数の違いによる率の変化を、サイズを揃えた改善として扱いません。",
                     "これらは統計的enrichmentの方向・q値の一致です。LLMの意味的正確性、full auditの優越性、独立ドナーによる検証へ読み替えません。",
                     "4hは同じcontrolを共有するsecondaryとしてSupplementに保持します。追加の有料モデルや専門家Round 2は実行しません。",
                     "症例追加、閾値変更、time point切替によって結果を有利にする作業は行いません。元P2Bの不利な比較、既存3名評価、R07の元endpointも保持します。",
                     "次は全解析を対応する主張と図へ統合し、未監査との比較、専門家評価、外部データの各回答を別々に完成させます。",
                     "現行R1 Wordの実物に合わせた図番号・ページ番号の統合は、この出力だけでは完了していません。採択の可否をこの実例だけで予測することはできません。"])
        (out / "REPORT_JA.txt").write_text("\n".join(text) + "\n", encoding="utf-8")
        (out / "MANUSCRIPT_DRAFT.txt").write_text(manuscript_text(review), encoding="utf-8")
        require(file_sha(archive) == original_sha, "Original archive changed during reporting")
        export = {str(p.relative_to(out)): file_sha(p) for p in sorted(out.rglob("*")) if p.is_file()}
        write_json(out / "REPORT_EXPORT_SHA256.json", {"files": export, "original_archive_sha256": original_sha})
        report_zip = Path(str(out) + ".zip")
        with zipfile.ZipFile(report_zip, "x", compression=zipfile.ZIP_DEFLATED) as handle:
            for name in [*export, "REPORT_EXPORT_SHA256.json"]:
                handle.write(out / name, name)
        return {**summary, "outdir": str(out), "results_archive": str(report_zip)}
    except Exception as error:
        write_json(out / "REPORT_STOPPED.json", {"error": str(error), "original_archive_sha256": original_sha, "original_result_modified": False})
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--archive", type=Path, help="Read-only returned result ZIP; default uses the original completed-analysis receipt")
    parser.add_argument("--outdir", type=Path, help="New report folder under CRM_R1/output; never overwrite")
    args = parser.parse_args(argv)
    try:
        root = args.data_root.expanduser().resolve()
        require((root / "input").is_dir() and (root / "output").is_dir(), "Use the existing CRM_R1 input/output directory")
        require(not root.is_relative_to(HERE.parents[4]), "Data root must be outside the repository")
        if args.archive:
            archive = args.archive.expanduser().resolve()
        else:
            record = json.loads((root / "output/revision_v17/external_dex_design_v1/COMPLETED_ANALYSIS.json").read_text())
            relative = Path(record["outdir_relative_path"])
            require(not relative.is_absolute() and ".." not in relative.parts, "Unsafe completed-analysis receipt path")
            archive = Path(str(root / relative) + ".zip")
            require(archive.resolve().is_relative_to(root / "output") and file_sha(archive) == record["archive_sha256"], "Completed result archive hash mismatch")
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        out = args.outdir.expanduser().resolve() if args.outdir else root / ("output/revision_v17/external_dex_report_" + stamp)
        require(out.resolve().is_relative_to(root / "output"), "New report must stay under CRM_R1/output")
        require(not any(p.is_symlink() for p in (out, *out.parents)), "Symlink in report output path")
        result = build_report(archive, out)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0
    except (ValueError, OSError, KeyError, ImportError, zipfile.BadZipFile) as error:
        print(f"[STOP] {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
