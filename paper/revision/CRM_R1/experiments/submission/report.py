"""Verify saved revision tables and render four main-figure drafts.

This is reporting code. It neither fits expression models nor queries an LLM,
retrieves literature, changes memberships, or adds expert labels. Inputs and
the author's original Word remain unchanged. Outputs stay outside public Git.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import shutil
import stat
import textwrap
import zipfile
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

P2B = "frozen_source_data/output/priority2b/final_v14_1_3/"
P1 = "frozen_source_data/output/priority1/GSE146225_TP53_v1/metrics/"
P5 = "frozen_source_data/output/priority5/PANCAN_TP53_v1_HNSC_R1_P5/ontology/"
R06 = "saved_results/r06/"
R07 = "saved_results/r07/results/"
DEX = "saved_results/external_dex/"
COLORS = {"raw_pool": "#7F8995", "q_value_matched": "#C46B26",
          "stability_matched": "#29836B", "full_audit": "#286E99"}
ORDER = ["raw_pool", "q_value_matched", "stability_matched", "full_audit"]
LABELS = {"raw_pool": "All 50 (descriptive)", "q_value_matched": "Matched q",
          "stability_matched": "Matched stability", "full_audit": "Legacy full audit"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def table(raw):
    return list(csv.DictReader(io.StringIO(raw.decode("utf-8-sig")), delimiter="\t"))


def write_table(path, rows, fields=None):
    require(bool(rows) or fields is not None, "An empty table needs explicit fields")
    fields = fields or list(rows[0])
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def verify_archive(path):
    """Validate the entire rootless compact ZIP before extracting any content."""
    path = Path(path)
    require(path.is_file() and path.stat().st_size < 50_000_000, "Missing or oversized compact ZIP")
    with zipfile.ZipFile(path) as z:
        infos = z.infolist()
        names = [i.filename for i in infos]
        require(len(names) == len(set(names)), "Duplicate ZIP paths")
        require(sum(i.file_size for i in infos) < 100_000_000, "Uncompressed ZIP limit")
        for i in infos:
            p = PurePosixPath(i.filename)
            require(not p.is_absolute() and ".." not in p.parts and "\\" not in i.filename
                    and not i.is_dir() and not stat.S_ISLNK(i.external_attr >> 16), "Unsafe ZIP entry")
        payload = {n: z.read(n) for n in names}
    manifest = json.loads(payload["EXPORT_SHA256.json"])["files"]
    require(set(payload) == set(manifest) | {"EXPORT_SHA256.json"}, "Unmanifested or missing exports")
    for name, expected in manifest.items():
        require(digest(payload[name]) == expected, f"Export hash mismatch: {name}")
    summary = json.loads(payload["SUMMARY.json"])
    require(summary["schema"] == "CRM_R1_SUBMISSION_ASSEMBLY_R10", "Unsupported source schema")
    require(not summary["missing_or_unverified_components"], "Incomplete source collection")
    return payload, summary, manifest


def close(a, b, name):
    require(math.isclose(float(a), float(b), rel_tol=0, abs_tol=1e-13), f"Numeric mismatch: {name}")


def verify_p2b(payload):
    """Reproduce the saved aggregation and fixed-seed cohort bootstrap only."""
    get = lambda name: table(payload[P2B + name])
    mraw = payload[P2B + "figure2_source_manifest.json"]
    sidecar = payload[P2B + "figure2_source_manifest.sha256"].decode().split()[0]
    require(digest(mraw) == sidecar, "P2B manifest sidecar mismatch")
    metadata = json.loads(mraw)
    require(metadata["inference"]["unit"] == "cohort", "Wrong inference unit")
    require(metadata["cohorts"] == 22 and metadata["repeats"] == 20, "Wrong P2B census")
    for item in metadata["outputs"].values():
        name = P2B + PurePosixPath(item["path"]).name
        if name in payload:
            require(digest(payload[name]) == item["sha256"], f"P2B output mismatch: {name}")
        else:
            require(name.endswith("figure2_claim_source.private.tsv"), "Missing small P2B output")
    splits = get("figure2_split_method_replication.tsv")
    cohorts = get("figure2_panelB_cohort_replication.tsv")
    require(len(splits) == 1760 and len(cohorts) == 88, "P2B row census")
    by_cohort = defaultdict(list)
    by_split = defaultdict(dict)
    keys = set()
    for r in splits:
        key = (r["cohort_id"], r["split_id"], r["method_id"])
        require(key not in keys and key[2] in ORDER, "Duplicate or unknown P2B row")
        keys.add(key)
        n, hits, signs = (int(r[k]) for k in ("n_selected", "n_replicated", "n_direction_concordant"))
        require(0 <= hits <= signs <= n and n > 0, "Invalid P2B counts")
        close(hits / n, r["replication_fraction"], key)
        by_cohort[(key[0], key[2])].append(r)
        by_split[key[:2]][key[2]] = n
    require(len(by_split) == 440 and len(by_cohort) == 88, "Missing P2B groups")
    for key, group in by_split.items():
        require(set(group) == set(ORDER) and group["raw_pool"] == 50, "P2B split census")
        require(len({group[m] for m in ORDER[1:]}) == 1, f"Unmatched K: {key}")
    lookup = {}
    for r in cohorts:
        key = (r["cohort_id"], r["method_id"])
        require(key not in lookup and key in by_cohort, "Duplicate cohort row")
        lookup[key] = r
        group = by_cohort[key]
        require(len(group) == int(r["n_splits"]) == 20, "P2B split count")
        values = [float(x["replication_fraction"]) for x in group]
        close(np.mean(values), r["mean_replication_fraction"], key)
        close(np.std(values, ddof=1), r["sd_replication_fraction_across_splits"], key)
        sizes = [int(x["n_selected"]) for x in group]
        for observed, field in ((np.mean(sizes), "mean_selected_k"), (min(sizes), "min_selected_k"), (max(sizes), "max_selected_k")):
            close(observed, r[field], (key, field))
    for r in get("figure2_panelD_matched_k.tsv"):
        src = lookup[(r["cohort_id"], "full_audit")]
        for a, b in (("mean_matched_k", "mean_selected_k"), ("min_matched_k", "min_selected_k"), ("max_matched_k", "max_selected_k")):
            close(r[a], src[b], a)
    included = {r["cohort_id"] for r in get("figure2_sensitivity_cohorts.tsv") if r["included_in_sensitivity"] == "True"}
    require(len(included) == 16, "Sensitivity cohort census")
    for r in get("figure2_sensitivity_cohorts.tsv"):
        require((int(r["minimum_expression_evaluable_group_n"]) >= 25) == (r["cohort_id"] in included), "Sensitivity inclusion changed")
    contrasts = get("figure2_panelC_contrast_by_cohort.tsv")
    sens_contrasts = get("figure2_sensitivity_contrast_by_cohort.tsv")
    require(len(contrasts) == 44 and len(sens_contrasts) == 32, "Contrast census")
    for r in contrasts + sens_contrasts:
        full = float(lookup[(r["cohort_id"], "full_audit")]["mean_replication_fraction"])
        other = float(lookup[(r["cohort_id"], r["method_b"])]["mean_replication_fraction"])
        close(full, r["full_audit_replication_fraction"], "full contrast source")
        close(other, r["comparator_replication_fraction"], "comparator contrast source")
        close(full - other, r["difference_a_minus_b"], "paired contrast")
    interval_checks = []
    jobs = [
        ("figure2_panelB_method_summary.tsv", cohorts, "method_id", "mean_replication_fraction", "mean_cohort_replication_fraction"),
        ("figure2_sensitivity_method_summary.tsv", [r for r in cohorts if r["cohort_id"] in included], "method_id", "mean_replication_fraction", "mean_cohort_replication_fraction"),
        ("figure2_panelC_contrast_summary.tsv", contrasts, "method_b", "difference_a_minus_b", "mean_cohort_difference_a_minus_b"),
        ("figure2_sensitivity_contrast_summary.tsv", sens_contrasts, "method_b", "difference_a_minus_b", "mean_cohort_difference_a_minus_b")]
    for filename, source, key, vkey, estimate in jobs:
        for r in get(filename):
            group = sorted([x for x in source if x[key] == r[key]], key=lambda x: x["cohort_id"])
            a = np.array([float(x[vkey]) for x in group])
            require(len(a) == int(r["n_cohorts"]), "Bootstrap cohort count")
            close(np.mean(a), r[estimate], "cohort macro mean")
            require(int(r["bootstrap_draws"]) == 10000, "Changed saved bootstrap draws")
            rng = np.random.default_rng(int(r["bootstrap_seed"]))
            bs = rng.choice(a, size=(10000, len(a)), replace=True).mean(axis=1)
            low, high = np.quantile(bs, [.025, .975])
            close(low, r["cohort_bootstrap_ci_low"], "cohort CI lower")
            close(high, r["cohort_bootstrap_ci_high"], "cohort CI upper")
            interval_checks.append({"table": filename, "method_or_comparator": r[key], "cohorts": len(a),
                                    "estimate": float(np.mean(a)), "ci_low": float(low), "ci_high": float(high), "saved_seed": int(r["bootstrap_seed"])})
    return {"cohorts": 22, "splits_per_cohort": 20, "method_split_rows": 1760,
            "matched_K_verified": True, "intervals_reproduced": interval_checks,
            "claim_ledger_rows_checked_here": False, "raw_expression_refitted": False,
            "new_protocol_or_inference": False}


def verify_other_components(payload):
    p1 = table(payload[P1 + "replication_by_pathway.tsv"])
    require(len(p1) == 50, "P1 candidate census")
    flags = ("empirical_selected", "q_value_matched_selected", "q_value_size_matched_selected")
    counts = []
    for flag in flags:
        selected = [r for r in p1 if r[flag] == "True"]
        require(len(selected) == 23, "P1 membership changed")
        counts.append(sum(r["replicated_primary"] == "True" for r in selected))
    require(counts == [18, 17, 16], "P1 endpoint changed")
    for r in p1:
        replicate = float(r["NES_48h"]) * float(r["NES_72h"]) > 0 and float(r["padj_72h"]) < .05
        require(replicate == (r["replicated_primary"] == "True"), "P1 endpoint mismatch")
    dex = table(payload[DEX + "TERM_LEDGER.tsv"])
    require(len(dex) == 50, "Dex census")
    require([r["term_id"] for r in dex if r["Q_PLUS_DONOR_LOO_3_OF_4"] == "True"] ==
            [r["term_id"] for r in dex if r["MATCHED_Q_K"] == "True"], "Dex selected IDs differ")
    for r in dex:
        replica = float(r["discovery_NES"]) * float(r["validation_NES"]) > 0 and float(r["validation_q"]) <= .05
        require(replica == (r["replicated_24h"] == "True"), "Dex endpoint mismatch")
    mapping = table(payload[P5 + "mapping_qc.tsv"])
    pairs = table(payload[P5 + "hierarchy_pairs.tsv"])
    for col, mapped in (("C5_GO_BP", 481), ("C2_CP_REACTOME", 492)):
        group = [r for r in mapping if r["collection"] == col]
        require(len(group) == 500 and sum(r["mapping_status"] == "MAPPED_UNIQUE" for r in group) == mapped, "P5 mapping mismatch")
    for r in pairs:
        require(r["status_pattern"] == r["child_status"] + "->" + r["parent_status"], "P5 child-to-parent label mismatch")
        require((r["parent_direction"] != r["child_direction"]) == bool(int(r["directional_contradiction"])), "P5 direction mismatch")
        intersection, parent, child = (int(r[k]) for k in ("intersection_n", "parent_gene_n", "child_gene_n"))
        close(intersection / (parent + child - intersection), r["leading_edge_jaccard"], "P5 Jaccard")
    for r in table(payload[P5 + "hierarchy_metrics.tsv"]):
        group = [x for x in pairs if (x["collection"], x["relation_scope"]) == (r["collection"], r["relation_scope"])]
        require(len(group) == int(r["n_pairs"]), "P5 pair census")
        require(sum(int(x["directional_contradiction"]) for x in group) == int(r["n_directional_contradictions"]), "P5 discordance mismatch")
        for field in ("leading_edge_jaccard", "child_covered_by_parent", "parent_covered_by_child"):
            close(np.median([float(x[field]) for x in group]), r["median_" + field], "P5 median")
    endpoints = table(payload[R06 + "rater_method_endpoints.tsv"])
    for r in endpoints:
        close(int(r["n_event_ratings"]) / int(r["n_selected_ratings"]), r["confirmed_fraction"], "R06 confirmed fraction")
    prose = table(payload[R07 + "source_fact_check.optional.private.tsv"])
    require(len(prose) == 50, "R07 first-text census")
    for r in prose:
        require(digest(r["unaudited_text"].encode()) == r["text_sha256"], "R07 text identity mismatch")
        require(not r["source_faithfulness"] and not r["annotator_id"], "New source labels detected")
    return {"P1_hits_at_K23": counts, "dex_matched_selections_identical": True,
            "P5_status_pattern_orientation": "child_to_parent", "P5_hierarchy_used_by_audit": False,
            "R07_texts_checked": 50, "new_source_or_expert_labels": 0}


def save_figure(fig, out, name):
    fig.savefig(out / (name + ".pdf"), metadata={"Title": name, "Subject": "Saved-result main-figure draft"})
    fig.savefig(out / (name + ".png"), dpi=240)
    plt.close(fig)


def figures(payload, out):
    get = lambda name: table(payload[name])
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                         "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42,
                         "svg.fonttype": "none"})
    # Figure 1: exact source comparisons, explicitly not automatic audit successes.
    fig = plt.figure(figsize=(11.7, 8.0))
    ax = fig.add_axes([.035, .72, .93, .23]); ax.axis("off")
    ax.text(0, 1.03, "A   Pathway-specific evidence contract", weight="bold", fontsize=13)
    ax.set_xlim(-.015,1.015)
    boxes = [(0, "Enrichment + context", "Term ID, statistics, supporting\ngenes and study comparison"),
             (.255, "Linked candidate", "Schema fields and tool-owned\nevidence identifiers"),
             (.51, "Recorded checks", "Integrity, stability, configured\ncontext and contradiction rules"),
             (.765, "Inspectable report", "Disposition + reasons + source\nlinks for analyst review")]
    for x, title, body in boxes:
        ax.add_patch(FancyBboxPatch((x, .30), .225, .52, boxstyle="round,pad=0.012", fc="#eef3f7", ec="#728ca1", lw=1))
        ax.text(x + .1125, .64, title, ha="center", weight="bold", fontsize=9)
        ax.text(x + .1125, .45, body, ha="center", va="center", fontsize=8.1)
        if x < .76:
            ax.add_patch(FancyArrowPatch((x+.228, .55), (x+.245, .55), arrowstyle="-|>", mutation_scale=12, color="#526574"))
    ax.text(0, .07, "A recorded PASS certifies implemented conditions. Biological validity and prose interpretation require separate evidence.", fontsize=9)
    texts = get(R07 + "source_fact_check.optional.private.tsv")
    panels = [("ADIPOGENESIS", "B", "Significance wording exceeds the provided q value", (.035,.405,.93,.275)),
              ("G2M_CHECKPOINT", "C", "Numeric agreement leaves a causal claim unchecked", (.035,.075,.93,.275))]
    for term, letter, title, position in panels:
        a = fig.add_axes(position); a.axis("off")
        r = next(x for x in texts if x["evidence_id"].endswith("_"+term))
        a.text(0, .97, f"{letter}   {title}", weight="bold", fontsize=12)
        a.text(0, .80, "HALLMARK_" + term, fontsize=10, weight="bold", color="#286E99")
        a.text(0, .63, f"Source: NES = {float(r['source_nes']):.4f}; BH q = {float(r['source_q']):.4g}", fontsize=10)
        if term == "ADIPOGENESIS":
            quote = "while the result is statistically significant, it may not be as robust as other findings in the study"
            diagnosis = "At q = 0.6644, a significance claim at q ≤ 0.05 is unsupported."
            scope = "Source-derived wording: negative enrichment statistic; does not meet the q ≤ 0.05 reporting threshold."
        else:
            quote = "the absence of functional TP53 leads to down-regulation of cell cycle checkpoint genes"
            diagnosis = "The observational contrast does not establish functional TP53 loss or causality."
            scope = "This is the sole numeric-gate checked text. The gate did not adjudicate the causal assertion."
        require(quote in r["unaudited_text"], "Vignette quote no longer matches")
        a.text(.40, .75, "Verbatim excerpt from the first local-model text:", fontsize=9, color="#5b6570")
        a.text(.40, .56, textwrap.fill('“'+quote+'”', 85), va="center", fontsize=10)
        a.text(0, .33, diagnosis, fontsize=10, color="#8b4d2b")
        a.text(0, .13, scope, fontsize=9)
        a.axhline(.015, color="#d7dfe5", lw=1)
    fig.text(.035,.03,"Post hoc source comparisons on previously examined HNSC discovery data; no new expert labels, accuracy rate or full-audit success claim.",fontsize=8)
    save_figure(fig, out, "Figure1_Evidence_Contract")
    # Figure 2: unfavorable external statistical contrast and rater disagreement.
    fig = plt.figure(figsize=(12.0, 9.8))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.14, 1.12, 1.1], hspace=.48, wspace=.68,
                          left=.07, right=.97, top=.94, bottom=.19)
    a = fig.add_subplot(gs[:,0])
    cohorts = get(P2B + "figure2_panelB_cohort_replication.tsv")
    ids = sorted({r["cohort_id"] for r in cohorts})
    lookup = {(r["cohort_id"],r["method_id"]):float(r["mean_replication_fraction"])*100 for r in cohorts}
    for i, co in enumerate(ids):
        a.plot([lookup[(co,"full_audit")],lookup[(co,"q_value_matched")]], [i,i], color="#d4dce2", lw=1.5)
    for method, marker in (("full_audit","o"),("q_value_matched","s"),("stability_matched","^")):
        a.scatter([lookup[(co,method)] for co in ids],range(len(ids)),s=21,c=COLORS[method],marker=marker,label=LABELS[method],zorder=3)
    a.set_yticks(range(len(ids)),ids); a.invert_yaxis(); a.set_xlim(-2,103)
    a.set_xlabel("Held-out replication (%)\nMean of 20 splits per cohort")
    a.set_title("A   Paired cohort outcomes",loc="left",weight="bold",fontsize=11)
    handles, labels = a.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower left", bbox_to_anchor=(.063,.085), ncol=3, frameon=False, fontsize=8)
    a = fig.add_subplot(gs[0,1])
    methods = {r["method_id"]:r for r in get(P2B + "figure2_panelB_method_summary.tsv")}
    for i, m in enumerate(ORDER):
        r=methods[m]; x=float(r["mean_cohort_replication_fraction"])*100
        lo=float(r["cohort_bootstrap_ci_low"])*100; hi=float(r["cohort_bootstrap_ci_high"])*100
        a.errorbar(x,i,xerr=[[x-lo],[hi-x]],fmt="o",color=COLORS[m],capsize=3)
        a.text(x,i-.20,f"{x:.1f}%",ha="center",fontsize=8,color=COLORS[m])
    a.set_yticks(range(4),[LABELS[x].replace(" (descriptive)","\n(descriptive)") for x in ORDER],fontsize=8)
    a.set_ylim(3.6,-.5); a.set_xlim(0,100); a.set_xlabel("Cohort macro-mean replication (%)\n95% cohort bootstrap interval",fontsize=8)
    a.set_title("B   Same-pool comparisons",loc="left",weight="bold",fontsize=11)
    a = fig.add_subplot(gs[1,1])
    contrasts = get(P2B + "figure2_panelC_contrast_summary.tsv") + get(P2B + "figure2_sensitivity_contrast_summary.tsv")
    for i,r in enumerate(contrasts):
        x=float(r["mean_cohort_difference_a_minus_b"])*100
        lo=float(r["cohort_bootstrap_ci_low"])*100; hi=float(r["cohort_bootstrap_ci_high"])*100
        a.errorbar(x,i,xerr=[[x-lo],[hi-x]],fmt="o",color=COLORS[r["method_b"]],capsize=3)
        a.text(x,i-.20,f"{x:+.1f} pp",ha="center",fontsize=8)
    a.axvline(0,c="#9aa6af",ls="--",lw=1)
    a.set_yticks(range(4),["vs q · 22 cohorts","vs stability · 22","vs q · 16 sensitivity","vs stability · 16"],fontsize=8)
    a.set_ylim(3.6,-.5); a.set_xlim(-30,4); a.set_xlabel("Legacy full audit − comparator\n95% paired cohort bootstrap interval",fontsize=8)
    a.set_title("C   Paired contrasts",loc="left",weight="bold",fontsize=11)
    a = fig.add_subplot(gs[0,2])
    ratings=[r for r in get(R06+"rater_method_endpoints.tsv") if r["endpoint"]=="confirmed_major_overstatement"]
    for j,method in enumerate(("q_value_matched","legacy_full_audit")):
        color=COLORS["q_value_matched" if j==0 else "full_audit"]
        for i, rater in enumerate(("P4_R1","P4_R2","P4_R3","FIXED_RATER_MEAN")):
            r=next(x for x in ratings if x["method"]==method and x["rater_id"]==rater)
            y=i + (-.11 if j==0 else .11); x=float(r["confirmed_fraction"])*100
            upper=float(r["uncertainty_bound_high"])*100
            a.plot([x,upper],[y,y],lw=3,alpha=.45,color=color)
            a.scatter(x,y,s=28,color=color,marker="s" if j==0 else "o",label=LABELS["q_value_matched" if j==0 else "full_audit"] if i==0 else None)
    a.set_yticks(range(4),["Rater 1","Rater 2","Rater 3","Fixed-rater mean"],fontsize=8); a.set_ylim(3.5,-.6); a.set_xlim(-4,104)
    a.set_xlabel("Confirmed major overstatement (%)\nLines: uncertain-label bounds, not CIs",fontsize=8)
    a.set_title("D   Original ratings, K = 25",loc="left",weight="bold",fontsize=11)
    a = fig.add_subplot(gs[1,2])
    agreement=get(R06+"interrater_agreement.tsv")
    for i,r in enumerate(agreement):
        x=float(r["fleiss_kappa"]); lo=float(r["kappa_ci_low"]); hi=float(r["kappa_ci_high"])
        a.errorbar(x,i,xerr=[[x-lo],[hi-x]],fmt="o",color="#425a70",capsize=3)
        a.text(x,i-.18,f"{x:.3f}",ha="center",fontsize=8)
    a.axvline(0,c="#9aa6af",ls="--",lw=1)
    a.set_yticks(range(3),["Statistical\nsupport","External\nevidence","Overstatement"],fontsize=8)
    a.set_ylim(2.6,-.5);a.set_xlim(-.43,.11);a.set_xlabel("Fleiss' κ\nDescriptive 95% claim bootstrap interval",fontsize=8)
    a.set_title("E   Agreement on 50 texts",loc="left",weight="bold",fontsize=11)
    fig.text(.07,.02,"A–C: 22 cohorts, not 440 independent validation cohorts. D–E: three fixed raters, 150 original ratings; not scores for new prose.\nLower overstatement is preferable; these data do not establish a general full-audit interpretive advantage.",fontsize=8)
    save_figure(fig,out,"Figure2_Matched_Baselines_and_Raters")
    # Figure 3: two statistical-component cases, no independent-term intervals.
    fig, axs=plt.subplots(1,2,figsize=(11.7,5.4))
    p1=get(P1+"replication_by_method.tsv")
    order=["raw_pool","empirical_stability_audit","q_value_matched","q_value_and_leading_edge_size_matched"]
    lookup={r["method"]:r for r in p1}
    a=axs[0]; rr=[lookup[m] for m in order]
    a.barh(range(4),[float(r["replication_fraction"])*100 for r in rr],color=[COLORS["raw_pool"],COLORS["stability_matched"],COLORS["q_value_matched"],"#B09685"])
    a.set_yticks(range(4),["All 50","Empirical stability","Matched q","Size-matched q"],fontsize=9);a.invert_yaxis();a.set_xlim(0,100)
    for i,r in enumerate(rr):a.text(float(r["replication_fraction"])*100+1.5,i,f"{r['n_replicated']}/{r['n_selected']}",va="center",fontsize=9)
    a.set_title("A   TP53 × DNA-damage response",loc="left",weight="bold",fontsize=12,pad=32)
    a.set_xlabel("72-h statistical replication (%)")
    a.text(0,1.025,"GSE146225 · ENDO 48 h discovery → 72 h",transform=a.transAxes,fontsize=9,color="#5b6570")
    a.text(0,-.25,"18/23 vs 17/23: +4.35 pp; exact reference p = 0.50.\nSame study; overlapping pathways; component evaluation.",transform=a.transAxes,fontsize=8,va="top")
    a=axs[1];dex=json.loads(payload[DEX+"SUMMARY.json"])["comparisons"]
    a.barh(range(4),[r["replication_rate"]*100 for r in dex],color=[COLORS["raw_pool"],"#B09685",COLORS["stability_matched"],COLORS["q_value_matched"]])
    a.set_yticks(range(4),["All 50","Discovery q ≤ 0.05","q + donor LOO","Matched q, K = 10"],fontsize=9);a.invert_yaxis();a.set_xlim(0,100)
    for i,r in enumerate(dex):a.text(r["replication_rate"]*100+1.5,i,f"{r['replicated_count']}/{r['selected_count']}",va="center",fontsize=9)
    a.set_title("B   Airway dexamethasone response",loc="left",weight="bold",fontsize=12,pad=32)
    a.text(0,1.025,"GSE52778 RNA-seq → GSE34313 microarray",transform=a.transAxes,fontsize=9,color="#5b6570")
    a.set_xlabel("24-h culture-level statistical replication (%)")
    a.text(0,-.25,"LOO and matched q retained the same ten terms.\nValidation: one cell line; 3 dex cultures + 4 controls.",transform=a.transAxes,fontsize=8,va="top")
    fig.subplots_adjust(left=.16,right=.97,top=.81,bottom=.28,wspace=.70)
    save_figure(fig,out,"Figure3_Temporal_and_Noncancer_Components")
    # Figure 4: independently defined hierarchy diagnostics, not correctness labels.
    fig,axs=plt.subplots(2,2,figsize=(11.7,8.2))
    pairs=get(P5+"hierarchy_pairs.tsv");metrics=get(P5+"hierarchy_metrics.tsv")
    direct=[r for r in metrics if r["relation_scope"]=="direct_parent_child"]
    a=axs[0,0]; x=np.arange(2)
    a.bar(x,[float(r["directional_contradiction_fraction"])*100 for r in direct],color=["#5D7896","#819D9B"],width=.55)
    for i,r in enumerate(direct):a.text(i,float(r["directional_contradiction_fraction"])*100+2,f"{r['n_directional_contradictions']}/{r['n_pairs']}",ha="center")
    a.set_xticks(x,["GO biological process","Reactome"],fontsize=9);a.set_ylim(0,55)
    a.set_ylabel("Opposite enrichment directions (%)");a.set_title("A   Direct hierarchy pairs",loc="left",weight="bold",fontsize=12)
    a=axs[0,1]
    for i,col in enumerate(("C5_GO_BP","C2_CP_REACTOME")):
        rows=[r for r in pairs if r["collection"]==col and r["relation_scope"]=="direct_parent_child"]
        values=[float(r["child_covered_by_parent"]) for r in rows]
        offsets=np.linspace(-.12,.12,len(values))
        a.scatter(np.full(len(values),i)+offsets,values,s=9,color=["#5D7896","#819D9B"][i],alpha=.30)
        med=float(np.median(values));a.plot([i-.17,i+.17],[med,med],c="#2c4359",lw=2)
        a.text(i,med+.07,f"Median {med:.2f}",ha="center",fontsize=9)
    a.set_xticks(x,["GO biological process","Reactome"],fontsize=9);a.set_ylim(-.06,1.18)
    a.set_ylabel("Child supporting genes covered by parent");a.set_title("B   Supporting-gene containment",loc="left",weight="bold",fontsize=12)
    status=["PASS","ABSTAIN","FAIL"]
    matrix_rows=[]
    for i,col in enumerate(("C5_GO_BP","C2_CP_REACTOME")):
        a=axs[1,i];rows=[r for r in pairs if r["collection"]==col and r["relation_scope"]=="direct_parent_child"]
        counts=Counter((r["parent_status"],r["child_status"]) for r in rows)
        matrix=np.array([[counts[(p,c)] for c in status] for p in status])
        a.imshow(matrix,cmap="Blues",vmin=0,vmax=100,aspect="equal")
        for yy,p in enumerate(status):
            for xx,c in enumerate(status):
                a.text(xx,yy,str(matrix[yy,xx]),ha="center",va="center",color="white" if matrix[yy,xx]>=60 else "#283d52",fontsize=11)
                matrix_rows.append({"collection":col,"parent_status":p,"child_status":c,"direct_pairs":int(matrix[yy,xx])})
        a.set_xticks(range(3),status,fontsize=9);a.set_yticks(range(3),status,fontsize=9)
        a.set_xlabel("Child disposition");a.set_ylabel("Parent disposition")
        a.set_title(("C   GO" if i==0 else "D   Reactome")+" disposition pairs",loc="left",weight="bold",fontsize=12)
        for spine in a.spines.values():spine.set_visible(False)
    fig.subplots_adjust(left=.10,right=.97,top=.94,bottom=.11,hspace=.56,wspace=.39)
    fig.text(.10,.02,"Hierarchy was not an audit input. Mapping: GO 481/500; Reactome 492/500. Pair counts share terms and genes.\nOpposite directions are not biological-error labels; a child PASS does not require its parent to PASS.",fontsize=8)
    write_table(out.parent/"Source_Data/Figure4_Disposition_Pairs.tsv",matrix_rows)
    save_figure(fig,out,"Figure4_Ontology_Hierarchy_Diagnostics")


def build_report(archive, out):
    archive, out = Path(archive).resolve(), Path(out).resolve()
    require(not out.exists() and not archive.is_relative_to(out), "Use a new output folder")
    before = digest(archive.read_bytes())
    payload, summary, exports = verify_archive(archive)
    p2b = verify_p2b(payload)
    other = verify_other_components(payload)
    out.mkdir(parents=True)
    source = out / "verified_source"; source.mkdir()
    for name, raw in payload.items():
        dest = source/name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(raw)
    figdir = out/"figures";figdir.mkdir()
    datadir = out/"Source_Data";datadir.mkdir()
    paths = [n for n in exports if n.startswith(P2B) and n.endswith('.tsv')]
    paths += [P1+"replication_by_pathway.tsv",P1+"replication_by_method.tsv",R06+"rater_method_endpoints.tsv",R06+"interrater_agreement.tsv",
              R07+"source_fact_check.optional.private.tsv",R07+"method_coverage.tsv",DEX+"TERM_LEDGER.tsv",DEX+"METHOD_COMPARISON.tsv",
              P5+"hierarchy_pairs.tsv",P5+"hierarchy_metrics.tsv",P5+"mapping_qc.tsv"]
    routes=[]
    for name in paths:
        label = ("P2B_" if name.startswith(P2B) else "P1_" if name.startswith(P1) else "P5_" if name.startswith(P5) else
                 "R06_" if name.startswith(R06) else "R07_" if name.startswith(R07) else "Dex_") + Path(name).name
        (datadir/label).write_bytes(payload[name])
        routes.append({"copied_file":label,"original_archive_entry":name,"sha256":digest(payload[name])})
    write_table(out/"SOURCE_FILE_MAP.tsv",routes)
    panel_routes = [
        ("1A", "Conceptual diagram", "No empirical denominator", "Contract fields and documented implementation boundary"),
        ("1B–C", "R07_source_fact_check.optional.private.tsv", "Two post hoc illustrations; all 50 texts supplied", "Exact text/hash, source NES/q, observational contrast"),
        ("2A", "P2B_figure2_panelB_cohort_replication.tsv", "22 cohorts; 20 splits averaged within cohort", "Paired full/q means and matched stability"),
        ("2B", "P2B_figure2_panelB_method_summary.tsv", "22 cohorts", "Saved means and cohort bootstrap limits"),
        ("2C", "P2B_figure2_panelC_contrast_summary.tsv; P2B_figure2_sensitivity_contrast_summary.tsv", "22 primary / 16 sensitivity cohorts", "Saved paired differences and intervals; values in percentage points"),
        ("2D", "R06_rater_method_endpoints.tsv", "25 selected statements/method; 3 fixed raters", "Confirmed major overstatement and uncertain-label bounds; not CIs"),
        ("2E", "R06_interrater_agreement.tsv", "50 unchanged statements; 3 fixed raters", "Saved Fleiss kappa and descriptive claim-bootstrap limits"),
        ("3A", "P1_replication_by_method.tsv; P1_replication_by_pathway.tsv", "50 all / 23 selected pathways", "Saved same-direction and 72h FDR<0.05 replication fractions"),
        ("3B", "Dex_METHOD_COMPARISON.tsv; Dex_TERM_LEDGER.tsv", "50, 11, 10 and 10 terms; one validation cell line", "Saved 24h same-direction q<=0.05 fractions and identical selected sets"),
        ("4A", "P5_hierarchy_metrics.tsv; P5_hierarchy_pairs.tsv", "41 GO / 213 Reactome direct dependent pairs", "Fixed directional-discordance counts, without biological-error labels"),
        ("4B", "P5_hierarchy_pairs.tsv", "Same direct pairs", "Per-pair child_covered_by_parent values and median"),
        ("4C–D", "P5_hierarchy_pairs.tsv; Figure4_Disposition_Pairs.tsv", "Same direct pairs", "Counts by named parent_status rows and child_status columns")]
    write_table(out/"FIGURE_PANEL_MAP.tsv",[dict(zip(("panel", "source_data_files", "unit_or_denominator", "rendering_rule"),r)) for r in panel_routes])
    figures(payload,figdir)
    require(digest(archive.read_bytes()) == before, "Input ZIP changed during reporting")
    for name, expected in exports.items():
        require(digest((source/name).read_bytes()) == expected, "Copied source changed")
    verification={"schema":"CRM_R1_MAIN_FIGURE_REPORT_R12", "status":"SAVED_TABLES_VERIFIED_FOUR_FIGURE_DRAFTS",
                  "source_archive_sha256":before,"source_export_files_verified":len(exports),
                  "original_Word_sha256":digest(payload["current_manuscript/LLM-PathwayCurator_CRM_R1.docx"]),
                  "p2b":p2b,"other_components":other,"source_collection_missing":summary["missing_or_unverified_components"],
                  "reporting_code_sha256":digest(Path(__file__).read_bytes()),
                  "numpy_version":np.__version__,"matplotlib_version":matplotlib.__version__,
                  "model_calls":0,"new_expert_ratings":0,"R_fitting_or_enrichment_rerun":False,
                  "original_Word_edited":False,"raw_input_bytes_authenticated_here":False,
                  "current_full_audit_advantage_established":False,"submission_ready":False}
    (out/"SOURCE_VERIFICATION.json").write_text(json.dumps(verification,indent=2)+"\n")
    return verification


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root",type=Path,required=True)
    parser.add_argument("--source-archive",type=Path,required=True,help="Exact collected ZIP; no automatic newest-run selection")
    args=parser.parse_args()
    root=args.data_root.expanduser().resolve()
    require((root/"input").is_dir() and (root/"output").is_dir(), "Use the existing CRM_R1 root")
    stamp=datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    out=root/"output/revision_v17"/f"manuscript_report_{stamp}"
    value=build_report(args.source_archive.expanduser(),out)
    print(json.dumps({"outdir":str(out),"status":value["status"],"P2B_intervals_reproduced":len(value["p2b"]["intervals_reproduced"]),
                      "model_calls":0,"new_expert_ratings":0,"submission_ready":False},indent=2))


if __name__=="__main__":
    main()
