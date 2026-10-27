#!/usr/bin/env python3
"""Render Figures S1-S4 and numerical tables from the deposited source records.

Plot definitions ported from the October 24 supplement renderer. Edited Word
layout is maintained separately; no manuscript file is written by this script.
"""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from _figure_io import finish
from matplotlib.colors import ListedColormap, TwoSlopeNorm
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


def render(DATA, OUT):
    OUT.mkdir(parents=True, exist_ok=False)
    FIG = OUT
    D1 = DATA / "Supplementary_Data_1_TCGA"
    D2 = DATA / "Supplementary_Data_2_DEX"
    D3 = DATA / "Supplementary_Data_3_External_HNSC"
    D4 = DATA / "Supplementary_Data_4_Comparisons"
    used = []

    def read(p):
        used.append(p)
        return pd.read_csv(p, sep="\t", keep_default_na=False)

    coh = read(D1 / "TCGA_cohorts.tsv")
    terms = read(D1 / "TCGA_350_terms.tsv")
    base = read(D1 / "q_baseline_comparison.tsv")
    dex = read(D2 / "DEX_text_checks_50.tsv")
    ratings = read(D2 / "DEX_case_ratings_24.tsv")
    endpoints = read(D2 / "DEX_endpoint_summary.tsv")
    interpret = read(D2 / "DEX_interpretation_counts.tsv")
    hc = read(D3 / "rater_category_counts.tsv")
    ha = read(D3 / "interrater_agreement.tsv")
    hr = read(D3 / "statement_ratings.tsv")
    split = read(D4 / "split_replication/figure2_panelB_cohort_replication.tsv")
    sm = read(D4 / "split_replication/figure2_panelB_method_summary.tsv")
    contrast = read(D4 / "split_replication/figure2_panelC_contrast_summary.tsv")
    temp = read(D4 / "temporal_replication/replication_by_method.tsv")
    cross = read(D4 / "dexamethasone_replication/METHOD_COMPARISON.tsv")
    mapping = read(D4 / "hierarchy/mapping_qc.tsv")
    hier = read(D4 / "hierarchy/hierarchy_metrics.tsv")
    checks = []

    def check(name, value):
        checks.append({"check": name, "passed": bool(value)})
        if not value:
            raise ValueError(name)

    check(
        "Seven cohorts and 350 unique cohort-term records",
        len(coh) == 7 and len(terms) == 350 and not terms.duplicated(["cohort", "term_id"]).any(),
    )
    check("159 records meet BH q <= 0.05", int((terms.qval <= 0.05).sum()) == 159)
    for r in coh.itertuples():
        x = terms[terms.cohort == r.cohort]
        check(
            f"{r.cohort} complete counts",
            len(x) == 50
            and (x.qval <= 0.05).sum() == r.q_le_0_05
            and r.MUT + r.call_negative + r.excluded == r.expression_samples,
        )
    check(
        "No selected identity differences in supplied baseline comparison",
        base.selection_difference_count.eq(0).all() and base.source_selected.sum() == 159,
    )
    check(
        "DEX original text hashes",
        all(hashlib.sha256(r.text.encode()).hexdigest() == r.text_sha256 for r in dex.itertuples()),
    )
    check(
        "DEX statuses 12 / 30 / 8",
        dex.limited_checks_status.value_counts().to_dict()
        == {"REVIEW_FLAGGED": 30, "VIOLATION_DETECTED": 12, "NO_LIMITED_FLAG": 8},
    )
    check(
        "DEX subset statuses 4 / 18 / 2",
        ratings.tool_status.value_counts().to_dict()
        == {"REVIEW_FLAGGED": 18, "VIOLATION_DETECTED": 4, "NO_LIMITED_FLAG": 2},
    )
    for a, b, n in [("J1_material", "J2_material", 9), ("J1_strength", "J2_strength", 13)]:
        check(a + " observed agreement", int((ratings[a] == ratings[b]).sum()) == n)
    check(
        "50 identical statements across three HNSC evaluators",
        len(hr) == 150
        and hr.claim_uid.nunique() == 50
        and hr.groupby("claim_uid").claim_text.nunique().eq(1).all(),
    )
    for r in hc.itertuples():
        check(
            f"HNSC category {r.rater_id}/{r.question}/{r.category}",
            int((hr.loc[hr.rater_id == r.rater_id, r.question] == r.category).sum()) == r.count,
        )
    pv = split.pivot(
        index="cohort_id", columns="method_id", values="mean_replication_fraction"
    ).sort_index()
    check("22 cohorts with 20 splits per method", len(pv) == 22 and split.n_splits.eq(20).all())
    for r in sm.itertuples():
        check(
            "Replication mean " + r.method_id,
            np.isclose(
                pv[r.method_id].mean(), r.mean_cohort_replication_fraction, rtol=0, atol=1e-12
            ),
        )
    for r in contrast.itertuples():
        check(
            "Paired contrast " + r.method_b,
            np.isclose(
                (pv[r.method_a] - pv[r.method_b]).mean(),
                r.mean_cohort_difference_a_minus_b,
                rtol=0,
                atol=1e-12,
            ),
        )
    for c, n in [("C5_GO_BP", 481), ("C2_CP_REACTOME", 492)]:
        check(
            c + " unique ontology mapping",
            ((mapping.collection == c) & (mapping.mapping_status == "MAPPED_UNIQUE")).sum() == n,
        )

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.facecolor": "white",
        }
    )
    BLUE = "#2166AC"
    ORANGE = "#C87532"
    GRAY = "#6D7680"
    TEAL = "#218C7B"

    def savefig(fig, name):
        for ext in ["png", "pdf", "svg"]:
            fig.savefig(FIG / f"{name}.{ext}", dpi=300, bbox_inches="tight", pad_inches=0.10)
        plt.close(fig)

    # Figure S1 is a software diagram, not a plot of empirical performance.
    fig, ax = plt.subplots(figsize=(7.0, 6.9))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")

    def box(x, y, w, h, title, body, color="#EEF3F6"):
        ax.add_patch(
            FancyBboxPatch(
                (x, y),
                w,
                h,
                boxstyle="round,pad=0.10,rounding_size=0.08",
                lw=0.8,
                edgecolor="#697680",
                facecolor=color,
            )
        )
        ax.text(x + w / 2, y + h - 0.28, title, ha="center", va="top", fontsize=10, weight="bold")
        ax.text(x + w / 2, y + h - 0.73, body, ha="center", va="top", fontsize=9, linespacing=1.4)

    def arrow(a, b):
        ax.add_patch(
            FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=12, color="#424C55", lw=1.1)
        )

    box(
        0.4,
        8.2,
        9.2,
        1.4,
        "Source records and study comparison",
        "Complete enrichment table • original file hash • record locator\n"
        "Source statistic and adjusted value • supporting genes",
    )
    box(
        0.4,
        5.7,
        4.35,
        1.7,
        "Statistical observations",
        "q ≤ 0.05; no reporting cap here\n"
        "PASS = reporting eligibility\nOther source records retained",
    )
    box(
        5.25,
        5.7,
        4.35,
        1.7,
        "Supporting-gene modules",
        "Shared genes ≥ 3; Jaccard ≥ 0.10\n"
        "Components include isolated terms\nDESCRIPTIVE support summaries",
    )
    arrow((2.6, 8.1), (2.6, 7.5))
    arrow((7.4, 8.1), (7.4, 7.5))
    box(
        0.4,
        3.05,
        9.2,
        1.65,
        "Imported proposals and exact evidence references",
        "Original JSONL retained; exact comparison and evidence references\n"
        "Single-source observations: limited text checks\n"
        "Hypotheses and multi-source text: review required",
        "#F7F4EE",
    )
    arrow((2.6, 5.6), (2.6, 4.8))
    arrow((7.4, 5.6), (7.4, 4.8))
    box(
        0.4,
        0.7,
        4.35,
        1.5,
        "FAIL",
        "Explicit violation detected\nFindings link to source and text spans",
        "#FBEEE8",
    )
    box(
        5.25,
        0.7,
        4.35,
        1.5,
        "ABSTAIN",
        "No explicit violation detected\nInterpretation still requires review",
        "#EFF5F1",
    )
    arrow((3, 2.95), (2.6, 2.3))
    arrow((7, 2.95), (7.4, 2.3))
    ax.text(
        5, 0.1, "No automatic acceptance of imported prose", ha="center", fontsize=10, weight="bold"
    )
    savefig(fig, "Figure_S1_source_contract")

    cohort_order = coh.cohort.tolist()
    term_order = sorted(terms.term_id.unique())
    nes = terms.pivot(index="term_id", columns="cohort", values="stat").loc[
        term_order, cohort_order
    ]
    qs = terms.pivot(index="term_id", columns="cohort", values="qval").loc[term_order, cohort_order]
    fig = plt.figure(figsize=(7.2, 8.35))
    ax = fig.add_axes([0.49, 0.08, 0.40, 0.84])
    cax = fig.add_axes([0.91, 0.28, 0.018, 0.40])
    im = ax.imshow(
        nes.to_numpy(),
        aspect="auto",
        cmap="RdBu_r",
        norm=TwoSlopeNorm(vmin=-4, vcenter=0, vmax=4),
        interpolation="nearest",
    )
    y, x = np.where(qs.to_numpy() <= 0.05)
    ax.scatter(x, y, s=12, c="black", marker="o", linewidths=0.25, edgecolors="white")
    ax.set_xticks(range(7), cohort_order, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(
        range(50), [t.replace("HALLMARK_", "").replace("_", " ") for t in term_order], fontsize=7.3
    )
    ax.tick_params(axis="both", length=0, pad=4)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title("All 50 Hallmark sets in each cohort", fontsize=11, pad=12)
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("NES  (mutation-positive minus call-negative)", fontsize=8)
    fig.text(0.50, 0.985, "●  BH q ≤ 0.05  (159 of 350 records)", ha="center", va="top", fontsize=9)
    savefig(fig, "Figure_S2_TCGA_Hallmark_census")

    r = ratings.sort_values("reference_id").copy()

    def term_label(term):
        words = term.replace("HALLMARK_", "").split("_")
        caps = {
            "DNA",
            "RNA",
            "IL6",
            "JAK",
            "STAT3",
            "UV",
            "TNFA",
            "NFKB",
            "WNT",
            "MYC",
            "V1",
            "V2",
            "E2F",
        }
        return " ".join(w if w in caps else w.lower() if w == "VIA" else w.title() for w in words)

    labels = [f"{a}  " + term_label(b) for a, b in zip(r.reference_id, r.term_id, strict=True)]
    vals = np.column_stack(
        [
            r.J1_material.eq("DEFINITE_MATERIAL_ERROR"),
            r.J2_material.eq("DEFINITE_MATERIAL_ERROR"),
            r.J1_strength.eq("OVERSTATED"),
            r.J2_strength.eq("OVERSTATED"),
        ]
    ).astype(int)
    fig = plt.figure(figsize=(7.2, 7.0))
    ax = fig.add_axes([0.52, 0.18, 0.40, 0.72])
    ax.imshow(
        vals,
        aspect="auto",
        cmap=ListedColormap(["#EDF0F3", "#436E91"]),
        vmin=0,
        vmax=1,
        interpolation="nearest",
    )
    ax.set_xticks(range(4), ["J1", "J2", "J1", "J2"], fontsize=10)
    ax.set_yticks(range(24), labels, fontsize=7.8)
    ax.tick_params(length=0, pad=5)
    ax.set_xticks(np.arange(-0.5, 4, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 24, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.2)
    ax.tick_params(which="minor", length=0)
    for i in range(24):
        for j in range(4):
            ax.text(
                j,
                i,
                "Yes" if vals[i, j] else "No",
                va="center",
                ha="center",
                fontsize=7.7,
                color="white" if vals[i, j] else "#454B52",
            )
    ax.text(0.5, -2.0, "Material source\ncontradiction", ha="center", va="center", fontsize=9)
    ax.text(2.5, -2.0, "Qualitative\noverstatement", ha="center", va="center", fontsize=9)
    ax.axvline(1.5, color="white", linewidth=6)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.text(0.52, 0.09, "J1: author     J2: external reviewer", ha="center", fontsize=9)
    fig.text(
        0.52, 0.05, "Returned categories; no adjudicated reference labels", ha="center", fontsize=9
    )
    savefig(fig, "Figure_S3_DEX_returned_ratings")

    fig, (ax, ax2) = plt.subplots(
        1, 2, figsize=(7.1, 6.9), gridspec_kw={"width_ratios": [1.65, 1], "wspace": 0.38}
    )
    y = np.arange(len(pv))
    for i, (_, row) in enumerate(pv.iterrows()):
        ax.plot([row.full_audit, row.q_value_matched], [i, i], color="#BDC4CB", lw=1.4, zorder=1)
    ax.scatter(pv.full_audit, y, c=BLUE, s=28, label="Full audit", zorder=2)
    ax.scatter(pv.q_value_matched, y, c=ORANGE, s=28, marker="s", label="Matched q value", zorder=2)
    ax.set_yticks(y, pv.index)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1], ["0", "25", "50", "75", "100"])
    ax.set_xlabel("Mean replication across splits (%)")
    ax.grid(axis="x", alpha=0.2)
    ax.set_title("A   Paired cohort results", loc="left", fontsize=11, pad=14)
    ax.legend(loc="lower left", bbox_to_anchor=(-0.02, -0.17), frameon=False, fontsize=8.5)
    for i, (mid, label, color) in enumerate(
        [
            ("full_audit", "Full audit", BLUE),
            ("q_value_matched", "Matched q value", ORANGE),
            ("stability_matched", "Matched stability", TEAL),
            ("raw_pool", "All candidates", GRAY),
        ]
    ):
        row = sm[sm.method_id == mid].iloc[0]
        v = row.mean_cohort_replication_fraction
        ax2.errorbar(
            v,
            i,
            xerr=[[v - row.cohort_bootstrap_ci_low], [row.cohort_bootstrap_ci_high - v]],
            fmt="o",
            color=color,
            capsize=3,
            ms=6,
        )
        ax2.text(0.03, i - 0.24, label, fontsize=8.5)
        ax2.text(v, i + 0.24, f"{v * 100:.1f}%", ha="center", fontsize=8.5)
    ax2.set_ylim(3.6, -0.6)
    ax2.set_yticks([])
    ax2.set_xlim(0, 1)
    ax2.set_xticks([0, 0.5, 1], ["0", "50", "100"])
    ax2.set_xlabel("Across-cohort mean (%)")
    ax2.grid(axis="x", alpha=0.2)
    ax2.set_title("B   Descriptive summaries", loc="left", fontsize=10.5, pad=14)
    fig.subplots_adjust(left=0.10, right=0.98, bottom=0.14, top=0.91)
    savefig(fig, "Figure_S4_replication_comparison")

    tables = OUT / "tables"
    tables.mkdir()
    exports = {
        "Table_S1A_samples.tsv": coh,
        "Table_S1B_selection_and_components.tsv": base,
        "Table_S4A_split_methods.tsv": sm,
        "Table_S4A_split_contrasts.tsv": contrast,
        "Table_S4A_temporal.tsv": temp,
        "Table_S4B_dexamethasone.tsv": cross,
        "Table_S4C_hierarchy.tsv": hier,
        "Table_S5A_DEX_endpoints.tsv": endpoints,
        "Table_S5A_DEX_interpretation.tsv": interpret,
        "Table_S5B_HNSC_categories.tsv": hc,
        "Table_S5B_HNSC_agreement.tsv": ha,
        "Table_S6_worked_examples.tsv": dex[
            dex.reference_id.isin(["SRC003", "SRC019"])
        ].sort_values("reference_id"),
    }
    for name, frame in exports.items():
        frame.to_csv(tables / name, sep="\t", index=False, float_format="%.17g")
    (OUT / "data_checks.json").write_text(json.dumps(checks, indent=2) + "\n")
    finish(DATA, OUT, [str(p.relative_to(DATA)) for p in sorted(set(used))], Path(__file__))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    args = parser.parse_args()
    render(args.data_dir.resolve(), args.outdir.resolve())


if __name__ == "__main__":
    main()
