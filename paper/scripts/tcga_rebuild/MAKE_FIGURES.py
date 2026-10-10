#!/usr/bin/env python3
"""Plot the completed diagnostic run. Does not change analytical results."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args()
    root = args.run_dir
    summary = pd.read_csv(root / "cohort_summary.tsv", sep="\t")
    results = pd.read_csv(root / "hallmark_results.tsv", sep="\t")
    dest = root / "figures"
    dest.mkdir(exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 5.2), gridspec_kw={"width_ratios": [1.45, 1]})
    y = np.arange(len(summary))
    a, b = axes
    a.barh(y, summary.n_mut, color="#245a81", label="TP53 MUT")
    a.barh(
        y, summary.n_wt, left=summary.n_mut, color="#85b6ca", label="MC3 call-negative comparator"
    )
    a.barh(
        y,
        summary.n_unknown,
        left=summary.n_mut + summary.n_wt,
        color="#d8dce0",
        label="Excluded / unknown",
    )
    a.set_yticks(y, summary.cancer)
    a.invert_yaxis()
    a.set_xlabel("Expression-matched primary tumor samples")
    a.set_title("A  Cohort composition", loc="left", fontweight="bold", pad=12)
    a.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=1, fontsize=8)
    a.set_xlim(0, 1200)
    for i, row in enumerate(summary.itertuples()):
        a.text(row.n_expression + 12, i, str(row.n_expression), va="center", fontsize=8)
        value = row.n_q_le_0_05
        if pd.notna(value):
            b.barh(i, value, color="#245a81")
            b.text(value + 0.8, i, str(int(value)), va="center", fontsize=9)
        else:
            b.text(1, i, "Not estimable", color="#656b72", va="center", fontsize=9)
    b.set_yticks(y, summary.cancer)
    b.set_ylim(a.get_ylim())
    b.set_xlim(0, 50)
    b.set_xlabel("Hallmark pathways with q ≤ 0.05 (of 50)")
    b.set_title("B  Enrichment results", loc="left", fontweight="bold", pad=12)
    for ax in axes:
        ax.grid(axis="x", color="#eceef0", linewidth=0.6)
        ax.set_axisbelow(True)
        ax.tick_params(axis="y", length=0)
    fig.suptitle(
        "TCGA reanalysis after input correction", x=0.075, ha="left", fontsize=14, fontweight="bold"
    )
    fig.text(
        0.075,
        0.91,
        "MC3 PASS; OV also accepts wga-only; RNA QC; GDC-positive/MC3-negative OV cases excluded",
        fontsize=9,
        color="#555b63",
    )
    fig.text(
        0.075,
        0.02,
        (
            "Comparator: no qualifying MC3 TP53 call; not confirmed "
            "biological WT. OV comparison group and SKCM MUT group are small; "
            "see sample ledger."
        ),
        fontsize=7.2,
        color="#555b63",
    )
    fig.subplots_adjust(left=0.075, right=0.975, top=0.80, bottom=0.27, wspace=0.36)
    for ext in ["png", "pdf"]:
        fig.savefig(dest / f"cohorts_and_enrichment.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)

    cohorts = summary.loc[summary.fit_eligible, "cancer"].tolist()
    nes = results.pivot(index="term_id", columns="cancer", values="stat").sort_index()[cohorts]
    q = results.pivot(index="term_id", columns="cancer", values="qval").reindex(
        index=nes.index, columns=cohorts
    )
    if nes.shape != (50, len(cohorts)) or not np.isfinite(nes.to_numpy()).all():
        raise RuntimeError("Expected a complete 50-pathway by eligible-cohort result matrix")
    fig, ax = plt.subplots(figsize=(9.4, 12.4))
    bound = float(np.ceil(np.max(np.abs(nes.to_numpy())) * 2) / 2)
    image = ax.imshow(
        nes, cmap="RdBu_r", norm=TwoSlopeNorm(vmin=-bound, vcenter=0, vmax=bound), aspect="auto"
    )
    ax.set_xticks(range(len(cohorts)), cohorts)
    ax.xaxis.tick_top()
    ax.tick_params(axis="both", length=0)
    ax.set_yticks(
        range(50), [x.replace("HALLMARK_", "").replace("_", " ") for x in nes.index], fontsize=8
    )
    for i in range(50):
        for j in range(len(cohorts)):
            if q.iloc[i, j] <= 0.05:
                ax.text(
                    j,
                    i,
                    "•",
                    ha="center",
                    va="center",
                    fontsize=12,
                    color="white" if abs(nes.iloc[i, j]) > bound * 0.55 else "#20252a",
                )
    colorbar = fig.colorbar(image, ax=ax, fraction=0.055, pad=0.045)
    colorbar.set_label("Normalized enrichment score (MUT − comparator)")
    fig.suptitle(
        "Hallmark enrichment across eligible TCGA cohorts",
        x=0.035,
        ha="left",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )
    fig.text(
        0.035,
        0.947,
        (
            "Dots: within-cohort adjusted q ≤ 0.05. All 50 Hallmark sets are "
            "shown. OV and SKCM comparisons include small groups."
        ),
        fontsize=8.5,
    )
    provenance = pd.read_csv(
        root / "evidence_tables" / f"{cohorts[0]}.evidence_table.tsv.provenance.tsv",
        sep="\t",
        dtype=str,
    ).set_index("key")["value"]
    caption = (
        f"MSigDB {provenance['msigdb']}; fgsea {provenance['fgsea']}; "
        f"seed {provenance['seed']}. Associations are unadjusted "
        "and do not establish a causal TP53 effect."
    )
    fig.text(0.035, 0.02, caption, fontsize=8, color="#555b63")
    fig.subplots_adjust(left=0.43, right=0.90, top=0.90, bottom=0.06)
    for ext in ["png", "pdf"]:
        fig.savefig(dest / f"hallmark_NES.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(dest.resolve())


if __name__ == "__main__":
    main()
