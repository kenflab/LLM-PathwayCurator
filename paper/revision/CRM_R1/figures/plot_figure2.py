import csv
import math
import textwrap
from pathlib import Path

import matplotlib
import numpy as np
from _figure_io import finish, prepare

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors

DATA, OUT = prepare(
    "Render Figure 2 from Supplementary Data 1",
    ["TCGA_cohorts.tsv", "enrichment/HNSC.evidence_table.tsv"],
)


def read(p):
    return list(csv.DictReader(p.open(), delimiter="\t"))


def table(name, rows):
    with (OUT / name).open("w") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), delimiter="\t")
        w.writeheader()
        w.writerows(rows)


base = read(DATA / "TCGA_cohorts.tsv")
expected_cohorts = {"BRCA", "HNSC", "LUAD", "LUSC", "OV", "SKCM", "UCEC"}
if len(base) != 7 or {r["cohort"] for r in base} != expected_cohorts:
    raise ValueError("Expected one row for each of the seven TCGA cohorts")
for row in base:
    counts = [int(row[k]) for k in ("MUT", "call_negative", "excluded")]
    if min(counts) < 0 or sum(counts) != int(row["expression_samples"]):
        raise ValueError("Cohort accounting does not match the expression census")
    if int(row["Hallmark_terms"]) != 50 or not 0 <= int(row["q_le_0_05"]) <= 50:
        raise ValueError("Expected 50 Hallmark candidates per cohort")
cohorts = [
    {
        "cancer": r["cohort"],
        "n_mut": r["MUT"],
        "n_wt": r["call_negative"],
        "n_unknown": r["excluded"],
        "n_q_le_0_05": r["q_le_0_05"],
    }
    for r in base
]
records = read(DATA / "enrichment/HNSC.evidence_table.tsv")
current = {r["term_id"]: r for r in records}
if len(records) != 50 or len(current) != 50:
    raise ValueError("Expected 50 unique HNSC source terms")
for row in records:
    if not math.isfinite(float(row["stat"])) or row["stat_kind"] != "NES":
        raise ValueError("Expected finite NES values")
    if not 0 < float(row["qval"]) <= 1:
        raise ValueError("This fixed panel requires finite positive adjusted values")
selected = sorted(
    [x for x in records if float(x["qval"]) <= 0.05],
    key=lambda x: (float(x["qval"]), x["term_id"]),
)
expected_selected = int(next(r for r in base if r["cohort"] == "HNSC")["q_le_0_05"])
if not selected or len(selected) != expected_selected:
    raise ValueError("HNSC shortlist disagrees with cohort accounting")
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 9.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
    }
)
fig = plt.figure(figsize=(8.4, 8.5), layout="constrained")
gs = fig.add_gridspec(2, 2, height_ratios=[0.8, 1.8], width_ratios=[1, 1.05])
a = fig.add_subplot(gs[0, 0])
b = fig.add_subplot(gs[0, 1])
c = fig.add_subplot(gs[1, 0])
d = fig.add_subplot(gs[1, 1])
y = np.arange(7)
mut = np.array([int(r["n_mut"]) for r in cohorts])
neg = np.array([int(r["n_wt"]) for r in cohorts])
ex = np.array([int(r["n_unknown"]) for r in cohorts])
a.barh(y, mut, color="#27677c", label="TP53 MUT")
a.barh(y, neg, left=mut, color="#91b6a5", label="Call-negative")
a.barh(y, ex, left=mut + neg, color="#d4d6d8", label="Excluded")
a.set_yticks(y, [r["cancer"] for r in cohorts])
a.invert_yaxis()
a.set_xlabel("Primary tumor expression samples")
a.legend(ncol=3, fontsize=7.5, loc="lower left", bbox_to_anchor=(0, 1.02))
a.set_title("A   Cohort accounting", loc="left", fontweight="bold", pad=30)
qcounts = [int(float(r["n_q_le_0_05"])) for r in cohorts]
b.barh(y, qcounts, color="#27677c")
b.set_yticks(y, [r["cancer"] for r in cohorts])
b.invert_yaxis()
b.set_xlim(0, 50)
for i, v in enumerate(qcounts):
    b.text(v + 0.5, i, str(v), va="center")
b.set_xlabel("Terms with BH q ≤ 0.05 out of 50", fontsize=8.5)
b.set_title("B   Eligible Hallmark terms", loc="left", fontweight="bold")
labs = [
    textwrap.fill(x["term_id"].replace("HALLMARK_", "").replace("_", " "), width=23)
    for x in selected
]
nes = np.array([float(x["stat"]) for x in selected])
q = np.array([float(x["qval"]) for x in selected])
nn = len(selected)
cm = plt.get_cmap("RdBu_r")
norm = colors.TwoSlopeNorm(vmin=-3.2, vcenter=0, vmax=3.2)
c.barh(range(nn), -np.log10(q), color=cm(norm(nes)))
c.set_yticks(range(nn), labs, fontsize=8.5)
c.invert_yaxis()
c.set_xlabel("−log10(BH q)")
c.set_title("C   HNSC source shortlist", loc="left", fontweight="bold")
sets = [
    {g.strip() for g in x["evidence_genes"].replace(";", ",").split(",") if g.strip()}
    for x in selected
]
jac = np.array([[(len(x & z) / len(x | z) if x | z else 0.0) for z in sets] for x in sets])
edge = np.array(
    [
        [len(x & z) >= 3 and (len(x & z) / len(x | z) if x | z else 0.0) >= 0.1 for z in sets]
        for x in sets
    ]
)
np.fill_diagonal(jac, np.nan)
im = d.imshow(jac, cmap="Blues", vmin=0, vmax=0.5, aspect="auto")
d.set_xticks(range(0, nn, 2), range(1, nn + 1, 2), fontsize=8)
d.set_yticks(range(0, nn, 2), range(1, nn + 1, 2), fontsize=8)
d.set_xlabel("Term rank in panel C")
d.set_ylabel("")
d.set_title("D   Gene overlap", loc="left", fontweight="bold")
for i in range(nn):
    for j in range(nn):
        if i != j and edge[i, j]:
            d.add_patch(
                plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, lw=0.6, color="#183747")
            )
fig.colorbar(im, ax=d, fraction=0.045, pad=0.04, label="Jaccard overlap")
fig.colorbar(
    plt.cm.ScalarMappable(norm=norm, cmap=cm),
    ax=c,
    fraction=0.045,
    pad=0.04,
    label="NES  MUT − call-negative",
)
for ext in ["png", "pdf", "svg"]:
    fig.savefig(OUT / f"Figure2_current_TCGA.{ext}", dpi=220, bbox_inches="tight")
plt.close(fig)
table("Figure2C_HNSC_source.tsv", [{"rank": i + 1, **x} for i, x in enumerate(selected)])
table(
    "Figure2D_HNSC_overlap.tsv",
    [
        {
            "term_i": selected[i]["term_id"],
            "term_j": selected[j]["term_id"],
            "shared_genes": len(sets[i] & sets[j]),
            "jaccard": (
                len(sets[i] & sets[j]) / len(sets[i] | sets[j]) if sets[i] | sets[j] else 0.0
            ),
            "edge": bool(edge[i, j]),
        }
        for i in range(nn)
        for j in range(i + 1, nn)
    ],
)

finish(DATA, OUT, ["TCGA_cohorts.tsv", "enrichment/HNSC.evidence_table.tsv"], Path(__file__))
