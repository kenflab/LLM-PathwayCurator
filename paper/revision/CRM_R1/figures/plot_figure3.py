import csv
import json
from pathlib import Path

from _figure_io import finish, prepare


def figures(out, s, data):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.0), gridspec_kw={"width_ratios": [1.12, 1]})
    colors = ["#ae453e", "#738aa0", "#dce1e5"]
    statuses = ["VIOLATION_DETECTED", "REVIEW_FLAGGED", "NO_LIMITED_FLAG"]
    labels = ["Violation detected", "Review flagged only", "No limited flag"]
    for scope, denom in [("counts_50", 50), ("counts_24", 24)]:
        counts = [s[scope][key] for key in statuses]
        if any(type(n) is not int or n < 0 for n in counts) or sum(counts) != denom:
            raise ValueError(f"Invalid fixed-text denominator in {scope}")
    for y, (scope, denom) in enumerate([("counts_50", 50), ("counts_24", 24)]):
        left = 0
        for st, col in zip(statuses, colors, strict=True):
            n = s[scope][st]
            axes[0].barh(y, n / denom * 100, left=left, color=col, height=0.48)
            axes[0].text(
                left + n / denom * 50,
                y,
                str(n),
                ha="center",
                va="center",
                color="white" if st != statuses[2] else "#222",
                fontsize=11,
                fontweight="bold",
            )
            left += n / denom * 100
    axes[0].set_yticks([0, 1], ["All texts\n(n=50)", "Rated subset\n(n=24)"])
    axes[0].invert_yaxis()
    axes[0].set_xlim(0, 100)
    axes[0].set_xlabel("Share of texts (%)")
    axes[0].set_title("A  Source checks", loc="left", fontweight="bold")
    from matplotlib.patches import Patch

    axes[0].legend(
        handles=[Patch(color=c, label=label) for c, label in zip(colors, labels, strict=True)],
        loc="upper left",
        bbox_to_anchor=(0, -0.24),
        frameon=False,
        fontsize=9,
    )
    rows = list(csv.DictReader((data / "DEX_endpoint_summary.tsv").open(), delimiter="\t"))
    endpoints = ["material source contradiction", "qualitative overstatement"]
    if len(rows) != 2 or {r["endpoint"] for r in rows} != set(endpoints):
        raise ValueError("Expected two distinct DEX endpoints")
    rows = [{r["endpoint"]: r for r in rows}[endpoint] for endpoint in endpoints]
    for row in rows:
        for judge in ("J1", "J2"):
            if int(row[judge + "_n"]) != 24 or not 0 <= int(row[judge + "_positive"]) <= 24:
                raise ValueError("Expected 24 ratings per endpoint and evaluator")
    for offset, j, col, lab in [
        (-0.18, "J1", "#294e71", "J1 author"),
        (0.18, "J2", "#bd7936", "J2 external reviewer"),
    ]:
        ns = [int(r[j + "_positive"]) for r in rows]
        ys = [x + offset for x in (0, 1)]
        axes[1].barh(ys, ns, color=col, height=0.32, label=lab)
        for y, n in zip(ys, ns, strict=True):
            axes[1].text(n + 0.35, y, str(n), va="center", fontsize=10)
    axes[1].set_yticks([0, 1], ["Material source\ncontradiction", "Qualitative\noverstatement"])
    axes[1].invert_yaxis()
    axes[1].set_xlim(0, 24)
    axes[1].set_xticks([0, 6, 12, 18, 24])
    axes[1].set_xlabel("Texts assigned the concern (of 24)")
    axes[1].set_title("B  Human judgments", loc="left", fontweight="bold")
    axes[1].legend(loc="upper left", bbox_to_anchor=(0, -0.24), frameon=False, fontsize=9)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(axis="y", length=0)
    fig.subplots_adjust(left=0.12, right=0.97, wspace=0.55, bottom=0.34, top=0.85)
    for ext in ["png", "svg", "pdf"]:
        fig.savefig(out / f"Figure3_DEX_review.{ext}", dpi=220, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    data, out = prepare(
        "Render Figure 3 from Supplementary Data 2",
        ["DEX_SUMMARY.json", "DEX_endpoint_summary.tsv"],
    )
    figures(out, json.loads((data / "DEX_SUMMARY.json").read_text()), data)
    finish(data, out, ["DEX_SUMMARY.json", "DEX_endpoint_summary.tsv"], Path(__file__))
