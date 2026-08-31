#!/usr/bin/env python3
"""Render Figure 3 only from frozen P5 figure-source tables."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch

BENCHMARK_ID = "PANCAN_TP53_v1_HNSC_R1_P5"
BLUE = "#0072B2"
ORANGE = "#D55E00"
GREEN = "#009E73"
PURPLE = "#CC79A7"
GRAY = "#6B7280"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_sources(final_dir: Path) -> dict[str, Path]:
    manifest_path = final_dir / "figure_manifest.tsv"
    require(manifest_path.is_file(), f"Missing Figure 3 manifest: {manifest_path}")
    manifest = pd.read_csv(manifest_path, sep="\t")
    ready = manifest.loc[manifest["status"].eq("READY")]
    sources: dict[str, Path] = {}
    for row in ready.to_dict("records"):
        path = final_dir / row["source_table"]
        require(path.is_file(), f"Missing Figure 3 source: {path}")
        require(sha256_file(path) == row["sha256"], f"Figure 3 source hash drift: {path}")
        sources[str(row["panel"])] = path
    require({"A", "B", "C", "D"}.issubset(sources), "Figure 3 panels A-D are incomplete")
    return sources


def panel_label(axis: plt.Axes, label: str) -> None:
    axis.text(
        -0.10,
        1.12,
        label,
        transform=axis.transAxes,
        fontsize=15,
        fontweight="bold",
        va="top",
    )


def draw_design(axis: plt.Axes) -> None:
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")
    boxes = [
        (0.02, 0.55, 0.25, 0.25, "Frozen audit\nGO BP + Reactome", BLUE),
        (0.375, 0.55, 0.25, 0.25, "Ontology\nedges + depth", GREEN),
        (0.73, 0.55, 0.25, 0.25, "External test\nno audit feedback", ORANGE),
    ]
    for x, y, width, height, text, color in boxes:
        patch = FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle="round,pad=0.02,rounding_size=0.025",
            linewidth=1.8,
            edgecolor=color,
            facecolor="white",
        )
        axis.add_patch(patch)
        axis.text(
            x + width / 2,
            y + height / 2,
            text,
            ha="center",
            va="center",
            fontsize=8.5,
        )
    for start, end in ((0.27, 0.375), (0.625, 0.73)):
        axis.annotate(
            "",
            xy=(end, 0.675),
            xytext=(start, 0.675),
            arrowprops={"arrowstyle": "->", "lw": 1.8, "color": GRAY},
        )
    axis.text(
        0.50,
        0.28,
        "Primary: direct parent-child\nSensitivity: all safe ancestor-descendant",
        ha="center",
        fontsize=11,
    )
    axis.text(
        0.50,
        0.12,
        "Matched nonedge reference\n(depth + evidence size matched)",
        ha="center",
        fontsize=10.5,
        color=GRAY,
    )
    axis.set_title("Ontology hierarchy validation design", fontsize=13, pad=10, loc="left")


def plot_contradiction(axis: plt.Axes, table: pd.DataFrame) -> None:
    labels = []
    observed = []
    low = []
    high = []
    null = []
    colors = []
    p_values = []
    estimate_statuses = []
    for row in table.to_dict("records"):
        collection = "GO BP" if row["collection"] == "C5_GO_BP" else "Reactome"
        scope = "Direct" if row["relation_scope"] == "direct_parent_child" else "All ancestors"
        labels.append(f"{collection}\n{scope}\nn={int(row['n_pairs'])}")
        value = row["directional_contradiction_fraction"]
        observed.append(value)
        low.append(value - row["directional_contradiction_ci_low"])
        high.append(row["directional_contradiction_ci_high"] - value)
        null.append(row["contradiction_null_mean"])
        colors.append(BLUE if collection == "GO BP" else ORANGE)
        p_values.append(row["contradiction_p_one_sided_lower"])
        estimate_statuses.append(row["estimate_status"])
    x = np.arange(len(labels))
    axis.errorbar(
        x,
        np.asarray(observed) * 100,
        yerr=np.asarray([low, high]) * 100,
        fmt="none",
        ecolor=GRAY,
        capsize=4,
        lw=1.5,
    )
    axis.scatter(
        x,
        np.asarray(observed) * 100,
        s=65,
        c=colors,
        label="Observed hierarchy edges",
        zorder=3,
    )
    axis.scatter(
        x,
        np.asarray(null) * 100,
        s=55,
        facecolors="white",
        edgecolors=GRAY,
        marker="D",
        label="Matched nonedge mean",
        zorder=3,
    )
    annotation_heights = []
    for index, (value, upper_error, null_value, p_value, status) in enumerate(
        zip(observed, high, null, p_values, estimate_statuses, strict=True)
    ):
        height = np.nanmax([value + upper_error, null_value]) * 100 + 2.5
        annotation_heights.append(height)
        annotation = (
            f"Pdesc={p_value:.3g}"
            if status == "ESTIMABLE" and np.isfinite(p_value)
            else "Primary NE"
        )
        axis.text(index, height, annotation, ha="center", va="bottom", fontsize=8.5)
    axis.set_xticks(x, labels, fontsize=8.5)
    axis.set_ylabel("Directional contradiction (%)")
    axis.set_ylim(0, max(10, max(annotation_heights, default=0) + 8))
    axis.set_title("Directional contradiction vs matched nonedges", fontsize=13, loc="left")
    axis.legend(frameon=False, fontsize=9, loc="lower right")
    axis.grid(axis="y", color="#E5E7EB", lw=0.8)
    axis.text(
        0.01,
        0.02,
        "Pdesc is a descriptive matched-reference P value",
        transform=axis.transAxes,
        fontsize=7.8,
        color=GRAY,
    )


def plot_gene_support(axis: plt.Axes, table: pd.DataFrame) -> None:
    data = []
    labels = []
    colors = []
    for collection in ("C5_GO_BP", "C2_CP_REACTOME"):
        for scope in ("direct_parent_child", "safe_ancestor_descendant"):
            values = table.loc[
                table["collection"].eq(collection) & table["relation_scope"].eq(scope),
                "child_covered_by_parent",
            ].dropna()
            if len(values):
                data.append(values.to_numpy() * 100)
                name = "GO BP" if collection == "C5_GO_BP" else "Reactome"
                scope_name = "Direct" if scope == "direct_parent_child" else "All ancestors"
                labels.append(f"{name}\n{scope_name}\n(n={len(values)})")
                colors.append(BLUE if collection == "C5_GO_BP" else ORANGE)
    if data:
        artists = axis.boxplot(data, patch_artist=True, showfliers=True, widths=0.62)
        for box, color in zip(artists["boxes"], colors, strict=True):
            box.set(facecolor=color, alpha=0.70, edgecolor=color)
        for median in artists["medians"]:
            median.set(color="black", linewidth=1.5)
        axis.set_xticks(np.arange(1, len(labels) + 1), labels, fontsize=9.5)
    else:
        axis.text(0.5, 0.5, "No mapped hierarchy pairs", ha="center", va="center")
    axis.set_ylabel("Child leading edge covered by parent (%)")
    axis.set_ylim(0, 105)
    axis.set_title("Leading-edge support across hierarchy pairs", fontsize=13, loc="left")
    axis.grid(axis="y", color="#E5E7EB", lw=0.8)


def plot_depth(axis: plt.Axes, table: pd.DataFrame) -> None:
    colors = {"PASS": GREEN, "ABSTAIN": ORANGE, "FAIL": PURPLE}
    data = []
    labels = []
    statuses = []
    for collection, collection_label in (
        ("C5_GO_BP", "GO BP"),
        ("C2_CP_REACTOME", "Reactome"),
    ):
        for status in ("PASS", "ABSTAIN", "FAIL"):
            values = table.loc[
                table["collection"].eq(collection) & table["status"].eq(status),
                "depth",
            ].dropna()
            if len(values):
                data.append(values.to_numpy())
                labels.append(f"{collection_label}\n{status}\n(n={len(values)})")
                statuses.append(status)
    if data:
        artists = axis.boxplot(data, patch_artist=True, widths=0.58)
        for box, status in zip(artists["boxes"], statuses, strict=True):
            box.set(facecolor=colors[status], alpha=0.70, edgecolor=colors[status])
        for median in artists["medians"]:
            median.set(color="black", linewidth=1.5)
        rng = np.random.default_rng(20260812)
        for index, values in enumerate(data, start=1):
            axis.scatter(
                rng.normal(index, 0.045, size=len(values)),
                values,
                s=18,
                color=colors[statuses[index - 1]],
                alpha=0.45,
                edgecolors="none",
            )
        axis.set_xticks(
            np.arange(1, len(labels) + 1),
            labels,
            fontsize=7.6,
            rotation=0,
            ha="center",
        )
    axis.set_ylabel("Minimum ontology depth")
    axis.set_title("Ontology depth by audit disposition", fontsize=13, loc="left")
    axis.grid(axis="y", color="#E5E7EB", lw=0.8)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    data_root = args.data_root.expanduser().resolve()
    p5_root = data_root / "output/priority5" / BENCHMARK_ID
    final_dir = p5_root / "final"
    sources = verify_sources(final_dir)
    panel_b = pd.read_csv(sources["B"], sep="\t")
    panel_c = pd.read_csv(sources["C"], sep="\t")
    panel_d = pd.read_csv(sources["D"], sep="\t")

    figure_dir = p5_root / "fig"
    figure_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = figure_dir / "Fig3_priority5_ontology_v2.pdf"
    png_path = figure_dir / "Fig3_priority5_ontology_v2.png"
    meta_path = figure_dir / "Fig3_priority5_ontology_v2.run_meta.json"
    for path in (pdf_path, png_path, meta_path):
        require(
            args.force or not path.exists(),
            f"Render output exists: {path}; use --force",
        )

    plt.rcParams.update(
        {
            "font.size": 12,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    figure, axes = plt.subplots(2, 2, figsize=(10.2, 8.6), constrained_layout=True)
    draw_design(axes[0, 0])
    plot_contradiction(axes[0, 1], panel_b)
    plot_gene_support(axes[1, 0], panel_c)
    plot_depth(axes[1, 1], panel_d)
    for axis, label in zip(axes.flat, "ABCD", strict=True):
        panel_label(axis, label)
    figure.savefig(pdf_path, bbox_inches="tight")
    figure.savefig(png_path, dpi=600, bbox_inches="tight")
    plt.close(figure)
    meta = {
        "schema_version": "CRM_R1_PRIORITY5_FIGURE3_RENDER_v2",
        "created_utc": datetime.now(UTC).isoformat(),
        "source_sha256": {panel: sha256_file(path) for panel, path in sources.items()},
        "pdf_sha256": sha256_file(pdf_path),
        "png_sha256": sha256_file(png_path),
        "analytical_endpoint_recomputed": False,
        "render_only_revision": True,
        "base_font_points": 12,
        "png_dpi": 600,
    }
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("[PASS] Figure 3 rendered from frozen source tables")
    print(f"[INFO] PDF: {pdf_path}")
    print(f"[INFO] PNG: {png_path}")


if __name__ == "__main__":
    main()
