#!/usr/bin/env python3
"""Render Figure 2 from locked V11 sources or make a result-free layout preview."""

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

BENCHMARK_ID = "PANCAN_TP53_v1_HNSC_R1"
BLUE = "#0072B2"
ORANGE = "#D55E00"
GREEN = "#009E73"
GRAY = "#6B7280"
LIGHT_GRAY = "#E5E7EB"
METHOD_ORDER = ["raw_pool", "q_value_matched", "stability_matched", "full_audit"]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def panel_label(axis: plt.Axes, label: str) -> None:
    axis.text(
        -0.10,
        1.06,
        label,
        transform=axis.transAxes,
        fontsize=16,
        fontweight="bold",
        va="top",
    )


def draw_design(axis: plt.Axes) -> None:
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")
    boxes = [
        (0.12, 0.70, 0.76, 0.14, "Frozen HNSC 50-claim pool", GRAY),
        (
            0.12,
            0.47,
            0.76,
            0.14,
            "Same-pool rules\nFull audit | q-value | stability (K=25)",
            ORANGE,
        ),
        (
            0.12,
            0.24,
            0.76,
            0.14,
            "Blinded P3/P4 review\nLock, then unblind once",
            GREEN,
        ),
    ]
    for x, y, width, height, label, color in boxes:
        axis.add_patch(
            FancyBboxPatch(
                (x, y),
                width,
                height,
                boxstyle="round,pad=0.015,rounding_size=0.02",
                facecolor="white",
                edgecolor=color,
                linewidth=1.8,
            )
        )
        axis.text(x + width / 2, y + height / 2, label, ha="center", va="center", fontsize=9.3)
    for start, end in ((0.70, 0.61), (0.47, 0.38)):
        axis.annotate(
            "",
            xy=(0.50, end),
            xytext=(0.50, start),
            arrowprops={"arrowstyle": "->", "lw": 1.7, "color": GRAY},
        )
    axis.text(
        0.50,
        0.09,
        "Raw pool is descriptive only; primary comparisons use identical K",
        ha="center",
        fontsize=10,
        color=GRAY,
    )
    axis.set_title("Same-pool, blinded validation design", fontsize=13.5, loc="left", pad=10)


def pending_panel(axis: plt.Axes, title: str, ylabel: str) -> None:
    axis.set_title(title, fontsize=13.5, loc="left")
    axis.set_ylabel(ylabel)
    axis.set_xlim(-0.5, 3.5)
    axis.set_ylim(0, 100)
    axis.set_xticks(range(4), ["Raw pool", "q-value", "Stability", "Full audit"], fontsize=9)
    axis.grid(axis="y", color=LIGHT_GRAY, lw=0.8)
    axis.text(
        0.5,
        0.53,
        "Pending locked P3/P4 results",
        transform=axis.transAxes,
        ha="center",
        va="center",
        fontsize=12,
        color=GRAY,
        bbox={
            "boxstyle": "round,pad=0.5",
            "facecolor": "white",
            "edgecolor": LIGHT_GRAY,
        },
    )


def method_panel(
    axis: plt.Axes,
    table: pd.DataFrame,
    outcome: str,
    title: str,
    ylabel: str,
    exact: pd.DataFrame,
) -> None:
    data = (
        table.loc[table["outcome"].eq(outcome)]
        .copy()
        .set_index("method_id")
        .loc[METHOD_ORDER]
        .reset_index()
    )
    x = np.arange(len(data))
    values = data["fraction"].to_numpy() * 100
    errors = np.vstack(
        (
            (data["fraction"] - data["ci_low"]).to_numpy() * 100,
            (data["ci_high"] - data["fraction"]).to_numpy() * 100,
        )
    )
    bars = axis.bar(x, values, color=data["color"], width=0.68, alpha=0.88)
    axis.errorbar(x, values, yerr=errors, fmt="none", ecolor="#374151", capsize=4, lw=1.4)
    for bar, row in zip(bars, data.to_dict("records"), strict=True):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            min(98, bar.get_height() + 4),
            f"{int(row['n_outcome'])}/{int(row['n_selected'])}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    labels = [
        "Raw pool\n(descriptive)",
        "q-value\nmatched",
        "Stability\nmatched",
        "Full audit",
    ]
    axis.set_xticks(x, labels, fontsize=9)
    axis.set_ylim(0, 108)
    axis.set_ylabel(ylabel)
    axis.set_title(title, fontsize=13.5, loc="left")
    axis.grid(axis="y", color=LIGHT_GRAY, lw=0.8)
    comparison = exact.loc[(exact["outcome"].eq(outcome)) & exact["method_b"].eq("q_value_matched")]
    if len(comparison) == 1:
        row = comparison.iloc[0]
        axis.text(
            0.98,
            0.95,
            f"Full audit vs q-value\nPdesc={row['p_one_sided']:.3g}",
            transform=axis.transAxes,
            ha="right",
            va="top",
            fontsize=8.5,
            color=GRAY,
            bbox={
                "boxstyle": "round,pad=0.25",
                "facecolor": "white",
                "edgecolor": "none",
            },
        )


def agreement_panel(axis: plt.Axes, agreement: pd.DataFrame | None) -> None:
    axis.set_title("Independent-rater agreement", fontsize=13.5, loc="left")
    axis.axvline(0, color=GRAY, lw=1, ls="--")
    axis.set_xlim(-0.45, 1.0)
    axis.set_xlabel("Fleiss' kappa")
    axis.grid(axis="x", color=LIGHT_GRAY, lw=0.8)
    if agreement is None:
        axis.set_yticks(range(3), ["Statistical support", "External evidence", "Overstatement"])
        axis.text(
            0.52,
            0.53,
            "Pending 3 locked rating files",
            transform=axis.transAxes,
            ha="center",
            va="center",
            fontsize=12,
            color=GRAY,
            bbox={
                "boxstyle": "round,pad=0.5",
                "facecolor": "white",
                "edgecolor": LIGHT_GRAY,
            },
        )
        return
    data = (
        agreement.set_index("question")
        .loc[["q1_statistical_support", "q2_external_evidence", "q3_overstatement"]]
        .reset_index()
    )
    y = np.arange(len(data))
    axis.scatter(data["fleiss_kappa"], y, s=75, color=BLUE, zorder=3)
    axis.set_yticks(y, data["question_label"])
    axis.invert_yaxis()
    for index, row in data.iterrows():
        to_left = row["fleiss_kappa"] > 0.65
        axis.text(
            row["fleiss_kappa"] - 0.04 if to_left else row["fleiss_kappa"] + 0.04,
            index,
            f"unanimous {row['exact_unanimous_fraction'] * 100:.0f}%",
            va="center",
            ha="right" if to_left else "left",
            fontsize=9,
            color=GRAY,
        )


def read_locked_sources(
    data_root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    root = data_root / "output" / "priority2" / BENCHMARK_ID / "final_v11"
    manifest_path = root / "figure2_source_manifest.json"
    digest_path = root / "figure2_source_manifest.sha256"
    require(
        manifest_path.is_file() and digest_path.is_file(),
        "Missing locked Figure 2 sources",
    )
    require(
        sha256_file(manifest_path) == digest_path.read_text().split()[0],
        "Figure 2 manifest drift",
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    outputs = {}
    for label, record in manifest["outputs"].items():
        path = Path(record["path"])
        require(
            path.is_file() and sha256_file(path) == record["sha256"],
            f"Figure 2 source drift: {label}",
        )
        outputs[label] = path
    return (
        pd.read_csv(outputs["panel_BC"], sep="\t"),
        pd.read_csv(outputs["panel_D"], sep="\t"),
        pd.read_csv(outputs["exact"], sep="\t"),
        manifest,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--layout-preview", action="store_true")
    parser.add_argument("--preview-outdir", type=Path)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    require(
        args.layout_preview or args.data_root is not None,
        "Provide --data-root or --layout-preview",
    )
    require(
        not args.layout_preview or args.preview_outdir is not None,
        "Layout preview requires --preview-outdir",
    )

    metrics = agreement = exact = manifest = None
    if args.layout_preview:
        outdir = args.preview_outdir.expanduser().resolve()
        stem = "Fig2_layout_only_v11"
    else:
        data_root = args.data_root.expanduser().resolve()
        metrics, agreement, exact, manifest = read_locked_sources(data_root)
        outdir = data_root / "output" / "priority2" / BENCHMARK_ID / "fig"
        stem = "Fig2_blinded_same_pool_v11"
    outdir.mkdir(parents=True, exist_ok=True)
    pdf_path = outdir / f"{stem}.pdf"
    png_path = outdir / f"{stem}.png"
    meta_path = outdir / f"{stem}.run_meta.json"
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
    if args.layout_preview:
        pending_panel(axes[0, 1], "Independent literature support", "Supported claims (%)")
        pending_panel(
            axes[1, 0],
            "Major-overstatement risk",
            "Claims rated major overstatement (%)",
        )
        agreement_panel(axes[1, 1], None)
        figure.text(
            0.5,
            0.505,
            "LAYOUT ONLY - NO STUDY RESULTS",
            ha="center",
            va="center",
            fontsize=22,
            color="#B91C1C",
            alpha=0.20,
            rotation=18,
            fontweight="bold",
        )
    else:
        method_panel(
            axes[0, 1],
            metrics,
            "primary_independent_support",
            "Independent literature support",
            "Supported claims (%)",
            exact,
        )
        method_panel(
            axes[1, 0],
            metrics,
            "major_overstatement_majority",
            "Major-overstatement risk",
            "Claims rated major overstatement (%)",
            exact,
        )
        agreement_panel(axes[1, 1], agreement)
    for axis, label in zip(axes.flat, "ABCD", strict=True):
        panel_label(axis, label)
    figure.savefig(pdf_path, bbox_inches="tight")
    figure.savefig(png_path, dpi=600, bbox_inches="tight")
    plt.close(figure)
    meta = {
        "schema_version": "CRM_R1_FIGURE2_RENDER_v11",
        "created_utc": datetime.now(UTC).isoformat(),
        "layout_only": bool(args.layout_preview),
        "analytical_endpoint_recomputed": False,
        "source_manifest_sha256": None
        if manifest is None
        else sha256_file(
            Path(manifest["outputs"]["panel_BC"]["path"]).parent / "figure2_source_manifest.json"
        ),
        "pdf_sha256": sha256_file(pdf_path),
        "png_sha256": sha256_file(png_path),
        "base_font_points": 12,
        "png_dpi": 600,
    }
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        "[PASS] Figure 2 layout preview rendered"
        if args.layout_preview
        else "[PASS] Figure 2 rendered from locked P3/P4 source tables"
    )
    print(f"[INFO] PDF: {pdf_path}")
    print(f"[INFO] PNG: {png_path}")


if __name__ == "__main__":
    main()
