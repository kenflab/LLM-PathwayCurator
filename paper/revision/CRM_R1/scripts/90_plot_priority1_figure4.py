#!/usr/bin/env python3
"""Render frozen Priority 1 Figure 4 without recomputing an analytical endpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

CRM_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = CRM_DIR.parents[2]
SCRIPT_BUILD = "CRM_R1_PRIORITY1_FIG4_V7_20260809"
FIGURE_MESSAGE = "Empirical discovery stability stratifies held-out temporal replication."

COLORS = {
    "empirical": "#0072B2",
    "q_value": "#D55E00",
    "size_matched": "#CC79A7",
    "replicated": "#009E73",
    "not_replicated": "#9A9A9A",
    "raw_reference": "#666666",
    "random_reference": "#B3B3B3",
    "ink": "#222222",
    "grid": "#D9D9D9",
    "workflow_discovery": "#DCEAF7",
    "workflow_resampling": "#DDF2EA",
    "workflow_freeze": "#F8E8B6",
    "workflow_validation": "#E9DDF2",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    require(isinstance(value, dict), f"JSON root must be an object: {path}")
    return value


def git_value(*arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def require_clean_tracked_worktree() -> None:
    status = git_value("status", "--porcelain", "--untracked-files=no")
    require(not status, f"Commit V7 and leave tracked files clean before rendering: {status}")


def boolean_series(values: pd.Series, *, column: str) -> pd.Series:
    if pd.api.types.is_bool_dtype(values.dtype):
        return values.fillna(False).astype(bool)
    normalized = values.astype("string").fillna("").str.strip().str.lower()
    require(set(normalized) <= {"true", "false"}, f"Invalid Boolean values in {column}")
    return normalized.eq("true")


def numeric_series(values: pd.Series, *, column: str) -> pd.Series:
    numeric = pd.to_numeric(values, errors="raise")
    require(np.isfinite(numeric).all(), f"Non-finite values in {column}")
    return numeric.astype(float)


def apply_publication_style(fontsize: float = 12.0) -> None:
    """Apply a clean, colorblind-accessible Cell Reports Methods-oriented style."""
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": fontsize,
            "axes.titlesize": fontsize + 0.5,
            "axes.titleweight": "bold",
            "axes.labelsize": fontsize,
            "xtick.labelsize": fontsize - 1.0,
            "ytick.labelsize": fontsize - 1.0,
            "legend.fontsize": fontsize - 1.5,
            "axes.linewidth": 1.0,
            "lines.linewidth": 1.8,
            "lines.markersize": 7.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
        }
    )


def verify_provenance(*, source_path: Path, summary_path: Path, run_meta: dict[str, Any]) -> None:
    outputs = run_meta.get("outputs")
    require(isinstance(outputs, dict), "Evaluation run metadata lacks outputs")
    require("figure4" in outputs and "summary" in outputs, "Evaluation output hashes are missing")
    require(
        outputs["figure4"]["sha256"] == sha256_file(source_path),
        "Figure 4 source hash differs from the frozen evaluation metadata",
    )
    require(
        outputs["summary"]["sha256"] == sha256_file(summary_path),
        "Priority 1 summary hash differs from the frozen evaluation metadata",
    )
    require(
        run_meta.get("stop_gate_p1") == "PASS_EMPIRICAL_POINT_ESTIMATE_IMPROVED",
        "Priority 1 Stop-gate status drift",
    )


def validate_source(source: pd.DataFrame, summary: dict[str, Any]) -> dict[str, pd.DataFrame]:
    require({"panel", "record_type"} <= set(source.columns), "Figure source contract drift")
    panels = {
        panel: source.loc[source["panel"].astype(str).eq(panel)].copy()
        for panel in ("A", "B", "C", "D")
    }
    require(all(not table.empty for table in panels.values()), "Figure source is missing a panel")

    workflow = panels["A"]
    require(set(workflow["record_type"].astype(str)) == {"workflow"}, "Panel A type drift")
    workflow_values = {
        str(row.metric): float(row.value)
        for row in workflow[["metric", "value"]].itertuples(index=False)
    }
    expected_workflow = {
        "discovery_samples": 12.0,
        "balanced_resamples": 81.0,
        "primary_tau": 0.8,
        "frozen_k": 23.0,
        "validation_samples": 12.0,
    }
    require(workflow_values == expected_workflow, "Panel A frozen workflow values drift")

    pathway = panels["B"]
    required_pathway = {
        "claim_id",
        "pathway",
        "empirical_survival_48h",
        "replicated_primary",
    }
    require(required_pathway <= set(pathway.columns), "Panel B pathway columns drift")
    require(len(pathway) == 50 and pathway["claim_id"].nunique() == 50, "Panel B pool drift")
    pathway["empirical_survival_48h"] = numeric_series(
        pathway["empirical_survival_48h"], column="empirical_survival_48h"
    )
    require(
        pathway["empirical_survival_48h"].between(0.0, 1.0).all(),
        "Empirical survival outside [0,1]",
    )
    pathway["replicated_primary"] = boolean_series(
        pathway["replicated_primary"], column="replicated_primary"
    )
    require(int(pathway["replicated_primary"].sum()) == 31, "Panel B replication count drift")
    panels["B"] = pathway.sort_values("claim_id").reset_index(drop=True)

    methods = panels["C"]
    required_methods = {
        "method",
        "n_selected",
        "n_replicated",
        "replication_fraction",
        "replication_ci_low",
        "replication_ci_high",
    }
    require(required_methods <= set(methods.columns), "Panel C method columns drift")
    focus_methods = {
        "empirical_stability_audit": (23, 18),
        "q_value_matched": (23, 17),
        "q_value_and_leading_edge_size_matched": (23, 16),
    }
    for method, (expected_n, expected_replicated) in focus_methods.items():
        row = methods.loc[methods["method"].astype(str).eq(method)]
        require(len(row) == 1, f"Panel C method missing or duplicated: {method}")
        require(int(float(row.iloc[0]["n_selected"])) == expected_n, f"K drift: {method}")
        require(
            int(float(row.iloc[0]["n_replicated"])) == expected_replicated,
            f"Replication-count drift: {method}",
        )
    panels["C"] = methods

    tau_grid = panels["D"]
    required_tau = {
        "tau",
        "coverage",
        "replication_fraction",
        "replication_ci_low",
        "replication_ci_high",
        "nonreplication_risk",
    }
    require(required_tau <= set(tau_grid.columns), "Panel D tau-grid columns drift")
    for column in required_tau:
        tau_grid[column] = numeric_series(tau_grid[column], column=column)
    require(
        set(np.round(tau_grid["tau"], 12)) == {0.8, 0.9, 0.95, 0.98},
        "Frozen tau grid drift",
    )
    panels["D"] = tau_grid.sort_values("coverage").reset_index(drop=True)

    exact = summary.get("overlap_exact_reference", {})
    continuous = summary.get("continuous_secondary", {})
    require(summary.get("stop_gate_p1") == "PASS_EMPIRICAL_POINT_ESTIMATE_IMPROVED", "P1 drift")
    require(
        math.isclose(
            float(summary["primary_replication_fraction_difference_empirical_minus_q_value"]),
            1.0 / 23.0,
            abs_tol=1e-12,
        ),
        "Frozen primary difference drift",
    )
    require(
        math.isclose(float(exact["p_one_sided_empirical_greater"]), 0.5, abs_tol=1e-12),
        "Frozen exact reference drift",
    )
    require(math.isclose(float(continuous["auroc"]), 0.713073, abs_tol=1e-6), "AUROC drift")
    return panels


def panel_label(axis: plt.Axes, label: str) -> None:
    axis.text(
        -0.12,
        1.08,
        label,
        transform=axis.transAxes,
        fontsize=16,
        fontweight="bold",
        va="top",
        ha="left",
        color=COLORS["ink"],
    )


def clean_axis(axis: plt.Axes) -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.tick_params(width=1.0, length=4.0, color=COLORS["ink"])
    axis.set_axisbelow(True)


def plot_workflow(axis: plt.Axes, workflow: pd.DataFrame) -> None:
    values = {
        str(row.metric): float(row.value)
        for row in workflow[["metric", "value"]].itertuples(index=False)
    }
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.0)
    axis.axis("off")
    axis.set_title("Frozen discovery-to-validation\ndesign", loc="left", pad=10)

    boxes = [
        (
            0.04,
            0.65,
            COLORS["workflow_discovery"],
            f"48 h discovery\nENDO, n={int(values['discovery_samples'])}\nfull model",
        ),
        (
            0.60,
            0.65,
            COLORS["workflow_resampling"],
            f"Balanced deletion\n{int(values['balanced_resamples'])} resamples\n48 h only",
        ),
        (
            0.60,
            0.22,
            COLORS["workflow_freeze"],
            f"Frozen audit\nτ={values['primary_tau']:.2f}\nPASS {int(values['frozen_k'])}/50",
        ),
        (
            0.04,
            0.22,
            COLORS["workflow_validation"],
            f"72 h held-out\nENDO, n={int(values['validation_samples'])}\none-time test",
        ),
    ]
    width = 0.36
    height = 0.23
    for x, y, color, text in boxes:
        patch = FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle="round,pad=0.015,rounding_size=0.025",
            linewidth=1.1,
            edgecolor=COLORS["ink"],
            facecolor=color,
        )
        axis.add_patch(patch)
        axis.text(
            x + width / 2,
            y + height / 2,
            text,
            ha="center",
            va="center",
            fontsize=9.2,
            linespacing=1.25,
            color=COLORS["ink"],
        )
    arrows = [
        ((0.41, 0.765), (0.59, 0.765)),
        ((0.78, 0.64), (0.78, 0.47)),
        ((0.59, 0.335), (0.41, 0.335)),
    ]
    for start, end in arrows:
        arrow = FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=12,
            linewidth=1.3,
            color=COLORS["ink"],
        )
        axis.add_patch(arrow)
    axis.text(
        0.50,
        0.53,
        "Membership fixed before 72 h",
        ha="center",
        va="center",
        fontsize=9.3,
        fontweight="bold",
        color=COLORS["ink"],
    )
    axis.text(
        0.50,
        0.08,
        "Replication: same direction + 72 h FDR < 0.05",
        ha="center",
        va="center",
        fontsize=8.8,
        color=COLORS["ink"],
    )


def deterministic_jitter(count: int, width: float = 0.075) -> np.ndarray:
    if count <= 1:
        return np.zeros(count)
    base = np.linspace(-width, width, count)
    order = np.argsort((np.arange(count) * 17) % count)
    return base[order]


def format_p(value: float) -> str:
    if value < 0.001:
        return f"P < {0.001:.3f}"
    if value < 0.01:
        return f"P = {value:.4f}"
    return f"P = {value:.2f}"


def plot_continuous(axis: plt.Axes, pathway: pd.DataFrame, summary: dict[str, Any]) -> None:
    clean_axis(axis)
    axis.set_title("Empirical stability stratifies\n72 h replication", loc="left", pad=10)
    for replicated, label, color, marker in (
        (False, "Not replicated", COLORS["not_replicated"], "o"),
        (True, "Replicated", COLORS["replicated"], "o"),
    ):
        group = pathway.loc[pathway["replicated_primary"].eq(replicated)].sort_values(
            ["empirical_survival_48h", "claim_id"]
        )
        y = float(replicated) + deterministic_jitter(len(group))
        axis.scatter(
            group["empirical_survival_48h"],
            y,
            s=42,
            facecolor=color if replicated else "white",
            edgecolor=color,
            linewidth=1.2,
            alpha=0.95,
            marker=marker,
            label=f"{label} (n={len(group)})",
            zorder=3,
        )
    axis.axvline(
        0.80,
        color=COLORS["empirical"],
        linestyle=(0, (4, 3)),
        linewidth=1.5,
        zorder=1,
    )
    continuous = summary["continuous_secondary"]
    annotation = (
        f"AUROC = {float(continuous['auroc']):.3f}\n"
        f"95% CI {float(continuous['bootstrap_ci_low']):.3f}–"
        f"{float(continuous['bootstrap_ci_high']):.3f}\n"
        f"Permutation {format_p(float(continuous['permutation_p_one_sided']))}"
    )
    axis.text(
        0.03,
        0.73,
        annotation,
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=9.8,
        bbox={"boxstyle": "round,pad=0.35", "fc": "white", "ec": COLORS["grid"]},
        zorder=4,
    )
    axis.set_xlim(-0.02, 1.02)
    axis.set_ylim(-0.22, 1.22)
    axis.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
    axis.set_yticks([0.0, 1.0])
    not_replicated_n = int((~pathway["replicated_primary"]).sum())
    replicated_n = int(pathway["replicated_primary"].sum())
    axis.set_yticklabels(
        [f"Not replicated\n(n={not_replicated_n})", f"Replicated\n(n={replicated_n})"]
    )
    axis.set_xlabel("Empirical survival across 81 resamples")
    axis.set_ylabel("Held-out 72 h outcome")
    axis.grid(axis="x", color=COLORS["grid"], linewidth=0.8)
    axis.text(
        0.80,
        0.05,
        "Frozen τ=0.80",
        transform=axis.transAxes,
        ha="right",
        va="bottom",
        fontsize=9.5,
        color=COLORS["empirical"],
        fontweight="bold",
    )


def plot_methods(axis: plt.Axes, methods: pd.DataFrame, summary: dict[str, Any]) -> None:
    clean_axis(axis)
    axis.set_title("Matched-coverage replication", loc="left", pad=10)
    method_specs = [
        ("empirical_stability_audit", "Empirical stability", COLORS["empirical"]),
        ("q_value_matched", "q-value matched", COLORS["q_value"]),
        (
            "q_value_and_leading_edge_size_matched",
            "q-value + size\nmatched",
            COLORS["size_matched"],
        ),
    ]
    y_positions = np.array([2.0, 1.0, 0.0])
    labels: list[str] = []
    for y, (method, label, color) in zip(y_positions, method_specs, strict=True):
        row = methods.loc[methods["method"].astype(str).eq(method)].iloc[0]
        fraction = float(row["replication_fraction"])
        low = float(row["replication_ci_low"])
        high = float(row["replication_ci_high"])
        replicated = int(float(row["n_replicated"]))
        selected = int(float(row["n_selected"]))
        axis.errorbar(
            fraction,
            y,
            xerr=np.array([[fraction - low], [high - fraction]]),
            fmt="o",
            markersize=8.0,
            markerfacecolor=color,
            markeredgecolor=color,
            ecolor=color,
            elinewidth=2.0,
            capsize=4.0,
            capthick=1.4,
            zorder=3,
        )
        axis.text(
            min(high + 0.025, 0.95),
            y,
            f"{replicated}/{selected}",
            va="center",
            ha="left",
            fontsize=9.7,
            color=COLORS["ink"],
        )
        labels.append(label)
    axis.axvline(0.5, color=COLORS["grid"], linestyle=":", linewidth=1.0, zorder=0)
    exact_p = float(summary["overlap_exact_reference"]["p_one_sided_empirical_greater"])
    axis.text(
        0.02,
        0.97,
        f"Primary empirical vs q-value:\nexact one-sided {format_p(exact_p)}",
        transform=axis.transAxes,
        va="top",
        ha="left",
        fontsize=9.5,
        bbox={"boxstyle": "round,pad=0.3", "fc": "white", "ec": COLORS["grid"]},
    )
    axis.set_xlim(0.0, 1.02)
    axis.set_ylim(-0.55, 2.55)
    axis.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
    axis.set_yticks(y_positions)
    axis.set_yticklabels(labels)
    axis.set_xlabel("Held-out replication fraction (Wilson 95% CI)")
    axis.grid(axis="x", color=COLORS["grid"], linewidth=0.8)


def plot_risk_coverage(axis: plt.Axes, tau_grid: pd.DataFrame) -> None:
    clean_axis(axis)
    axis.set_title("Frozen risk–coverage sensitivity", loc="left", pad=10)
    coverage = tau_grid["coverage"].to_numpy(dtype=float)
    risk = tau_grid["nonreplication_risk"].to_numpy(dtype=float)
    risk_low = 1.0 - tau_grid["replication_ci_high"].to_numpy(dtype=float)
    risk_high = 1.0 - tau_grid["replication_ci_low"].to_numpy(dtype=float)
    axis.plot(coverage, risk, color=COLORS["empirical"], linewidth=1.7, zorder=1)
    for index, row in tau_grid.iterrows():
        tau = float(row["tau"])
        x = float(row["coverage"])
        y = float(row["nonreplication_risk"])
        primary = math.isclose(tau, 0.80, abs_tol=1e-12)
        axis.errorbar(
            x,
            y,
            yerr=np.array([[y - risk_low[index]], [risk_high[index] - y]]),
            fmt="o",
            markersize=8.5 if primary else 7.0,
            markerfacecolor=COLORS["empirical"] if primary else "white",
            markeredgecolor=COLORS["empirical"],
            markeredgewidth=1.5,
            ecolor=COLORS["empirical"],
            elinewidth=1.6,
            capsize=3.5,
            capthick=1.2,
            zorder=3,
        )
        x_offset = 0.015 if primary else 0.012
        y_offset = 0.025 if tau in {0.90, 0.98} else -0.055
        axis.text(
            x + x_offset,
            y + y_offset,
            f"τ={tau:.2f}",
            fontsize=9.5,
            fontweight="bold" if primary else "normal",
            color=COLORS["ink"],
        )
    axis.set_xlim(0.04, 0.52)
    axis.set_ylim(0.0, 0.70)
    axis.set_xticks([0.1, 0.2, 0.3, 0.4, 0.5])
    axis.set_yticks([0.0, 0.2, 0.4, 0.6])
    axis.set_xlabel("Reporting coverage")
    axis.set_ylabel("Non-replication risk")
    axis.grid(color=COLORS["grid"], linewidth=0.8)
    axis.text(
        0.98,
        0.97,
        "Filled marker: frozen primary",
        transform=axis.transAxes,
        ha="right",
        va="top",
        fontsize=9.3,
        color=COLORS["empirical"],
    )


def render_figure(
    source: pd.DataFrame,
    summary: dict[str, Any],
    *,
    fontsize: float = 12.0,
) -> plt.Figure:
    panels = validate_source(source, summary)
    apply_publication_style(fontsize)
    figure, axes = plt.subplots(
        2,
        2,
        figsize=(8.0, 8.0),
        gridspec_kw={"height_ratios": [0.90, 1.05]},
    )
    plot_workflow(axes[0, 0], panels["A"])
    plot_continuous(axes[0, 1], panels["B"], summary)
    plot_methods(axes[1, 0], panels["C"], summary)
    plot_risk_coverage(axes[1, 1], panels["D"])
    for axis, label in zip(axes.ravel(), ("A", "B", "C", "D"), strict=True):
        panel_label(axis, label)
    figure.subplots_adjust(left=0.13, right=0.98, bottom=0.08, top=0.96, wspace=0.52, hspace=0.48)
    return figure


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if not math.isfinite(float(value)) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--benchmark-id", default="GSE146225_TP53_v1")
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--basename", default="Fig4_priority1_temporal_replication_v1")
    parser.add_argument("--fontsize", type=float, default=12.0)
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace only the rendered figure and its render metadata; never analytical outputs.",
    )
    args = parser.parse_args()

    require(args.fontsize >= 10.0, "Use fontsize >= 10 for publication readability")
    require(args.dpi >= 300, "Use dpi >= 300")
    require_clean_tracked_worktree()

    data_root = args.data_root.expanduser().resolve()
    benchmark_dir = data_root / "output" / "priority1" / args.benchmark_id
    source_path = benchmark_dir / "source_data" / "figure4.tsv"
    summary_path = benchmark_dir / "metrics" / "priority1_replication_summary.json"
    evaluation_meta_path = benchmark_dir / "metrics" / "priority1_replication.run_meta.json"
    for path in (source_path, summary_path, evaluation_meta_path):
        require(path.is_file(), f"Missing frozen Figure 4 input: {path}")

    summary = read_json(summary_path)
    evaluation_meta = read_json(evaluation_meta_path)
    verify_provenance(
        source_path=source_path,
        summary_path=summary_path,
        run_meta=evaluation_meta,
    )
    source = pd.read_csv(source_path, sep="\t", low_memory=False)
    validate_source(source, summary)

    out_dir = (
        args.out_dir.expanduser().resolve() if args.out_dir is not None else benchmark_dir / "fig"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "pdf": out_dir / f"{args.basename}.pdf",
        "png": out_dir / f"{args.basename}.png",
        "metadata": out_dir / f"{args.basename}.run_meta.json",
    }
    existing = [path for path in outputs.values() if path.exists()]
    require(args.force or not existing, f"Rendered outputs already exist: {existing}")

    with tempfile.TemporaryDirectory(prefix=".fig4_v7_", dir=out_dir) as temporary:
        staging_dir = Path(temporary)
        staged = {name: staging_dir / path.name for name, path in outputs.items()}
        figure = render_figure(source, summary, fontsize=args.fontsize)
        figure.savefig(staged["pdf"], bbox_inches="tight", facecolor="white")
        figure.savefig(
            staged["png"],
            dpi=args.dpi,
            bbox_inches="tight",
            facecolor="white",
        )
        plt.close(figure)

        metadata = {
            "script_build": SCRIPT_BUILD,
            "figure_message": FIGURE_MESSAGE,
            "analysis_recomputed": False,
            "source_contract": "figure4.tsv generated by frozen script 19",
            "benchmark_id": args.benchmark_id,
            "created_at_utc": datetime.now(UTC).isoformat(),
            "code": {
                "branch": git_value("branch", "--show-current"),
                "commit": git_value("rev-parse", "HEAD"),
                "tracked_worktree_clean": True,
            },
            "inputs": {
                "figure4_source": {"path": str(source_path), "sha256": sha256_file(source_path)},
                "priority1_summary": {
                    "path": str(summary_path),
                    "sha256": sha256_file(summary_path),
                },
                "evaluation_run_meta": {
                    "path": str(evaluation_meta_path),
                    "sha256": sha256_file(evaluation_meta_path),
                },
            },
            "outputs": {
                "pdf": {"path": str(outputs["pdf"]), "sha256": sha256_file(staged["pdf"])},
                "png": {"path": str(outputs["png"]), "sha256": sha256_file(staged["png"])},
            },
            "render": {
                "figure_size_inches": [8.0, 8.0],
                "fontsize": args.fontsize,
                "png_dpi": args.dpi,
                "palette": COLORS,
                "pdf_fonttype": 42,
            },
            "python": {
                "implementation": platform.python_implementation(),
                "version": sys.version,
                "matplotlib": matplotlib.__version__,
                "pandas": pd.__version__,
                "numpy": np.__version__,
            },
            "script": {
                "path": str(Path(__file__).resolve()),
                "sha256": sha256_file(Path(__file__).resolve()),
            },
            "stop_gate_p1": summary["stop_gate_p1"],
        }
        staged["metadata"].write_text(
            json.dumps(json_safe(metadata), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        for name, destination in outputs.items():
            if args.force and destination.exists():
                destination.unlink()
            os.replace(staged[name], destination)

    print("[PASS] Priority 1 Figure 4 rendered from frozen source data")
    print("[INFO] Analytical endpoints recomputed: false")
    print(f"[INFO] PDF: {outputs['pdf']}")
    print(f"[INFO] PNG: {outputs['png']}")
    print(f"[INFO] Render metadata: {outputs['metadata']}")


if __name__ == "__main__":
    main()
