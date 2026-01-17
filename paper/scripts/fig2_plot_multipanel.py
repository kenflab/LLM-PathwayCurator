#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def apply_pub_style(fontsize: int = 16) -> None:
    plt.rcParams.update(
        {
            "font.size": fontsize,
            "axes.titlesize": fontsize + 1,
            "axes.labelsize": fontsize + 2,
            "xtick.labelsize": fontsize,
            "ytick.labelsize": fontsize,
            "legend.fontsize": fontsize,
            "figure.titlesize": fontsize + 4,
            "lines.linewidth": 2.0,
            "lines.markersize": 5.5,
            "axes.linewidth": 1.1,
        }
    )


def method_style(method: str) -> dict[str, object]:
    m = method.lower()
    if m in ("ours", "method_ours"):
        return {"linestyle": "-", "marker": "o"}
    if m in ("baseline_qonly", "qonly"):
        return {"linestyle": "--", "marker": "s"}
    if m in ("baseline_nonllm_selector", "nonllm_selector"):
        return {"linestyle": "-.", "marker": "^"}
    if m in ("shuffled_labels", "shuffled"):
        return {"linestyle": ":", "marker": "x"}
    if m in ("no_modules", "ablation_no_modules"):
        return {"linestyle": (0, (3, 1, 1, 1)), "marker": "d"}
    return {"linestyle": "-", "marker": None}


def load_and_validate(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    required = {
        "benchmark_id",
        "cancer",
        "method",
        "tau",
        "coverage_pass",
        "risk_human_reject",
        "n_pass_labeled",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"risk_coverage.tsv missing columns: {sorted(missing)}")

    for c in ["tau", "coverage_pass", "risk_human_reject", "n_pass_labeled"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df["benchmark_id"] = df["benchmark_id"].astype(str)
    df["cancer"] = df["cancer"].astype(str)
    df["method"] = df["method"].astype(str)

    df = df.dropna(subset=["coverage_pass", "tau"])
    df = df[(df["coverage_pass"] >= 0) & (df["coverage_pass"] <= 1)]
    df = df[(df["tau"] >= 0) & (df["tau"] <= 1)]
    return df


def _panel_plot(ax, d: pd.DataFrame, methods: list[str], cancer: str) -> None:
    d = d.sort_values(["method", "tau"], kind="mergesort")

    for method in methods:
        g = d[d["method"] == method].copy()
        if g.empty:
            continue
        g = g.sort_values("tau", kind="mergesort")
        g = g.dropna(subset=["risk_human_reject"])
        g = g[g["n_pass_labeled"] > 0]
        if g.empty:
            continue
        ax.plot(g["coverage_pass"], g["risk_human_reject"], label=method, **method_style(method))

    ax.set_title(cancer)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, linewidth=0.5, alpha=0.35)
    ax.set_xticks([0.0, 0.5, 1.0])
    ax.set_yticks([0.0, 0.5, 1.0])


def main() -> None:
    ap = argparse.ArgumentParser(description="Supplement multipanel Fig2 from risk_coverage.tsv")
    ap.add_argument("--in", dest="inp", required=True, help="risk_coverage.tsv")
    ap.add_argument("--out", required=True, help="output figure path (pdf/png)")
    ap.add_argument("--benchmark-id", default="", help="optional benchmark_id filter")
    ap.add_argument("--cols", type=int, default=4, help="grid columns (default 4)")
    ap.add_argument("--fontsize", type=int, default=16, help="base font size (default 16)")
    ap.add_argument("--title", default="Risk–coverage curves across cancers", help="figure title")
    args = ap.parse_args()

    df = load_and_validate(Path(args.inp))
    if args.benchmark_id:
        df = df[df["benchmark_id"] == str(args.benchmark_id)]
        if df.empty:
            raise ValueError(f"No rows match benchmark_id={args.benchmark_id}")

    cancers = sorted(df["cancer"].unique().tolist())
    methods = sorted(df["method"].unique().tolist())
    if not cancers:
        raise ValueError("No cancers to plot.")

    cols = max(1, int(args.cols))
    rows = int(math.ceil(len(cancers) / cols))

    apply_pub_style(fontsize=int(args.fontsize))

    fig_w = max(8.5, cols * 3.2)
    fig_h = max(6.5, rows * 2.8)
    fig, axes = plt.subplots(rows, cols, figsize=(fig_w, fig_h), dpi=150)
    axes_list = list(axes.ravel()) if hasattr(axes, "ravel") else [axes]

    for idx, cancer in enumerate(cancers):
        ax = axes_list[idx]
        _panel_plot(ax, df[df["cancer"] == cancer].copy(), methods, cancer)

        if idx % cols == 0:
            ax.set_ylabel("Risk (Human REJECT among PASS)")
        else:
            ax.set_ylabel("")
        if idx // cols == rows - 1:
            ax.set_xlabel("Coverage (PASS rate)")
        else:
            ax.set_xlabel("")

    for j in range(len(cancers), len(axes_list)):
        axes_list[j].axis("off")

    # shared legend (outside)
    handles, labels = [], []
    for ax in axes_list:
        h, lbls = ax.get_legend_handles_labels()
        if h:
            handles, labels = h, lbls
            break

    fig.suptitle(str(args.title), y=0.995)

    if handles:
        fig.legend(handles, labels, loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False)

    fig.tight_layout(rect=[0.0, 0.0, 0.86, 0.96])

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
