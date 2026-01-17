#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def apply_pub_style(fontsize: int = 20) -> None:
    plt.rcParams.update(
        {
            "font.size": fontsize,
            "axes.titlesize": fontsize + 2,
            "axes.labelsize": fontsize + 2,
            "xtick.labelsize": fontsize,
            "ytick.labelsize": fontsize,
            "legend.fontsize": fontsize - 2,
            "figure.titlesize": fontsize + 4,
            "lines.linewidth": 2.2,
            "lines.markersize": 6.0,
            "axes.linewidth": 1.2,
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

    df = df.dropna(subset=["coverage_pass", "tau"])
    df = df[(df["coverage_pass"] >= 0) & (df["coverage_pass"] <= 1)]
    df = df[(df["tau"] >= 0) & (df["tau"] <= 1)]
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot Fig2 risk–coverage (single cancer).")
    ap.add_argument("--in", dest="inp", required=True, help="risk_coverage.tsv")
    ap.add_argument("--out", required=True, help="output figure path (pdf/png)")
    ap.add_argument("--benchmark-id", default="", help="optional benchmark_id filter")
    ap.add_argument("--cancer", required=True, help="cancer code (e.g., HNSC)")
    ap.add_argument("--fontsize", type=int, default=20, help="base font size")
    args = ap.parse_args()

    df = load_and_validate(Path(args.inp))
    if args.benchmark_id:
        df = df[df["benchmark_id"].astype(str) == str(args.benchmark_id)]
    df = df[df["cancer"].astype(str) == str(args.cancer)]
    if df.empty:
        raise ValueError(f"No rows for cancer={args.cancer}")

    apply_pub_style(fontsize=int(args.fontsize))

    fig = plt.figure(figsize=(7.5, 6.5), dpi=150)
    ax = fig.add_subplot(111)

    df = df.sort_values(["method", "tau"], kind="mergesort")
    for method, g in df.groupby("method", sort=True):
        g = g.sort_values("tau", kind="mergesort")
        g = g.dropna(subset=["risk_human_reject"])
        g = g[g["n_pass_labeled"] > 0]
        if g.empty:
            continue
        ax.plot(
            g["coverage_pass"],
            g["risk_human_reject"],
            label=str(method),
            **method_style(str(method)),
        )

    ax.set_xlabel("Coverage (PASS rate)")
    ax.set_ylabel("Risk (Human REJECT rate among PASS)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, linewidth=0.6, alpha=0.4)
    ax.set_title("Selective prediction for pathway narratives")

    ax.legend(loc="upper right", frameon=False)

    fig.tight_layout()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
