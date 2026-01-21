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


def _canon_method(m: str) -> str:
    return str(m or "").strip().lower()


def method_label(method: str) -> str:
    m = _canon_method(method)
    # paper-facing labels (edit as you like)
    if m in ("ours", "method_ours"):
        return "Ours"
    if m in ("shuffled_context", "shuffled-context", "context_shuffled"):
        return "Shuffled context"
    if m in ("baseline_qonly", "qonly"):
        return "Q-only"
    if m in ("baseline_nonllm_selector", "nonllm_selector"):
        return "Non-LLM selector"
    if m in ("shuffled_labels", "shuffled"):
        return "Shuffled labels"
    if m in ("no_modules", "ablation_no_modules"):
        return "No modules"
    return str(method)


def method_style(method: str) -> dict[str, object]:
    m = _canon_method(method)
    if m in ("ours", "method_ours"):
        return {"linestyle": "-", "marker": "o"}
    if m in ("shuffled_context", "shuffled-context", "context_shuffled"):
        return {"linestyle": ":", "marker": "x"}
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
        df = df[df["benchmark_id"] == str(args.benchmark_id)]
    df = df[df["cancer"] == str(args.cancer)]
    if df.empty:
        raise ValueError(f"No rows for cancer={args.cancer}")

    apply_pub_style(fontsize=int(args.fontsize))

    fig = plt.figure(figsize=(7.5, 6.5), dpi=150)
    ax = fig.add_subplot(111)

    # stable order: ours first, then shuffled_context, then others
    order = [
        "ours",
        "shuffled_context",
        "baseline_qonly",
        "baseline_nonllm_selector",
        "no_modules",
        "shuffled_labels",
    ]
    df["_method_key"] = df["method"].map(_canon_method)
    df["_order"] = df["_method_key"].apply(lambda x: order.index(x) if x in order else 999)
    df = df.sort_values(["_order", "method", "tau"], kind="mergesort")

    for method, g in df.groupby("method", sort=False):
        g = g.sort_values("tau", kind="mergesort")
        g = g.dropna(subset=["risk_human_reject"])
        g = g[g["n_pass_labeled"] > 0]
        if g.empty:
            continue
        ax.plot(
            g["coverage_pass"],
            g["risk_human_reject"],
            label=method_label(method),
            **method_style(method),
        )

    ax.set_xlabel("Coverage (PASS rate)")
    ax.set_ylabel("Risk (Human REJECT rate among PASS)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, linewidth=0.6, alpha=0.4)
    ax.set_title(f"Selective prediction for pathway narratives ({args.cancer})")

    ax.legend(loc="upper right", frameon=False)

    fig.tight_layout()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
