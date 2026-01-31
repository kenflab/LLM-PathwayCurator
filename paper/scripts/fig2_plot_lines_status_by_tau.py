#!/usr/bin/env python3
# paper/scripts/fig2_lines_status_by_tau.py
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def apply_pub_style(fontsize: int = 16) -> None:
    plt.rcParams.update(
        {
            "font.size": fontsize,
            "axes.titlesize": fontsize + 2,
            "axes.labelsize": fontsize + 2,
            "xtick.labelsize": fontsize,
            "ytick.labelsize": fontsize,
            "legend.fontsize": fontsize,
            "figure.titlesize": fontsize + 4,
            "axes.linewidth": 1.1,
        }
    )


def variant_style(variant: str) -> dict[str, object]:
    v = str(variant or "").strip().lower()
    if v == "ours":
        return {"linestyle": "-", "marker": "o"}
    if v in ("context_swap", "shuffled_context", "shuffled-context"):
        return {"linestyle": ":", "marker": "x"}
    if v == "stress":
        return {"linestyle": "--", "marker": "s"}
    return {"linestyle": "-", "marker": None}


def _canon(s: str) -> str:
    return str(s or "").strip().lower()


def variant_label(variant: str, *, mode: str = "proposed") -> str:
    """
    Paper-facing legend label.
    mode:
      - "proposed": Proposed (τ-audit)  [recommended]
      - "ours":     Ours (audit+τ)
    """
    v = _canon(variant)

    if v == "ours":
        return "Proposed (τ-audit)" if mode == "proposed" else "Ours (audit+τ)"
    if v in ("context_swap", "shuffled_context", "shuffled-context"):
        return "Context swap"
    if v == "stress":
        return "Evidence dropout"
    return str(variant)


def _parse_csv(s: str) -> list[str]:
    return [x.strip() for x in str(s).split(",") if x.strip()]


def load_table(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")

    required = {
        "condition",
        "variant",
        "gate_mode",
        "tau",
        "n_total",
        "n_pass",
        "n_abstain",
        "n_fail",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"risk_coverage.tsv missing columns: {sorted(missing)}")

    for c in ["tau", "n_total", "n_pass", "n_abstain", "n_fail"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df["condition"] = df["condition"].astype(str)
    df["variant"] = df["variant"].astype(str)
    df["gate_mode"] = df["gate_mode"].astype(str)

    df = df.dropna(subset=["tau", "n_total", "n_pass", "n_abstain", "n_fail"])
    df = df[df["n_total"] > 0]
    return df


def summarize(
    df: pd.DataFrame,
    *,
    condition: str,
    gate_mode: str,
    variants: list[str],
) -> pd.DataFrame:
    d = df[(df["condition"] == condition) & (df["gate_mode"] == gate_mode)].copy()
    if variants:
        d = d[d["variant"].isin(set(variants))]

    if d.empty:
        msg = (
            "No rows after filtering "
            f"condition={condition} gate_mode={gate_mode} variants={variants}"
        )
        raise ValueError(msg)

    # In principle there should be exactly one row per (tau, variant, gate_mode, condition).
    # If duplicates exist, we aggregate by sum (safe for counts).
    g = (
        d.groupby(["tau", "variant"], as_index=False)[["n_total", "n_pass", "n_abstain", "n_fail"]]
        .sum()
        .sort_values(["variant", "tau"], kind="mergesort")
    )

    # Convert to rates for plotting
    g["pass_rate"] = g["n_pass"] / g["n_total"]
    g["abstain_rate"] = g["n_abstain"] / g["n_total"]
    g["fail_rate"] = g["n_fail"] / g["n_total"]

    # Sanity: rates should sum to ~1
    g["_sum"] = g["pass_rate"] + g["abstain_rate"] + g["fail_rate"]
    return g


def plot_lines(
    g: pd.DataFrame,
    *,
    out: Path,
    title: str,
    fontsize: int,
    metric: str = "abstain",  # "abstain" | "pass" | "both"
    label_mode: str = "proposed",  # "proposed" | "ours"
    xmargin: float = 0.03,  # tau-units padding
) -> None:
    apply_pub_style(fontsize=fontsize)

    metric = str(metric).strip().lower()
    if metric not in {"abstain", "pass", "both"}:
        raise ValueError(f"invalid metric={metric} (use abstain|pass|both)")

    # Keep legend/style order consistent with multipanel
    preferred = ["ours", "context_swap", "stress"]
    present = {_canon(v) for v in g["variant"].unique().tolist()}
    variants = [v for v in preferred if v in present]
    # Add any unexpected variants at the end (stable)
    variants += [v for v in sorted(present) if v not in set(preferred)]

    taus = sorted(g["tau"].unique().tolist())
    tau_labels = [f"{t:.2f}" for t in taus]

    fig_w = max(10.1, 1.05 * len(taus) + 5.0)
    fig_h = 5.8 if metric != "both" else 6.4
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=300)
    ax = fig.add_subplot(111)

    for v in variants:
        vv = g[g["variant"] == v].copy().sort_values("tau", kind="mergesort")
        vv = vv.set_index("tau").reindex(taus).reset_index()

        x = vv["tau"].to_numpy()

        if metric in {"abstain", "both"}:
            y = vv["abstain_rate"].to_numpy()
            ax.plot(
                x,
                y,
                label=variant_label(v, mode=label_mode),
                **variant_style(v),
            )

        if metric in {"pass", "both"}:
            st = dict(variant_style(v))
            st["linestyle"] = "-"  # PASS solid (kept simple)
            y = vv["pass_rate"].to_numpy()
            ax.plot(
                x,
                y,
                label=f"{variant_label(v, mode=label_mode)}: PASS",
                **st,
            )

    ax.set_ylim(0, 1)

    xmin = float(min(taus))
    xmax = float(max(taus))
    m = float(xmargin) if xmargin is not None else 0.0
    if m < 0:
        m = 0.0
    ax.set_xlim(xmin - m, xmax + m)

    ax.set_xticks(taus)
    ax.set_xticklabels(tau_labels)

    ax.set_xlabel("τ (audit threshold)")

    # Nature-style: make denominators explicit, keep text minimal.
    if metric == "abstain":
        ax.set_ylabel("ABSTAIN rate (fraction of total claims)")
    elif metric == "pass":
        ax.set_ylabel("PASS rate (fraction of total claims)")
    else:
        ax.set_ylabel("Fraction of total claims")

    ax.grid(True, axis="y", linewidth=0.5, alpha=0.3)

    # No in-figure title by default (Nature-style).
    # Only draw a title if explicitly provided.
    if str(title or "").strip():
        ax.set_title(str(title).strip())

    # De-duplicate legend entries while preserving insertion order
    handles, labels = ax.get_legend_handles_labels()
    seen = set()
    handles2, labels2 = [], []
    for handle, label in zip(handles, labels, strict=False):
        if label in seen:
            continue
        seen.add(label)
        handles2.append(handle)
        labels2.append(label)

    ax.legend(handles2, labels2, loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False)

    fig.tight_layout(rect=[0.0, 0.0, 0.84, 1.0])
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Line plot of status rates across τ (variant curves on one axis)."
    )
    ap.add_argument(
        "--in",
        dest="inp",
        required=True,
        help="risk_coverage.tsv (from fig2_collect_risk_coverage.py)",
    )
    ap.add_argument("--out", required=True, help="output figure path (pdf/png)")
    ap.add_argument("--condition", required=True, help="e.g., HNSC")
    ap.add_argument("--gate-mode", default="hard", help="gate mode to plot (default: hard)")
    ap.add_argument(
        "--variants", default="ours,context_swap,stress", help="comma list of variants to include"
    )
    ap.add_argument("--fontsize", type=int, default=16)
    ap.add_argument("--title", default="", help="optional title override")
    ap.add_argument(
        "--metric",
        default="abstain",
        choices=["abstain", "pass", "both"],
        help="Which rate(s) to plot as lines (default: abstain).",
    )
    ap.add_argument(
        "--label-mode",
        default="proposed",
        choices=["proposed", "ours"],
        help="Legend label for ours (default: proposed=Proposed (τ-audit))",
    )
    ap.add_argument(
        "--xmargin",
        type=float,
        default=0.03,
        help="Extra x-axis margin in tau units to avoid clipped markers (default: 0.03)",
    )

    args = ap.parse_args()

    df = load_table(Path(args.inp))

    condition = str(args.condition).strip()
    gate_mode = str(args.gate_mode).strip()
    variants = _parse_csv(args.variants)

    g = summarize(df, condition=condition, gate_mode=gate_mode, variants=variants)

    # Default: no title. Use --title only if you really want one.
    title = args.title.strip()

    plot_lines(
        g,
        out=Path(args.out),
        title=title,
        fontsize=int(args.fontsize),
        metric=str(args.metric),
        label_mode=str(args.label_mode),
        xmargin=float(args.xmargin),
    )


if __name__ == "__main__":
    main()
