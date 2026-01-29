#!/usr/bin/env python3
# paper/scripts/fig2_plot_multipanel.py
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


def variant_style(variant: str) -> dict[str, object]:
    v = str(variant or "").strip().lower()
    if v in ("ours",):
        return {"linestyle": "-", "marker": "o"}
    if v in ("context_swap", "shuffled_context", "shuffled-context"):
        return {"linestyle": ":", "marker": "x"}
    if v in ("stress",):
        return {"linestyle": "--", "marker": "s"}
    return {"linestyle": "-", "marker": None}


def _canon(s: str) -> str:
    return str(s or "").strip().lower()


def variant_label(variant: str, *, mode: str = "proposed") -> str:
    """
    Paper-facing legend label.
    - Keep code identifiers (ours/context_swap/stress) unchanged elsewhere.
    - Only rewrite what appears in the figure legend.

    mode:
      - "proposed": use Proposed (τ-audit) for ours (recommended)
      - "ours": use Ours (audit+τ) for ours
    """
    v = _canon(variant)

    if v == "ours":
        return "Proposed (τ-audit)" if mode == "proposed" else "Ours (audit+τ)"
    if v in ("context_swap", "shuffled_context", "shuffled-context"):
        return "Context swap"
    if v == "stress":
        # your stress is evidence dropout (and optionally contradiction); keep short
        return "Evidence dropout"
    return str(variant)


def method_label(method: str) -> str:
    """
    Optional: only used when label_mode=variant_method.
    Keep minimal and safe.
    """
    m = _canon(method)
    if m in ("ours", "method_ours"):
        return "Ours"
    if m in ("baseline_qonly", "qonly"):
        return "Q-only"
    if m in ("baseline_nonllm_selector", "nonllm_selector"):
        return "Non-LLM selector"
    if m in ("no_modules", "ablation_no_modules"):
        return "No modules"
    if m in ("shuffled_labels", "shuffled"):
        return "Shuffled labels"
    return str(method)


def load_and_validate(path: Path, *, ycol: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")

    required = {"benchmark_id", "condition", "variant", "method", "tau", "coverage_pass", ycol}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"risk_coverage.tsv missing columns: {sorted(missing)}")

    numeric_cols = ["tau", "coverage_pass", ycol]

    # Only require these if present/needed (human risk mode)
    if ycol in ("risk_human_reject", "risk_human_nonaccept"):
        if "n_pass_labeled" not in df.columns:
            raise ValueError(
                "risk_coverage.tsv missing columns: ['n_pass_labeled'] (human risk mode)"
            )
        numeric_cols.append("n_pass_labeled")

    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df["benchmark_id"] = df["benchmark_id"].astype(str)
    df["condition"] = df["condition"].astype(str)
    df["variant"] = df["variant"].astype(str)
    df["method"] = df["method"].astype(str)

    df = df.dropna(subset=["coverage_pass", "tau"])
    df = df[(df["coverage_pass"] >= 0) & (df["coverage_pass"] <= 1)]
    df = df[(df["tau"] >= 0) & (df["tau"] <= 1)]
    return df


def mm_to_inch(mm: float) -> float:
    return float(mm) / 25.4


def resolve_figsize(
    *,
    cols: int,
    rows: int,
    figsize: tuple[float, float] | None,
    width_mm: float | None,
    height_mm: float | None,
) -> tuple[float, float]:
    """
    Priority:
      1) --figsize W H (inches)
      2) --width-mm / --height-mm (mm; if one missing, infer from grid aspect)
      3) fallback auto sizing (current behavior)
    """
    if figsize is not None:
        w, h = float(figsize[0]), float(figsize[1])
        if w <= 0 or h <= 0:
            raise ValueError("--figsize must be positive: e.g. --figsize 7.2 6.0")
        return (w, h)

    if width_mm is not None or height_mm is not None:
        if width_mm is None and height_mm is None:
            pass
        # infer missing side from grid aspect ratio
        aspect = (rows * 2.8) / max(1e-6, (cols * 3.2))  # matches your current heuristic
        if width_mm is None:
            width_mm = float(height_mm) / max(1e-6, aspect)
        if height_mm is None:
            height_mm = float(width_mm) * aspect
        return (mm_to_inch(float(width_mm)), mm_to_inch(float(height_mm)))

    # fallback (existing behavior)
    fig_w = max(8.5, cols * 3.2)
    fig_h = max(6.5, rows * 2.8)
    return (fig_w, fig_h)


def _panel_plot(
    ax,
    d: pd.DataFrame,
    variants: list[str],
    condition: str,
    *,
    ycol: str,
    label_mode: str = "variant",  # "variant" or "variant_method"
    ours_label_mode: str = "proposed",  # "proposed" or "ours"
) -> None:
    d = d.sort_values(["variant", "method", "tau"], kind="mergesort")

    for variant in variants:
        g = d[d["variant"] == variant].copy()
        if g.empty:
            continue

        g = g.sort_values("tau", kind="mergesort")
        g = g.dropna(subset=[ycol])
        g = g[(g[ycol] >= 0) & (g[ycol] <= 1)]

        # Human-label risk requires labeled PASS count (only computed among PASS)
        if ycol in ("risk_human_reject", "risk_human_nonaccept"):
            if "n_pass_labeled" not in g.columns:
                continue
            g = g[g["n_pass_labeled"] > 0]

        if g.empty:
            continue

        # If multiple methods exist inside a variant, optionally split label
        methods_in = sorted(g["method"].unique().tolist())
        if label_mode == "variant_method" and len(methods_in) > 1:
            for method in methods_in:
                gg = g[g["method"] == method].copy()
                if gg.empty:
                    continue
                lbl = f"{variant_label(variant, mode=ours_label_mode)} / {method_label(method)}"
                ax.plot(
                    gg["coverage_pass"],
                    gg[ycol],
                    label=lbl,
                    **variant_style(variant),
                )
        else:
            ax.plot(
                g["coverage_pass"],
                g[ycol],
                label=variant_label(variant, mode=ours_label_mode),
                **variant_style(variant),
            )

    ax.set_title(condition)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, linewidth=0.5, alpha=0.35)
    ax.set_xticks([0.0, 0.5, 1.0])
    ax.set_yticks([0.0, 0.5, 1.0])


def _ylabel_for(ycol: str) -> str:
    if ycol == "fail_rate_answered":
        return "Non-accept risk (FAIL among answered = PASS+FAIL)"
    if ycol == "fail_rate_total":
        return "Non-accept rate (FAIL among total)"
    if ycol == "abstain_rate_total":
        return "ABSTAIN rate (fraction of total claims)"
    if ycol == "risk_human_reject":
        return "Risk (Human REJECT among PASS)"
    if ycol == "risk_human_nonaccept":
        return "Risk (Human non-accept among PASS)"
    if ycol == "risk_proxy":
        return "Risk proxy (flagged PASS among PASS)"
    return str(ycol)


def main() -> None:
    ap = argparse.ArgumentParser(description="Supplement multipanel Fig2 from risk_coverage.tsv")
    ap.add_argument("--in", dest="inp", required=True, help="risk_coverage.tsv")
    ap.add_argument("--out", required=True, help="output figure path (pdf/png)")
    ap.add_argument("--benchmark-id", default="", help="optional benchmark_id filter")
    ap.add_argument("--cols", type=int, default=4, help="grid columns (default 4)")
    ap.add_argument("--fontsize", type=int, default=16, help="base font size (default 16)")
    ap.add_argument(
        "--title",
        default="",
        help="optional figure title (default: none; Nature-style)",
    )
    ap.add_argument(
        "--ycol",
        default="fail_rate_answered",
        choices=[
            "fail_rate_answered",
            "fail_rate_total",
            "abstain_rate_total",
            "risk_human_reject",
            "risk_human_nonaccept",
            "risk_proxy",
        ],
        help="y-axis column (default: fail_rate_answered)",
    )
    ap.add_argument(
        "--variants",
        default="ours,context_swap,stress",
        help="Comma list of variants to plot (default: ours,context_swap,stress)",
    )
    ap.add_argument(
        "--label-mode",
        default="variant",
        choices=["variant", "variant_method"],
        help="Legend label mode (default: variant). Use variant_method only if needed.",
    )
    ap.add_argument(
        "--ours-label",
        default="proposed",
        choices=["proposed", "ours"],
        help="Legend label for variant=ours (default: proposed -> 'Proposed (τ-audit)')",
    )
    ap.add_argument(
        "--condition",
        default="",
        help="If set, plot only this condition (single-panel mode). e.g., HNSC",
    )
    ap.add_argument(
        "--figsize",
        nargs=2,
        type=float,
        default=None,
        metavar=("W_IN", "H_IN"),
        help="Figure size in inches (overrides auto sizing). e.g. --figsize 7.2 6.0",
    )
    ap.add_argument(
        "--width-mm",
        type=float,
        default=None,
        help="Figure width in mm (Nature: 89 single-col, 183 double-col). Overrides auto sizing.",
    )
    ap.add_argument(
        "--height-mm",
        type=float,
        default=None,
        help="Figure height in mm (Nature max ~170). Overrides auto sizing.",
    )
    args = ap.parse_args()

    ycol = str(args.ycol).strip()
    df = load_and_validate(Path(args.inp), ycol=ycol)

    if args.benchmark_id:
        df = df[df["benchmark_id"] == str(args.benchmark_id)]
        if df.empty:
            raise ValueError(f"No rows match benchmark_id={args.benchmark_id}")

    # Optional single-condition filter (recommended for human-label plots)
    if str(args.condition).strip():
        df = df[df["condition"] == str(args.condition).strip()]
        if df.empty:
            raise ValueError(f"No rows match condition={args.condition}")

    conditions = sorted(df["condition"].unique().tolist())
    variants_all = sorted(df["variant"].unique().tolist())
    if not conditions:
        raise ValueError("No conditions to plot.")

    want = [x.strip() for x in str(args.variants).split(",") if x.strip()]
    variants = [v for v in want if v in set(variants_all)]
    if not variants:
        raise ValueError(
            f"No matching variants to plot. available={variants_all}, requested={want}"
        )

    # Single-panel mode if only one condition remains
    if len(conditions) == 1:
        cols = 1
    else:
        cols = max(1, int(args.cols))
    rows = int(math.ceil(len(conditions) / cols))

    apply_pub_style(fontsize=int(args.fontsize))

    fig_w, fig_h = resolve_figsize(
        cols=cols,
        rows=rows,
        figsize=tuple(args.figsize) if args.figsize is not None else None,
        width_mm=args.width_mm,
        height_mm=args.height_mm,
    )
    fig, axes = plt.subplots(rows, cols, figsize=(fig_w, fig_h), dpi=300)

    axes_list = list(axes.ravel()) if hasattr(axes, "ravel") else [axes]

    for idx, condition in enumerate(conditions):
        ax = axes_list[idx]
        _panel_plot(
            ax,
            df[df["condition"] == condition].copy(),
            variants,
            condition,
            ycol=ycol,
            label_mode=str(args.label_mode),
            ours_label_mode=str(args.ours_label),
        )

        ax.set_ylabel("")
        ax.set_xlabel("")

    for j in range(len(conditions), len(axes_list)):
        axes_list[j].axis("off")

    # shared legend (outside)
    handles, labels = [], []
    for ax in axes_list:
        h, lbls = ax.get_legend_handles_labels()
        if h:
            handles, labels = h, lbls
            break

    if str(args.title).strip():
        fig.suptitle(str(args.title).strip(), y=0.995)

    fig.supxlabel(
        "PASS rate (n_pass / n_total)\n(fraction of total claims passing audit)",
        y=0.02,
    )

    fig.supylabel(_ylabel_for(ycol), x=0.01)

    if handles:
        fig.legend(
            handles,
            labels,
            loc="center left",
            bbox_to_anchor=(1.01, 0.5),
            frameon=False,
        )

    # fig.tight_layout(rect=[0.06, 0.06, 0.86, 0.95])
    # fig.subplots_adjust(wspace=0.35, hspace=0.45)
    fig.tight_layout()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
