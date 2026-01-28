#!/usr/bin/env python3
# paper/scripts/fig2_plot_scatter_human_risk.py
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
            "lines.linewidth": 1.8,
            "lines.markersize": 7.0,
        }
    )


def _canon(s: str) -> str:
    return str(s or "").strip().lower()


def variant_label(variant: str, *, mode: str = "proposed") -> str:
    v = _canon(variant)
    if v == "ours":
        return "Proposed (τ-audit)" if mode == "proposed" else "Ours (audit+τ)"
    if v in ("context_swap", "shuffled_context", "shuffled-context"):
        return "Context swap"
    if v == "stress":
        return "Evidence dropout"
    return str(variant)


def variant_style(variant: str) -> dict[str, object]:
    v = _canon(variant)
    # Match multipanel marker/linestyle identity BUT draw points only (no connecting lines).
    if v == "ours":
        return {"linestyle": "None", "marker": "o"}
    if v in ("context_swap", "shuffled_context", "shuffled-context"):
        return {"linestyle": "None", "marker": "x"}
    if v == "stress":
        return {"linestyle": "None", "marker": "s"}
    return {"linestyle": "None", "marker": "o"}


def _parse_csv(s: str) -> list[str]:
    return [x.strip() for x in str(s).split(",") if x.strip()]


def load_table(path: Path, *, ycol: str, min_npass_labeled: int = 1) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")

    required = {
        "condition",
        "variant",
        "gate_mode",
        "tau",
        "coverage_pass",
        ycol,
        "n_pass_labeled",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Input TSV missing columns: {sorted(missing)}")

    # numeric
    for c in ["tau", "coverage_pass", ycol, "n_pass_labeled"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    for c in ["condition", "variant", "gate_mode"]:
        df[c] = df[c].astype(str)

    # keep sane rows
    df = df.dropna(subset=["tau", "coverage_pass", ycol, "n_pass_labeled"])
    df = df[(df["tau"] >= 0) & (df["tau"] <= 1)]
    df = df[(df["coverage_pass"] >= 0) & (df["coverage_pass"] <= 1)]
    df = df[(df[ycol] >= 0) & (df[ycol] <= 1)]

    # Only points where the human-risk estimate is defined and not too noisy.
    k = int(min_npass_labeled)
    if k < 1:
        k = 1
    df = df[df["n_pass_labeled"] >= k]
    return df


def _ylabel(ycol: str) -> str:
    y = str(ycol).strip()
    if y == "risk_human_reject":
        return "Human reject rate\n (among human-labeled PASS)"
    if y == "risk_human_nonaccept":
        return "Human non-accept rate\n (among human-labeled PASS)"
    if y == "risk_proxy":
        return "Risk proxy\n (flagged PASS among PASS)"
    if y == "fail_rate_answered":
        return "Non-accept risk\n (FAIL among answered)"
    if y == "abstain_rate_total":
        return "Abstain rate"
    return y


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Scatter: coverage_pass vs human risk (from risk_coverage.label.tsv)"
    )
    ap.add_argument("--in", dest="inp", required=True, help="risk_coverage.label.tsv")
    ap.add_argument("--out", required=True, help="output figure path (pdf/png)")
    ap.add_argument("--condition", default="", help="e.g., HNSC (default: all conditions)")
    ap.add_argument("--gate-mode", default="hard", help="e.g., hard (default: hard)")
    ap.add_argument(
        "--ycol",
        default="risk_human_nonaccept",
        choices=["risk_human_reject", "risk_human_nonaccept"],
        help="human-risk column (default: risk_human_nonaccept)",
    )
    ap.add_argument("--variants", default="ours,context_swap,stress")
    ap.add_argument(
        "--ours-label",
        default="proposed",
        choices=["proposed", "ours"],
        help="Legend label for ours (default: proposed -> Proposed (τ-audit))",
    )
    ap.add_argument("--fontsize", type=int, default=16)
    ap.add_argument("--title", default="", help="optional title override")
    ap.add_argument("--annotate-tau", action="store_true", help="annotate each point with τ")
    ap.add_argument(
        "--annotate-npass",
        action="store_true",
        help="annotate each point with n_pass_labeled",
    )
    ap.add_argument(
        "--min-npass-labeled",
        type=int,
        default=1,
        help="plot only points with n_pass_labeled >= this threshold (default: 1)",
    )

    args = ap.parse_args()

    ycol = str(args.ycol).strip()
    df = load_table(Path(args.inp), ycol=ycol, min_npass_labeled=int(args.min_npass_labeled))

    if str(args.condition).strip():
        df = df[df["condition"] == str(args.condition).strip()]
        if df.empty:
            raise ValueError(f"No rows match condition={args.condition}")

    if str(args.gate_mode).strip():
        df = df[df["gate_mode"] == str(args.gate_mode).strip()]
        if df.empty:
            raise ValueError(f"No rows match gate_mode={args.gate_mode}")

    want = _parse_csv(args.variants)
    if want:
        df = df[df["variant"].isin(set(want))]
        if df.empty:
            raise ValueError(f"No rows match variants={want}")

    # enforce legend/order consistency with multipanel:
    order = ["ours", "context_swap", "stress"]
    present = [v for v in order if v in set(df["variant"].tolist())]
    # if something else exists, append deterministically
    extras = sorted([v for v in df["variant"].unique().tolist() if v not in set(present)])
    variants = present + extras

    apply_pub_style(fontsize=int(args.fontsize))

    fig = plt.figure(figsize=(11.1, 6.2), dpi=300)
    ax = fig.add_subplot(111)

    # Optional point-size encoding for uncertainty proxy:
    # bigger point => more labeled PASS used for human-risk estimate
    # (keeps colors/styles consistent without adding clutter)
    size_min = 30.0
    size_max = 120.0
    npass_max = float(df["n_pass_labeled"].max()) if len(df) else 1.0
    if npass_max <= 0:
        npass_max = 1.0

    def _size_from_npass(n: float) -> float:
        # linear map into [size_min, size_max]
        x = float(n) if n is not None else 0.0
        if x < 0:
            x = 0.0
        t = x / npass_max
        return size_min + (size_max - size_min) * t

    for v in variants:
        d = df[df["variant"] == v].copy().sort_values(["tau"], kind="mergesort")
        if d.empty:
            continue

        st = dict(variant_style(v))
        label = variant_label(v, mode=str(args.ours_label))

        x = d["coverage_pass"].to_numpy()
        y = d[ycol].to_numpy()
        sizes = d["n_pass_labeled"].to_numpy()

        # map size by labeled PASS count (avoids text clutter)
        s = [float(_size_from_npass(n)) for n in sizes]

        ax.scatter(
            x,
            y,
            s=s,
            label=label,
            marker=str(st.get("marker", "o")),
        )

        # Optional annotations (OFF by default). Keep them sparse to avoid overlap.
        if args.annotate_tau or args.annotate_npass:
            # annotate only endpoints to reduce clutter
            idxs = []
            if len(d) >= 1:
                idxs.append(d.index[0])
            if len(d) >= 2:
                idxs.append(d.index[-1])

            for i in idxs:
                r = d.loc[i]
                parts = []
                if args.annotate_tau:
                    parts.append(f"{float(r['tau']):.2f}")
                if args.annotate_npass:
                    parts.append(f" n={int(r['n_pass_labeled'])}")
                txt = ",".join(parts)
                if not txt:
                    continue
                ax.annotate(
                    txt,
                    (float(r["coverage_pass"]), float(r[ycol])),
                    textcoords="offset points",
                    xytext=(6, 6),
                    ha="left",
                    va="bottom",
                    fontsize=max(9, int(args.fontsize) - 6),
                )

    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)

    # coverage_pass is PASS fraction among TOTAL (n_pass / n_total)
    ax.set_xlabel("PASS rate (n_pass / n_total)\n (fraction of total claims passing audit)")
    ax.set_ylabel(_ylabel(ycol))

    ax.grid(True, linewidth=0.5, alpha=0.3)

    # cond = str(args.condition).strip() or "all conditions"
    # gate = str(args.gate_mode).strip() or "all gates"
    # No in-figure title by default.
    # Only draw a title if explicitly provided.
    if str(args.title).strip():
        ax.set_title(str(args.title).strip())
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False)

    fig.tight_layout(rect=[0.0, 0.0, 0.82, 1.0])
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
