#!/usr/bin/env python3
# paper/scripts/fig2_plot_abstain_reasons.py
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def apply_pub_style(fontsize: int = 16) -> None:
    """Apply publication-style matplotlib rcParams for stacked bar plots.

    Parameters
    ----------
    fontsize : int, optional
        Base font size for axes, ticks, and legend (default 16).

    Notes
    -----
    This mutates global matplotlib rcParams for the current process.
    """
    plt.rcParams.update(
        {
            "font.size": fontsize,
            "axes.titlesize": fontsize + 1,
            "axes.labelsize": fontsize + 2,
            "xtick.labelsize": fontsize,
            "ytick.labelsize": fontsize,
            "legend.fontsize": max(9, fontsize - 2),
            "figure.titlesize": fontsize + 3,
            "axes.linewidth": 1.1,
        }
    )


def infer_variant_from_path(p: Path) -> str:
    """Infer the variant directory name from an audit_log.tsv path.

    The function supports paths like:
    ``.../out*/<condition>/<variant>/gate_*/tau_*/audit_log.tsv``

    Parameters
    ----------
    p : pathlib.Path
        Path to an ``audit_log.tsv`` file.

    Returns
    -------
    str
        Inferred variant name. Falls back to a parent directory heuristic
        if the expected layout is not found.

    Notes
    -----
    This is a heuristic intended for plotting convenience, not a strict
    contract.
    """
    parts = [x for x in p.parts]
    # variant is the segment after condition (HNSC)
    # find ".../out*/<condition>/<variant>/gate_*/tau_*/audit_log.tsv"
    for i in range(len(parts) - 1):
        if parts[i] in ("out", "out_llm") and i + 2 < len(parts):
            return parts[i + 2]  # variant
    return p.parent.parent.parent.name


def canon_variant(v: str, *, is_llm: bool = False) -> str:
    """Canonicalize a variant identifier to a paper-facing label.

    Parameters
    ----------
    v : str
        Raw variant identifier.
    is_llm : bool, optional
        If True, label the proposed variant as an LLM run (default False).

    Returns
    -------
    str
        Canonicalized variant label used on the x-axis.
    """
    v = (v or "").strip().lower()

    if v == "ours":
        return "Proposed (LLM)" if is_llm else "Proposed"
    if v in ("context_swap", "shuffled_context", "shuffled-context"):
        return "Context swap"
    if v == "stress":
        return "Evidence dropout"
    return v or "Unknown"


def canon_reason(r: str) -> str:
    """Canonicalize an abstain reason code to a readable label.

    Parameters
    ----------
    r : str
        Raw abstain reason string from audit logs.

    Returns
    -------
    str
        Human-readable reason label suitable for legends.

    Notes
    -----
    Known codes are mapped to curated labels. Unknown values are converted
    from snake_case to Title-like text.
    """
    s = (r or "").strip()
    if not s:
        return "Unknown"

    k = s.strip().lower()

    # common codes seen in audit logs
    mapping = {
        "context_gate": "Context gate",
        "context_nonspecific": "Context-nonspecific",
        "inconclusive_stress": "Inconclusive stress",
        "unstable": "Unstable",
        "low_survival": "Low stability",
        "stability": "Low stability",
        "stability_gate": "Low stability",
    }
    if k in mapping:
        return mapping[k]

    # if mixed-case or snake_case appears, make a readable fallback
    k2 = k.replace("_", " ").strip()
    if not k2:
        return "Unknown"
    return k2[:1].upper() + k2[1:]


def main() -> None:
    """Plot ABSTAIN reason composition as a 100% stacked bar chart.

    Reads one or more ``audit_log.tsv`` files (typically one per variant),
    extracts ABSTAIN rows, normalizes abstain reasons, collapses reasons to a
    global top-k (others -> "Other"), and plots per-variant proportions among
    ABSTAIN.

    Command-line arguments
    ----------------------
    --audit-logs : list[str]
        One or more audit_log.tsv paths.
    --out : str
        Output figure path (pdf/png).
    --title : str, optional
        Optional title.
    --topk : int, optional
        Keep top-k abstain reasons globally (default 5).
    --fontsize : int, optional
        Base font size (default 16).

    Raises
    ------
    ValueError
        If required columns are missing or if no rows are available to plot.
    """
    ap = argparse.ArgumentParser(description="Plot ABSTAIN reason composition (stacked bar).")
    ap.add_argument(
        "--audit-logs", nargs="+", required=True, help="audit_log.tsv paths (one per variant)"
    )
    ap.add_argument("--out", required=True, help="output path (pdf/png)")
    ap.add_argument("--title", default="", help="optional title")
    ap.add_argument(
        "--topk", type=int, default=5, help="keep top-k abstain reasons (others -> Other)"
    )
    ap.add_argument("--fontsize", type=int, default=16)
    args = ap.parse_args()

    apply_pub_style(int(args.fontsize))

    rows = []
    for s in args.audit_logs:
        p = Path(s)
        df = pd.read_csv(p, sep="\t")

        if "status" not in df.columns or "abstain_reason" not in df.columns:
            raise ValueError(f"Missing required columns in {p}: need status + abstain_reason")

        is_llm = "out_llm" in str(p)

        variant = canon_variant(infer_variant_from_path(p), is_llm=is_llm)
        n_total = int(len(df))
        abd = df[df["status"] == "ABSTAIN"].copy()
        n_abstain = int(len(abd))

        # If no abstains, still keep row (all zeros)
        if n_abstain == 0:
            rows.append(
                {
                    "variant": variant,
                    "reason": "None",
                    "count": 0,
                    "n_total": n_total,
                    "n_abstain": 0,
                }
            )
            continue

        vc = abd["abstain_reason"].fillna("NA").astype(str).map(canon_reason).value_counts()
        for reason, c in vc.items():
            rows.append(
                {
                    "variant": variant,
                    "reason": reason,
                    "count": int(c),
                    "n_total": n_total,
                    "n_abstain": n_abstain,
                }
            )

    tall = pd.DataFrame(rows)
    # collapse to topk reasons globally (by count)
    if not tall.empty:
        top = (
            tall.groupby("reason", as_index=False)["count"]
            .sum()
            .sort_values("count", ascending=False)
            .head(int(args.topk))["reason"]
            .tolist()
        )
        tall["reason2"] = tall["reason"].where(tall["reason"].isin(set(top)), other="Other")
    else:
        raise ValueError("No rows to plot.")

    # pivot to variant x reason2
    grp = tall.groupby(["variant", "reason2"], as_index=False)["count"].sum()
    piv = grp.pivot(index="variant", columns="reason2", values="count").fillna(0.0)

    # order variants (match your figure identity)
    order = ["Proposed", "Context swap", "Evidence dropout", "Proposed (LLM)"]

    idx = [v for v in order if v in piv.index] + [v for v in piv.index if v not in set(order)]
    piv = piv.loc[idx]

    # convert to proportions among ABSTAIN (100% stacked)
    meta = tall.groupby("variant", as_index=False).agg(
        n_total=("n_total", "max"), n_abstain=("n_abstain", "max")
    )
    meta = meta.set_index("variant").loc[piv.index]
    denom = meta["n_abstain"].replace(0, 1)
    prop = piv.div(denom, axis=0)

    fig = plt.figure(figsize=(7.6, 5.5), dpi=300)
    ax = fig.add_subplot(111)

    bottom = None
    for col in prop.columns.tolist():
        y = prop[col].to_numpy()
        if bottom is None:
            ax.bar(prop.index.tolist(), y, label=str(col))
            bottom = y
        else:
            ax.bar(prop.index.tolist(), y, bottom=bottom, label=str(col))
            bottom = bottom + y

    ax.set_ylim(0, 1.0)
    ax.set_ylabel("ABSTAIN reason composition\n (fraction among ABSTAIN)")
    ax.set_xlabel("")
    ax.grid(True, axis="y", linewidth=0.5, alpha=0.3)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    # annotate n_abstain / n_total above bars
    for i, v in enumerate(prop.index.tolist()):
        na = int(meta.loc[v, "n_abstain"])
        nt = int(meta.loc[v, "n_total"])
        ax.annotate(
            f"{na}/{nt}",
            (i, 1.02),
            textcoords="offset points",
            xytext=(0, 0),
            ha="center",
            va="bottom",
            fontsize=max(9, int(args.fontsize) - 6),
        )

    if str(args.title).strip():
        ax.set_title(str(args.title).strip())

    # legend outside (keeps panel clean)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False)

    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
