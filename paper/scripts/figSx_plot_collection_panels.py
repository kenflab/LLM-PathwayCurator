#!/usr/bin/env python3
# paper/scripts/figSx_plot_collection_panels.py
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# -------------------------
# Style (paper-consistent, minimal)
# -------------------------
def apply_pub_style(fontsize: int = 16) -> None:
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
            "lines.linewidth": 1.8,
        }
    )


def _die(msg: str) -> None:
    raise SystemExit(msg)


def _canon(s: str) -> str:
    return str(s or "").strip().lower()


def _parse_csv(s: str) -> list[str]:
    return [x.strip() for x in str(s).split(",") if x.strip()]


def _ensure_file(p: Path, label: str) -> None:
    if not p.exists():
        _die(f"[figSx_plot] missing {label}: {p}")
    if not p.is_file():
        _die(f"[figSx_plot] not a file {label}: {p}")
    if p.stat().st_size == 0:
        _die(f"[figSx_plot] empty {label}: {p}")


def _read_tsv(p: Path) -> pd.DataFrame:
    _ensure_file(p, "TSV")
    return pd.read_csv(p, sep="\t")


def _filter_df(
    df: pd.DataFrame,
    *,
    conditions: list[str] | None,
    variants: list[str] | None,
    gate_modes: list[str] | None,
    taus: list[float] | None,
) -> pd.DataFrame:
    out = df.copy()

    def has(col: str) -> bool:
        return col in out.columns

    if conditions and has("condition"):
        want = set([c.strip().upper() for c in conditions])
        out = out[out["condition"].astype(str).str.upper().isin(want)]

    if variants and has("variant"):
        want = set([v.strip() for v in variants])
        out = out[out["variant"].astype(str).isin(want)]

    if gate_modes and has("gate_mode"):
        want = set([g.strip() for g in gate_modes])
        out = out[out["gate_mode"].astype(str).isin(want)]

    if taus and has("tau"):
        # robust float compare (allow string formatting)
        out["tau"] = pd.to_numeric(out["tau"], errors="coerce")
        want = set([float(t) for t in taus])
        out = out[out["tau"].round(6).isin({round(x, 6) for x in want})]

    return out


def _infer_collection_col(df: pd.DataFrame) -> str:
    for c in ["collection", "gene_set_collection", "msig_collection", "collection_name"]:
        if c in df.columns:
            return c
    _die(
        "[figSx_plot] cannot find a collection column (expected one of: "
        "collection, gene_set_collection, msig_collection, collection_name)"
    )


def _nice_collection_label(x: str) -> str:
    # Keep “Less is more”: don’t over-format.
    return str(x).replace("_", " ").strip()


def _infer_tumor_col(df: pd.DataFrame) -> str | None:
    candidates = [
        "conditiondiseasetumor_type",
        "tcga_cancer",
        "tcga_cancer_type",
        "cancer",
        "cancer_type",
        "cohort",
        "collection",
    ]
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _make_palette(keys: list[str]) -> dict[str, tuple[float, float, float]]:
    cmap = plt.get_cmap("tab20")
    colors = [cmap(i % cmap.N)[:3] for i in range(len(keys))]
    return {k: colors[i] for i, k in enumerate(keys)}


# -------------------------
# Panel A: module survival distribution (by collection)
# -------------------------


def plot_panel_A(
    module_survival_long: pd.DataFrame,
    *,
    out_path: Path,
    conditions: list[str] | None,
    variants: list[str] | None,
    gate_modes: list[str] | None,
    taus: list[float] | None,
    fontsize: int,
    title: str,
    dot_size: float = 10.0,
    dot_alpha: float = 0.55,
    jitter: float = 0.12,
    box_fill_alpha: float = 0.22,
) -> None:
    df = module_survival_long.copy()

    coll_col = _infer_collection_col(df)
    if "module_survival" not in df.columns:
        _die("[figSx_plot] module_survival_long.tsv missing column: module_survival")

    df = _filter_df(df, conditions=conditions, variants=variants, gate_modes=gate_modes, taus=taus)
    df["module_survival"] = pd.to_numeric(df["module_survival"], errors="coerce")
    df = df.dropna(subset=["module_survival", coll_col])

    if df.empty:
        _die("[figSx_plot] Panel A: no rows after filtering")

    # ---- group data ----
    colls = [str(c) for c in sorted(df[coll_col].astype(str).unique())]
    labels = [_nice_collection_label(c) for c in colls]
    groups = []
    for c in colls:
        vals = df.loc[df[coll_col].astype(str) == c, "module_survival"].to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        groups.append(vals)

    if not any(len(g) for g in groups):
        _die("[figSx_plot] Panel A: no valid module_survival values to plot")

    # ---- colors: fill by collection ----
    cmap = plt.get_cmap("tab10")
    fill = {c: cmap(i % cmap.N)[:3] for i, c in enumerate(colls)}

    apply_pub_style(fontsize)
    fig, ax = plt.subplots(figsize=(7.1, 4.8), dpi=300)

    bp = ax.boxplot(
        groups,
        tick_labels=labels,
        showfliers=False,
        patch_artist=True,
        widths=0.6,
        # visible: box outline + median (both black)
        boxprops={"color": "black", "linewidth": 1.1},
        medianprops={"color": "black", "linewidth": 1.4},
        # hidden: whiskers + caps
        whiskerprops={"linewidth": 0.0, "alpha": 0.0},
        capprops={"linewidth": 0.0, "alpha": 0.0},
    )

    # Fill each box by collection (outline stays black)
    for c, b in zip(colls, bp["boxes"], strict=True):
        r, g, bcol = fill[c]
        b.set_facecolor((r, g, bcol, box_fill_alpha))
        b.set_edgecolor("black")

    # Dots: black only
    rng = np.random.default_rng(0)
    for i, c in enumerate(colls, start=1):
        sub = df[df[coll_col].astype(str) == c]
        y = sub["module_survival"].to_numpy(dtype=float)
        y = y[np.isfinite(y)]
        if len(y) == 0:
            continue
        x = i + rng.uniform(-jitter, jitter, size=len(y))
        ax.scatter(x, y, s=dot_size, alpha=dot_alpha, linewidths=0, color="black")

    ax.set_ylim(-0.02, 1.02)

    ax.set_ylabel("Module survival (0–1)")
    ax.grid(True, axis="y", linewidth=0.5, alpha=0.25)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


# -------------------------
# Panel B: module structure stats (by collection)
# -------------------------
def _pick_proxy_col(df: pd.DataFrame) -> str:
    # Prefer a single “hub/bridge proxy” column if present.
    # Adjust this list if your collector uses different names.
    candidates = [
        "bridge_gene_rate",
        "bridge_rate",
        "hub_gene_rate",
        "hub_rate",
        "hubness",
        "hub_filter_n_hubs",
        "hub_filter_frac_hubs",
        "hub_filter_max_gene_term_degree",
        "module_jaccard_min",
    ]
    for c in candidates:
        if c in df.columns:
            return c
    # If nothing found, we’ll return empty and plot only the 2 core metrics.
    return ""


def plot_panel_B(
    collections_summary_wide: pd.DataFrame,
    *,
    out_path: Path,
    conditions: list[str] | None,
    variants: list[str] | None,
    gate_modes: list[str] | None,
    taus: list[float] | None,
    fontsize: int,
    title: str,
) -> None:
    df = collections_summary_wide.copy()
    coll_col = _infer_collection_col(df)

    df = _filter_df(df, conditions=conditions, variants=variants, gate_modes=gate_modes, taus=taus)
    if df.empty:
        _die("[figSx_plot] Panel B: no rows after filtering")

    required_any = ["n_modules", "median_module_genes", "median_module_terms"]
    missing = [c for c in required_any if c not in df.columns]
    if missing:
        _die(
            f"[figSx_plot] collections_summary_wide.tsv missing required columns "
            f" for Panel B: {missing}"
        )

    # numeric coercion
    for c in ["n_modules", "median_module_genes", "median_module_terms"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    proxy_col = _pick_proxy_col(df)
    if proxy_col:
        df[proxy_col] = pd.to_numeric(df[proxy_col], errors="coerce")

    # Aggregate across filtered slice → one row per collection (median across runs/conditions)
    agg_cols = ["n_modules", "median_module_genes", "median_module_terms"] + (
        [proxy_col] if proxy_col else []
    )
    g = df.groupby(coll_col, as_index=False)[agg_cols].median(numeric_only=True)
    if g.empty:
        _die("[figSx_plot] Panel B: empty aggregation result")

    g = g.sort_values(coll_col, kind="mergesort")
    labels = [_nice_collection_label(x) for x in g[coll_col].astype(str).tolist()]

    apply_pub_style(fontsize)

    fig = plt.figure(figsize=(9.6, 4.8), dpi=300)
    ax1 = fig.add_subplot(111)
    ax2 = ax1.twinx()

    x = np.arange(len(g))
    width = 0.28

    # Left axis: genes/terms (large scale)
    ax1.bar(
        x - width / 2, g["median_module_genes"].to_numpy(), width=width, label="Genes per module"
    )
    ax1.bar(
        x + width / 2, g["median_module_terms"].to_numpy(), width=width, label="Terms per module"
    )
    ax1.set_ylabel("Per-module counts (median)")
    ax1.grid(True, axis="y", linewidth=0.5, alpha=0.3)

    # Right axis: n_modules (small scale, previously looked like 0)
    # Plot as points+line to avoid bar clutter
    ax2.plot(
        x,
        g["n_modules"].to_numpy(dtype=float),
        color="black",
        marker="o",
        markersize=12,
        linestyle="-",
        label="Number of modules",
    )
    ax2.set_ylabel("Number of modules\n (median)")

    # Optional proxy shown on left axis as points (keeps 2-axis simple)
    if proxy_col:
        ax1.plot(
            x,
            g[proxy_col].to_numpy(dtype=float),
            marker="s",
            linestyle="--",
            label=f"{proxy_col} (median)",
        )

    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=30, ha="right")

    # Merge legends from both axes
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="lower left", bbox_to_anchor=(0.45, 0.03), frameon=False)

    fig.tight_layout(rect=[0.0, 0.0, 0.80, 1.0])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


# -------------------------
# Panel C: PASS/ABSTAIN/FAIL distribution (by collection) – 100% stacked
# -------------------------
def plot_panel_C(
    collections_summary_wide: pd.DataFrame,
    *,
    out_path: Path,
    conditions: list[str] | None,
    variants: list[str] | None,
    gate_modes: list[str] | None,
    taus: list[float] | None,
    fontsize: int,
    title: str,
) -> None:
    df = collections_summary_wide.copy()
    coll_col = _infer_collection_col(df)
    # Accept collector naming
    if "n_total" not in df.columns and "n_total_claims" in df.columns:
        df["n_total"] = df["n_total_claims"]

    df = _filter_df(df, conditions=conditions, variants=variants, gate_modes=gate_modes, taus=taus)
    if df.empty:
        _die("[figSx_plot] Panel C: no rows after filtering")

    needed = ["n_total", "n_pass", "n_abstain", "n_fail"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        _die(
            f"[figSx_plot] collections_summary_wide.tsv missing required columns "
            f"for Panel C: {missing}"
        )

    for c in needed:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Aggregate across filtered slice → sum counts per collection
    g = df.groupby(coll_col, as_index=False)[needed].sum(numeric_only=True)
    g = g.sort_values(coll_col, kind="mergesort")

    # Proportions
    denom = g["n_total"].replace(0, np.nan).to_numpy(dtype=float)
    p_pass = g["n_pass"].to_numpy(dtype=float) / denom
    p_abst = g["n_abstain"].to_numpy(dtype=float) / denom
    p_fail = g["n_fail"].to_numpy(dtype=float) / denom

    labels = [_nice_collection_label(x) for x in g[coll_col].astype(str).tolist()]

    apply_pub_style(fontsize)

    # --- Outcome colors (user-specified) ---
    OUTCOME_COLORS = {
        "PASS": "#4C72B0",
        "ABSTAIN": "#E6C229",
        "FAIL": "#C44E52",
    }

    fig = plt.figure(figsize=(9.6, 4.8), dpi=300)
    ax = fig.add_subplot(111)

    x = np.arange(len(g))
    ax.bar(
        x,
        p_pass,
        label="PASS",
        color=OUTCOME_COLORS["PASS"],
        edgecolor="white",
        linewidth=0.6,
    )
    ax.bar(
        x,
        p_abst,
        bottom=p_pass,
        label="ABSTAIN",
        color=OUTCOME_COLORS["ABSTAIN"],
        edgecolor="white",
        linewidth=0.6,
    )
    ax.bar(
        x,
        p_fail,
        bottom=p_pass + p_abst,
        label="FAIL",
        color=OUTCOME_COLORS["FAIL"],
        edgecolor="white",
        linewidth=0.6,
    )

    ax.set_ylim(0, 1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("Fraction of claims")
    ax.grid(True, axis="y", linewidth=0.5, alpha=0.3)

    # annotate total counts (and flag non-uniform denominators)
    totals = g["n_total"].to_numpy(dtype=float)
    uniq = sorted(set([int(t) for t in totals if np.isfinite(t)]))
    varying = len(uniq) > 1
    for i, nt in enumerate(totals):
        if not np.isfinite(nt):
            continue
        ax.annotate(
            f"n={int(nt)}",
            (i, 1.02),
            textcoords="offset points",
            xytext=(0, 0),
            ha="center",
            va="bottom",
            fontsize=max(9, fontsize - 6),
        )

    if varying:
        ax.text(
            0.0,
            -0.22,
            "Note: n_total varies across collections (proposal exhaustion / pre-audit filtering).",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=max(9, fontsize - 6),
        )

    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False)

    fig.tight_layout(rect=[0.0, 0.0, 0.80, 1.0])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


# -------------------------
# Panel D: Top ABSTAIN reasons (primary), optional FAIL reasons (secondary)
# -------------------------
def _infer_reason_cols(df: pd.DataFrame) -> tuple[str, str]:
    """
    Returns (status_col, reason_col).

    Supports two formats:
      (A) collector output: reason_type + reason
      (B) legacy: status/audit_status/outcome + reason/reason_code/...
    """
    # Prefer collector format
    if "reason_type" in df.columns:
        status_col = "reason_type"
    else:
        status_candidates = ["status", "audit_status", "outcome"]
        status_col = ""
        for c in status_candidates:
            if c in df.columns:
                status_col = c
                break
        if not status_col:
            _die(
                "[figSx_plot] audit_reasons_long.tsv missing a status column (expected: "
                "reason_type or status/audit_status/outcome)"
            )

    # Reason column
    if "reason" in df.columns:
        reason_col = "reason"
    else:
        reason_candidates = [
            "reason_code",
            "audit_reason",
            "abstain_reason",
            "fail_reason",
            "reason2",
        ]
        reason_col = ""
        for c in reason_candidates:
            if c in df.columns:
                reason_col = c
                break
        if not reason_col:
            _die(
                "[figSx_plot] audit_reasons_long.tsv missing a reason column (expected: "
                "reason or reason_code/audit_reason/...)"
            )

    return status_col, reason_col


def plot_panel_D(
    audit_reasons_long: pd.DataFrame,
    *,
    out_path: Path,
    conditions: list[str] | None,
    variants: list[str] | None,
    gate_modes: list[str] | None,
    taus: list[float] | None,
    fontsize: int,
    title: str,
    topk: int = 5,
    include_fail_if_any: bool = True,
) -> None:
    df = audit_reasons_long.copy()
    coll_col = _infer_collection_col(df)

    df = _filter_df(df, conditions=conditions, variants=variants, gate_modes=gate_modes, taus=taus)
    if df.empty:
        _die("[figSx_plot] Panel D: no rows after filtering")

    status_col, reason_col = _infer_reason_cols(df)

    # Count column
    if "count" in df.columns:
        df["count"] = pd.to_numeric(df["count"], errors="coerce").fillna(0).astype(int)
    else:
        df["count"] = 1

    # Normalize status (ABSTAIN/FAIL)
    df[status_col] = df[status_col].astype(str).str.upper().str.strip()
    df[reason_col] = df[reason_col].fillna("NA").astype(str)

    # Primary: ABSTAIN
    abd = df[df[status_col] == "ABSTAIN"].copy()
    if abd.empty:
        _die("[figSx_plot] Panel D: no ABSTAIN rows (after filtering)")

    top = (
        abd.groupby(reason_col, as_index=False)["count"]
        .sum()
        .sort_values("count", ascending=False)
        .head(int(topk))[reason_col]
        .tolist()
    )
    abd["reason2"] = abd[reason_col].where(abd[reason_col].isin(set(top)), other="Other")

    piv_a = (
        abd.groupby([coll_col, "reason2"], as_index=False)["count"]
        .sum()
        .pivot(index=coll_col, columns="reason2", values="count")
        .fillna(0.0)
    )
    denom_a = piv_a.sum(axis=1).replace(0, np.nan)
    prop_a = piv_a.div(denom_a, axis=0).fillna(0.0)

    # Secondary: FAIL
    fail = df[df[status_col] == "FAIL"].copy()
    has_fail = (fail["count"].sum() if not fail.empty else 0) > 0
    do_fail = bool(include_fail_if_any and has_fail)

    if do_fail:
        top_f = (
            fail.groupby(reason_col, as_index=False)["count"]
            .sum()
            .sort_values("count", ascending=False)
            .head(int(topk))[reason_col]
            .tolist()
        )
        fail["reason2"] = fail[reason_col].where(fail[reason_col].isin(set(top_f)), other="Other")

        piv_f = (
            fail.groupby([coll_col, "reason2"], as_index=False)["count"]
            .sum()
            .pivot(index=coll_col, columns="reason2", values="count")
            .fillna(0.0)
        )
        denom_f = piv_f.sum(axis=1).replace(0, np.nan)
        prop_f = piv_f.div(denom_f, axis=0).fillna(0.0)
    else:
        prop_f = pd.DataFrame()

    # Align collections
    collections = sorted(set(prop_a.index.astype(str).tolist()))
    prop_a = prop_a.reindex(collections).fillna(0.0)
    if do_fail:
        prop_f = prop_f.reindex(collections).fillna(0.0)

    labels = [_nice_collection_label(x) for x in collections]

    apply_pub_style(fontsize)

    fig = plt.figure(figsize=(11.2, 4.8), dpi=300)
    ax = fig.add_subplot(111)

    x = np.arange(len(collections))
    w = 0.38 if do_fail else 0.65

    bottom = np.zeros(len(collections), dtype=float)
    for col in prop_a.columns.tolist():
        y = prop_a[col].to_numpy(dtype=float)
        ax.bar(x - (w / 2 if do_fail else 0.0), y, width=w, bottom=bottom, label=f"{col}")
        bottom = bottom + y

    if do_fail:
        bottom = np.zeros(len(collections), dtype=float)
        for col in prop_f.columns.tolist():
            y = prop_f[col].to_numpy(dtype=float)
            ax.bar(x + (w / 2), y, width=w, bottom=bottom, label=f"FAIL: {col}")
            bottom = bottom + y

    ax.set_ylim(0, 1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("ABSTAIN reason composition\n (fraction among ABSTAIN)")
    ax.grid(True, axis="y", linewidth=0.5, alpha=0.3)

    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False)
    fig.tight_layout(rect=[0.0, 0.0, 0.78, 1.0])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


# -------------------------
# Main
# -------------------------
def main() -> None:
    ap = argparse.ArgumentParser(
        description="Plot Supp Fig Sx panels (A–D) from collection_metrics TSVs."
    )
    ap.add_argument(
        "--collections-summary-wide", required=True, help="collections_summary_wide.tsv"
    )
    ap.add_argument("--module-survival-long", required=True, help="module_survival_long.tsv")
    ap.add_argument("--audit-reasons-long", required=True, help="audit_reasons_long.tsv")
    ap.add_argument("--outdir", required=True, help="output directory for PDFs/PNGs")
    ap.add_argument(
        "--fmt", default="pdf", choices=["pdf", "png"], help="output format (default: pdf)"
    )
    ap.add_argument("--fontsize", type=int, default=16)

    ap.add_argument("--conditions", default="ALL", help="Comma list (e.g., HNSC,LUAD) or ALL")
    ap.add_argument("--variants", default="ALL", help="Comma list (e.g., ours) or ALL")
    ap.add_argument("--gate-modes", default="ALL", help="Comma list (e.g., hard) or ALL")
    ap.add_argument("--taus", default="ALL", help="Comma list (e.g., 0.8) or ALL")

    ap.add_argument("--topk-reasons", type=int, default=5)
    ap.add_argument(
        "--include-fail-reasons",
        action="store_true",
        help="Also plot FAIL reasons (if any) alongside ABSTAIN reasons",
    )

    ap.add_argument("--title-A", default="Module survival by collection")
    ap.add_argument(
        "--title-B", default="Module structure metrics by collection (median across runs)"
    )
    ap.add_argument("--title-C", default="Audit outcomes by collection (100% stacked)")
    ap.add_argument("--title-D", default="Top audit reasons by collection (ABSTAIN primary)")

    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Filters
    conditions = (
        None
        if str(args.conditions).strip().upper() == "ALL"
        else [c.upper() for c in _parse_csv(args.conditions)]
    )
    variants = None if str(args.variants).strip().upper() == "ALL" else _parse_csv(args.variants)
    gate_modes = (
        None if str(args.gate_modes).strip().upper() == "ALL" else _parse_csv(args.gate_modes)
    )
    taus = None
    if str(args.taus).strip().upper() != "ALL":
        taus = [float(x) for x in _parse_csv(args.taus)]

    # Load
    p_wide = Path(args.collections_summary_wide)
    p_surv = Path(args.module_survival_long)
    p_reas = Path(args.audit_reasons_long)

    wide = _read_tsv(p_wide)
    surv = _read_tsv(p_surv)
    reas = _read_tsv(p_reas)

    fmt = str(args.fmt).strip().lower()
    fs = int(args.fontsize)

    # Panel A
    plot_panel_A(
        surv,
        out_path=outdir / f"FigSx_PanelA_module_survival.{fmt}",
        conditions=conditions,
        variants=variants,
        gate_modes=gate_modes,
        taus=taus,
        fontsize=fs,
        title=str(args.title_A),
    )

    # Panel B
    plot_panel_B(
        wide,
        out_path=outdir / f"FigSx_PanelB_module_stats.{fmt}",
        conditions=conditions,
        variants=variants,
        gate_modes=gate_modes,
        taus=taus,
        fontsize=fs,
        title=str(args.title_B),
    )

    # Panel C
    plot_panel_C(
        wide,
        out_path=outdir / f"FigSx_PanelC_outcomes.{fmt}",
        conditions=conditions,
        variants=variants,
        gate_modes=gate_modes,
        taus=taus,
        fontsize=fs,
        title=str(args.title_C),
    )

    # Panel D
    plot_panel_D(
        reas,
        out_path=outdir / f"FigSx_PanelD_top_reasons.{fmt}",
        conditions=conditions,
        variants=variants,
        gate_modes=gate_modes,
        taus=taus,
        fontsize=fs,
        title=str(args.title_D),
        topk=int(args.topk_reasons),
        include_fail_if_any=bool(args.include_fail_reasons),
    )

    print("[figSx_plot] WROTE:")
    for p in sorted(outdir.glob(f"FigSx_Panel*.{fmt}")):
        print(" ", p)


if __name__ == "__main__":
    main()
