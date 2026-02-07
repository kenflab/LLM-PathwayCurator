#!/usr/bin/env python3
# paper/scripts/figS2_collections_plot_hallmark_threshold.py
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from llm_pathway_curator.modules import build_term_gene_edges, filter_hub_genes


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


def _ensure_file(p: Path, label: str) -> None:
    if not p.exists():
        _die(f"[figSx_hallmark_thresholds] missing {label}: {p}")
    if not p.is_file():
        _die(f"[figSx_hallmark_thresholds] not a file {label}: {p}")
    if p.stat().st_size == 0:
        _die(f"[figSx_hallmark_thresholds] empty {label}: {p}")


def _read_tsv(p: Path) -> pd.DataFrame:
    _ensure_file(p, "TSV")
    return pd.read_csv(p, sep="\t")


# -------------------------
# Core helpers
# -------------------------
def empirical_percentile(dist: np.ndarray, x: float) -> float:
    dist = dist[np.isfinite(dist)]
    if dist.size == 0:
        return float("nan")
    return float((dist <= x).mean())


def _ecdf_xy(dist: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    d = dist[np.isfinite(dist)].astype(float)
    if d.size == 0:
        return np.array([]), np.array([])
    d = np.sort(d)
    y = np.arange(1, d.size + 1, dtype=float) / float(d.size)
    return d, y


def sample_term_pairs(terms: list[str], *, max_pairs: int = 200_000, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = len(terms)
    if n < 2:
        return np.empty((0, 2), dtype=int)

    total = n * (n - 1) // 2
    if total <= max_pairs:
        pairs = []
        for i in range(n):
            for j in range(i + 1, n):
                pairs.append((i, j))
        return np.array(pairs, dtype=int)

    # rejection sampling, unique pairs
    seen = set()
    pairs = []
    while len(pairs) < int(max_pairs):
        i = int(rng.integers(0, n))
        j = int(rng.integers(0, n))
        if i == j:
            continue
        a, b = (i, j) if i < j else (j, i)
        key = a * n + b
        if key in seen:
            continue
        seen.add(key)
        pairs.append((a, b))
    return np.array(pairs, dtype=int)


def _build_term_to_genes(edges_f: pd.DataFrame) -> dict[str, set[str]]:
    term_to_genes: dict[str, set[str]] = {}
    if edges_f.empty:
        return term_to_genes
    for t, g in edges_f[["term_uid", "gene_id"]].itertuples(index=False):
        ts = str(t).strip()
        gs = str(g).strip()
        if not ts or not gs:
            continue
        term_to_genes.setdefault(ts, set()).add(gs)
    return term_to_genes


def compute_hallmark_distributions(
    evidence_df: pd.DataFrame,
    *,
    term_id_col: str = "term_uid",
    genes_col: str = "evidence_genes",
    max_gene_term_degree_value: int = 200,
    min_shared_genes_value: int = 3,
    jaccard_min_value: float = 0.10,
    hub_degree_for_filter: int = 200,
    pair_sample_max: int = 200_000,
    seed: int = 0,
    include_zero_shared: bool = False,
) -> dict[str, object]:
    # 1) edges (pre-filter): gene term-degree distribution
    edges = build_term_gene_edges(evidence_df, term_id_col=term_id_col, genes_col=genes_col)
    if edges.empty:
        return {"error": "edges empty"}

    if "gene_id" not in edges.columns or "term_uid" not in edges.columns:
        return {"error": "edges missing required columns"}

    gene_deg = edges.groupby("gene_id")["term_uid"].nunique().astype(int).to_numpy()
    pct_hub = empirical_percentile(gene_deg, float(max_gene_term_degree_value))

    # 2) edges_f (post-filter): term-term stats
    edges_f = filter_hub_genes(edges, max_gene_term_degree=int(hub_degree_for_filter))
    term_to_genes = _build_term_to_genes(edges_f)
    terms = sorted(term_to_genes.keys())

    n_terms = int(len(terms))
    n_pairs_total = int(n_terms * (n_terms - 1) // 2)

    pairs = sample_term_pairs(terms, max_pairs=int(pair_sample_max), seed=int(seed))
    n_pairs_sampled = int(len(pairs))

    shared_list: list[float] = []
    jacc_list: list[float] = []

    # If include_zero_shared=False, we keep only pairs with inter>0 (conditioned distribution).
    # If include_zero_shared=True, we include zeros too (unconditional over sampled pairs).
    for i, j in pairs:
        a = terms[int(i)]
        b = terms[int(j)]
        ga = term_to_genes.get(a, set())
        gb = term_to_genes.get(b, set())
        if not ga or not gb:
            continue
        inter = len(ga & gb)
        union = len(ga | gb)
        jac = (inter / union) if union else 0.0

        if include_zero_shared:
            shared_list.append(float(inter))
            jacc_list.append(float(jac))
        else:
            if inter <= 0:
                continue
            shared_list.append(float(inter))
            jacc_list.append(float(jac))

    shared = np.array(shared_list, dtype=float)
    jacc = np.array(jacc_list, dtype=float)

    pct_shared = empirical_percentile(shared, float(min_shared_genes_value))
    pct_jacc = empirical_percentile(jacc, float(jaccard_min_value))

    # shared>0 rate among sampled pairs (computed regardless of include_zero_shared setting)
    # We recompute cheaply using the already-collected data if include_zero_shared=True,
    # otherwise approximate from conditioned list size is not possible. So compute directly.
    # This is still O(n_pairs_sampled) which we already paid.
    shared_pos = 0
    used_for_rate = 0
    for i, j in pairs:
        a = terms[int(i)]
        b = terms[int(j)]
        ga = term_to_genes.get(a, set())
        gb = term_to_genes.get(b, set())
        if not ga or not gb:
            continue
        used_for_rate += 1
        if len(ga & gb) > 0:
            shared_pos += 1
    shared_pos_rate = (shared_pos / used_for_rate) if used_for_rate else float("nan")

    out = {
        "meta": {
            "n_terms": n_terms,
            "n_pairs_total": n_pairs_total,
            "n_pairs_sampled": n_pairs_sampled,
            "n_pairs_rate_used": int(used_for_rate),
            "shared_pos_rate_among_used_pairs": float(shared_pos_rate),
            "include_zero_shared": bool(include_zero_shared),
            "pair_sample_max": int(pair_sample_max),
            "seed": int(seed),
            "hub_degree_for_filter": int(hub_degree_for_filter),
        },
        "gene_term_degree": {
            "n_genes": int(len(gene_deg)),
            "p50": float(np.nanmedian(gene_deg)),
            "p90": float(np.nanpercentile(gene_deg, 90)),
            "p95": float(np.nanpercentile(gene_deg, 95)),
            "p99": float(np.nanpercentile(gene_deg, 99)),
            "value": int(max_gene_term_degree_value),
            "value_percentile": float(pct_hub),
            "dist": gene_deg.astype(float),
        },
        "term_pair_shared_genes": {
            "n_pairs_used": int(shared.size),
            "p50": float(np.nanmedian(shared)) if shared.size else float("nan"),
            "p75": float(np.nanpercentile(shared, 75)) if shared.size else float("nan"),
            "p90": float(np.nanpercentile(shared, 90)) if shared.size else float("nan"),
            "value": int(min_shared_genes_value),
            "value_percentile": float(pct_shared),
            "dist": shared,
        },
        "term_pair_jaccard": {
            "n_pairs_used": int(jacc.size),
            "p50": float(np.nanmedian(jacc)) if jacc.size else float("nan"),
            "p75": float(np.nanpercentile(jacc, 75)) if jacc.size else float("nan"),
            "p90": float(np.nanpercentile(jacc, 90)) if jacc.size else float("nan"),
            "value": float(jaccard_min_value),
            "value_percentile": float(pct_jacc),
            "dist": jacc,
        },
    }
    return out


# -------------------------
# Plotting (ECDF panels)
# -------------------------
def _annotate_meta(ax: plt.Axes, meta: dict[str, object], *, fontsize: int) -> None:
    lines = []
    n_terms = meta.get("n_terms", None)
    n_pairs_total = meta.get("n_pairs_total", None)
    n_pairs_sampled = meta.get("n_pairs_sampled", None)
    rate_used = meta.get("n_pairs_rate_used", None)
    shared_rate = meta.get("shared_pos_rate_among_used_pairs", None)
    include_zero = meta.get("include_zero_shared", None)

    if n_terms is not None:
        lines.append(f"n_terms={int(n_terms)}")
    if n_pairs_total is not None:
        lines.append(f"pairs_total={int(n_pairs_total):,}")
    if n_pairs_sampled is not None:
        lines.append(f"pairs_sampled={int(n_pairs_sampled):,}")
    if rate_used is not None:
        lines.append(f"pairs_used={int(rate_used):,}")
    if shared_rate is not None and np.isfinite(float(shared_rate)):
        lines.append(f"shared>0 rate={float(shared_rate):.3f}")
    if include_zero is not None:
        lines.append(f"zeros included={bool(include_zero)}")

    txt = "\n".join(lines)
    ax.text(
        0.98,
        0.02,
        txt,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=max(9, fontsize - 6),
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 4.0},
    )


def plot_ecdf_panel(
    dist: np.ndarray,
    *,
    out_path: Path,
    title: str,
    xlabel: str,
    vline_value: float,
    vline_label: str,
    meta: dict[str, object] | None,
    fontsize: int,
) -> None:
    x, y = _ecdf_xy(dist)
    if x.size == 0:
        _die(f"[figSx_hallmark_thresholds] no finite values for panel: {title}")

    apply_pub_style(fontsize)
    fig, ax = plt.subplots(figsize=(6.6, 4.8), dpi=300)

    ax.plot(x, y, color="black", linestyle="-", label="Empirical CDF")

    ax.axvline(float(vline_value), color="black", linestyle="--", linewidth=1.4, label=vline_label)

    ax.set_title(str(title))
    ax.set_xlabel(str(xlabel))
    ax.set_ylabel("Empirical CDF")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, axis="both", linewidth=0.5, alpha=0.25)

    if meta is not None:
        _annotate_meta(ax, meta, fontsize=fontsize)

    ax.legend(loc="lower right", frameon=False)
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _write_summary_tables(res: dict[str, object], *, outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)

    # JSON (full, without huge arrays)
    slim = json.loads(json.dumps(res, default=str))
    for k in ["gene_term_degree", "term_pair_shared_genes", "term_pair_jaccard"]:
        if k in slim and isinstance(slim[k], dict) and "dist" in slim[k]:
            slim[k].pop("dist", None)

    with open(outdir / "FigS2_hallmark_threshold_percentiles.json", "w") as f:
        json.dump(slim, f, indent=2)

    # TSV (one row per metric)
    rows = []
    meta = res.get("meta", {})
    for metric_key, label in [
        ("gene_term_degree", "gene_term_degree (pre-filter)"),
        ("term_pair_shared_genes", "term_pair_shared_genes (post-filter)"),
        ("term_pair_jaccard", "term_pair_jaccard (post-filter)"),
    ]:
        d = res.get(metric_key, {})
        if not isinstance(d, dict):
            continue
        rows.append(
            {
                "metric": metric_key,
                "label": label,
                "value": d.get("value", np.nan),
                "value_percentile_ecdf": d.get("value_percentile", np.nan),
                "p50": d.get("p50", np.nan),
                "p75": d.get("p75", np.nan),
                "p90": d.get("p90", np.nan),
                "p95": d.get("p95", np.nan),
                "p99": d.get("p99", np.nan),
                "n": d.get("n_genes", d.get("n_pairs_used", np.nan)),
                "n_terms": meta.get("n_terms", np.nan),
                "n_pairs_total": meta.get("n_pairs_total", np.nan),
                "n_pairs_sampled": meta.get("n_pairs_sampled", np.nan),
                "pairs_used_for_rate": meta.get("n_pairs_rate_used", np.nan),
                "shared_pos_rate": meta.get("shared_pos_rate_among_used_pairs", np.nan),
                "include_zero_shared": meta.get("include_zero_shared", False),
                "hub_degree_for_filter": meta.get("hub_degree_for_filter", np.nan),
                "pair_sample_max": meta.get("pair_sample_max", np.nan),
                "seed": meta.get("seed", np.nan),
            }
        )

    pd.DataFrame(rows).to_csv(
        outdir / "FigS2_hallmark_threshold_percentiles.tsv", sep="\t", index=False
    )


# -------------------------
# Main
# -------------------------
def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Generate Supp Fig Sx panels: ECDFs showing where Hallmark module thresholds sit "
            "within empirical distributions (HALLMARK distilled evidence)."
        )
    )
    ap.add_argument(
        "--distilled", required=True, help="HALLMARK distilled TSV (needs term_uid, evidence_genes)"
    )
    ap.add_argument(
        "--outdir", required=True, help="Output directory for PDFs/PNGs and summary TSV/JSON"
    )
    ap.add_argument(
        "--fmt", default="pdf", choices=["pdf", "png"], help="Output format (default: pdf)"
    )
    ap.add_argument("--fontsize", type=int, default=16)

    ap.add_argument("--term-id-col", default="term_uid")
    ap.add_argument("--genes-col", default="evidence_genes")

    ap.add_argument(
        "--max-gene-term-degree", type=int, default=200, help="Vertical line for hub threshold"
    )
    ap.add_argument(
        "--min-shared-genes", type=int, default=3, help="Vertical line for min shared genes"
    )
    ap.add_argument("--jaccard-min", type=float, default=0.10, help="Vertical line for Jaccard min")

    ap.add_argument(
        "--hub-degree-for-filter",
        type=int,
        default=200,
        help="Hub filter applied before term-pair stats",
    )
    ap.add_argument("--pair-sample-max", type=int, default=200_000)
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument(
        "--include-zero-shared",
        action="store_true",
        help=(
            "Include zero-shared pairs in shared/jaccard ECDFs (unconditional). "
            "Default is conditioned on shared>0."
        ),
    )

    ap.add_argument("--title-A", default="Hallmark: Gene term-degree ECDF (pre-filter)")
    ap.add_argument("--title-B", default="Hallmark: Shared-gene count ECDF (post-filter)")
    ap.add_argument("--title-C", default="Hallmark: Jaccard ECDF (post-filter)")

    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    p_in = Path(args.distilled)
    df = _read_tsv(p_in)

    term_id_col = str(args.term_id_col)
    genes_col = str(args.genes_col)
    for c in [term_id_col, genes_col]:
        if c not in df.columns:
            _die(f"[figSx_hallmark_thresholds] input missing required column: {c}")

    res = compute_hallmark_distributions(
        df,
        term_id_col=term_id_col,
        genes_col=genes_col,
        max_gene_term_degree_value=int(args.max_gene_term_degree),
        min_shared_genes_value=int(args.min_shared_genes),
        jaccard_min_value=float(args.jaccard_min),
        hub_degree_for_filter=int(args.hub_degree_for_filter),
        pair_sample_max=int(args.pair_sample_max),
        seed=int(args.seed),
        include_zero_shared=bool(args.include_zero_shared),
    )

    if "error" in res:
        _die(f"[figSx_hallmark_thresholds] {res['error']}")

    # Write summary TSV/JSON
    _write_summary_tables(res, outdir=outdir)

    fmt = str(args.fmt).strip().lower()
    fs = int(args.fontsize)

    meta = res.get("meta", {})
    if not isinstance(meta, dict):
        meta = {}

    # Panel A
    dA = res.get("gene_term_degree", {})
    if not isinstance(dA, dict) or "dist" not in dA:
        _die("[figSx_hallmark_thresholds] missing gene_term_degree.dist")
    plot_ecdf_panel(
        np.asarray(dA["dist"], dtype=float),
        out_path=outdir / f"FigSx_HallmarkThresholds_PanelA_gene_degree.{fmt}",
        title=str(args.title_A),
        xlabel="Gene term-degree (#terms per gene)",
        vline_value=float(args.max_gene_term_degree),
        vline_label=f"threshold={int(args.max_gene_term_degree)}",
        meta=meta,
        fontsize=fs,
    )

    # Panel B
    dB = res.get("term_pair_shared_genes", {})
    if not isinstance(dB, dict) or "dist" not in dB:
        _die("[figSx_hallmark_thresholds] missing term_pair_shared_genes.dist")
    plot_ecdf_panel(
        np.asarray(dB["dist"], dtype=float),
        out_path=outdir / f"FigSx_HallmarkThresholds_PanelB_shared_genes.{fmt}",
        title=str(args.title_B),
        xlabel="Shared genes between term pairs",
        vline_value=float(args.min_shared_genes),
        vline_label=f"threshold={int(args.min_shared_genes)}",
        meta=meta,
        fontsize=fs,
    )

    # Panel C
    dC = res.get("term_pair_jaccard", {})
    if not isinstance(dC, dict) or "dist" not in dC:
        _die("[figSx_hallmark_thresholds] missing term_pair_jaccard.dist")
    plot_ecdf_panel(
        np.asarray(dC["dist"], dtype=float),
        out_path=outdir / f"FigSx_HallmarkThresholds_PanelC_jaccard.{fmt}",
        title=str(args.title_C),
        xlabel="Jaccard similarity between term pairs",
        vline_value=float(args.jaccard_min),
        vline_label=f"threshold={float(args.jaccard_min):.2f}",
        meta=meta,
        fontsize=fs,
    )

    print("[figSx_hallmark_thresholds] WROTE:")
    for p in sorted(outdir.glob(f"FigSx_HallmarkThresholds_Panel*.{fmt}")):
        print(" ", p)
    print(" ", outdir / "FigSx_hallmark_threshold_percentiles.tsv")
    print(" ", outdir / "FigSx_hallmark_threshold_percentiles.json")


if __name__ == "__main__":
    main()
