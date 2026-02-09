#!/usr/bin/env python3
# paper/scripts/figS2_export_panel_source_csv.py
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


# -------------------------
# Small helpers (minimal, deterministic)
# -------------------------
def _die(msg: str) -> None:
    raise SystemExit(msg)


def _parse_csv(s: str) -> list[str]:
    return [x.strip() for x in str(s).split(",") if x.strip()]


def _ensure_file(p: Path, label: str) -> None:
    if not p.exists():
        _die(f"[figS2_export] missing {label}: {p}")
    if not p.is_file():
        _die(f"[figS2_export] not a file {label}: {p}")
    if p.stat().st_size == 0:
        _die(f"[figS2_export] empty {label}: {p}")


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
    """
    Match the plot script behavior:
    - condition: case-insensitive (upper compare)
    - variant/gate_mode: exact string compare after stripping
    - tau: numeric compare (rounded to 6 decimals)
    - Missing columns => ignore that filter dimension
    """
    out = df.copy()

    def has(col: str) -> bool:
        return col in out.columns

    if conditions and has("condition"):
        want = {c.strip().upper() for c in conditions}
        out = out[out["condition"].astype(str).str.upper().isin(want)]

    if variants and has("variant"):
        want = {v.strip() for v in variants}
        out = out[out["variant"].astype(str).isin(want)]

    if gate_modes and has("gate_mode"):
        want = {g.strip() for g in gate_modes}
        out = out[out["gate_mode"].astype(str).isin(want)]

    if taus and has("tau"):
        out["tau"] = pd.to_numeric(out["tau"], errors="coerce")
        want = {round(float(t), 6) for t in taus}
        out = out[out["tau"].round(6).isin(want)]

    return out


def _infer_collection_col(df: pd.DataFrame) -> str:
    for c in ["collection", "gene_set_collection", "msig_collection", "collection_name"]:
        if c in df.columns:
            return c
    _die(
        "[figS2_export] cannot find a collection column (expected one of: "
        "collection, gene_set_collection, msig_collection, collection_name)"
    )


def _pick_proxy_col(df: pd.DataFrame) -> str:
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
    return ""


def _infer_reason_cols(df: pd.DataFrame) -> tuple[str, str]:
    # Status column
    if "reason_type" in df.columns:
        status_col = "reason_type"
    else:
        for c in ["status", "audit_status", "outcome"]:
            if c in df.columns:
                status_col = c
                break
        else:
            _die(
                "[figS2_export] audit_reasons_long.tsv missing a status column "
                "(expected: reason_type or status/audit_status/outcome)"
            )

    # Reason column
    if "reason" in df.columns:
        reason_col = "reason"
    else:
        for c in ["reason_code", "audit_reason", "abstain_reason", "fail_reason", "reason2"]:
            if c in df.columns:
                reason_col = c
                break
        else:
            _die(
                "[figS2_export] audit_reasons_long.tsv missing a reason column "
                "(expected: reason or reason_code/audit_reason/...)"
            )

    return status_col, reason_col


def _order_collections(values: list[str]) -> list[str]:
    """
    Paper-stable ordering (optional). Unknown labels go to the end (lexicographic).
    """
    preferred = ["Hallmark", "GO", "Reactome", "KEGG", "Other"]
    vset = set(values)
    known = [v for v in preferred if v in vset]
    rest = sorted([v for v in values if v not in set(preferred)])
    return known + rest


def _write_csv(df: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)


# -------------------------
# Panel b (EDFig2b): outcome composition per collection
# -------------------------
def build_edfig2b_outcomes_by_collection(
    collections_summary_wide: pd.DataFrame,
    *,
    conditions: list[str] | None,
    variants: list[str] | None,
    gate_modes: list[str] | None,
    taus: list[float] | None,
    strict_nonempty: bool,
) -> pd.DataFrame:
    df = collections_summary_wide.copy()
    coll_col = _infer_collection_col(df)

    # Accept collector legacy naming
    if "n_total" not in df.columns and "n_total_claims" in df.columns:
        df["n_total"] = df["n_total_claims"]

    df = _filter_df(df, conditions=conditions, variants=variants, gate_modes=gate_modes, taus=taus)
    if df.empty and strict_nonempty:
        _die("[figS2_export] EDFig2c: no rows after filtering")

    needed = ["n_total", "n_pass", "n_abstain", "n_fail"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        _die(f"[figS2_export] EDFig2b missing required columns: {missing}")

    for c in needed:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    g = df.groupby(coll_col, as_index=False)[needed].sum(numeric_only=True)

    if g.empty and strict_nonempty:
        _die("[figS2_export] EDFig2b: empty aggregation result")

    g[coll_col] = g[coll_col].astype(str)
    ordered = _order_collections(sorted(g[coll_col].unique().tolist()))
    g[coll_col] = pd.Categorical(g[coll_col], categories=ordered, ordered=True)
    g = g.sort_values(coll_col, kind="mergesort").reset_index(drop=True)

    denom = g["n_total"].replace(0, np.nan).to_numpy(dtype=float)
    out = pd.DataFrame(
        {
            "collection": g[coll_col].astype(str).tolist(),
            "n_total": g["n_total"].astype(float),
            "n_pass": g["n_pass"].astype(float),
            "n_abstain": g["n_abstain"].astype(float),
            "n_fail": g["n_fail"].astype(float),
            "frac_pass": g["n_pass"].to_numpy(dtype=float) / denom,
            "frac_abstain": g["n_abstain"].to_numpy(dtype=float) / denom,
            "frac_fail": g["n_fail"].to_numpy(dtype=float) / denom,
        }
    )
    return out


# -------------------------
# Panel C (EDFig2c): median module stats per collection
# -------------------------
def build_edfig2c_module_stats(
    collections_summary_wide: pd.DataFrame,
    *,
    conditions: list[str] | None,
    variants: list[str] | None,
    gate_modes: list[str] | None,
    taus: list[float] | None,
    strict_nonempty: bool,
) -> pd.DataFrame:
    df = collections_summary_wide.copy()
    coll_col = _infer_collection_col(df)

    df = _filter_df(df, conditions=conditions, variants=variants, gate_modes=gate_modes, taus=taus)
    if df.empty and strict_nonempty:
        _die("[figS2_export] EDFig2c: no rows after filtering")

    required = ["n_modules", "median_module_genes", "median_module_terms"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        _die(f"[figS2_export] EDFig2c: missing required columns: {missing}")

    for c in required:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    proxy_col = _pick_proxy_col(df)
    if proxy_col:
        df[proxy_col] = pd.to_numeric(df[proxy_col], errors="coerce")

    agg_cols = required + ([proxy_col] if proxy_col else [])
    g = df.groupby(coll_col, as_index=False)[agg_cols].median(numeric_only=True)

    if g.empty and strict_nonempty:
        _die("[figS2_export] EDFig2c: empty aggregation result")

    g[coll_col] = g[coll_col].astype(str)
    ordered = _order_collections(sorted(g[coll_col].unique().tolist()))
    g[coll_col] = pd.Categorical(g[coll_col], categories=ordered, ordered=True)
    g = g.sort_values(coll_col, kind="mergesort").reset_index(drop=True)

    out = g.rename(
        columns={
            coll_col: "collection",
            "median_module_genes": "median_genes_per_module",
            "median_module_terms": "median_terms_per_module",
            "n_modules": "n_modules_median",
        }
    )
    if proxy_col and proxy_col in out.columns:
        out = out.rename(columns={proxy_col: f"{proxy_col}_median"})

    cols = ["collection", "median_genes_per_module", "median_terms_per_module", "n_modules_median"]
    extra = [c for c in out.columns if c not in cols]
    return out[cols + extra].copy()


# -------------------------
# Panel D (EDFig2d): ABSTAIN reason composition per collection (topK + Other)
# -------------------------
def build_edfig2d_abstain_reasons_by_collection(
    audit_reasons_long: pd.DataFrame,
    *,
    conditions: list[str] | None,
    variants: list[str] | None,
    gate_modes: list[str] | None,
    taus: list[float] | None,
    topk: int,
    strict_nonempty: bool,
    include_fail_reasons: bool,
) -> pd.DataFrame:
    df = audit_reasons_long.copy()
    coll_col = _infer_collection_col(df)

    df = _filter_df(df, conditions=conditions, variants=variants, gate_modes=gate_modes, taus=taus)
    if df.empty and strict_nonempty:
        _die("[figS2_export] EDFig2d: no rows after filtering")

    status_col, reason_col = _infer_reason_cols(df)

    # Count column (collector typically provides "count")
    if "count" in df.columns:
        df["count"] = pd.to_numeric(df["count"], errors="coerce").fillna(0).astype(int)
    else:
        df["count"] = 1

    df[status_col] = df[status_col].astype(str).str.upper().str.strip()
    df[reason_col] = df[reason_col].fillna("NA").astype(str)
    df[coll_col] = df[coll_col].astype(str)

    parts: list[pd.DataFrame] = []

    def _collapse_topk(sub: pd.DataFrame, label: str) -> pd.DataFrame:
        top = (
            sub.groupby(reason_col, as_index=False)["count"]
            .sum()
            .sort_values("count", ascending=False)
            .head(int(topk))[reason_col]
            .tolist()
        )
        sub = sub.copy()
        sub["reason2"] = sub[reason_col].where(sub[reason_col].isin(set(top)), other="Other")

        g = (
            sub.groupby([coll_col, "reason2"], as_index=False)["count"]
            .sum()
            .rename(columns={"reason2": "reason"})
        )
        tot = (
            g.groupby(coll_col, as_index=False)["count"]
            .sum()
            .rename(columns={"count": "n_type_total"})
        )
        g = g.merge(tot, on=coll_col, how="left")
        g["fraction_within_type"] = g["count"] / g["n_type_total"].replace(0, np.nan)
        g.insert(0, "reason_type", label)
        return g

    abd = df[df[status_col] == "ABSTAIN"].copy()
    if abd.empty and strict_nonempty:
        _die("[figS2_export] EDFig2d: no ABSTAIN rows (after filtering)")
    if not abd.empty:
        parts.append(_collapse_topk(abd, "ABSTAIN"))

    if include_fail_reasons:
        fail = df[df[status_col] == "FAIL"].copy()
        if not fail.empty and int(fail["count"].sum()) > 0:
            parts.append(_collapse_topk(fail, "FAIL"))

    if not parts:
        # strict_nonempty already handled above; here we allow empty if not strict
        return pd.DataFrame(
            columns=["reason_type", "collection", "reason", "count", "fraction_within_type"]
        )

    out = pd.concat(parts, ignore_index=True)

    ordered = _order_collections(sorted(out[coll_col].astype(str).unique().tolist()))
    out[coll_col] = pd.Categorical(out[coll_col].astype(str), categories=ordered, ordered=True)
    out = out.sort_values(
        ["reason_type", coll_col, "count", "reason"],
        ascending=[True, True, False, True],
        kind="mergesort",
    ).reset_index(drop=True)

    out = out.rename(columns={coll_col: "collection"})
    return out[["reason_type", "collection", "reason", "count", "fraction_within_type"]].copy()


# -------------------------
# Main
# -------------------------
def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Export EDFig2b/c/d source-data CSVs from collection_metrics TSVs (no plotting)."
        )
    )
    ap.add_argument(
        "--collections-summary-wide", required=True, help="collections_summary_wide.tsv"
    )
    ap.add_argument("--audit-reasons-long", required=True, help="audit_reasons_long.tsv")
    ap.add_argument("--outdir", required=True, help="output directory for CSVs")

    ap.add_argument("--conditions", default="ALL", help="Comma list (e.g., HNSC,LUAD) or ALL")
    ap.add_argument("--variants", default="ALL", help="Comma list (e.g., ours) or ALL")
    ap.add_argument("--gate-modes", default="ALL", help="Comma list (e.g., hard) or ALL")
    ap.add_argument("--taus", default="ALL", help="Comma list (e.g., 0.2,0.4) or ALL")

    ap.add_argument("--topk-reasons", type=int, default=5, help="Top K reasons kept; rest -> Other")
    ap.add_argument(
        "--include-fail-reasons",
        action="store_true",
        help="Also export FAIL reasons (in addition to ABSTAIN) in EDFig2d CSV",
    )
    ap.add_argument(
        "--strict-nonempty",
        action="store_true",
        help="Error if filtering yields empty tables for EDFig2b/c/d",
    )

    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    conditions = (
        None
        if str(args.conditions).strip().upper() == "ALL"
        else [c.upper() for c in _parse_csv(args.conditions)]
    )
    variants = None if str(args.variants).strip().upper() == "ALL" else _parse_csv(args.variants)
    gate_modes = (
        None if str(args.gate_modes).strip().upper() == "ALL" else _parse_csv(args.gate_modes)
    )

    taus: list[float] | None = None
    if str(args.taus).strip().upper() != "ALL":
        taus = [float(x) for x in _parse_csv(args.taus)]

    wide = _read_tsv(Path(args.collections_summary_wide))
    reas = _read_tsv(Path(args.audit_reasons_long))

    # EDFig2b
    b_df = build_edfig2b_outcomes_by_collection(
        wide,
        conditions=conditions,
        variants=variants,
        gate_modes=gate_modes,
        taus=taus,
        strict_nonempty=bool(args.strict_nonempty),
    )
    # EDFig2c
    c_df = build_edfig2c_module_stats(
        wide,
        conditions=conditions,
        variants=variants,
        gate_modes=gate_modes,
        taus=taus,
        strict_nonempty=bool(args.strict_nonempty),
    )

    # EDFig2d
    d_df = build_edfig2d_abstain_reasons_by_collection(
        reas,
        conditions=conditions,
        variants=variants,
        gate_modes=gate_modes,
        taus=taus,
        topk=int(args.topk_reasons),
        strict_nonempty=bool(args.strict_nonempty),
        include_fail_reasons=bool(args.include_fail_reasons),
    )

    p_b = outdir / "EDFig2b_data.csv"
    p_c = outdir / "EDFig2c_data.csv"
    p_d = outdir / "EDFig2d_data.csv"

    _write_csv(b_df, p_b)
    _write_csv(c_df, p_c)
    _write_csv(d_df, p_d)

    print("[figS2_export] WROTE:")
    print(" ", p_b)
    print(" ", p_c)
    print(" ", p_d)


if __name__ == "__main__":
    main()
