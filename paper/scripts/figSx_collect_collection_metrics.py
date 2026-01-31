#!/usr/bin/env python3
# paper/scripts/figSx_collect_collection_metrics.py
from __future__ import annotations

import argparse
import re
from collections.abc import Iterable
from pathlib import Path

import pandas as pd


# -----------------------------------------------------------------------------
# Collection + reason canonicalization
# -----------------------------------------------------------------------------
def _canon(s: str) -> str:
    return str(s or "").strip()


def _canon_l(s: str) -> str:
    return _canon(s).lower()


def infer_collection_from_source(source: str, term_id: str = "", term_uid: str = "") -> str:
    """
    Robust, spec-level mapping:
      - Prefer `source` (audit_log/distilled) or prefix inside term_uid / term_id.
      - Fall back to "Other" (never drop rows).
    """
    s = _canon_l(source)
    tu = _canon_l(term_uid)
    tid = _canon_l(term_id)

    blob = " | ".join([s, tu, tid])

    # Hallmark (MSigDB H)
    if "msigdb_h" in blob or "hallmark_" in blob or "fgsea_msigdb_h" in blob:
        return "Hallmark"

    # Reactome
    if "reactome" in blob or tid.startswith("r-hsa-"):
        return "Reactome"

    # KEGG
    if "kegg" in blob or tid.startswith("hsa") or tid.startswith("kegg_"):
        return "KEGG"

    # GO
    # (Prefer to keep broad; if you later want BP-only, tighten here.)
    if "go" in s or tid.startswith("go:") or "go:" in blob:
        return "GO"

    return "Other"


def infer_collection_from_term_ids_str(term_ids_str: str) -> str:
    """
    modules.tsv has term_ids_str like:
      fgsea_msigdb_H_entrez:HALLMARK_X;reactome:R-HSA-...
    We assign collection by majority vote across term prefixes (ties -> Mixed/Other).
    """
    t = _canon(term_ids_str)
    if not t:
        return "Other"
    parts = [p.strip() for p in t.split(";") if p.strip()]
    if not parts:
        return "Other"

    counts: dict[str, int] = {}
    for p in parts:
        # Prefer prefix before ":" (source-like)
        src = p.split(":", 1)[0] if ":" in p else p
        coll = infer_collection_from_source(src, term_id="", term_uid=p)
        counts[coll] = counts.get(coll, 0) + 1

    # majority
    items = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    if not items:
        return "Other"
    top_coll, top_n = items[0]
    # detect mixed tie
    if len(items) >= 2 and items[1][1] == top_n and items[1][0] != top_coll:
        return "Other"
    return top_coll


def canon_reason(reason: str) -> str:
    """
    Paper-facing normalization.
    Keeps bounded set readable; everything else becomes Title Case-ish.
    """
    s = _canon(reason)
    if not s or s.lower() in ("na", "none", "nan"):
        return "Unknown"

    k = s.strip().lower()
    mapping = {
        # abstain-like
        "context_gate": "Context gate",
        "context_nonspecific": "Context-nonspecific",
        "abstain_context_nonspecific": "Context-nonspecific",
        "abstain_context_missing": "Context-missing",
        "abstain_inconclusive_stress": "Inconclusive stress",
        "inconclusive_stress": "Inconclusive stress",
        "abstain_unstable": "Unstable",
        "unstable": "Unstable",
        "abstain_under_supported": "Under-supported",
        "under_supported": "Under-supported",
        "abstain_missing_evidence_genes": "Missing evidence genes",
        "missing_evidence_genes": "Missing evidence genes",
        "abstain_missing_survival": "Missing stability",
        "missing_survival": "Missing stability",
        "low_survival": "Low stability",
        "stability": "Low stability",
        "stability_gate": "Low stability",
        # fail-like
        "fail_contradiction": "Contradiction",
        "contradiction": "Contradiction",
        "fail_evidence_drift": "Evidence drift",
        "evidence_drift": "Evidence drift",
        "fail_schema_violation": "Schema violation",
        "schema_violation": "Schema violation",
        "fail_context": "Context violation",
    }
    if k in mapping:
        return mapping[k]

    # readable fallback
    k2 = re.sub(r"[_\s]+", " ", k).strip()
    if not k2:
        return "Unknown"
    return k2[:1].upper() + k2[1:]


# -----------------------------------------------------------------------------
# IO helpers
# -----------------------------------------------------------------------------
def _parse_csv_list(x: str) -> list[str]:
    return [a.strip() for a in str(x or "").split(",") if a.strip()]


def _looks_like_condition_dir(p: Path, conditions: list[str]) -> bool:
    # Heuristic: directory name matches one of requested conditions (case-insensitive)
    name = p.name.strip().upper()
    return name in {c.strip().upper() for c in conditions}


def _infer_collection_from_dirname(dirname: str) -> str:
    # Map your pipeline folder names to paper-facing labels
    d = _canon(dirname)
    dl = d.lower()

    # Matches: H, C5_GO_BP, C2_CP_REACTOME, C2_CP_KEGG_MEDICUS, etc.
    if d == "H" or dl.endswith("_h") or "msigdb_h" in dl:
        return "Hallmark"
    if "go" in dl:
        return "GO"
    if "reactome" in dl:
        return "Reactome"
    if "kegg" in dl:
        return "KEGG"
    return d or "Other"


def iter_runs(
    out_root: Path,
    conditions: Iterable[str],
    variants: Iterable[str],
    gate_modes: Iterable[str],
    taus: Iterable[float],
) -> Iterable[dict[str, object]]:
    conditions_l = [c.strip().upper() for c in conditions]
    variants_l = [v.strip() for v in variants]
    gate_modes_l = [g.strip() for g in gate_modes]
    taus_l = [float(t) for t in taus]

    # Case 1: out_root/COND/... exists  => "flat" layout
    has_flat = (
        any((out_root / c / variants_l[0]).exists() for c in conditions_l) if variants_l else False
    )

    # Case 2: out_root/COLLECTION/COND/... exists => "collection-first" layout
    # Detect by checking subdirs that contain at least one condition dir
    collection_roots: list[tuple[Path, str]] = []
    if not has_flat:
        for sub in sorted([p for p in out_root.iterdir() if p.is_dir()]):
            if any(_looks_like_condition_dir(sub / c, conditions_l) for c in conditions_l):
                collection_roots.append((sub, _infer_collection_from_dirname(sub.name)))

    # Iterate either layout
    if has_flat:
        roots = [(out_root, "")]  # no fixed collection
    else:
        roots = collection_roots

    for root, coll_run in roots:
        for cond in conditions_l:
            for var in variants_l:
                for gate in gate_modes_l:
                    for tau in taus_l:
                        run_dir = root / cond / var / f"gate_{gate}" / f"tau_{tau:.2f}"
                        yield {
                            "condition": cond,
                            "variant": var,
                            "gate_mode": gate,
                            "tau": float(tau),
                            "run_dir": run_dir,
                            # If collection-first layout: run-level fixed collection label
                            "collection_run": coll_run,
                        }


def read_tsv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t")


def ensure_cols(df: pd.DataFrame, cols: Iterable[str], *, where: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {where}: {missing}")


_TRUE = {"true", "1", "yes", "y", "t"}
_FALSE = {"false", "0", "no", "n", "f"}


def _parse_bool_series(s: pd.Series) -> pd.Series:
    x = s.copy()
    # Already boolean
    if x.dtype == bool:
        return x
    # Numbers: 0/1
    if pd.api.types.is_numeric_dtype(x):
        return x.fillna(0).astype(int).astype(bool)
    # Strings
    xl = x.astype(str).str.strip().str.lower()
    out = pd.Series(pd.NA, index=x.index, dtype="boolean")
    out[xl.isin(_TRUE)] = True
    out[xl.isin(_FALSE)] = False
    # Anything else stays NA (so mean ignores it if we dropna)
    return out


def _true_rate(series: pd.Series) -> float:
    b = _parse_bool_series(series)
    b2 = b.dropna()
    if b2.empty:
        return float("nan")
    return float(b2.astype(float).mean())


# -----------------------------------------------------------------------------
# Core collection builders
# -----------------------------------------------------------------------------
def build_audit_tables(
    run: dict[str, object],
    audit_path: Path,
    benchmark_id: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      - audit_outcomes_wide: one row per collection with PASS/ABSTAIN/FAIL counts + rates
      - audit_reasons_long: one row per collection x reason_type x reason
      - audit_gate_summary: one row per collection with key gate diagnostic rates
    """
    df = read_tsv(audit_path)

    # ---- Backward compatibility: fill missing mandatory keys from run ----
    if "condition" not in df.columns:
        df["condition"] = str(run["condition"])
    if "status" not in df.columns:
        raise ValueError(f"Missing required column 'status' in {audit_path}")
    if "source" not in df.columns:
        df["source"] = ""
    if "tau_used" not in df.columns:
        df["tau_used"] = float(run["tau"])

    # optional (may be absent)
    if "term_uid" not in df.columns:
        df["term_uid"] = ""
    if "term_id" not in df.columns:
        df["term_id"] = ""

    # normalize keys from run dir (authoritative for variant/gate), but keep df.condition
    df["benchmark_id"] = benchmark_id
    df["variant"] = str(run["variant"])
    df["gate_mode"] = str(run["gate_mode"])
    # use tau_used (truth) but ensure numeric + consistent
    df["tau"] = pd.to_numeric(df["tau_used"], errors="coerce")
    df["status"] = df["status"].astype(str)

    # collection
    coll_run = _canon(str(run.get("collection_run", "")))
    if coll_run:
        df["collection"] = coll_run
    else:
        df["collection"] = [
            infer_collection_from_source(s, term_id=tid, term_uid=tu)
            for s, tid, tu in zip(
                df["source"].astype(str),
                df["term_id"].astype(str),
                df["term_uid"].astype(str),
                strict=True,
            )
        ]

    # outcomes wide
    grp_keys = ["benchmark_id", "condition", "variant", "gate_mode", "tau", "collection"]
    g = (
        df.groupby(grp_keys, as_index=False)["status"]
        .value_counts()
        .pivot(index=grp_keys, columns="status", values="count")
        .fillna(0.0)
        .reset_index()
    )

    for c in ["PASS", "ABSTAIN", "FAIL"]:
        if c not in g.columns:
            g[c] = 0.0

    g["n_total_claims"] = g["PASS"] + g["ABSTAIN"] + g["FAIL"]
    # avoid div0
    denom = g["n_total_claims"].replace(0, 1.0)
    g["n_pass"] = g["PASS"].astype(int)
    g["n_abstain"] = g["ABSTAIN"].astype(int)
    g["n_fail"] = g["FAIL"].astype(int)
    g["pass_rate"] = (g["PASS"] / denom).astype(float)
    g["abstain_rate"] = (g["ABSTAIN"] / denom).astype(float)
    g["fail_rate"] = (g["FAIL"] / denom).astype(float)
    g = g.drop(columns=["PASS", "ABSTAIN", "FAIL"])

    # gate diagnostic summary (collection-level)
    gate_cols_bool = [
        "context_gate_blocked",
        "eligible_context",
        "link_ok",
        "stability_ok",
        "stress_evaluated",
    ]
    gate_cols_tri = [
        "stress_ok",
        "contradiction_ok",
    ]

    gate_df = df.copy()
    for c in gate_cols_bool:
        if c in gate_df.columns:
            gate_df[c] = _parse_bool_series(gate_df[c])

    # For tri-state columns, treat True as 1, False as 0, ignore NA.
    def _rate_true(series: pd.Series) -> float:
        s = series.dropna()
        if s.empty:
            return float("nan")
        return float((s.astype(str).str.upper() == "TRUE").mean())

    rows = []
    for keys, sub in gate_df.groupby(grp_keys, dropna=False):
        row = dict(zip(grp_keys, keys, strict=True))
        for c in gate_cols_bool:
            if c in sub.columns:
                row[f"{c}_rate"] = (
                    float(sub[c].dropna().astype(float).mean())
                    if sub[c].notna().any()
                    else float("nan")
                )
        for c in gate_cols_tri:
            if c in sub.columns:
                row[f"{c}_true_rate"] = _true_rate(sub[c])
        rows.append(row)

    gate_summary = pd.DataFrame(rows)

    # reasons long
    # ABSTAIN reasons
    out_rows = []

    def _add_reason_rows(reason_type: str, col: str, status_value: str) -> None:
        if col not in df.columns:
            return
        sub = df[df["status"] == status_value].copy()
        if sub.empty:
            return
        sub[col] = sub[col].fillna("Unknown").astype(str).map(canon_reason)
        rc = (
            sub.groupby(grp_keys + [col], as_index=False)
            .size()
            .rename(columns={"size": "count", col: "reason"})
        )
        # totals for fraction
        totals = (
            rc.groupby(grp_keys, as_index=False)["count"]
            .sum()
            .rename(columns={"count": "n_type_total"})
        )
        rc = rc.merge(totals, on=grp_keys, how="left")
        rc["fraction_within_type"] = rc["count"] / rc["n_type_total"].replace(0, 1)
        rc["reason_type"] = reason_type
        out_rows.append(rc)

    _add_reason_rows("ABSTAIN", "abstain_reason", "ABSTAIN")
    _add_reason_rows("FAIL", "fail_reason", "FAIL")

    reasons_long = (
        pd.concat(out_rows, ignore_index=True)
        if out_rows
        else pd.DataFrame(
            columns=grp_keys
            + ["reason_type", "reason", "count", "n_type_total", "fraction_within_type"]
        )
    )

    return g, reasons_long, gate_summary


def build_module_tables(
    run: dict[str, object],
    modules_path: Path,
    distilled_with_modules_path: Path,
    benchmark_id: str,
) -> pd.DataFrame:
    """
    Returns module_survival_long (one row per module) with:
      module_id, collection, module_survival, n_genes, n_terms, hub_filter_n_hubs, hub_gene_rate
    """
    mod = read_tsv(modules_path)
    ensure_cols(
        mod,
        ["module_id", "n_terms", "n_genes", "term_ids_str", "hub_filter_n_hubs"],
        where=str(modules_path),
    )

    # ---- collection (attach to the SAME table we will output) ----
    coll_run = _canon(str(run.get("collection_run", "")))
    if coll_run:
        mod["collection"] = coll_run
    else:
        mod["collection"] = mod["term_ids_str"].astype(str).map(infer_collection_from_term_ids_str)

    # ---- module_survival (prefer modules.tsv; fallback to distilled.with_modules) ----
    if "module_survival" in mod.columns:
        mod["module_survival"] = pd.to_numeric(mod["module_survival"], errors="coerce")
        base = mod.copy()
    else:
        d = read_tsv(distilled_with_modules_path)
        ensure_cols(
            d,
            ["module_id", "module_survival", "keep_term", "module_id_missing", "source"],
            where=str(distilled_with_modules_path),
        )
        d2 = d.copy()
        d2["keep_term"] = d2["keep_term"].astype(str).str.strip().str.lower().isin(_TRUE)
        d2["module_id_missing"] = (
            d2["module_id_missing"].astype(str).str.strip().str.lower().isin(_TRUE)
        )
        d2 = d2[(d2["keep_term"]) & (~d2["module_id_missing"]) & d2["module_id"].notna()].copy()

        d2["module_survival"] = pd.to_numeric(d2["module_survival"], errors="coerce")
        d2 = d2.dropna(subset=["module_survival"])

        if d2.empty:
            base = mod.copy()
            base["module_survival"] = pd.NA
        else:
            ms = (
                d2.groupby("module_id", as_index=False)["module_survival"]
                .agg(["min", "max", "first"])
                .reset_index()
            )
            ms["module_survival"] = ms["first"]
            ms = ms[["module_id", "module_survival"]]
            # no conflict because mod doesn't have module_survival here
            base = mod.merge(ms, on="module_id", how="left")

    # ---- hub gene rate (size-normalized) ----
    base["n_genes"] = pd.to_numeric(base["n_genes"], errors="coerce")
    base["n_terms"] = pd.to_numeric(base["n_terms"], errors="coerce")
    base["hub_filter_n_hubs"] = pd.to_numeric(base["hub_filter_n_hubs"], errors="coerce").fillna(
        0.0
    )
    denom = base["n_genes"].replace(0, pd.NA)
    base["hub_gene_rate"] = base["hub_filter_n_hubs"] / denom

    # ---- attach run keys ----
    base["benchmark_id"] = benchmark_id
    base["condition"] = str(run["condition"])
    base["variant"] = str(run["variant"])
    base["gate_mode"] = str(run["gate_mode"])
    base["tau"] = float(run["tau"])

    keep = [
        "benchmark_id",
        "condition",
        "variant",
        "gate_mode",
        "tau",
        "collection",
        "module_id",
        "module_survival",
        "n_genes",
        "n_terms",
        "hub_filter_n_hubs",
        "hub_gene_rate",
    ]
    # helpful error if something regresses
    ensure_cols(base, keep, where="build_module_tables(base)")

    return base[keep].copy()


def summarize_modules(module_long: pd.DataFrame) -> pd.DataFrame:
    """
    One row per (benchmark_id, condition, variant, gate_mode, tau, collection).
    """
    keys = ["benchmark_id", "condition", "variant", "gate_mode", "tau", "collection"]

    df = module_long.copy()
    df["module_survival"] = pd.to_numeric(df["module_survival"], errors="coerce")
    df["hub_gene_rate"] = pd.to_numeric(df["hub_gene_rate"], errors="coerce")
    df["n_genes"] = pd.to_numeric(df["n_genes"], errors="coerce")
    df["n_terms"] = pd.to_numeric(df["n_terms"], errors="coerce")

    def q(x: pd.Series, p: float) -> float:
        x2 = x.dropna()
        if x2.empty:
            return float("nan")
        return float(x2.quantile(p))

    agg = df.groupby(keys, as_index=False).agg(
        n_modules=("module_id", "nunique"),
        median_module_genes=("n_genes", "median"),
        median_module_terms=("n_terms", "median"),
        hub_gene_rate_median=("hub_gene_rate", "median"),
        module_survival_p10=("module_survival", lambda s: q(s, 0.10)),
        module_survival_p50=("module_survival", lambda s: q(s, 0.50)),
        module_survival_p90=("module_survival", lambda s: q(s, 0.90)),
    )
    return agg


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Collect Hallmark vs GO vs Reactome vs KEGG collection metrics "
            "from audit_log/modules outputs."
        )
    )
    ap.add_argument(
        "--out-root", required=True, help="e.g., /work/paper/source_data/PANCAN_TP53_v1/out"
    )
    ap.add_argument(
        "--benchmark-id", default="PANCAN_TP53_v1", help="benchmark_id to record in outputs"
    )
    ap.add_argument("--conditions", required=True, help="comma-separated cancers, e.g. HNSC,LUAD")
    ap.add_argument("--variants", default="ours", help="comma-separated variants (default: ours)")
    ap.add_argument(
        "--gate-modes", default="hard", help="comma-separated gate modes (default: hard)"
    )
    ap.add_argument("--taus", default="0.80", help="comma-separated taus, e.g. 0.2,0.4,0.6,0.8,0.9")
    ap.add_argument(
        "--out-dir",
        default="",
        help="output directory (default: <out-root>/../collection_metrics)",
    )
    ap.add_argument("--strict", action="store_true", help="fail if any run_dir is missing files")
    args = ap.parse_args()

    out_root = Path(args.out_root)
    if not out_root.exists():
        raise ValueError(f"--out-root does not exist: {out_root}")

    benchmark_id = str(args.benchmark_id).strip() or "benchmark"

    conditions = _parse_csv_list(args.conditions)
    variants = _parse_csv_list(args.variants)
    gate_modes = _parse_csv_list(args.gate_modes)
    taus = [float(x) for x in _parse_csv_list(args.taus)]

    if not conditions:
        raise ValueError("--conditions must be non-empty")
    if not variants:
        raise ValueError("--variants must be non-empty")
    if not gate_modes:
        raise ValueError("--gate-modes must be non-empty")
    if not taus:
        raise ValueError("--taus must be non-empty")

    out_dir = (
        Path(args.out_dir)
        if str(args.out_dir).strip()
        else (out_root.parent / "collection_metrics")
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    audit_outcomes_all = []
    audit_reasons_all = []
    module_long_all = []
    audit_gate_all = []

    missing_runs = []

    for run in iter_runs(out_root, conditions, variants, gate_modes, taus):
        run_dir: Path = run["run_dir"]  # type: ignore[assignment]

        audit_path = run_dir / "audit_log.tsv"
        modules_path = run_dir / "modules.tsv"
        distilled_with_modules_path = run_dir / "distilled.with_modules.tsv"

        if not (
            audit_path.exists() and modules_path.exists() and distilled_with_modules_path.exists()
        ):
            missing_runs.append(str(run_dir))
            if args.strict:
                raise ValueError(f"Missing required files in run_dir: {run_dir}")
            continue

        # audit
        audit_wide, reasons_long, gate_summary = build_audit_tables(run, audit_path, benchmark_id)
        audit_outcomes_all.append(audit_wide)
        audit_reasons_all.append(reasons_long)
        # collect gate summaries
        audit_gate_all.append(gate_summary)

        # modules + survival
        module_long = build_module_tables(
            run, modules_path, distilled_with_modules_path, benchmark_id
        )
        module_long_all.append(module_long)

    if not audit_outcomes_all:
        raise ValueError("No runs collected (check --out-root/--conditions/--variants/--taus).")

    audit_outcomes = pd.concat(audit_outcomes_all, ignore_index=True)
    audit_reasons = (
        pd.concat(audit_reasons_all, ignore_index=True) if audit_reasons_all else pd.DataFrame()
    )

    module_long = (
        pd.concat(module_long_all, ignore_index=True)
        if module_long_all
        else pd.DataFrame(
            columns=[
                "benchmark_id",
                "condition",
                "variant",
                "gate_mode",
                "tau",
                "collection",
                "module_id",
                "module_survival",
                "n_genes",
                "n_terms",
                "hub_filter_n_hubs",
                "hub_gene_rate",
            ]
        )
    )
    module_summary = summarize_modules(module_long) if not module_long.empty else pd.DataFrame()

    # merge outcomes + module summary into collections_summary_wide
    keys = ["benchmark_id", "condition", "variant", "gate_mode", "tau", "collection"]
    audit_gate = pd.concat(audit_gate_all, ignore_index=True) if audit_gate_all else pd.DataFrame()
    collections_summary = audit_outcomes.merge(module_summary, on=keys, how="left")
    if not audit_gate.empty:
        collections_summary = collections_summary.merge(audit_gate, on=keys, how="left")

    # deterministic ordering
    col_order = ["Hallmark", "GO", "Reactome", "KEGG", "Other"]
    collections_summary["collection"] = pd.Categorical(
        collections_summary["collection"], categories=col_order, ordered=True
    )
    collections_summary = collections_summary.sort_values(
        ["condition", "variant", "gate_mode", "tau", "collection"], kind="mergesort"
    )

    if not audit_reasons.empty:
        audit_reasons["collection"] = pd.Categorical(
            audit_reasons["collection"], categories=col_order, ordered=True
        )
        audit_reasons = audit_reasons.sort_values(
            ["condition", "variant", "gate_mode", "tau", "collection", "reason_type", "count"],
            ascending=[True, True, True, True, True, True, False],
            kind="mergesort",
        )

    if not module_long.empty:
        module_long["collection"] = pd.Categorical(
            module_long["collection"], categories=col_order, ordered=True
        )
        module_long = module_long.sort_values(
            ["condition", "variant", "gate_mode", "tau", "collection", "module_id"],
            kind="mergesort",
        )

    # write
    p1 = out_dir / "collections_summary_wide.tsv"
    p2 = out_dir / "module_survival_long.tsv"
    p3 = out_dir / "audit_reasons_long.tsv"
    p4 = out_dir / "missing_runs.txt"

    collections_summary.to_csv(p1, sep="\t", index=False)
    module_long.to_csv(p2, sep="\t", index=False)
    if not audit_reasons.empty:
        audit_reasons.to_csv(p3, sep="\t", index=False)
    else:
        # still write an empty file with header for reproducibility
        pd.DataFrame(
            columns=keys
            + ["reason_type", "reason", "count", "n_type_total", "fraction_within_type"]
        ).to_csv(p3, sep="\t", index=False)

    if missing_runs:
        Path(p4).write_text("\n".join(missing_runs) + "\n", encoding="utf-8")

    print("WROTE:")
    print("  ", p1)
    print("  ", p2)
    print("  ", p3)
    if missing_runs:
        print("NOTE: some run_dirs were missing required files; see:", p4)


if __name__ == "__main__":
    main()
