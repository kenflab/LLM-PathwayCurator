#!/usr/bin/env python3
# paper/scripts/fig2_sanity_check.py

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd


def die(msg: str, code: int = 1) -> None:
    print(f"[SANITY][ERROR] {msg}", file=sys.stderr)
    raise SystemExit(code)


def warn(msg: str) -> None:
    print(f"[SANITY][WARN] {msg}", file=sys.stderr)


def info(msg: str) -> None:
    print(f"[SANITY][INFO] {msg}")


def read_jsonl(p: Path) -> pd.DataFrame:
    rows = []
    with p.open("r", encoding="utf-8") as f:
        for ln, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception as e:
                die(f"invalid JSONL at {p} line {ln}: {e}")
    if not rows:
        return pd.DataFrame()
    return pd.json_normalize(rows)


def head_text(path: Path, n: int = 3) -> None:
    print(f"\n== head: {path.name} (n={n}) ==")
    with path.open("r", encoding="utf-8") as f:
        for _i, line in zip(range(n), f, strict=False):
            print(line.rstrip())


def _maybe_show_df(df: pd.DataFrame, n: int = 10, cols: list[str] | None = None) -> None:
    if df is None or df.empty:
        print("(empty)")
        return
    if cols is not None:
        missing = [c for c in cols if c not in df.columns]
        if missing:
            warn(f"missing columns: {missing}")
        cols2 = [c for c in cols if c in df.columns]
        df = df[cols2]
    with pd.option_context(
        "display.max_rows",
        n,
        "display.max_columns",
        200,
        "display.width",
        200,
        "display.max_colwidth",
        80,
    ):
        print(df.head(n).to_string(index=False))


def main() -> None:
    # Allow overriding base on CLI (useful for tau sweep dirs)
    default_base = "/work/paper/source_data/PANCAN_TP53_v1/out/HNSC/ours"
    base = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(default_base)
    base = base.expanduser().resolve()

    paths = {
        "report": base / "report.jsonl",
        "audit": base / "audit_log.tsv",
        "modules": base / "modules.tsv",
        "edges": base / "term_gene_edges.tsv",
        "terms": base / "term_modules.tsv",
        "distilled": base / "distilled.tsv",
        "risk": base / "risk_coverage.tsv",
        "card": base / "sample_card.resolved.json",
        "meta": base / "run_meta.json",  # proof that pipeline ran
    }

    print("BASE:", base)
    for k, p in paths.items():
        ok = p.exists()
        size = p.stat().st_size if ok else None
        print(f"{k:8s}", ok, size)

    # ---- 0) required files
    required = ["report", "audit", "risk", "card", "meta"]
    for k in required:
        p = paths[k]
        if not p.exists():
            die(f"missing required artifact: {k} -> {p}")
        if p.stat().st_size == 0:
            die(f"empty required artifact: {k} -> {p}")

    # ---- 0b) run_meta.json (strongest proof of run_pipeline)
    meta = json.loads(paths["meta"].read_text(encoding="utf-8"))
    meta_tool = meta.get("tool", "")
    meta_status = meta.get("status", "")
    meta_step = meta.get("step", "")
    print("\n[run_meta] tool/status/step:", meta_tool, meta_status, meta_step)

    if meta_tool != "llm-pathway-curator":
        warn(f"run_meta.tool unexpected: {meta_tool!r}")
    if meta_status != "ok":
        warn(f"run_meta.status is not ok: {meta_status!r}")
    # We expect to have reached the jsonl step by end
    if meta_step and meta_step != "report_jsonl":
        warn(f"run_meta.step is not report_jsonl (maybe mid-run or older): {meta_step!r}")

    # ---- 1) report.jsonl (core)
    rep = read_jsonl(paths["report"])
    print("\n[report] shape:", rep.shape)
    if rep.empty:
        die("report.jsonl parsed as empty")

    if "decision" in rep.columns:
        print("[report] decisions:", rep["decision"].value_counts(dropna=False).to_dict())
    else:
        warn("report.jsonl missing column: decision")

    if "tau" in rep.columns:
        tau_vals = sorted(pd.to_numeric(rep["tau"], errors="coerce").dropna().unique().tolist())
        print("[report] tau unique:", tau_vals)
    else:
        warn("report.jsonl missing column: tau")

    print("[report] columns (top 25):", list(rep.columns)[:25])

    contract_cols = [
        "schema_version",
        "benchmark_id",
        "run_id",
        "method",
        "cancer",
        "comparison",
        "tau",
        "claim_id",
        "decision",
        "survival",
        "claim.entity_type",
        "claim.entity_id",
        "claim.direction",
        "claim.context.disease",
        "claim.context.tissue",
        "claim.context.perturbation",
        "evidence_refs.gene_set_hash",
    ]
    print("\n[report] contract view (head):")
    _maybe_show_df(rep, n=10, cols=contract_cols)

    # ---- 2) audit_log.tsv (PASS/ABSTAIN/FAIL + reasons)
    aud = pd.read_csv(paths["audit"], sep="\t")
    print("\n[audit] shape:", aud.shape)
    print("[audit] columns:", aud.columns.tolist())

    # your tool uses "status" as decision
    decision_col = None
    for c in aud.columns:
        if c.lower() in ("status", "decision"):
            decision_col = c
            break

    if decision_col:
        counts = aud[decision_col].astype(str).value_counts(dropna=False).to_dict()
        print("[audit] decision counts:", counts)
        if counts.get("FAIL", 0) == 0 and counts.get("ABSTAIN", 0) == 0:
            warn("audit has no FAIL/ABSTAIN (bench may be too easy for Fig2)")
    else:
        warn("audit_log.tsv missing status/decision column")

    # reason fields in v1: abstain_reason / fail_reason / audit_notes
    for rc in ["abstain_reason", "fail_reason"]:
        if rc in aud.columns:
            top = aud[rc].astype(str).value_counts().head(10).to_dict()
            print(f"[audit] top {rc}:", top)
        else:
            warn(f"audit_log.tsv missing column: {rc}")

    if "audit_notes" in aud.columns:
        n_empty = (
            aud["audit_notes"].isna().sum()
            + (aud["audit_notes"].astype(str).str.strip() == "").sum()
        )
        if n_empty == len(aud):
            warn("audit_notes all empty/NA (explanations may be too thin for paper)")
    else:
        warn("audit_log.tsv missing column: audit_notes")

    print("\n[audit] head:")
    _maybe_show_df(aud, n=5)

    # ---- 3) modules + edges sanity
    if paths["modules"].exists() and paths["edges"].exists():
        mods = pd.read_csv(paths["modules"], sep="\t")
        edges = pd.read_csv(paths["edges"], sep="\t")
        print("\n[modules] n:", len(mods), "cols:", mods.columns.tolist())
        print("[edges]   n:", len(edges), "cols:", edges.columns.tolist())

        if len(mods) == 0:
            warn("modules.tsv is empty")
        if len(edges) == 0:
            warn("term_gene_edges.tsv is empty")

        print("\n[modules] head:")
        _maybe_show_df(mods, n=5)
        print("\n[edges] head:")
        _maybe_show_df(edges, n=5)

    # ---- 4) risk_coverage.tsv (Fig2 payload)
    risk = pd.read_csv(paths["risk"], sep="\t")
    print("\n[risk] shape:", risk.shape)
    print("[risk] columns:", risk.columns.tolist())
    print("\n[risk] head:")
    _maybe_show_df(risk, n=20)

    if risk.empty:
        warn("risk_coverage.tsv empty")
    else:
        # Optional: sanity expectations
        for c in ["n_total", "n_pass", "n_fail", "n_abstain", "coverage_pass_total"]:
            if c not in risk.columns:
                warn(f"risk_coverage.tsv missing column: {c}")

    # ---- 5) sample card
    card = json.loads(paths["card"].read_text(encoding="utf-8"))
    print("\n[sample_card] keys:", sorted(card.keys()))
    # show a compact excerpt
    excerpt = {
        "disease": card.get("disease"),
        "tissue": card.get("tissue"),
        "perturbation": card.get("perturbation"),
        "comparison": card.get("comparison"),
        "notes": card.get("notes"),
        "extra": card.get("extra"),
    }
    print(json.dumps(excerpt, indent=2, ensure_ascii=False)[:1200], "...\n")

    info("OK (sanity check complete)")


if __name__ == "__main__":
    main()
