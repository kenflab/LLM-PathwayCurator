#!/usr/bin/env python3
# paper/scripts/collect_fig2_risk_coverage.py
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd


def _parse_tau_from_path(p: Path) -> float | None:
    m = re.search(r"tau_([0-9]*\.?[0-9]+)", str(p))
    if not m:
        return None
    try:
        return float(m.group(1))
    except Exception:
        return None


def _infer_cancer_method_from_path(p: Path, root: Path) -> tuple[str, str]:
    # root/<CANCER>/<METHOD>/tau_xx/audit_log.tsv
    rel = p.relative_to(root)
    parts = rel.parts
    cancer = parts[0] if len(parts) >= 1 else ""
    method = parts[1] if len(parts) >= 2 else ""
    return (str(cancer), str(method))


def _read_card_benchmark_id(run_dir: Path) -> str:
    # run_dir は tau_xx ディレクトリ（audit_log.tsv の親）
    for name in ["sample_card.json", "sample_card.tau.json"]:
        cp = run_dir / name
        if cp.exists():
            try:
                obj = json.loads(cp.read_text(encoding="utf-8"))
                extra = obj.get("extra") or {}
                if isinstance(extra, dict) and extra.get("benchmark_id"):
                    return str(extra["benchmark_id"])
            except Exception:
                pass
    return ""


def _compute_from_audit(audit: pd.DataFrame) -> dict[str, float]:
    if "status" not in audit.columns:
        raise ValueError("audit_log.tsv missing column: status")

    st = audit["status"].astype(str).str.upper().str.strip()
    n_total = float(len(audit))
    n_pass = float((st == "PASS").sum())
    coverage_pass = (n_pass / n_total) if n_total > 0 else float("nan")

    # ---- human label (optional but needed for risk_human_reject) ----
    label_col = None
    for c in ["human_label", "human_decision", "label"]:
        if c in audit.columns:
            label_col = c
            break

    n_pass_labeled = 0.0
    risk_human_reject = float("nan")

    if label_col is not None:
        lab = audit[label_col].astype(str).str.upper().str.strip()
        missing = lab.isin(["", "NAN", "NONE", "NA"])
        labeled_pass = (st == "PASS") & (~missing)
        n_pass_labeled = float(labeled_pass.sum())
        if n_pass_labeled > 0:
            risk_human_reject = float(((lab == "REJECT") & labeled_pass).sum() / n_pass_labeled)

    return {
        "coverage_pass": float(coverage_pass),
        "risk_human_reject": float(risk_human_reject)
        if pd.notna(risk_human_reject)
        else float("nan"),
        "n_pass_labeled": float(n_pass_labeled),
        "n_total": float(n_total),
        "n_pass": float(n_pass),
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Collect Fig2 risk–coverage points (cancer×method×tau) from out/*/*/tau_*/audit_log.tsv"
        )
    )
    ap.add_argument("--root", required=True, help="e.g. /work/paper/source_data/PANCAN_TP53_v1/out")
    ap.add_argument("--out", required=True, help="output TSV (for fig2_plot.py)")
    ap.add_argument("--benchmark-id", default="", help="fallback benchmark_id if not in cards")
    args = ap.parse_args()

    root = Path(args.root)
    files = sorted(root.glob("*/*/tau_*/audit_log.tsv"))
    if not files:
        raise SystemExit(f"No audit_log.tsv found under {root} (expected */*/tau_*/audit_log.tsv)")

    rows: list[dict[str, object]] = []
    for audit_path in files:
        tau = _parse_tau_from_path(audit_path)
        if tau is None:
            continue

        cancer, method = _infer_cancer_method_from_path(audit_path, root)
        if not cancer or not method:
            continue

        run_dir = audit_path.parent
        bench = _read_card_benchmark_id(run_dir) or str(args.benchmark_id or "")

        audit = pd.read_csv(audit_path, sep="\t")
        m = _compute_from_audit(audit)

        rows.append(
            {
                "benchmark_id": bench,
                "cancer": cancer,
                "method": method,
                "tau": float(tau),
                "coverage_pass": m["coverage_pass"],
                "risk_human_reject": m["risk_human_reject"],
                "n_pass_labeled": m["n_pass_labeled"],
                "n_total": m["n_total"],
                "n_pass": m["n_pass"],
                "_src": str(audit_path),
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit("Collected 0 rows (unexpected).")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, sep="\t", index=False)
    print(f"[OK] wrote: {out_path} rows={len(df)}")


if __name__ == "__main__":
    main()
