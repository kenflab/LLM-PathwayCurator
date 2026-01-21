#!/usr/bin/env python3
# paper/scripts/fig2_collect_risk_coverage.py
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd


# -------------------------
# Path helpers
# -------------------------
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
    for name in ["sample_card.json", "sample_card.tau.json", "sample_card.tau.json"]:
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


# -------------------------
# Labels (optional)
# -------------------------
def _load_labels(path: Path) -> pd.DataFrame:
    """
    Expected TSV columns (minimal):
      - claim_id
      - human_label  (ACCEPT/REJECT)   (or: label / human_decision)
    Optional:
      - benchmark_id, cancer, method, tau
    """
    df = pd.read_csv(path, sep="\t")

    # Normalize columns
    col_map = {c: c.strip() for c in df.columns}
    df = df.rename(columns=col_map)

    if "claim_id" not in df.columns:
        raise ValueError(f"labels.tsv missing required column: claim_id ({path})")

    label_col = None
    for c in ["human_label", "human_decision", "label"]:
        if c in df.columns:
            label_col = c
            break
    if label_col is None:
        raise ValueError(
            f"labels.tsv must have one of: human_label / human_decision / label ({path})"
        )

    df = df.copy()
    df["claim_id"] = df["claim_id"].astype(str)

    df["human_label"] = (
        df[label_col].astype(str).str.upper().str.strip().replace({"": pd.NA, "NA": pd.NA})
    )

    # Keep only 1 label per claim_id deterministically: last non-null wins (stable sort)
    df = df.sort_values(["claim_id"], kind="mergesort")
    df = df.dropna(subset=["human_label"])
    df = df.drop_duplicates(subset=["claim_id"], keep="last").reset_index(drop=True)

    return df[["claim_id", "human_label"]]


def _attach_labels(audit: pd.DataFrame, labels: pd.DataFrame | None) -> pd.DataFrame:
    if labels is None or labels.empty:
        return audit

    if "claim_id" not in audit.columns:
        # Can't join -> return unchanged
        return audit

    out = audit.copy()
    out["claim_id"] = out["claim_id"].astype(str)

    out = out.merge(labels, on="claim_id", how="left", validate="m:1")
    return out


# -------------------------
# Metrics
# -------------------------
def _compute_from_audit(audit: pd.DataFrame) -> dict[str, float]:
    if "status" not in audit.columns:
        raise ValueError("audit_log.tsv missing column: status")

    st = audit["status"].astype(str).str.upper().str.strip()
    n_total = float(len(audit))
    n_pass = float((st == "PASS").sum())
    coverage_pass = (n_pass / n_total) if n_total > 0 else float("nan")

    # ---- human label (joined from labels.tsv) ----
    n_pass_labeled = 0.0
    risk_human_reject = float("nan")
    if "human_label" in audit.columns:
        lab = audit["human_label"].astype(str).str.upper().str.strip()
        missing = lab.isin(["", "NAN", "NONE", "NA"])
        labeled_pass = (st == "PASS") & (~missing)
        n_pass_labeled = float(labeled_pass.sum())
        if n_pass_labeled > 0:
            risk_human_reject = float(((lab == "REJECT") & labeled_pass).sum() / n_pass_labeled)

    # ---- risk proxy (label-free, dev/debug) ----
    #
    # Define proxy risk among PASS as:
    #   fraction of PASS rows whose audit_notes contains "warning-like" tokens
    # This lets you debug whether tau/method/stress changes anything *without* labels.
    #
    warn_tokens = [
        "under_supported",
        "context_nonspecific",
        "stress_",
        "contradiction",
        "evidence_drift",
        "schema_violation",
        "missing_survival",
    ]

    risk_proxy = float("nan")
    n_pass_proxy = float((st == "PASS").sum())
    if n_pass_proxy > 0:
        if "audit_notes" in audit.columns:
            notes = audit["audit_notes"].astype(str)
            pass_notes = notes[st == "PASS"]
            bad = pd.Series(False, index=pass_notes.index)
            for tok in warn_tokens:
                bad = bad | pass_notes.str.contains(tok, regex=False, na=False)
            risk_proxy = float(bad.sum() / n_pass_proxy)
        else:
            risk_proxy = float("nan")

    return {
        "coverage_pass": float(coverage_pass),
        "risk_human_reject": float(risk_human_reject)
        if pd.notna(risk_human_reject)
        else float("nan"),
        "n_pass_labeled": float(n_pass_labeled),
        "risk_proxy": float(risk_proxy) if pd.notna(risk_proxy) else float("nan"),
        "n_total": float(n_total),
        "n_pass": float(n_pass),
    }


# -------------------------
# Main
# -------------------------
def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Collect Fig2 risk–coverage points (cancer×method×tau) from "
            "out/*/*/tau_*/audit_log.tsv, optionally joining labels.tsv."
        )
    )
    ap.add_argument(
        "--root",
        required=True,
        help="e.g. /work/paper/source_data/PANCAN_TP53_v1/out",
    )
    ap.add_argument("--out", required=True, help="output TSV (for fig2_plot*.py)")
    ap.add_argument("--benchmark-id", default="", help="fallback benchmark_id if not in cards")
    ap.add_argument(
        "--labels",
        default="",
        help=(
            "Optional labels.tsv with columns claim_id + human_label "
            "(or label/human_decision). If provided, enables risk_human_reject."
        ),
    )
    args = ap.parse_args()

    root = Path(args.root)
    files = sorted(root.glob("*/*/tau_*/audit_log.tsv"))
    if not files:
        raise SystemExit(f"No audit_log.tsv found under {root} (expected */*/tau_*/audit_log.tsv)")

    labels_df: pd.DataFrame | None = None
    if str(args.labels).strip():
        lp = Path(str(args.labels)).expanduser()
        if not lp.exists():
            raise SystemExit(f"--labels not found: {lp}")
        labels_df = _load_labels(lp)

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

        # join labels (claim_id -> human_label)
        audit = _attach_labels(audit, labels_df)

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
                "risk_proxy": m["risk_proxy"],
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

    # Helpful hint for empty plots
    n_labeled = int(pd.to_numeric(df["n_pass_labeled"], errors="coerce").fillna(0).sum())
    if n_labeled == 0:
        print(
            "[WARN] n_pass_labeled is 0 for all rows: "
            "fig2_plot*.py will likely plot nothing if it filters on labeled PASS only. "
            "Use risk_proxy for dev plots or provide --labels."
        )


if __name__ == "__main__":
    main()
