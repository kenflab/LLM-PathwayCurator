#!/usr/bin/env python3
# paper/scripts/fig2_collect_risk_coverage.py
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd


def _parse_tau_from_path(p: Path) -> float | None:
    m = re.search(r"tau_([0-9]*\.?[0-9]+)", str(p))
    if not m:
        return None
    try:
        return float(m.group(1))
    except Exception:
        return None


def _infer_keys_from_path(audit_path: Path, root: Path) -> dict[str, str]:
    """
    Supports BOTH layouts:

    New (preferred):
      root/<CANCER>/<VARIANT>/gate_<GATE_MODE>/tau_xx/audit_log.tsv

    Old:
      root/<CANCER>/<VARIANT>/tau_xx/audit_log.tsv

    Returns: cancer, variant, gate_mode (may be "")
    """
    rel = audit_path.relative_to(root)
    parts = rel.parts

    out = {"cancer": "", "variant": "", "gate_mode": ""}

    # Must at least have: <CANCER>/<VARIANT>/...
    if len(parts) < 3:
        return out

    out["cancer"] = str(parts[0])
    out["variant"] = str(parts[1])

    # New: third component is gate_<MODE>
    third = str(parts[2])
    if third.startswith("gate_"):
        out["gate_mode"] = third.replace("gate_", "", 1)

    # Old: no gate folder -> gate_mode stays ""
    return out


def _read_run_meta(run_dir: Path) -> dict[str, Any]:
    rp = run_dir / "run_meta.json"
    if not rp.exists():
        return {}
    try:
        return json.loads(rp.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _read_benchmark_id(run_dir: Path, fallback: str) -> str:
    meta = _read_run_meta(run_dir)
    bid = str(meta.get("benchmark_id", "")).strip()
    return bid if bid else str(fallback or "").strip()


def _read_method(run_dir: Path, fallback: str) -> str:
    meta = _read_run_meta(run_dir)
    m = str(meta.get("method", "")).strip()
    return m if m else str(fallback or "").strip()


def _load_labels(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", dtype=str).fillna("")
    df.columns = [c.strip() for c in df.columns]

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
    df["claim_id"] = df["claim_id"].astype(str).str.strip()
    df["human_label"] = df[label_col].astype(str).str.upper().str.strip()
    df["human_label"] = df["human_label"].replace({"": pd.NA, "NA": pd.NA, "NAN": pd.NA})

    df = df.dropna(subset=["human_label"])
    df = df.sort_values(["claim_id"], kind="mergesort")
    df = df.drop_duplicates(subset=["claim_id"], keep="last").reset_index(drop=True)
    return df[["claim_id", "human_label"]]


def _attach_labels(audit: pd.DataFrame, labels: pd.DataFrame | None) -> pd.DataFrame:
    if labels is None or labels.empty:
        return audit
    if "claim_id" not in audit.columns:
        return audit
    out = audit.copy()
    out["claim_id"] = out["claim_id"].astype(str).str.strip()
    return out.merge(labels, on="claim_id", how="left", validate="m:1")


def _compute_from_audit(audit: pd.DataFrame) -> dict[str, float]:
    if "status" not in audit.columns:
        raise ValueError("audit_log.tsv missing column: status")

    st = audit["status"].astype(str).str.upper().str.strip()
    n_total = float(len(audit))
    n_pass = float((st == "PASS").sum())
    coverage_pass = (n_pass / n_total) if n_total > 0 else float("nan")

    n_pass_labeled = 0.0
    risk_human_reject = float("nan")
    risk_human_nonaccept = float("nan")

    if "human_label" in audit.columns:
        lab = audit["human_label"].astype(str).str.upper().str.strip()
        missing = lab.isin(["", "NAN", "NONE", "NA"])
        pass_labeled = (st == "PASS") & (~missing)

        n_pass_labeled = float(pass_labeled.sum())
        if n_pass_labeled > 0:
            n_reject = float(((lab == "REJECT") & pass_labeled).sum())
            n_should_abstain = float(((lab == "SHOULD_ABSTAIN") & pass_labeled).sum())
            risk_human_reject = n_reject / n_pass_labeled
            risk_human_nonaccept = (n_reject + n_should_abstain) / n_pass_labeled

    warn_tokens = [
        "under_supported",
        "context_nonspecific",
        "stress_",
        "contradiction",
        "evidence_drift",
        "schema_violation",
        "missing_survival",
        "fail_",
    ]

    risk_proxy = float("nan")
    n_pass_proxy = float((st == "PASS").sum())
    if n_pass_proxy > 0 and "audit_notes" in audit.columns:
        notes = audit["audit_notes"].astype(str)
        pass_notes = notes[st == "PASS"]
        bad = pd.Series(False, index=pass_notes.index)
        for tok in warn_tokens:
            bad = bad | pass_notes.str.contains(tok, regex=False, na=False)
        risk_proxy = float(bad.sum() / n_pass_proxy)

    return {
        "n_total": float(n_total),
        "n_pass": float(n_pass),
        "coverage_pass": float(coverage_pass),
        "n_pass_labeled": float(n_pass_labeled),
        "risk_human_reject": float(risk_human_reject)
        if pd.notna(risk_human_reject)
        else float("nan"),
        "risk_human_nonaccept": float(risk_human_nonaccept)
        if pd.notna(risk_human_nonaccept)
        else float("nan"),
        "risk_proxy": float(risk_proxy) if pd.notna(risk_proxy) else float("nan"),
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Collect Fig2 points from audit_log.tsv (dev/debug). "
            "Supports both layouts:\n"
            "  new: out/<CANCER>/<VARIANT>/gate_<MODE>/tau_*/audit_log.tsv\n"
            "  old: out/<CANCER>/<VARIANT>/tau_*/audit_log.tsv"
        )
    )
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--benchmark-id", default="")
    ap.add_argument("--method", default="ours")
    ap.add_argument("--labels", default="")
    args = ap.parse_args()

    root = Path(args.root)

    # Pick up both layouts (union)
    files_new = list(root.glob("*/*/gate_*/tau_*/audit_log.tsv"))
    files_old = list(root.glob("*/*/tau_*/audit_log.tsv"))
    files = sorted({p.resolve(): p for p in (files_new + files_old)}.values())

    if not files:
        raise SystemExit(f"No audit_log.tsv found under {root}")

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

        keys = _infer_keys_from_path(audit_path, root)
        cancer = keys["cancer"].strip()
        variant = keys["variant"].strip()
        gate_mode = keys["gate_mode"].strip()  # may be ""

        if not cancer or not variant:
            continue

        run_dir = audit_path.parent
        benchmark_id = _read_benchmark_id(run_dir, str(args.benchmark_id))
        method = _read_method(run_dir, str(args.method))

        audit = pd.read_csv(audit_path, sep="\t")
        audit = _attach_labels(audit, labels_df)

        m = _compute_from_audit(audit)

        rows.append(
            {
                "benchmark_id": benchmark_id,
                "cancer": cancer,
                "method": method,
                "variant": variant,
                "gate_mode": gate_mode,
                "tau": float(tau),
                "coverage_pass": m["coverage_pass"],
                "risk_human_reject": m["risk_human_reject"],
                "risk_human_nonaccept": m["risk_human_nonaccept"],
                "n_pass_labeled": m["n_pass_labeled"],
                "risk_proxy": m["risk_proxy"],
                "n_total": m["n_total"],
                "n_pass": m["n_pass"],
                "_src": str(audit_path),
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit("Collected 0 rows (unexpected). Check directory layout and inputs.")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, sep="\t", index=False)
    print(f"[OK] wrote: {out_path} rows={len(df)}")

    n_labeled = int(pd.to_numeric(df["n_pass_labeled"], errors="coerce").fillna(0).sum())
    if n_labeled == 0:
        print("[WARN] n_pass_labeled is 0 for all rows. Use risk_proxy or provide --labels.")


if __name__ == "__main__":
    main()
