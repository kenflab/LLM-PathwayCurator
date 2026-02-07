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
    """Parse tau value from a path.

    The function looks for a substring like ``tau_0.8`` within the path
    string and returns it as a float.

    Parameters
    ----------
    p : pathlib.Path
        Path that may include a ``tau_<number>`` segment.

    Returns
    -------
    float or None
        Parsed tau value if present and valid; otherwise None.
    """
    m = re.search(r"tau_([0-9]*\.?[0-9]+)", str(p))
    if not m:
        return None
    try:
        return float(m.group(1))
    except Exception:
        return None


def _infer_keys_from_path(audit_path: Path, root: Path) -> dict[str, str]:
    """Infer run keys from an audit_log.tsv path.

    Supports both directory layouts:

    - New:
      ``root/<condition>/<VARIANT>/gate_<MODE>/tau_*/audit_log.tsv``
    - Old:
      ``root/<condition>/<VARIANT>/tau_*/audit_log.tsv``

    Parameters
    ----------
    audit_path : pathlib.Path
        Path to ``audit_log.tsv``.
    root : pathlib.Path
        Root directory used to compute the relative path.

    Returns
    -------
    dict of str
        Mapping with keys ``condition``, ``variant``, ``gate_mode``.
        ``gate_mode`` is an empty string for the old layout.

    Notes
    -----
    If the relative path is shorter than expected, empty strings are
    returned for all keys.
    """
    rel = audit_path.relative_to(root)
    parts = rel.parts

    out = {"condition": "", "variant": "", "gate_mode": ""}

    # Must at least have: <condition>/<VARIANT>/...
    if len(parts) < 3:
        return out

    out["condition"] = str(parts[0])
    out["variant"] = str(parts[1])

    # New: third component is gate_<MODE>
    third = str(parts[2])
    if third.startswith("gate_"):
        out["gate_mode"] = third.replace("gate_", "", 1)

    # Old: no gate folder -> gate_mode stays ""
    return out


def _read_run_meta(run_dir: Path) -> dict[str, Any]:
    """Read run metadata JSON from a run directory.

    Parameters
    ----------
    run_dir : pathlib.Path
        Directory expected to contain ``run_meta.json``.

    Returns
    -------
    dict
        Parsed JSON object if present and valid; otherwise an empty dict.
    """
    rp = run_dir / "run_meta.json"
    if not rp.exists():
        return {}
    try:
        return json.loads(rp.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _read_benchmark_id(run_dir: Path, fallback: str) -> str:
    """Get benchmark_id from run metadata with a fallback.

    Parameters
    ----------
    run_dir : pathlib.Path
        Run directory containing ``run_meta.json``.
    fallback : str
        Fallback benchmark id used when metadata is missing or empty.

    Returns
    -------
    str
        Benchmark id string (trimmed). Falls back to ``fallback``.
    """
    meta = _read_run_meta(run_dir)
    bid = str(meta.get("benchmark_id", "")).strip()
    return bid if bid else str(fallback or "").strip()


def _read_method(run_dir: Path, fallback: str) -> str:
    """Get method name from run metadata with a fallback.

    Parameters
    ----------
    run_dir : pathlib.Path
        Run directory containing ``run_meta.json``.
    fallback : str
        Fallback method name used when metadata is missing or empty.

    Returns
    -------
    str
        Method string (trimmed). Falls back to ``fallback``.
    """
    meta = _read_run_meta(run_dir)
    m = str(meta.get("method", "")).strip()
    return m if m else str(fallback or "").strip()


def _load_labels(path: Path) -> pd.DataFrame:
    """Load human labels from a TSV file.

    The input must include ``claim_id`` and one of:
    ``human_label``, ``human_decision``, or ``label``.
    Labels are normalized to upper-case and empty/NA-like values are
    dropped. Duplicate claim_ids keep the last occurrence.

    Parameters
    ----------
    path : pathlib.Path
        Path to a labels TSV.

    Returns
    -------
    pandas.DataFrame
        Two-column DataFrame with ``claim_id`` and ``human_label``.
        ``claim_id`` is stripped; ``human_label`` is upper-case.

    Raises
    ------
    ValueError
        If required columns are missing.
    """
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
    """Attach human labels to an audit log DataFrame.

    This performs a left join on ``claim_id`` and keeps audit rows even when
    a label is missing.

    Parameters
    ----------
    audit : pandas.DataFrame
        Audit log DataFrame (expects ``claim_id`` for joining).
    labels : pandas.DataFrame or None
        Output of ``_load_labels`` (may be None/empty).

    Returns
    -------
    pandas.DataFrame
        Audit DataFrame with an added ``human_label`` column when possible.

    Notes
    -----
    If ``claim_id`` is missing from the audit DataFrame, the input is
    returned unchanged.
    """
    if labels is None or labels.empty:
        return audit
    if "claim_id" not in audit.columns:
        return audit
    out = audit.copy()
    out["claim_id"] = out["claim_id"].astype(str).str.strip()
    return out.merge(labels, on="claim_id", how="left", validate="m:1")


def _compute_from_audit(audit: pd.DataFrame) -> dict[str, float]:
    """Compute risk/coverage metrics from an audit log.

    Required column:
    - ``status`` in {PASS, ABSTAIN, FAIL}

    Optional columns:
    - ``human_label`` (evaluated among PASS only)
    - ``audit_notes`` (for a debug-only proxy risk)

    Parameters
    ----------
    audit : pandas.DataFrame
        Audit log table.

    Returns
    -------
    dict of str to float
        Summary metrics including:
        - counts: n_total, n_pass, n_abstain, n_fail, n_answered
        - rates: coverage_pass, abstain_rate_total, fail_rate_total,
          fail_rate_answered
        - human risk (PASS only): risk_human_reject, risk_human_nonaccept,
          n_pass_labeled
        - debug proxy: risk_proxy

    Raises
    ------
    ValueError
        If the audit table lacks the required ``status`` column.

    Notes
    -----
    - ``answered`` is defined as PASS or FAIL (ABSTAIN excluded).
    - ``risk_proxy`` is for dev/debug only and should not be used as a paper
      risk metric.
    """
    if "status" not in audit.columns:
        raise ValueError("audit_log.tsv missing column: status")

    st = audit["status"].astype(str).str.upper().str.strip()

    n_total = float(len(audit))
    n_pass = float((st == "PASS").sum())
    n_abstain = float((st == "ABSTAIN").sum())
    n_fail = float((st == "FAIL").sum())

    coverage_pass = (n_pass / n_total) if n_total > 0 else float("nan")
    abstain_rate_total = (n_abstain / n_total) if n_total > 0 else float("nan")
    fail_rate_total = (n_fail / n_total) if n_total > 0 else float("nan")

    n_answered = float(n_pass + n_fail)  # answered = PASS or FAIL (exclude ABSTAIN)
    fail_rate_answered = (n_fail / n_answered) if n_answered > 0 else float("nan")

    # -------------------------
    # Optional: human labels (only evaluated among PASS)
    # -------------------------
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

    # -------------------------
    # Debug-only proxy: PASS notes containing "warning-ish" tokens
    # (Keep, but do NOT use as paper risk.)
    # -------------------------
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
        "n_abstain": float(n_abstain),
        "n_fail": float(n_fail),
        "n_answered": float(n_answered),
        "coverage_pass": float(coverage_pass),
        "abstain_rate_total": float(abstain_rate_total)
        if pd.notna(abstain_rate_total)
        else float("nan"),
        "fail_rate_total": float(fail_rate_total) if pd.notna(fail_rate_total) else float("nan"),
        "fail_rate_answered": float(fail_rate_answered)
        if pd.notna(fail_rate_answered)
        else float("nan"),
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
    """Collect Fig. 2 risk/coverage points from audit_log.tsv files.

    The script scans a root directory for audit logs in either layout:

    - New:
      ``<condition>/<VARIANT>/gate_<MODE>/tau_*/audit_log.tsv``
    - Old:
      ``<condition>/<VARIANT>/tau_*/audit_log.tsv``

    It parses tau from the path, attaches optional human labels, computes
    summary metrics, and writes a TSV table.

    Command-line arguments
    ----------------------
    --root : str
        Root directory to scan for audit logs.
    --out : str
        Output TSV path.
    --benchmark-id : str, optional
        Fallback benchmark id when run_meta.json is missing.
    --method : str, optional
        Fallback method name when run_meta.json is missing.
    --labels : str, optional
        Labels TSV path to compute human risk among PASS.

    Raises
    ------
    SystemExit
        If no audit logs are found, labels path is invalid, or 0 rows are
        collected unexpectedly.
    """
    ap = argparse.ArgumentParser(
        description=(
            "Collect Fig2 points from audit_log.tsv (dev/debug). "
            "Supports both layouts:\n"
            "  new: out/<condition>/<VARIANT>/gate_<MODE>/tau_*/audit_log.tsv\n"
            "  old: out/<condition>/<VARIANT>/tau_*/audit_log.tsv"
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
        condition = keys["condition"].strip()
        variant = keys["variant"].strip()
        gate_mode = keys["gate_mode"].strip()  # may be ""

        if not condition or not variant:
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
                "condition": condition,
                "method": method,
                "variant": variant,
                "gate_mode": gate_mode,
                "tau": float(tau),
                "coverage_pass": m["coverage_pass"],
                "abstain_rate_total": m["abstain_rate_total"],
                "fail_rate_total": m["fail_rate_total"],
                "fail_rate_answered": m["fail_rate_answered"],
                "n_total": m["n_total"],
                "n_pass": m["n_pass"],
                "n_abstain": m["n_abstain"],
                "n_fail": m["n_fail"],
                "n_answered": m["n_answered"],
                "risk_human_reject": m["risk_human_reject"],
                "risk_human_nonaccept": m["risk_human_nonaccept"],
                "n_pass_labeled": m["n_pass_labeled"],
                "risk_proxy": m["risk_proxy"],
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
