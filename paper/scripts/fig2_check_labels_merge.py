#!/usr/bin/env python3
# paper/scripts/fig2_check_labels_merge.py
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

_ALLOWED = {"", "ACCEPT", "REJECT", "SHOULD_ABSTAIN"}


def _iter_audit_logs(root: Path) -> list[Path]:
    """Iterate audit_log.tsv files under a Fig. 2 output root.

    This collects both supported directory layouts:

    - new: ``*/*/gate_*/tau_*/audit_log.tsv``
    - old: ``*/*/tau_*/audit_log.tsv``

    Parameters
    ----------
    root : pathlib.Path
        Root directory containing per-condition run outputs.

    Returns
    -------
    list of pathlib.Path
        Sorted list of unique audit_log.tsv paths.
    """
    files_new = list(root.glob("*/*/gate_*/tau_*/audit_log.tsv"))
    files_old = list(root.glob("*/*/tau_*/audit_log.tsv"))
    return sorted({p.resolve(): p for p in (files_new + files_old)}.values())


def _infer_condition_from_path(audit_path: Path, root: Path) -> str:
    """Infer condition name from an audit_log.tsv path.

    Parameters
    ----------
    audit_path : pathlib.Path
        Path to ``audit_log.tsv`` under ``root``.
    root : pathlib.Path
        Root directory used to compute the relative path.

    Returns
    -------
    str
        Condition name inferred as the first component of the relative path.
        Returns an empty string if inference fails.
    """
    rel = audit_path.relative_to(root)
    return str(rel.parts[0]) if len(rel.parts) >= 1 else ""


def _load_labels(path: Path) -> pd.DataFrame:
    """Load labels.tsv for merge checks.

    The input must contain at least ``claim_id`` and ``human_label``.
    Values are normalized (strip + upper-case) and missing values are
    filled with empty strings.

    Parameters
    ----------
    path : pathlib.Path
        Path to ``labels.tsv``.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing ``claim_id`` and ``human_label`` (and any other
        columns present in the input).

    Raises
    ------
    ValueError
        If required columns are missing.
    """
    df = pd.read_csv(path, sep="\t", dtype=str).fillna("")
    if "claim_id" not in df.columns:
        raise ValueError(f"labels.tsv missing claim_id: {path}")
    if "human_label" not in df.columns:
        raise ValueError(f"labels.tsv missing human_label: {path}")

    df["claim_id"] = df["claim_id"].astype(str).str.strip()
    df["human_label"] = df["human_label"].astype(str).str.upper().str.strip()

    return df


def main() -> None:
    """Check labels.tsv quality and merge-ability against audit_log.tsv.

    The script performs:
    1) Duplicate ``claim_id`` detection in labels.
    2) Validation of ``human_label`` values against an allowed set.
    3) Collection of claim_ids from audit logs under ``--root`` (optionally
       filtered by ``--condition``).
    4) Coverage checks: labeled claim_ids found vs missing in audit logs.
    5) Counts of non-blank labels targeting a preferred audit status
       (default PASS).

    Command-line arguments
    ----------------------
    --labels : str
        Path to labels.tsv.
    --root : str
        Fig2 output root containing audit_log.tsv files.
    --condition : str, optional
        If set, restrict checks to a single condition (e.g., HNSC).
    --prefer-status : str, optional
        Audit status used for the "labeled among status" report
        (default PASS).

    Raises
    ------
    SystemExit
        If no audit_log.tsv files are found under ``--root``.
    """
    ap = argparse.ArgumentParser(
        description="Check labels.tsv quality and merge-ability against audit_log.tsv."
    )
    ap.add_argument("--labels", required=True, help="labels.tsv")
    ap.add_argument("--root", required=True, help="Fig2 out root (contains audit_log.tsv)")
    ap.add_argument(
        "--condition", default="", help="If set, only check this condition (e.g., HNSC)."
    )
    ap.add_argument(
        "--prefer-status",
        default="PASS",
        help="Report labeled counts among this status (default PASS).",
    )
    args = ap.parse_args()

    labels_path = Path(args.labels)
    root = Path(args.root).resolve()

    lab = _load_labels(labels_path)

    # 1) duplicates in labels
    dup = lab["claim_id"].duplicated(keep=False)
    n_dup = int(dup.sum())
    if n_dup:
        print(f"[WARN] labels has duplicated claim_id rows: {n_dup}")
        print(lab.loc[dup, ["claim_id", "human_label"]].head(20).to_string(index=False))
    else:
        print("[OK] labels claim_id unique")

    # 2) allowed label values
    bad_val = ~lab["human_label"].isin(_ALLOWED)
    if int(bad_val.sum()):
        print(f"[WARN] labels has invalid human_label values: {int(bad_val.sum())}")
        print(lab.loc[bad_val, ["claim_id", "human_label"]].head(20).to_string(index=False))
        print(f"[INFO] allowed = {sorted(_ALLOWED)}")
    else:
        print("[OK] labels human_label values look valid (or blank)")

    # 3) gather audit claim_ids (optionally filtered by condition)
    logs = _iter_audit_logs(root)
    if not logs:
        raise SystemExit(f"No audit_log.tsv found under: {root}")

    prefer = str(args.prefer_status).upper().strip()

    all_claims = set()
    prefer_claims = set()
    n_logs_used = 0

    for p in logs:
        cond = _infer_condition_from_path(p, root)
        if str(args.condition).strip() and cond != str(args.condition).strip():
            continue

        df = pd.read_csv(p, sep="\t", dtype=str).fillna("")
        if "claim_id" not in df.columns:
            continue

        n_logs_used += 1
        df["claim_id"] = df["claim_id"].astype(str).str.strip()
        all_claims.update(set(df["claim_id"].tolist()))

        if "status" in df.columns:
            st = df["status"].astype(str).str.upper().str.strip()
            prefer_claims.update(set(df.loc[st == prefer, "claim_id"].tolist()))

    print(f"[OK] audit logs used: {n_logs_used}")
    print(f"[OK] unique claim_id in audit (all statuses): {len(all_claims)}")
    if prefer_claims:
        print(f"[OK] unique claim_id in audit (status={prefer}): {len(prefer_claims)}")

    # 4) coverage: how many labels map to audit claims?
    lab_claims = set(lab["claim_id"].tolist())
    in_audit = lab_claims & all_claims
    missing_in_audit = lab_claims - all_claims

    print(f"[OK] labels rows: {len(lab)}")
    print(f"[OK] labels claim_id found in audit: {len(in_audit)}")
    if missing_in_audit:
        print(
            f"[WARN] labels claim_id NOT found in audit: {len(missing_in_audit)} (showing up to 20)"
        )
        for x in list(sorted(missing_in_audit))[:20]:
            print(f"  - {x}")

    # 5) labeled among preferred status (PASS usually)
    if prefer_claims:
        # labeled = non-blank label
        labeled_nonblank = set(lab.loc[lab["human_label"] != "", "claim_id"].tolist())
        labeled_in_prefer = labeled_nonblank & prefer_claims
        print(f"[OK] non-blank labels total: {len(labeled_nonblank)}")
        print(f"[OK] non-blank labels among audit status={prefer}: {len(labeled_in_prefer)}")
        if len(labeled_nonblank) > 0:
            frac = len(labeled_in_prefer) / len(labeled_nonblank)
            print(f"[INFO] fraction of non-blank labels that target status={prefer}: {frac:.3f}")

    print("[DONE] If this is clean, pass --labels to fig2_collect_risk_coverage.py")


if __name__ == "__main__":
    main()
