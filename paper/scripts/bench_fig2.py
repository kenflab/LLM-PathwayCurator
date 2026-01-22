#!/usr/bin/env python3
# paper/scripts/bench_fig2.py
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd


def _iter_report_paths(inputs: list[str]) -> list[Path]:
    paths: list[Path] = []
    for s in inputs:
        p = Path(s)
        if any(ch in s for ch in ["*", "?", "["]):
            paths.extend([Path(x) for x in sorted(p.parent.glob(p.name))])
            continue
        if p.is_dir():
            paths.extend(sorted(p.rglob("report.jsonl")))
            continue
        paths.append(p)

    seen = set()
    out: list[Path] = []
    for p in paths:
        rp = p.resolve()
        if rp not in seen:
            seen.add(rp)
            out.append(p)
    return out


def _get_nested(d: dict[str, Any], keys: tuple[str, ...], default: Any = None) -> Any:
    cur: Any = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _is_fail(rec: dict[str, Any]) -> bool:
    """
    v1-safe FAIL detection:
    - Prefer audit_result / decision if present.
    - Else: audit_flags keys starting with FAIL_ and True count as FAIL.
    """
    ar = str(rec.get("audit_result", "") or rec.get("decision", "")).strip().upper()
    if ar == "FAIL":
        return True
    if ar in {"PASS", "ABSTAIN"}:
        return False

    audit_flags = rec.get("audit_flags", None)
    if not isinstance(audit_flags, dict):
        return False

    for k, v in audit_flags.items():
        kk = str(k).strip().upper()
        if kk.startswith("FAIL_") and bool(v):
            return True
    return False


def _parse_taus(arg: str) -> list[float]:
    s = str(arg).strip()
    if ":" in s:
        parts = s.split(":")
        if len(parts) != 3:
            raise ValueError("--taus must be start:stop:step (e.g., 0:1:0.05)")
        start, stop, step = map(float, parts)
        if step <= 0:
            raise ValueError("tau step must be > 0")
        taus: list[float] = []
        t = start
        while t <= stop + 1e-12:
            taus.append(round(t, 6))
            t += step
        return taus
    vals = [v.strip() for v in s.split(",") if v.strip()]
    if not vals:
        raise ValueError("--taus must not be empty")
    return [round(float(v), 6) for v in vals]


def load_reports(report_paths: list[Path]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for path in report_paths:
        if not path.exists():
            raise FileNotFoundError(f"report.jsonl not found: {path}")

        with path.open("r", encoding="utf-8") as f:
            for ln, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError as e:
                    raise ValueError(f"Invalid JSON in {path} line {ln}: {e}") from e

                benchmark_id = str(rec.get("benchmark_id", "")).strip()
                cancer = str(rec.get("cancer", "")).strip()
                method = str(rec.get("method", "")).strip()
                claim_id = str(rec.get("claim_id", "")).strip()

                # New recommended fields (may be empty for older outputs)
                variant = str(rec.get("variant", "")).strip()
                gate_mode = str(rec.get("gate_mode", "")).strip()

                if not benchmark_id or not cancer or not method or not claim_id:
                    raise ValueError(
                        f"Missing required fields in {path} line {ln}: "
                        "benchmark_id/cancer/method/claim_id"
                    )

                # v1 schema: metrics.survival is required for tau sweep
                survival = _get_nested(rec, ("metrics", "survival"), None)
                if survival is None:
                    raise ValueError(
                        f"Missing metrics.survival in {path} line {ln} (claim_id={claim_id})"
                    )
                try:
                    survival_f = float(survival)
                except Exception as e:
                    raise ValueError(
                        f"metrics.survival not float-like in {path} line {ln} (claim_id={claim_id})"
                    ) from e
                if math.isnan(survival_f):
                    raise ValueError(
                        f"metrics.survival is NaN in {path} line {ln} (claim_id={claim_id})"
                    )

                is_fail = _is_fail(rec)

                rows.append(
                    dict(
                        benchmark_id=benchmark_id,
                        cancer=cancer,
                        method=method,
                        variant=variant,
                        gate_mode=gate_mode,
                        claim_id=claim_id,
                        survival=survival_f,
                        is_fail=bool(is_fail),
                    )
                )

    df = pd.DataFrame.from_records(rows)
    if df.empty:
        raise ValueError("No records loaded from report.jsonl inputs.")

    df["variant"] = df["variant"].fillna("").astype(str)
    df["gate_mode"] = df["gate_mode"].fillna("").astype(str)

    df = df.drop_duplicates(
        subset=["benchmark_id", "cancer", "method", "variant", "gate_mode", "claim_id"],
        keep="first",
    )
    return df


def load_labels(labels_path: Path, rater_id: str = "R1") -> pd.DataFrame:
    """
    labels.tsv contract (v1):
      required columns: claim_id, human_label, rater_id, label_version
      allowed human_label: ACCEPT / REJECT / SHOULD_ABSTAIN

    IMPORTANT:
      - Partial labeling is allowed: empty/NA human_label rows are dropped.
      - This enables labeling a small subset of claims without breaking the pipeline.
    """
    if not labels_path.exists():
        raise FileNotFoundError(f"labels.tsv not found: {labels_path}")

    lab = pd.read_csv(labels_path, sep="\t", dtype=str).fillna("")
    required = {"claim_id", "human_label", "rater_id", "label_version"}
    missing = required - set(lab.columns)
    if missing:
        raise ValueError(f"labels.tsv missing columns: {sorted(missing)}")

    lab = lab.copy()
    lab["rater_id"] = lab["rater_id"].astype(str).str.strip()
    lab = lab[lab["rater_id"] == str(rater_id)].copy()
    if lab.empty:
        raise ValueError(f"No rows for rater_id={rater_id} in {labels_path}")

    lab["claim_id"] = lab["claim_id"].astype(str).str.strip()
    lab["human_label"] = lab["human_label"].astype(str).str.upper().str.strip()

    # Allow partial labels: drop blanks/NA tokens
    na = lab["human_label"].isin(["", "NA", "NAN", "NONE"])
    lab = lab[~na].copy()
    if lab.empty:
        raise ValueError(
            f"No non-empty human_label for rater_id={rater_id} in {labels_path} "
            "(fill at least 1 label: ACCEPT/REJECT/SHOULD_ABSTAIN)"
        )

    allowed = {"ACCEPT", "REJECT", "SHOULD_ABSTAIN"}
    bad = sorted(set(lab["human_label"]) - allowed)
    if bad:
        raise ValueError(f"labels.tsv contains invalid human_label values: {bad}")

    lab = lab.drop_duplicates(subset=["claim_id"], keep="last")
    return lab[["claim_id", "human_label"]]


def compute_risk_coverage(
    reports: pd.DataFrame, labels: pd.DataFrame, taus: list[float]
) -> pd.DataFrame:
    rep = reports.merge(labels, on="claim_id", how="left", validate="many_to_one")
    rep["human_label"] = rep["human_label"].fillna("")
    rep["is_labeled"] = rep["human_label"].ne("")

    group_cols = ["benchmark_id", "cancer", "method", "variant", "gate_mode"]

    out_rows: list[dict[str, Any]] = []
    for keys, g in rep.groupby(group_cols, sort=True):
        benchmark_id, cancer, method, variant, gate_mode = keys
        n_total = int(len(g))
        if n_total == 0:
            continue

        n_fail = int(g["is_fail"].sum())
        fail_rate = n_fail / n_total
        ok = ~g["is_fail"]

        n_labeled_total = int(g["is_labeled"].sum())

        for tau in taus:
            pass_mask = ok & (g["survival"] >= tau)
            abstain_mask = ok & (g["survival"] < tau)

            n_pass = int(pass_mask.sum())
            n_abstain = int(abstain_mask.sum())
            coverage_pass = n_pass / n_total

            pass_labeled = pass_mask & g["is_labeled"]
            n_pass_labeled = int(pass_labeled.sum())

            n_reject_in_pass = int((pass_labeled & (g["human_label"] == "REJECT")).sum())
            n_should_abstain_in_pass = int(
                (pass_labeled & (g["human_label"] == "SHOULD_ABSTAIN")).sum()
            )
            n_nonaccept_in_pass = n_reject_in_pass + n_should_abstain_in_pass

            risk_reject_in_pass = (
                (n_reject_in_pass / n_pass_labeled) if n_pass_labeled > 0 else float("nan")
            )
            risk_nonaccept_in_pass = (
                (n_nonaccept_in_pass / n_pass_labeled) if n_pass_labeled > 0 else float("nan")
            )

            out_rows.append(
                dict(
                    benchmark_id=benchmark_id,
                    cancer=cancer,
                    method=method,
                    variant=variant,
                    gate_mode=gate_mode,
                    tau=float(tau),
                    n_total=n_total,
                    n_fail=n_fail,
                    fail_rate=fail_rate,
                    n_pass=n_pass,
                    n_abstain=n_abstain,
                    coverage_pass=coverage_pass,
                    n_labeled_total=n_labeled_total,
                    n_pass_labeled=n_pass_labeled,
                    n_reject_in_pass=n_reject_in_pass,
                    n_should_abstain_in_pass=n_should_abstain_in_pass,
                    risk_reject_in_pass=risk_reject_in_pass,
                    risk_nonaccept_in_pass=risk_nonaccept_in_pass,
                    label_policy=f"{str(getattr(labels, 'name', 'R1_only') or 'R1_only')}",
                )
            )

    out = pd.DataFrame(out_rows)
    if out.empty:
        raise ValueError("No output rows produced. Check inputs.")
    return out.sort_values(group_cols + ["tau"], kind="mergesort").reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Fig2 v1: tau sweep risk–coverage from report.jsonl + labels.tsv "
            "(partial labels allowed)."
        )
    )
    ap.add_argument("--reports", nargs="+", required=True, help="report.jsonl file/dir/glob")
    ap.add_argument("--labels", required=True, help="labels.tsv")
    ap.add_argument("--out", required=True, help="output risk_coverage.tsv")
    ap.add_argument("--taus", default="0:1:0.05", help="comma list or start:stop:step")
    ap.add_argument("--rater-id", default="R1", help="Rater ID to use (v1 default: R1)")
    args = ap.parse_args()

    report_paths = _iter_report_paths(args.reports)
    if not report_paths:
        raise SystemExit("No report.jsonl files found from --reports.")

    reports_df = load_reports(report_paths)
    labels_df = load_labels(Path(args.labels), rater_id=str(args.rater_id))
    taus = _parse_taus(str(args.taus))

    rc = compute_risk_coverage(reports_df, labels_df, taus)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rc.to_csv(out_path, sep="\t", index=False)


if __name__ == "__main__":
    main()
