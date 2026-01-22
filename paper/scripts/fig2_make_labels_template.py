#!/usr/bin/env python3
# paper/scripts/fig2_make_labels_template.py
from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


def _infer_keys_from_path(audit_path: Path, root: Path) -> dict[str, str]:
    """
    Supports BOTH layouts:
      new: root/<CANCER>/<VARIANT>/gate_<MODE>/tau_*/audit_log.tsv
      old: root/<CANCER>/<VARIANT>/tau_*/audit_log.tsv
    """
    rel = audit_path.relative_to(root)
    parts = rel.parts
    out = {"cancer": "", "variant": "", "gate_mode": "", "tau": ""}

    if len(parts) < 3:
        return out

    out["cancer"] = str(parts[0])
    out["variant"] = str(parts[1])

    third = str(parts[2])
    if third.startswith("gate_"):
        out["gate_mode"] = third.replace("gate_", "", 1)

    m = re.search(r"tau_([0-9]*\.?[0-9]+)", str(audit_path))
    if m:
        out["tau"] = m.group(1)
    return out


def _iter_audit_logs(root: Path) -> list[Path]:
    files_new = list(root.glob("*/*/gate_*/tau_*/audit_log.tsv"))
    files_old = list(root.glob("*/*/tau_*/audit_log.tsv"))
    files = sorted({p.resolve(): p for p in (files_new + files_old)}.values())
    return files


def _pick_candidates(df: pd.DataFrame, n: int, prefer_status: str = "PASS") -> pd.DataFrame:
    """
    Pick labeling candidates per audit_log:
    - Prefer a given status (default PASS) to match Fig2 risk-in-pass.
    - Fall back to all rows if not enough.
    """
    if n <= 0:
        return df
    if "status" in df.columns:
        s = df["status"].astype(str).str.upper().str.strip()
        preferred = df[s == prefer_status].copy()
        if len(preferred) >= n:
            return preferred.head(n)
        # If not enough PASS, include remaining from all rows (stable order)
        rest = df[~df.index.isin(preferred.index)]
        return pd.concat([preferred, rest], axis=0).head(n)
    return df.head(n)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Make Fig2 labels.tsv template from audit_log.tsv files (bench-compatible)."
    )
    ap.add_argument(
        "--root",
        required=True,
        help="Fig2 out root (e.g., paper/source_data/PANCAN_TP53_v1/out)",
    )
    ap.add_argument("--out", required=True, help="Output labels.tsv path")
    ap.add_argument("--rater-id", default="R1")
    ap.add_argument("--label-version", default="v1")
    ap.add_argument(
        "--per-run",
        type=int,
        default=0,
        help="If >0, sample this many candidates per audit_log.tsv (0 = include all claim_ids).",
    )
    ap.add_argument(
        "--prefer-status",
        default="PASS",
        help="When --per-run>0, prefer this status first (PASS/ABSTAIN/FAIL). Default PASS.",
    )
    ap.add_argument(
        "--with-hints",
        action="store_true",
        help="Add helper columns for faster human labeling (ignored by bench).",
    )
    args = ap.parse_args()

    root = Path(args.root).resolve()
    logs = _iter_audit_logs(root)
    if not logs:
        raise SystemExit(f"No audit_log.tsv found under: {root}")

    rows: list[dict[str, str]] = []
    for p in logs:
        keys = _infer_keys_from_path(p, root)
        audit = pd.read_csv(p, sep="\t", dtype=str).fillna("")
        if "claim_id" not in audit.columns:
            continue

        audit = _pick_candidates(
            audit,
            n=int(args.per_run),
            prefer_status=str(args.prefer_status).upper().strip(),
        )

        for _, r in audit.iterrows():
            cid = str(r.get("claim_id", "")).strip()
            if not cid:
                continue

            rec = {
                # bench_fig2.py expects these columns
                "claim_id": cid,
                # fill later: ACCEPT/REJECT/SHOULD_ABSTAIN (partial labels allowed)
                "human_label": "",
                "rater_id": str(args.rater_id).strip(),
                "label_version": str(args.label_version).strip(),
            }

            if args.with_hints:
                # Optional helper columns (do not affect bench)
                rec.update(
                    {
                        "cancer": keys["cancer"],
                        "variant": keys["variant"],
                        "gate_mode": keys["gate_mode"],
                        "tau": keys["tau"],
                        "entity": str(r.get("entity", "")),
                        "direction": str(r.get("direction", "")),
                        "term_name": str(r.get("term_name", "")),
                        "module_id_effective": str(
                            r.get("module_id_effective", r.get("module_id", ""))
                        ),
                        "gene_symbols_str": str(r.get("gene_symbols_str", "")),
                        "term_ids_str": str(r.get("term_ids_str", "")),
                        "term_survival_agg": str(r.get("term_survival_agg", "")),
                        "context_score": str(r.get("context_score", "")),
                        "status": str(r.get("status", "")),
                        "abstain_reason": str(r.get("abstain_reason", "")),
                        "fail_reason": str(r.get("fail_reason", "")),
                        "audit_notes": str(r.get("audit_notes", "")),
                        "claim_json": str(r.get("claim_json", "")),
                        "_src": str(p),
                    }
                )

            rows.append(rec)

    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit("No claim_id rows collected (unexpected).")

    df["claim_id"] = df["claim_id"].astype(str).str.strip()
    df = df.drop_duplicates(subset=["claim_id"], keep="first").reset_index(drop=True)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, sep="\t", index=False)

    print(f"[OK] wrote: {out_path} rows={len(df)}")
    print("[NEXT] Fill human_label with: ACCEPT / REJECT / SHOULD_ABSTAIN (partial labels ok)")


if __name__ == "__main__":
    main()
