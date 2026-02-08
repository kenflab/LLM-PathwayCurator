#!/usr/bin/env python3
# paper/scripts/fig2_collect_abstain_reasons.py
from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


def _die(msg: str) -> None:
    raise SystemExit(msg)


def canon_variant(v: str, *, is_llm: bool = False) -> str:
    v = (v or "").strip().lower()
    if v == "ours":
        return "Proposed (LLM)" if is_llm else "Proposed"
    if v in ("context_swap", "shuffled_context", "shuffled-context"):
        return "Context swap"
    if v == "stress":
        return "Evidence dropout"
    return v or "Unknown"


def canon_reason(r: str) -> str:
    s = (r or "").strip()
    if not s:
        return "Unknown"
    k = s.lower()
    mapping = {
        "context_gate": "Context gate",
        "context_nonspecific": "Context-nonspecific",
        "inconclusive_stress": "Inconclusive stress",
        "unstable": "Unstable",
        "low_survival": "Low stability",
        "stability": "Low stability",
        "stability_gate": "Low stability",
    }
    if k in mapping:
        return mapping[k]
    k2 = k.replace("_", " ").strip()
    return (k2[:1].upper() + k2[1:]) if k2 else "Unknown"


def infer_condition_variant_tau(p: Path) -> tuple[str, str, float | None]:
    """
    Infer condition/variant/tau from an audit_log.tsv path.

    Supports:
      .../<out_like>/<condition>/<variant>/gate_*/tau_*/audit_log.tsv
    without requiring <out_like> to be literally 'out' or 'out_llm'.
    """
    parts = list(p.parts)
    condition = ""
    variant = ""
    tau: float | None = None

    # Preferred: locate gate_* and take the two preceding segments.
    # .../<condition>/<variant>/gate_*/tau_*/audit_log.tsv
    gate_idx = None
    for i, seg in enumerate(parts):
        if str(seg).startswith("gate_"):
            gate_idx = i
            break
    if gate_idx is not None and gate_idx >= 2:
        condition = str(parts[gate_idx - 2])
        variant = str(parts[gate_idx - 1])
    else:
        # Fallback: parent chain heuristic (variant is 3-levels up)
        # audit_log -> tau_* -> gate_* -> <variant> -> <condition>
        try:
            variant = p.parent.parent.parent.name
            condition = p.parent.parent.parent.parent.name
        except Exception:
            condition, variant = "", ""

    # tau from directory name like tau_0.80
    m = re.search(r"tau_([0-9]*\.[0-9]+|[0-9]+)", str(p))
    if m:
        try:
            tau = float(m.group(1))
        except Exception:
            tau = None

    return condition, variant, tau


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Collect Fig2d ABSTAIN reason source data from audit_log.tsv."
    )
    ap.add_argument(
        "--audit-logs", nargs="+", required=True, help="audit_log.tsv paths (one per variant)"
    )
    ap.add_argument("--out-tsv", required=True, help="output TSV path (Fig2d_abstain_reasons.tsv)")
    ap.add_argument(
        "--benchmark-id", default="", help="optional benchmark_id to fill (e.g., PANCAN_TP53_v1)"
    )
    ap.add_argument("--condition", default="", help="optional override condition (e.g., HNSC)")
    ap.add_argument(
        "--topk", type=int, default=5, help="keep top-k reasons globally; others -> Other"
    )
    args = ap.parse_args()

    rows = []
    for s in args.audit_logs:
        p = Path(s)
        df = pd.read_csv(p, sep="\t")

        if "status" not in df.columns or "abstain_reason" not in df.columns:
            _die(f"Missing required columns in {p}: need status + abstain_reason")

        # robust: any path segment containing 'out_llm'
        is_llm = any("out_llm" in str(seg) for seg in p.parts)
        cond_infer, var_infer, tau_infer = infer_condition_variant_tau(p)

        condition = str(args.condition).strip() or cond_infer or "NA"
        variant = canon_variant(var_infer, is_llm=is_llm)
        tau = tau_infer

        n_total = int(len(df))
        abd = df[df["status"] == "ABSTAIN"].copy()
        n_abstain = int(len(abd))

        if n_abstain == 0:
            rows.append(
                {
                    "benchmark_id": str(args.benchmark_id).strip() or "",
                    "condition": condition,
                    "tau": tau if tau is not None else "",
                    "variant": variant,
                    "reason": "None",
                    "n_total": n_total,
                    "n_abstain_total": 0,
                    "n_abstain_reason": 0,
                }
            )
            continue

        vc = abd["abstain_reason"].fillna("NA").astype(str).map(canon_reason).value_counts()
        for reason, c in vc.items():
            rows.append(
                {
                    "benchmark_id": str(args.benchmark_id).strip() or "",
                    "condition": condition,
                    "tau": tau if tau is not None else "",
                    "variant": variant,
                    "reason": str(reason),
                    "n_total": n_total,
                    "n_abstain_total": n_abstain,
                    "n_abstain_reason": int(c),
                }
            )

    tall = pd.DataFrame(rows)
    if tall.empty:
        _die("No rows generated.")

    # collapse to topk reasons globally (by count)
    top = (
        tall.groupby("reason", as_index=False)["n_abstain_reason"]
        .sum()
        .sort_values("n_abstain_reason", ascending=False)
        .head(int(args.topk))["reason"]
        .tolist()
    )
    tall["reason"] = tall["reason"].where(tall["reason"].isin(set(top)), other="Other")

    # regroup after collapsing
    out = tall.groupby(
        ["benchmark_id", "condition", "tau", "variant", "reason", "n_total", "n_abstain_total"],
        as_index=False,
    )["n_abstain_reason"].sum()
    denom = out["n_abstain_total"].replace(0, 1)
    out["frac_within_abstain"] = out["n_abstain_reason"] / denom

    out_path = Path(args.out_tsv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, sep="\t", index=False)
    print(f"[collect_abstain_reasons] OK -> {out_path}")


if __name__ == "__main__":
    main()
