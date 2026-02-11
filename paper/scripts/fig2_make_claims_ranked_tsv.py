#!/usr/bin/env python3
# paper/scripts/fig2_make_claims_ranked_tsv.py
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _ensure_src_on_path() -> None:
    # paper/scripts -> paper -> repo root
    root = Path(__file__).resolve().parents[2]
    src = root / "src"
    if src.exists():
        sys.path.insert(0, str(src))


def main(argv: list[str] | None = None) -> int:
    _ensure_src_on_path()

    from llm_pathway_curator.ranked import write_claims_ranked_tsv  # noqa: PLC0415

    ap = argparse.ArgumentParser(
        description=(
            "Build claims_ranked.tsv (figure-friendly shortlist) from audit_log.tsv "
            "(+ optional evidence.normalized.tsv). Delegates to src/llm_pathway_curator/ranked.py"
        )
    )
    ap.add_argument(
        "--run-dir", type=str, default="", help="Directory containing audit_log.tsv, etc."
    )
    ap.add_argument(
        "--audit-log", type=str, default="", help="audit_log.tsv (optional if --run-dir)."
    )
    ap.add_argument(
        "--evidence-tsv",
        type=str,
        default="",
        help="evidence.normalized.tsv (optional if --run-dir).",
    )
    ap.add_argument("--out-tsv", type=str, required=True, help="Output claims_ranked.tsv")
    ap.add_argument("--top-n", type=int, default=0, help="If >0, keep only top-N after ranking.")
    ap.add_argument(
        "--prefer-q",
        action="store_true",
        help="Prefer -log10(q) for evidence strength if available.",
    )
    ap.add_argument(
        "--evidence-score-col",
        type=str,
        default="",
        help="Override evidence score column in evidence TSV.",
    )
    ap.add_argument(
        "--default-direction",
        type=str,
        default="na",
        choices=["up", "down", "na"],
        help="Fill direction when missing/NA (useful for undirected ORA such as Metascape).",
    )
    ap.add_argument(
        "--fill-direction-if-na",
        action="store_true",
        help="If set, apply --default-direction only where direction is NA/missing.",
    )
    a = ap.parse_args(argv)

    out = write_claims_ranked_tsv(
        out_tsv=Path(a.out_tsv),
        run_dir=Path(a.run_dir) if a.run_dir else None,
        audit_log=Path(a.audit_log) if a.audit_log else None,
        evidence_tsv=Path(a.evidence_tsv) if a.evidence_tsv else None,
        top_n=int(a.top_n),
        prefer_q=bool(a.prefer_q),
        evidence_score_col=str(a.evidence_score_col or ""),
        default_direction=str(a.default_direction or "na"),
        fill_direction_if_na=bool(a.fill_direction_if_na),
    )
    print(f"[OK] wrote: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
