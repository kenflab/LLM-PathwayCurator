# LLM-PathwayCurator/src/llm_pathway_curator/report.py

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .sample_card import SampleCard


def _safe_table_md(df: pd.DataFrame, n: int = 20) -> str:
    """
    Always return a text table.
    Prefer markdown if available; fallback to TSV-like text.
    """
    head = df.head(n).copy()
    try:
        return head.to_markdown(index=False)
    except Exception:
        return head.to_csv(sep="\t", index=False)


def _subset_cols(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    keep = [c for c in cols if c in df.columns]
    return df[keep].copy() if keep else df.copy()


def write_report(
    audit_log: pd.DataFrame, distilled: pd.DataFrame, card: SampleCard, outdir: str
) -> None:
    """
    Write a minimal decision-grade report. Must not fail silently.
    Produces:
      - report.md
      - audit_log.tsv
      - distilled.tsv
    """
    out_path = Path(outdir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Always persist machine-readable artifacts (reproducible output)
    audit_tsv = out_path / "audit_log.tsv"
    distilled_tsv = out_path / "distilled.tsv"
    audit_log.to_csv(audit_tsv, sep="\t", index=False)
    distilled.to_csv(distilled_tsv, sep="\t", index=False)

    status_col = "status" if "status" in audit_log.columns else None
    if status_col is None:
        n_pass = n_abs = n_fail = 0
    else:
        n_pass = int((audit_log[status_col] == "PASS").sum())
        n_abs = int((audit_log[status_col] == "ABSTAIN").sum())
        n_fail = int((audit_log[status_col] == "FAIL").sum())

    lines: list[str] = []
    lines.append("# LLM-PathwayCurator report")
    lines.append("")
    lines.append("## Sample Card")
    lines.append(f"- disease: {card.disease}")
    lines.append(f"- tissue: {card.tissue}")
    lines.append(f"- perturbation: {card.perturbation}")
    lines.append(f"- comparison: {card.comparison}")
    if card.notes:
        lines.append(f"- notes: {card.notes}")
    lines.append("")
    lines.append(f"## Decisions (PASS/ABSTAIN/FAIL): {n_pass}/{n_abs}/{n_fail}")
    lines.append("")
    lines.append("## PASS (top)")
    lines.append("")
    if status_col:
        pass_df = audit_log[audit_log[status_col] == "PASS"]
    else:
        pass_df = audit_log.iloc[0:0]
    lines.append(
        _safe_table_md(
            _subset_cols(
                pass_df,
                ["claim_id", "entity", "direction", "module_id", "term_ids", "audit_notes"],
            ),
            n=5,
        )
    )
    lines.append("")
    lines.append("## ABSTAIN (top)")
    lines.append("")
    if status_col:
        abs_df = audit_log[audit_log[status_col] == "ABSTAIN"]
    else:
        abs_df = audit_log.iloc[0:0]
    lines.append(
        _safe_table_md(
            _subset_cols(
                abs_df,
                ["claim_id", "entity", "direction", "module_id", "abstain_reason", "audit_notes"],
            ),
            n=5,
        )
    )
    lines.append("")
    lines.append("## FAIL (top)")
    lines.append("")
    if status_col:
        fail_df = audit_log[audit_log[status_col] == "FAIL"]
    else:
        fail_df = audit_log.iloc[0:0]
    lines.append(
        _safe_table_md(
            _subset_cols(
                fail_df,
                ["claim_id", "entity", "direction", "module_id", "fail_reason", "audit_notes"],
            ),
            n=5,
        )
    )
    lines.append("")
    lines.append("## Audit log (top)")
    lines.append("")
    lines.append(_safe_table_md(audit_log, n=20))
    lines.append("")
    lines.append("## Reproducible artifacts")
    lines.append("")
    lines.append(f"- audit_log.tsv: {audit_tsv.name}")
    lines.append(f"- distilled.tsv: {distilled_tsv.name}")
    lines.append("")

    report_path = out_path / "report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
