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
        # pandas.DataFrame.to_markdown may rely on optional deps; keep it safe.
        return head.to_markdown(index=False)
    except Exception:
        # Minimal fallback: tab-separated text
        return head.to_csv(sep="\t", index=False)


def write_report(
    audit_log: pd.DataFrame, distilled: pd.DataFrame, card: SampleCard, outdir: str
) -> None:
    """
    Write a minimal decision-grade report. Must not fail silently.
    """
    out_path = Path(outdir)
    out_path.mkdir(parents=True, exist_ok=True)

    report_path = out_path / "report.md"

    n_pass = int((audit_log["status"] == "PASS").sum()) if "status" in audit_log.columns else 0
    n_abs = int((audit_log["status"] == "ABSTAIN").sum()) if "status" in audit_log.columns else 0
    n_fail = int((audit_log["status"] == "FAIL").sum()) if "status" in audit_log.columns else 0

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
    lines.append("## Audit log (top)")
    lines.append("")
    lines.append(_safe_table_md(audit_log, n=20))
    lines.append("")

    report_path.write_text("\n".join(lines), encoding="utf-8")
