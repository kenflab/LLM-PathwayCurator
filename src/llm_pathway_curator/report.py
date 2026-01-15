# LLM-PathwayCurator/src/llm_pathway_curator/report.py
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .sample_card import SampleCard


def _safe_table_md(df: pd.DataFrame, n: int = 20) -> str:
    head = df.head(n).copy()
    try:
        return head.to_markdown(index=False)
    except Exception:
        return head.to_csv(sep="\t", index=False)


def _subset_cols(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    keep = [c for c in cols if c in df.columns]
    return df[keep].copy() if keep else df.copy()


def _stringify_list_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    TSV-safe conversion for list-like columns.
    - If a <col>_str already exists, prefer it.
    - Otherwise stringify list/tuple/set as comma-joined.
    """
    out = df.copy()
    for c in list(out.columns):
        if c.endswith("_str"):
            continue
        c_str = f"{c}_str"
        if c_str in out.columns:
            continue

        # detect list-like objects in this column (cheap sample)
        s = out[c]
        if s.dtype == "object":
            sample = s.dropna().head(20).tolist()
            if any(isinstance(x, (list, tuple, set)) for x in sample):
                out[c_str] = s.map(
                    lambda x: ",".join(map(str, x))
                    if isinstance(x, (list, tuple, set))
                    else ("" if pd.isna(x) else str(x))
                )
    return out


def _reason_summary(audit_log: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Return (abstain_summary, fail_summary) as small tables.
    """
    if "status" not in audit_log.columns:
        return (pd.DataFrame(), pd.DataFrame())

    abs_df = audit_log[audit_log["status"] == "ABSTAIN"]
    fail_df = audit_log[audit_log["status"] == "FAIL"]

    abs_sum = (
        abs_df["abstain_reason"]
        .fillna("")
        .replace({"": "(missing)"})
        .value_counts()
        .rename_axis("abstain_reason")
        .reset_index(name="n")
        if "abstain_reason" in abs_df.columns
        else pd.DataFrame()
    )
    fail_sum = (
        fail_df["fail_reason"]
        .fillna("")
        .replace({"": "(missing)"})
        .value_counts()
        .rename_axis("fail_reason")
        .reset_index(name="n")
        if "fail_reason" in fail_df.columns
        else pd.DataFrame()
    )
    return abs_sum, fail_sum


def write_report(
    audit_log: pd.DataFrame, distilled: pd.DataFrame, card: SampleCard, outdir: str
) -> None:
    out_path = Path(outdir)
    out_path.mkdir(parents=True, exist_ok=True)

    # TSV-safe artifacts
    audit_out = _stringify_list_columns(audit_log)
    dist_out = _stringify_list_columns(distilled)

    # Prefer canonical string columns if present
    if "evidence_genes_str" in dist_out.columns:
        # keep original list column for in-memory use, but TSV uses *_str
        pass

    audit_tsv = out_path / "audit_log.tsv"
    distilled_tsv = out_path / "distilled.tsv"
    audit_out.to_csv(audit_tsv, sep="\t", index=False)
    dist_out.to_csv(distilled_tsv, sep="\t", index=False)

    status_col = "status" if "status" in audit_log.columns else None
    if status_col is None:
        n_pass = n_abs = n_fail = 0
    else:
        n_pass = int((audit_log[status_col] == "PASS").sum())
        n_abs = int((audit_log[status_col] == "ABSTAIN").sum())
        n_fail = int((audit_log[status_col] == "FAIL").sum())

    # small reason summaries
    abs_sum, fail_sum = _reason_summary(audit_log)

    # Stable “top” ordering
    def _top(df: pd.DataFrame) -> pd.DataFrame:
        cols = df.columns
        if "term_survival" in cols:
            return df.sort_values(
                ["term_survival", "claim_id"], ascending=[False, True], kind="mergesort"
            )
        if "stat" in cols:
            return df.sort_values(["stat", "claim_id"], ascending=[False, True], kind="mergesort")
        return df.sort_values(["claim_id"], ascending=[True], kind="mergesort")

    lines: list[str] = []
    lines.append("# LLM-PathwayCurator report")
    lines.append("")
    lines.append("## Sample Card")
    lines.append(f"- disease: {getattr(card, 'disease', '')}")
    lines.append(f"- tissue: {getattr(card, 'tissue', '')}")
    lines.append(f"- perturbation: {getattr(card, 'perturbation', '')}")
    lines.append(f"- comparison: {getattr(card, 'comparison', '')}")
    notes = getattr(card, "notes", None)
    if notes:
        lines.append(f"- notes: {notes}")
    lines.append("")
    lines.append(f"## Decisions (PASS/ABSTAIN/FAIL): {n_pass}/{n_abs}/{n_fail}")
    lines.append("")

    if not abs_sum.empty or not fail_sum.empty:
        lines.append("## Reason summary")
        lines.append("")
        if not abs_sum.empty:
            lines.append("### ABSTAIN reasons")
            lines.append(_safe_table_md(abs_sum, n=50))
            lines.append("")
        if not fail_sum.empty:
            lines.append("### FAIL reasons")
            lines.append(_safe_table_md(fail_sum, n=50))
            lines.append("")

    # PASS
    lines.append("## PASS (top)")
    lines.append("")
    pass_df = audit_log[audit_log[status_col] == "PASS"] if status_col else audit_log.iloc[0:0]
    pass_df = _top(pass_df)
    lines.append(
        _safe_table_md(
            _subset_cols(
                pass_df, ["claim_id", "entity", "direction", "module_id", "term_ids", "audit_notes"]
            ),
            n=5,
        )
    )
    lines.append("")

    # ABSTAIN
    lines.append("## ABSTAIN (top)")
    lines.append("")
    abs_df = audit_log[audit_log[status_col] == "ABSTAIN"] if status_col else audit_log.iloc[0:0]
    abs_df = _top(abs_df)
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

    # FAIL
    lines.append("## FAIL (top)")
    lines.append("")
    fail_df = audit_log[audit_log[status_col] == "FAIL"] if status_col else audit_log.iloc[0:0]
    fail_df = _top(fail_df)
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
    lines.append(_safe_table_md(_stringify_list_columns(audit_log), n=20))
    lines.append("")
    lines.append("## Reproducible artifacts")
    lines.append("")
    lines.append(f"- audit_log.tsv: {audit_tsv.name}")
    lines.append(f"- distilled.tsv: {distilled_tsv.name}")
    lines.append("")

    report_path = out_path / "report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
