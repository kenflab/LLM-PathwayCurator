# LLM-PathwayCurator/src/llm_pathway_curator/schema.py
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

import pandas as pd

from . import _shared

# v1 contract (tool-facing): we MUST preserve term×gene.
# Practical reality:
# - ORA often has no direction
# - some exports have no qval (or only p-value)
# We therefore enforce a "core required" set and auto-fill the rest.

CORE_REQUIRED_EVIDENCE_COLS = [
    "term_id",
    "term_name",
    "stat",
    "evidence_genes",
]

# Columns that should exist after normalization (auto-filled if missing).
NORMALIZED_COLS = [
    "term_id",
    "term_name",
    "source",
    "stat",
    "qval",
    "direction",
    "evidence_genes",
]

# NOTE:
# Gene parsing is spec-defined in _shared.py (parse_genes / split_gene_string / clean_gene_token).
# Do not duplicate parsing regexes here to avoid contract drift.

# Excel formula injection starters.
_EXCEL_FORMULA_START = ("=", "+", "-", "@")


ALIASES = {
    # -------------------------
    # term id/name
    # -------------------------
    "term": "term_id",
    "termid": "term_id",
    "term_id": "term_id",
    "id": "term_id",
    "term_identifier": "term_id",
    "termid_": "term_id",
    # names/descriptions
    "description": "term_name",
    "name": "term_name",
    "term_name": "term_name",
    "desc": "term_name",
    "term_description": "term_name",
    "termname": "term_name",
    "gene_set_name": "term_name",
    # -------------------------
    # source / database
    # -------------------------
    "source": "source",
    "database": "source",
    "db": "source",
    "category": "source",
    "collection": "source",
    # -------------------------
    # stat (effect/strength)
    # -------------------------
    "nes": "stat",
    "score": "stat",
    "stat": "stat",
    "logp": "stat",
    "log_p": "stat",
    "log(p)": "stat",
    "_log10p": "stat",
    "_log10_p": "stat",
    "_log10(p)": "stat",
    "_log10q": "stat",
    "_log10_q": "stat",
    "_log10(q)": "stat",
    "logq": "stat",
    "log(q)": "stat",
    "log_q": "stat",
    # -------------------------
    # qval / pval (allow both; we normalize to qval)
    # -------------------------
    "qval": "qval",
    "q_value": "qval",
    "qvalue": "qval",
    "fdr": "qval",
    "fdr_q": "qval",
    "padj": "qval",
    "p.adjust": "qval",
    "adj_p": "qval",
    "adj_pval": "qval",
    "adj_p_value": "qval",
    # some tools only export p-value
    "pval": "pval",
    "p_value": "pval",
    "pvalue": "pval",
    "p": "pval",
    # -------------------------
    # direction
    # -------------------------
    "direction": "direction",
    "dir": "direction",
    "sign": "direction",
    # -------------------------
    # evidence genes (common EA exports)
    # -------------------------
    "leadingedge": "evidence_genes",
    "leading_edge": "evidence_genes",
    "leadingedgegenes": "evidence_genes",
    "core_enrichment": "evidence_genes",
    "genes": "evidence_genes",
    "gene": "evidence_genes",
    "symbols": "evidence_genes",
    "overlap": "evidence_genes",
    "overlap_genes": "evidence_genes",
    # -------------------------
    # optional flags (not required; used to tolerate summary rows)
    # -------------------------
    "is_summary": "is_summary",
    "summary": "is_summary",
    "group_id": "group_id",
    "groupid": "group_id",
}

ReadMode = Literal["tsv", "sniff", "whitespace"]


@dataclass(frozen=True)
class EvidenceReadResult:
    df: pd.DataFrame
    read_mode: ReadMode


@dataclass(frozen=True)
class EvidenceTable:
    df: pd.DataFrame

    @staticmethod
    def _normalize_col(c: str) -> str:
        # Normalize common messiness while keeping it conservative.
        s = str(c).strip().lstrip("\ufeff")
        s = s.replace(" ", "_").replace("-", "_")
        s = re.sub(r"_+", "_", s)
        return s.lower()

    @staticmethod
    def _is_na_scalar(x: object) -> bool:
        return _shared.is_na_scalar(x)

    @staticmethod
    def _clean_required_str(x: object) -> str:
        if EvidenceTable._is_na_scalar(x):
            return ""
        s = str(x).strip()
        if _shared.is_na_token(s):
            return ""
        return s

    @staticmethod
    def _normalize_direction(x: object) -> str:
        if EvidenceTable._is_na_scalar(x):
            return "na"
        s = str(x).strip().lower()
        if s in {
            "up",
            "upregulated",
            "increase",
            "increased",
            "activated",
            "+",
            "pos",
            "positive",
            "1",
        }:
            return "up"
        if s in {
            "down",
            "downregulated",
            "decrease",
            "decreased",
            "suppressed",
            "-",
            "neg",
            "negative",
            "-1",
        }:
            return "down"
        return "na"

    @staticmethod
    def _clean_gene_token(g: str) -> str:
        return _shared.clean_gene_token(g)

    @classmethod
    def _split_gene_string(cls, s: str) -> list[str]:
        return _shared.split_gene_string(s)

    @classmethod
    def _parse_genes(cls, x: object) -> list[str]:
        return _shared.parse_genes(x)

    @staticmethod
    def _normalize_bool(x: object) -> bool:
        if EvidenceTable._is_na_scalar(x):
            return False
        s = str(x).strip().lower()
        return s in {"1", "true", "t", "yes", "y"}

    @classmethod
    def _read_flexible(cls, path: str) -> EvidenceReadResult:
        # robust defaults: BOM, comments, gzip, do not auto-convert NA tokens
        common_kwargs = dict(
            encoding="utf-8-sig",
            comment="#",
            compression="infer",
            keep_default_na=False,
            na_values=[],
            low_memory=False,
        )

        # 1) TSV first (expected)
        df = pd.read_csv(path, sep="\t", **common_kwargs)
        if df.shape[1] > 1:
            return EvidenceReadResult(df=df, read_mode="tsv")

        # 2) sniff delimiter (csv/tsv/others)
        df2 = pd.read_csv(path, sep=None, engine="python", **common_kwargs)
        if df2.shape[1] > 1:
            return EvidenceReadResult(df=df2, read_mode="sniff")

        # 3) whitespace fallback
        df3 = pd.read_csv(path, sep=r"\s+", engine="python", **common_kwargs)
        return EvidenceReadResult(df=df3, read_mode="whitespace")

    @classmethod
    def read_tsv(
        cls, path: str, *, strict: bool = False, drop_invalid: bool = True
    ) -> EvidenceTable:
        """
        Read an evidence table and normalize it to the tool-facing contract.

        strict:
          - if True, any invalid row triggers ValueError.
          - if False (default), invalid rows are dropped/kept per drop_invalid and marked.

        drop_invalid:
          - if True (default), drop rows that violate term×gene (e.g., empty evidence_genes),
            unless strict=True (then we raise).
          - if False, keep rows but mark them invalid; downstream should ignore them via is_valid.
        """
        rr = cls._read_flexible(path)
        df_raw = rr.df

        cols_norm = [cls._normalize_col(c) for c in df_raw.columns]
        cols_mapped = [ALIASES.get(c, c) for c in cols_norm]

        df = df_raw.copy()
        df.columns = cols_mapped

        dup = df.columns[df.columns.duplicated()].tolist()
        if dup:
            raise ValueError(
                "EvidenceTable has duplicate columns after aliasing: "
                f"{sorted(set(dup))}. "
                f"read_mode={rr.read_mode}. "
                f"raw_columns={list(df_raw.columns)} "
                f"mapped_columns={list(df.columns)}"
            )

        # ---- auto-fill optional normalized columns ----
        if "source" not in df.columns:
            df["source"] = "unknown"

        qval_in_input = "qval" in df.columns
        pval_in_input = "pval" in df.columns

        if "qval" not in df.columns:
            df["qval"] = pd.NA
        if "pval" not in df.columns:
            df["pval"] = pd.NA

        if "direction" not in df.columns:
            df["direction"] = "na"

        if "is_summary" in df.columns:
            df["is_summary"] = df["is_summary"].map(cls._normalize_bool)
        else:
            df["is_summary"] = False

        if "group_id" not in df.columns:
            df["group_id"] = pd.NA

        # evidence_genes: must exist
        # term_id / term_name / stat: must exist
        missing_core = [c for c in CORE_REQUIRED_EVIDENCE_COLS if c not in df.columns]
        if missing_core:
            raise ValueError(
                f"EvidenceTable missing core columns: {missing_core}. "
                f"read_mode={rr.read_mode}. "
                f"raw_columns={list(df_raw.columns)} "
                f"mapped_columns={list(df.columns)}"
            )

        for c in NORMALIZED_COLS:
            if c not in df.columns:
                df[c] = pd.NA

        # ---- clean required string fields ----
        df["term_id"] = df["term_id"].map(cls._clean_required_str)
        df["term_name"] = df["term_name"].map(cls._clean_required_str)
        df["source"] = df["source"].map(cls._clean_required_str)
        if (df["source"] == "").any():
            df.loc[df["source"] == "", "source"] = "unknown"

        # ---- salvage term_id/term_name conservatively (BEFORE invalid marking) ----
        # Rationale: some exports use "term" to mean name, not id.
        #
        # IMPORTANT:
        #   Salvage is an input hygiene step that can change downstream identities.
        #   We therefore record salvage flags for auditability.
        #
        # Rules:
        # - If term_id is empty and term_name is present -> copy term_name into term_id.
        # - If term_name is empty but term_id contains whitespace (looks like a name)
        #   -> copy term_id into term_name.
        df["term_id_salvaged"] = False
        df["term_name_salvaged"] = False

        term_id_empty = df["term_id"].eq("")
        term_name_empty = df["term_name"].eq("")

        salvage_1 = term_id_empty & (~term_name_empty)
        n_salvage_1 = int(salvage_1.sum())
        if n_salvage_1 > 0:
            df.loc[salvage_1, "term_id"] = df.loc[salvage_1, "term_name"]
            df.loc[salvage_1, "term_id_salvaged"] = True

        looks_like_name = df["term_id"].astype(str).str.contains(r"\s+", regex=True)
        salvage_2 = term_name_empty & (~term_id_empty) & looks_like_name
        n_salvage_2 = int(salvage_2.sum())
        if n_salvage_2 > 0:
            df.loc[salvage_2, "term_name"] = df.loc[salvage_2, "term_id"]
            df.loc[salvage_2, "term_name_salvaged"] = True

        # ---- direction normalization ----
        df["direction"] = df["direction"].map(cls._normalize_direction)

        # ---- evidence genes parsing ----
        df["evidence_genes"] = df["evidence_genes"].map(cls._parse_genes)
        df["evidence_genes_str"] = df["evidence_genes"].map(lambda xs: ",".join(xs))

        # ---- numeric normalization ----
        df["stat"] = pd.to_numeric(df["stat"], errors="coerce")
        df["qval"] = pd.to_numeric(df["qval"], errors="coerce")
        df["pval"] = pd.to_numeric(df["pval"], errors="coerce")

        # ---- row validity bookkeeping (always present) ----
        df["is_valid"] = True
        df["invalid_reason"] = ""

        # Required string fields
        bad_required = df["term_id"].eq("") | df["term_name"].eq("")
        if bad_required.any():
            df.loc[bad_required, "is_valid"] = False
            df.loc[bad_required, "invalid_reason"] = "EMPTY_REQUIRED_FIELDS"
            if strict:
                i = int(df.index[bad_required][0])
                bad = df.loc[i].to_dict()
                raise ValueError(
                    f"EvidenceTable has empty required fields at row index={i}. "
                    f"read_mode={rr.read_mode}. "
                    "Fix: ensure term_id/term_name are non-empty. "
                    f"row_term_id={bad.get('term_id')!r} row_term_name={bad.get('term_name')!r}"
                )

        # stat validity
        bad_stat = df["stat"].isna()
        if bad_stat.any():
            df.loc[bad_stat, "is_valid"] = False
            df.loc[bad_stat & df["invalid_reason"].eq(""), "invalid_reason"] = "NON_NUMERIC_STAT"
            if strict:
                i = int(df.index[bad_stat][0])
                bad = df.loc[i, ["term_id", "term_name"]].to_dict()
                raise ValueError(
                    f"EvidenceTable has non-numeric stat at row index={i} (row={bad}). "
                    f"read_mode={rr.read_mode}. "
                    "Fix: provide a numeric stat (e.g., NES, -log10(q), LogP). "
                    "Tip: check your input column mapped to 'stat'."
                )

        # ---- enforce term×gene contract (tolerate + drop/mark invalid) ----
        empty_ev = df["evidence_genes"].map(len).eq(0)
        if empty_ev.any():
            df.loc[empty_ev, "is_valid"] = False
            df.loc[empty_ev & df["invalid_reason"].eq(""), "invalid_reason"] = (
                "EMPTY_EVIDENCE_GENES"
            )
            if strict:
                i = int(df.index[empty_ev][0])
                raise ValueError(
                    f"EvidenceTable has empty evidence_genes at row index={i} "
                    f"(term_id={df.loc[i, 'term_id']}). "
                    f"read_mode={rr.read_mode}. "
                    "Fix: provide overlap genes (ORA) or leadingEdge/core_enrichment (GSEA/fgsea), "
                    "or set strict=False to drop/mark summary rows."
                )

        # ---- pval/qval range checks (prevent silent mis-mapping) ----
        # If the user mistakenly mapped logP into pval, it can exceed 1.
        pval_bad_range = (~df["pval"].isna()) & ((df["pval"] < 0.0) | (df["pval"] > 1.0))
        if pval_bad_range.any():
            df.loc[pval_bad_range, "is_valid"] = False
            df.loc[pval_bad_range & df["invalid_reason"].eq(""), "invalid_reason"] = (
                "INVALID_PVAL_RANGE"
            )
            if strict:
                i = int(df.index[pval_bad_range][0])
                raise ValueError(
                    f"EvidenceTable has pval outside [0,1] at row index={i} "
                    f"(term_id={df.loc[i, 'term_id']}, pval={df.loc[i, 'pval']}). "
                    f"read_mode={rr.read_mode}. "
                    "Fix: ensure 'pval' column is a true p-value, not logP/-log10(P)."
                )

        qval_bad_range = (~df["qval"].isna()) & ((df["qval"] < 0.0) | (df["qval"] > 1.0))
        if qval_bad_range.any():
            df.loc[qval_bad_range, "is_valid"] = False
            df.loc[qval_bad_range & df["invalid_reason"].eq(""), "invalid_reason"] = (
                "INVALID_QVAL_RANGE"
            )
            if strict:
                i = int(df.index[qval_bad_range][0])
                raise ValueError(
                    f"EvidenceTable has qval outside [0,1] at row index={i} "
                    f"(term_id={df.loc[i, 'term_id']}, qval={df.loc[i, 'qval']}). "
                    f"read_mode={rr.read_mode}. "
                    "Fix: ensure 'qval' column is a true FDR/q-value in [0,1]."
                )

        # ---- qval fallback policy ----
        # Provenance:
        #   - "qval": provided in input
        #   - "bh(pval)": computed from p-values (BH)
        #   - "missing": neither qval nor pval available
        df["qval_source"] = "missing"

        if qval_in_input:
            df.loc[~df["qval"].isna(), "qval_source"] = "qval"

        def _bh_qvalues(p: pd.Series) -> pd.Series:
            p = pd.to_numeric(p, errors="coerce")
            # Guard again: p must be in [0,1]
            p = p.where((p >= 0.0) & (p <= 1.0))
            m = int(p.notna().sum())
            if m == 0:
                return pd.Series([pd.NA] * len(p), index=p.index, dtype="float64")

            p_non = p.dropna()
            order = p_non.sort_values().index

            ranks = pd.Series(range(1, len(order) + 1), index=order, dtype="float64")
            q_non = p_non.loc[order] * (m / ranks.loc[order])

            q_non = q_non.iloc[::-1].cummin().iloc[::-1]
            q_non = q_non.clip(lower=0.0, upper=1.0)

            q = pd.Series(pd.NA, index=p.index, dtype="float64")
            q.loc[q_non.index] = q_non
            return q

        needs_q = df["qval"].isna() & (~df["pval"].isna())
        # Only compute qvals for rows that are otherwise valid enough to be used.
        needs_q = needs_q & df["is_valid"]

        bh_group_cols = ["source", "direction"]  # conservative grouping
        if needs_q.any():
            for _, idx in df.loc[needs_q].groupby(bh_group_cols).groups.items():
                idx = list(idx)
                q_calc = _bh_qvalues(df.loc[idx, "pval"])
                df.loc[idx, "qval"] = q_calc
                df.loc[idx, "qval_source"] = "bh(pval)"

        df["qval_provided"] = df["qval_source"].eq("qval")

        df["pval_source"] = "missing"
        if pval_in_input:
            df.loc[~df["pval"].isna(), "pval_source"] = "pval"
        df["pval_provided"] = df["pval_source"].eq("pval")

        # ---- drop invalid rows if requested ----
        n_invalid_total = int((~df["is_valid"]).sum())
        n_invalid_empty_ev = int(empty_ev.sum())

        # Capture a small sample of invalid rows for debugging/repro.
        invalid_sample = []
        if n_invalid_total > 0:
            cols = ["term_id", "term_name", "source", "invalid_reason"]
            cols = [c for c in cols if c in df.columns]
            invalid_sample = df.loc[~df["is_valid"], cols].head(5).to_dict(orient="records")

        if drop_invalid:
            df = df.loc[df["is_valid"]].copy()

        # provenance
        df.attrs["read_mode"] = rr.read_mode

        genes_n = (
            df["evidence_genes"].map(len)
            if "evidence_genes" in df.columns
            else pd.Series([], dtype="int64")
        )
        df.attrs["health"] = {
            "n_terms": int(df.shape[0]),
            "n_terms_genes_le1": int((genes_n <= 1).sum()) if len(genes_n) else 0,
            "genes_per_term_median": float(genes_n.median()) if len(genes_n) else 0.0,
            "qval_source_counts": df["qval_source"].value_counts(dropna=False).to_dict()
            if "qval_source" in df.columns
            else {},
            "n_invalid_total": int(n_invalid_total),
            "n_invalid_empty_evidence_genes": int(n_invalid_empty_ev),
            "invalid_sample": invalid_sample,
            "drop_invalid": bool(drop_invalid),
            "strict": bool(strict),
            "bh_group_cols": list(bh_group_cols),
            "read_mode": rr.read_mode,
            "n_salvage_term_id_from_name": int(n_salvage_1),
            "n_salvage_term_name_from_id": int(n_salvage_2),
            "n_term_id_salvaged": int(df["term_id_salvaged"].sum())
            if "term_id_salvaged" in df.columns
            else 0,
            "n_term_name_salvaged": int(df["term_name_salvaged"].sum())
            if "term_name_salvaged" in df.columns
            else 0,
        }

        return cls(df=df)

    def summarize(self) -> dict[str, object]:
        df = self.df
        genes_n = df["evidence_genes"].map(len)
        return {
            "n_terms": int(df.shape[0]),
            "n_sources": int(df["source"].nunique()),
            "sources": sorted(df["source"].astype(str).unique().tolist()),
            "direction_counts": df["direction"].value_counts(dropna=False).to_dict(),
            "genes_per_term_median": float(genes_n.median()) if len(genes_n) else 0.0,
            "genes_per_term_p90": float(genes_n.quantile(0.9)) if len(genes_n) else 0.0,
            "genes_per_term_p10": float(genes_n.quantile(0.1)) if len(genes_n) else 0.0,
            "n_terms_genes_le1": int((genes_n <= 1).sum()),
            "read_mode": df.attrs.get("read_mode", "unknown"),
            "qval_source_counts": df["qval_source"].value_counts(dropna=False).to_dict()
            if "qval_source" in df.columns
            else {},
            "health": df.attrs.get("health", {}),
        }

    def write_tsv(self, path: str) -> None:
        out = self.df.copy()

        def _excel_safe_cell(s: str) -> str:
            s = str(s or "")
            if not s:
                return ""
            # Only guard if it could be interpreted as a formula in Excel-like tools.
            if s.startswith(_EXCEL_FORMULA_START):
                return "'" + s
            return s

        def _excel_safe_series(col: str) -> None:
            if col in out.columns:
                out[col] = out[col].map(
                    lambda x: _excel_safe_cell(str(x).strip())
                    if not EvidenceTable._is_na_scalar(x)
                    else ""
                )

        def _join_genes(x: object) -> str:
            if EvidenceTable._is_na_scalar(x):
                return ""
            if isinstance(x, (list, tuple, set)):
                s = ",".join([str(g) for g in x if str(g).strip()])
            else:
                s = str(x).strip()
            return _excel_safe_cell(s)

        # Excel-safe for common text columns (people paste these into spreadsheets)
        for c in ["term_id", "term_name", "source", "direction"]:
            _excel_safe_series(c)

        if "evidence_genes" in out.columns:
            out["evidence_genes_str"] = out["evidence_genes"].map(_join_genes)
        elif "evidence_genes_str" in out.columns:
            out["evidence_genes_str"] = out["evidence_genes_str"].map(
                lambda x: _excel_safe_cell(str(x).strip())
            )
        else:
            out["evidence_genes_str"] = ""

        out = out.drop(columns=["evidence_genes"], errors="ignore")
        out = out.rename(columns={"evidence_genes_str": "evidence_genes"})

        out.to_csv(path, sep="\t", index=False)
