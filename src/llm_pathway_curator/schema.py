# LLM-PathwayCurator/src/llm_pathway_curator/schema.py
"""
EvidenceTable schema gate for LLM-PathwayCurator.
This module defines the tool-facing EvidenceTable contract (v1) that preserves
term×gene relationships across enrichment analysis tools (ORA, fgsea/GSEA, etc.).
It provides robust IO, conservative column aliasing, spec-owned evidence parsing
(delegated to _shared), and provenance metadata (df.attrs) for auditability.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

import pandas as pd

from . import _shared

# -----------------------------------------------------------------------------
# EvidenceTable schema gate (tool-facing contract)
# -----------------------------------------------------------------------------
# Contract: we MUST preserve term×gene.
#
# Practical reality:
# - ORA often has no direction
# - some exports have no qval (or only p-value)
#
# Policy:
# - enforce a "core required" set and auto-fill the rest
# - record provenance (aliasing, read_mode, salvage) in df.attrs for auditability
# - keep parsing rules in _shared.py to avoid contract drift
# -----------------------------------------------------------------------------

EVIDENCE_TABLE_CONTRACT_VERSION = "v1"

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

# Excel formula injection starters (output-only defense).
_EXCEL_FORMULA_START = ("=", "+", "-", "@")

# NOTE:
# Gene parsing is spec-defined in _shared.py (parse_genes / split_gene_string / clean_gene_token).
# Do not duplicate parsing regexes here to avoid contract drift.

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
    """
    Result of a flexible EvidenceTable read.

    Attributes
    ----------
    df : pandas.DataFrame
        Raw dataframe loaded from file (no contract normalization applied yet).
    read_mode : {'tsv', 'sniff', 'whitespace'}
        Reader mode that successfully parsed the file.
    """

    df: pd.DataFrame
    read_mode: ReadMode


@dataclass(frozen=True)
class EvidenceTable:
    """
    Tool-facing EvidenceTable wrapper.

    This class normalizes heterogeneous enrichment analysis outputs into a stable,
    auditable internal representation that preserves term×gene relationships.

    Notes
    -----
    - The contract requires non-empty `term_id`, `term_name`, and `evidence_genes`.
    - Parsing/normalization of gene tokens is spec-owned by `llm_pathway_curator._shared`
      (e.g., `parse_genes`, `clean_gene_token`), to avoid contract drift.
    - Provenance and health summaries are recorded in `df.attrs`.
    """

    df: pd.DataFrame

    # -------------------------
    # column / scalar helpers
    # -------------------------
    @staticmethod
    def _normalize_col(c: str) -> str:
        """
        Normalize a raw column header conservatively.

        Parameters
        ----------
        c : str
            Raw column name.

        Returns
        -------
        str
            Normalized column name:
            - stripped
            - BOM removed
            - spaces/dashes converted to underscores
            - repeated underscores collapsed
            - lowercased
        """
        # Normalize common messiness while keeping it conservative.
        s = str(c).strip().lstrip("\ufeff")
        s = s.replace(" ", "_").replace("-", "_")
        s = re.sub(r"_+", "_", s)
        return s.lower()

    @staticmethod
    def _is_na_scalar(x: object) -> bool:
        """
        Check whether a scalar should be treated as NA.

        Parameters
        ----------
        x : object
            Input scalar.

        Returns
        -------
        bool
            True if `x` is NA-like according to `_shared.is_na_scalar`.
        """
        return _shared.is_na_scalar(x)

    @staticmethod
    def _clean_required_str(x: object) -> str:
        """
        Clean a required string field to a non-NA canonical form.

        Parameters
        ----------
        x : object
            Input value.

        Returns
        -------
        str
            Cleaned string. Returns an empty string for NA-like values or NA tokens.

        Notes
        -----
        This is used for required fields such as `term_id`, `term_name`, and `source`.
        """
        if EvidenceTable._is_na_scalar(x):
            return ""
        s = str(x).strip()
        if _shared.is_na_token(s):
            return ""
        return s

    @staticmethod
    def _normalize_direction(x: object) -> str:
        """
        Normalize direction vocabulary to the contract.

        Parameters
        ----------
        x : object
            Direction-like value (e.g., 'up', 'down', '+', '-', NA).

        Returns
        -------
        str
            Normalized direction string (e.g., 'up', 'down', 'na').

        See Also
        --------
        llm_pathway_curator._shared.normalize_direction
        """
        # Spec-level vocabulary normalization lives in _shared to avoid contract drift.
        return _shared.normalize_direction(x)

    @staticmethod
    def _normalize_bool(x: object) -> bool:
        """
        Parse a boolean-like scalar.

        Parameters
        ----------
        x : object
            Boolean-like value.

        Returns
        -------
        bool
            True for {'1','true','t','yes','y'} (case-insensitive), otherwise False.

        Notes
        -----
        NA-like values are treated as False.
        """
        if EvidenceTable._is_na_scalar(x):
            return False
        s = str(x).strip().lower()
        return s in {"1", "true", "t", "yes", "y"}

    # -------------------------
    # IO
    # -------------------------
    @classmethod
    def _read_flexible(cls, path: str) -> EvidenceReadResult:
        """
        Read a delimited table with robust fallbacks.

        The reader attempts, in order:
        1) TSV (`sep='\\t'`)
        2) Sniffed delimiter (`sep=None`, `engine='python'`)
        3) Whitespace-delimited (`sep='\\s+'`)

        Parameters
        ----------
        path : str
            Input file path. Compression is inferred (e.g., .gz).

        Returns
        -------
        EvidenceReadResult
            Loaded dataframe and the read mode that succeeded.

        Notes
        -----
        - Uses `encoding='utf-8-sig'` to tolerate BOM.
        - Uses `keep_default_na=False` to avoid accidental NA coercion of gene tokens.
        """

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

    # -------------------------
    # main contract gate
    # -------------------------
    @classmethod
    def read_tsv(
        cls, path: str, *, strict: bool = False, drop_invalid: bool = True
    ) -> EvidenceTable:
        """
        Read and normalize an evidence table to the contract (v1).

        This is the main schema gate that:
        - aliases common column variants to contract names
        - cleans required fields
        - parses `evidence_genes` via `_shared.parse_genes`
        - normalizes numeric fields (`stat`, `qval`, `pval`)
        - validates the term×gene contract
        - optionally computes q-values from p-values (BH) when q-values are missing
        - records provenance and health metrics in `df.attrs`

        Parameters
        ----------
        path : str
            Input evidence table path.
        strict : bool, optional
            If True, the first invalid row raises ValueError.
            If False, invalid rows are marked (and optionally dropped). Default is False.
        drop_invalid : bool, optional
            If True, drop rows with `is_valid=False`. Default is True.
            If False, keep invalid rows and rely on `is_valid` downstream.

        Returns
        -------
        EvidenceTable
            Normalized evidence table wrapper.

        Raises
        ------
        ValueError
            If core required columns are missing after aliasing.
            If `strict=True` and an invalid row is encountered.

        Notes
        -----
        Contract-required columns (core)
        - term_id
        - term_name
        - stat
        - evidence_genes

        Output guarantees (post-normalization)
        - `evidence_genes` is a list-like object per row (and `evidence_genes_str` is TSV-safe)
        - `direction` is normalized (typically 'up', 'down', 'na')
        - `df.attrs` contains: `contract_version`, `read_mode`, `aliasing`, `health`

        Examples
        --------
        >>> et = EvidenceTable.read_tsv("evidence_table.tsv")
        >>> info = et.summarize()
        >>> et.write_tsv("normalized_evidence_table.tsv")
        """
        rr = cls._read_flexible(path)
        df_raw = rr.df

        raw_columns = list(df_raw.columns)
        cols_norm = [cls._normalize_col(c) for c in raw_columns]
        cols_mapped = [ALIASES.get(c, c) for c in cols_norm]

        df = df_raw.copy()
        df.columns = cols_mapped
        # Snapshot attrs for safe propagation across slice/copy steps.
        _attrs_snapshot: dict[str, object] = dict(getattr(df, "attrs", {}))

        dup = df.columns[df.columns.duplicated()].tolist()
        if dup:
            # Coalesce duplicate columns deterministically instead of hard-failing.
            #
            # Rationale:
            #   Real-world EA exports often contain multiple synonymous columns
            #   (e.g., genes + symbols, description + name). Hard failure here
            #   reduces usability and causes avoidable pipeline aborts.
            #
            # Policy:
            #   - For each duplicated name, keep the leftmost column as the target.
            #   - Fill empty/NA-like cells in the target from subsequent duplicates (left-to-right).
            #   - Record the coalescing action in df.attrs for auditability.
            dup_names = sorted(set(dup))
            coalesced: dict[str, list[str]] = {}

            for name in dup_names:
                idxs = [i for i, c in enumerate(df.columns) if c == name]
                if len(idxs) <= 1:
                    continue

                # Column labels may repeat; build explicit positional column references.
                # Use iloc to access by position safely.
                target_pos = idxs[0]
                src_positions = idxs[1:]

                target = df.iloc[:, target_pos]
                used_sources: list[str] = []

                # Define "empty" conservatively: NA scalar or empty string after strip,
                # or NA tokens defined by spec.
                def _is_empty_cell(v: object) -> bool:
                    if _shared.is_na_scalar(v):
                        return True
                    s = str(v).strip()
                    return (not s) or _shared.is_na_token(s)

                empty_mask = target.map(_is_empty_cell)

                for pos in src_positions:
                    src = df.iloc[:, pos]
                    fill_mask = empty_mask & (~src.map(_is_empty_cell))
                    if fill_mask.any():
                        target = target.mask(fill_mask, src)
                        empty_mask = target.map(_is_empty_cell)
                        used_sources.append(f"pos{pos}")

                # Write back merged target into the original target position.
                df.iloc[:, target_pos] = target

                # Drop duplicate columns by *position* (safe even with duplicate labels).
                keep_positions = [i for i in range(df.shape[1]) if i not in src_positions]
                df = df.iloc[:, keep_positions]

                coalesced[name] = used_sources if used_sources else ["no_nonempty_fills"]

            # Attach provenance for auditability (attrs preserved downstream in summarize()).
            df.attrs["coalesced_duplicate_columns"] = coalesced
            _attrs_snapshot = dict(df.attrs)

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

        # core required columns must exist (after aliasing)
        missing_core = [c for c in CORE_REQUIRED_EVIDENCE_COLS if c not in df.columns]
        if missing_core:
            raise ValueError(
                f"EvidenceTable missing core columns: {missing_core}. "
                f"read_mode={rr.read_mode}. "
                f"raw_columns={raw_columns} "
                f"normalized_columns={cols_norm} "
                f"mapped_columns={cols_mapped}"
            )

        # ensure normalized cols exist (even if unused by some tools)
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
        # IMPORTANT: salvage can change downstream identities; record flags.
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

        # ---- evidence genes parsing (spec-defined) ----
        df["evidence_genes"] = df["evidence_genes"].map(_shared.parse_genes)
        df["evidence_genes_str"] = df["evidence_genes"].map(_shared.join_genes_tsv)

        # ---- numeric normalization ----
        df["stat"] = pd.to_numeric(df["stat"], errors="coerce")
        df["qval"] = pd.to_numeric(df["qval"], errors="coerce")
        df["pval"] = pd.to_numeric(df["pval"], errors="coerce")

        # ---- row validity bookkeeping (always present) ----
        df["is_valid"] = True
        df["invalid_reason"] = ""

        # required string fields
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

        # enforce term×gene contract
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

        # pval/qval range checks (prevent silent mis-mapping)
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
        #   - "bh(pval)": computed from p-values (BH) within deterministic groups
        #   - "missing": neither qval nor pval available
        df["qval_source"] = "missing"
        if qval_in_input:
            df.loc[~df["qval"].isna(), "qval_source"] = "qval"

        def _bh_qvalues(p: pd.Series) -> pd.Series:
            p = pd.to_numeric(p, errors="coerce")
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

        # Fill qval only when missing, using BH computed over ALL valid p-values in the group.
        needs_q = df["qval"].isna() & (~df["pval"].isna()) & df["is_valid"]

        # Conservative grouping (documented, deterministic)
        bh_group_cols = ["source", "direction"]

        if needs_q.any():
            # Iterate groups based on rows that need filling,
            # but compute BH using all valid pvals in group.
            groups = df.loc[needs_q, bh_group_cols].drop_duplicates()

            for _, row in groups.iterrows():
                key_source = row["source"]
                key_dir = row["direction"]

                in_group = (df["source"] == key_source) & (df["direction"] == key_dir)
                idx_all = df.index[in_group & df["is_valid"] & (~df["pval"].isna())]
                if len(idx_all) == 0:
                    continue

                q_all = _bh_qvalues(df.loc[idx_all, "pval"])

                idx_need = df.index[in_group & needs_q]
                if len(idx_need) == 0:
                    continue

                df.loc[idx_need, "qval"] = q_all.loc[idx_need]
                df.loc[idx_need, "qval_source"] = "bh(pval)"

        df["qval_provided"] = df["qval_source"].eq("qval")

        df["pval_source"] = "missing"
        if pval_in_input:
            df.loc[~df["pval"].isna(), "pval_source"] = "pval"
        df["pval_provided"] = df["pval_source"].eq("pval")

        # ---- capture invalid/salvage samples for auditability ----
        n_invalid_total = int((~df["is_valid"]).sum())
        n_invalid_empty_ev = int(empty_ev.sum())

        invalid_sample = []
        if n_invalid_total > 0:
            cols = ["term_id", "term_name", "source", "invalid_reason"]
            cols = [c for c in cols if c in df.columns]
            invalid_sample = df.loc[~df["is_valid"], cols].head(5).to_dict(orient="records")

        salvage_sample = []
        salvaged_any = df["term_id_salvaged"] | df["term_name_salvaged"]
        if salvaged_any.any():
            cols = ["term_id", "term_name", "source", "term_id_salvaged", "term_name_salvaged"]
            cols = [c for c in cols if c in df.columns]
            salvage_sample = df.loc[salvaged_any, cols].head(5).to_dict(orient="records")

        if drop_invalid:
            df = df.loc[df["is_valid"]].copy()
            # Ensure attrs survive copy/slice across pandas versions.
            df.attrs.update(_attrs_snapshot)

        # provenance (attrs)
        df.attrs["read_mode"] = rr.read_mode
        df.attrs["contract_version"] = EVIDENCE_TABLE_CONTRACT_VERSION
        df.attrs["aliasing"] = {
            "raw_columns": raw_columns,
            "normalized_columns": cols_norm,
            "mapped_columns": cols_mapped,
        }

        genes_n = (
            df["evidence_genes"].map(len)
            if "evidence_genes" in df.columns
            else pd.Series([], dtype="int64")
        )
        df.attrs["health"] = {
            "contract_version": EVIDENCE_TABLE_CONTRACT_VERSION,
            "n_terms": int(df.shape[0]),
            "n_terms_genes_le1": int((genes_n <= 1).sum()) if len(genes_n) else 0,
            "genes_per_term_median": float(genes_n.median()) if len(genes_n) else 0.0,
            "qval_source_counts": df["qval_source"].value_counts(dropna=False).to_dict()
            if "qval_source" in df.columns
            else {},
            "n_invalid_total": int(n_invalid_total),
            "n_invalid_empty_evidence_genes": int(n_invalid_empty_ev),
            "invalid_sample": invalid_sample,
            "salvage_sample": salvage_sample,
            "drop_invalid": bool(drop_invalid),
            "strict": bool(strict),
            "bh_group_cols": list(bh_group_cols),
            "read_mode": rr.read_mode,
            "n_salvage_term_id_from_name": int(n_salvage_1),
            "n_salvage_term_name_from_id": int(n_salvage_2),
            "n_term_id_salvaged_kept": int(df["term_id_salvaged"].sum())
            if "term_id_salvaged" in df.columns
            else 0,
            "n_term_name_salvaged_kept": int(df["term_name_salvaged"].sum())
            if "term_name_salvaged" in df.columns
            else 0,
        }

        return cls(df=df)

    def summarize(self) -> dict[str, object]:
        """
        Summarize the normalized EvidenceTable for logging and QA.

        Returns
        -------
        dict[str, object]
            Summary dictionary including:
            - contract version
            - number of terms and sources
            - direction counts
            - evidence genes per term quantiles
            - q-value provenance counts
            - `df.attrs['health']` and `df.attrs['aliasing']` (if present)
        """
        df = self.df
        genes_n = df["evidence_genes"].map(len)
        return {
            "contract_version": df.attrs.get("contract_version", "unknown"),
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
            "aliasing": df.attrs.get("aliasing", {}),
        }

    def write_tsv(self, path: str) -> None:
        """
        Write the normalized EvidenceTable to a TSV file.

        This writer:
        - applies a small Excel formula-injection defense for common text fields
        - serializes `evidence_genes` as a TSV-friendly string column
        - emits a stable column order for reproducibility

        Parameters
        ----------
        path : str
            Output TSV path.

        Notes
        -----
        - `evidence_genes` is written as a joined string under the column name
          `evidence_genes` (list form is dropped).
        - Normalized contract columns are emitted first; remaining columns are sorted.
        """
        out = self.df.copy()

        def _excel_safe_cell(txt: object) -> str:
            if self._is_na_scalar(txt):
                return ""
            s = str(txt).strip()
            if (not s) or _shared.is_na_token(s):
                return ""
            if s.startswith(_EXCEL_FORMULA_START):
                return "'" + s
            return s

        # Excel-safe for common text columns (people paste these into spreadsheets)
        for c in ["term_id", "term_name", "source", "direction"]:
            if c in out.columns:
                out[c] = out[c].map(
                    lambda x: _excel_safe_cell(x) if not self._is_na_scalar(x) else ""
                )

        # evidence_genes: write as TSV-friendly joined string; do NOT re-parse on write
        if "evidence_genes" in out.columns:
            out["evidence_genes_str"] = out["evidence_genes"].map(
                lambda x: _excel_safe_cell(_shared.join_genes_tsv(list(x)))
                if isinstance(x, (list, tuple, set))
                else _excel_safe_cell(x)
            )
        elif "evidence_genes_str" in out.columns:
            out["evidence_genes_str"] = out["evidence_genes_str"].map(_excel_safe_cell)
        else:
            out["evidence_genes_str"] = ""

        out = out.drop(columns=["evidence_genes"], errors="ignore")
        out = out.rename(columns={"evidence_genes_str": "evidence_genes"})

        # Stable column order for reproducibility:
        # - normalized contract columns first
        # - then the rest (sorted) to reduce drift across versions
        first = [c for c in NORMALIZED_COLS if c in out.columns]
        rest = [c for c in out.columns if c not in first]
        out = out.loc[:, first + sorted(rest)]

        out.to_csv(path, sep="\t", index=False)
