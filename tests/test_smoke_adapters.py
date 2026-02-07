# LLM-PathwayCurator/tests/test_smoke_adapters.py

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

"""
Smoke test: adapters (Metascape + fgsea) -> EvidenceTable TSV.

Goals
- Generate tiny synthetic inputs (xlsx/tsv) in a temp dir
- Run adapters to produce EvidenceTable TSVs
- Validate minimal EvidenceTable contract + delimiter policy

This is a developer-facing contract test (NOT a reviewer demo).
"""


def _ensure_src_on_path() -> None:
    """Allow running tests from a repo checkout without `pip install -e .`."""
    root = Path(__file__).resolve().parents[1]
    src = root / "src"
    if src.exists():
        sys.path.insert(0, str(src))


_ensure_src_on_path()


def _require_openpyxl() -> None:
    if importlib.util.find_spec("openpyxl") is None:
        pytest.skip("openpyxl is required for Metascape xlsx smoke test")


def _check_min_contract(df: pd.DataFrame, name: str, gene_delim: str) -> None:
    required = ["term_id", "term_name", "source", "stat", "qval", "direction", "evidence_genes"]

    missing = [c for c in required if c not in df.columns]
    assert not missing, f"{name}: missing required columns: {missing}"
    assert len(df) > 0, f"{name}: produced 0 rows"

    # qval: allow NA, but not all NA; if numeric, must be in (0, 1]
    q = pd.to_numeric(df["qval"], errors="coerce")
    assert q.notna().sum() > 0, f"{name}: qval all NA/invalid"
    q_ok = q.dropna()
    bad = ~((q_ok > 0) & (q_ok <= 1))
    assert not bad.any(), f"{name}: qval out of (0,1] at rows: {list(q_ok.index[bad])[:10]}"

    # evidence_genes: non-empty strings; enforce canonical delimiter (no other delimiters)
    eg = df["evidence_genes"].astype(str)

    assert (~eg.str.fullmatch(r"\s*")).all(), f"{name}: evidence_genes contains empty/blank entries"

    # Reject obvious leftover list syntax from messy inputs
    bad_syntax = eg.str.contains(r"[\[\]\(\)]")
    assert not bad_syntax.any(), (
        f"{name}: evidence_genes contains bracket/paren syntax at rows: "
        f"{list(df.index[bad_syntax])[:10]}"
    )

    # Enforce canonical delimiter: evidence_genes must not contain alternative delimiters
    alt_delims = {",", ";", "|"}
    if gene_delim in alt_delims:
        alt_delims.remove(gene_delim)

    alt_pat = r"[{}]".format("".join(map(lambda x: "\\" + x if x in r"\^-" else x, alt_delims)))
    has_alt = eg.str.contains(alt_pat)
    assert not has_alt.any(), (
        f"{name}: evidence_genes contains non-canonical delimiter(s) {sorted(alt_delims)} "
        f"but spec delimiter is '{gene_delim}'. Offending term_ids: "
        f"{df.loc[has_alt, 'term_id'].astype(str).head(10).tolist()}"
    )

    # If multiple genes present, they must be split by gene_delim into clean tokens
    if gene_delim:
        multi = eg.str.contains(gene_delim)
        if multi.any():
            for idx, s in eg[multi].items():
                toks = [t.strip() for t in s.split(gene_delim)]
                assert all(toks), f"{name}: empty token in evidence_genes at row {idx}: {s!r}"
                # Reject tokens that still look like they carry separators
                assert not any("," in t or "|" in t or ";" in t for t in toks), (
                    f"{name}: token contains delimiter residue at row {idx}: {s!r}"
                )


@pytest.mark.smoke
def test_smoke_adapters_metascape_and_fgsea_to_evidence_table(tmp_path: Path) -> None:
    _require_openpyxl()

    # Import shared constants
    from llm_pathway_curator import _shared

    gene_delim = getattr(_shared, "GENE_JOIN_DELIM", ";")

    # -----------------------------
    # 1) Build tiny Metascape xlsx
    # -----------------------------
    xlsx_path = tmp_path / "metascape_result.demo.xlsx"
    metascape_out = tmp_path / "evidence_table.metascape.demo.tsv"

    enrichment = pd.DataFrame(
        [
            # Member row; Log(q-value) negative (log10(q))
            {
                "GroupID": "1_Member",
                "Category": "Reactome Gene Sets",
                "Term": "R-HSA-397014",
                "Description": "Muscle contraction",
                "LogP": -10.9658904,
                "Log(q-value)": -6.705,
                "InTerm_InList": "8/197",
                "Genes": "72,487,4624,4626",
                "Symbols": "ACTG2,ATP2A1,MYH6,MYH8",
            },
            # Member row; Log(q-value) positive (-log10(q))
            {
                "GroupID": "1_Member",
                "Category": "KEGG Pathway",
                "Term": "hsa04261",
                "Description": "Adrenergic signaling in cardiomyocytes",
                "LogP": -5.009092925,
                "Log(q-value)": 2.269,
                "InTerm_InList": "6/150",
                "Genes": "487,4624,4634,4635",
                "Symbols": "ATP2A1,MYH6,MYL3,MYL4",
            },
            # Summary row (should be excluded by default)
            {
                "GroupID": "2_Summary",
                "Category": "GO Biological Processes",
                "Term": "GO:0003015",
                "Description": "heart process",
                "LogP": -5.465503832,
                "Log(q-value)": -2.492,
                "InTerm_InList": "4/108",
                "Genes": "4624,4634,4635,58498",
                "Symbols": "MYH6,MYL3,MYL4,MYL7",
            },
            # Weird delimiters/brackets (should still parse)
            {
                "GroupID": "3_Member",
                "Category": "KEGG Pathway",
                "Term": "hsa04020",
                "Description": "Calcium signaling pathway",
                "LogP": -4.863429607,
                "Log(q-value)": -2.198,
                "InTerm_InList": "7/240",
                "Genes": "487|2248|5260",
                "Symbols": "['ATP2A1','FGF3','PHKG1']",
            },
        ]
    )

    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as w:
        pd.DataFrame({"dummy": [1]}).to_excel(w, sheet_name="Annotation", index=False)
        enrichment.to_excel(w, sheet_name="Enrichment", index=False)

    from llm_pathway_curator.adapters.metascape import (
        MetascapeAdapterConfig,
        convert_metascape_table_to_evidence_tsv,
    )

    cfg_m = MetascapeAdapterConfig(
        sheet_name="Enrichment",
        include_summary=False,
        prefer_symbols=True,
    )

    ev_m = convert_metascape_table_to_evidence_tsv(str(xlsx_path), str(metascape_out), config=cfg_m)
    assert metascape_out.exists(), "metascape adapter did not write output TSV"
    _check_min_contract(ev_m, "metascape", gene_delim=gene_delim)

    # Ensure Summary row excluded when include_summary=False
    assert "GO:0003015" not in set(ev_m["term_id"].astype(str)), (
        "metascape: summary row was not excluded"
    )

    # -----------------------------
    # 2) Build tiny fgsea TSV
    # -----------------------------
    fgsea_in = tmp_path / "fgsea_results.demo.tsv"
    fgsea_out = tmp_path / "evidence_table.fgsea.demo.tsv"

    fgsea_res = pd.DataFrame(
        [
            {
                "pathway": "HALLMARK_P53_PATHWAY",
                "NES": 2.1,
                "padj": 1e-4,
                "pval": 1e-6,
                "leadingEdge": "1,2,3,4",
            },
            {
                "pathway": "HALLMARK_E2F_TARGETS",
                "NES": -1.9,
                "padj": 2e-3,
                "pval": 5e-4,
                "leadingEdge": "c(10, 20, 30)",
            },
            {
                "pathway": "HALLMARK_OXIDATIVE_PHOSPHORYLATION",
                "NES": 1.5,
                "padj": 5e-2,
                "pval": 2e-2,
                "leadingEdge": "[100;200;300]",
            },
        ]
    )
    fgsea_res.to_csv(fgsea_in, sep="\t", index=False)

    from llm_pathway_curator.adapters.fgsea import (
        FgseaAdapterConfig,
        convert_fgsea_table_to_evidence_tsv,
    )

    cfg_f = FgseaAdapterConfig(source_name="fgsea_demo")
    ev_f = convert_fgsea_table_to_evidence_tsv(str(fgsea_in), str(fgsea_out), config=cfg_f)

    assert fgsea_out.exists(), "fgsea adapter did not write output TSV"
    _check_min_contract(ev_f, "fgsea", gene_delim=gene_delim)

    # Direction sanity (do not over-constrain exact strings)
    dirs = set(ev_f["direction"].astype(str).str.lower())
    assert any(d in dirs for d in {"up", "pos", "positive"}), (
        f"fgsea: expected an 'up' direction, got {sorted(dirs)}"
    )
    assert any(d in dirs for d in {"down", "neg", "negative"}), (
        f"fgsea: expected a 'down' direction, got {sorted(dirs)}"
    )
