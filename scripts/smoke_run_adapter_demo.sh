#!/usr/bin/env bash
# scripts/smoke_run_demo.sh scripts/smoke_run_adapter_demo.sh
set -euo pipefail


# Smoke test for adapters (Metascape + fgsea) → EvidenceTable
# - Creates tiny demo inputs under out/smoke_adapters/
# - Runs adapters to produce EvidenceTable TSVs
# - Validates minimal contract and delimiter policy

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Avoid requiring pip install -e . (works from repo checkout)
export PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}"

OUTDIR="${ROOT}/out/smoke_adapters"
mkdir -p "${OUTDIR}"

python - <<'PY'
import os
import sys
import pandas as pd

OUTDIR = os.path.abspath("out/smoke_adapters")
os.makedirs(OUTDIR, exist_ok=True)

# -----------------------------
# 0) Import + shared constants
# -----------------------------
try:
    from llm_pathway_curator import _shared
except Exception as e:
    raise SystemExit(f"ERROR: cannot import llm_pathway_curator._shared ({e}). "
                     f"Try: python -m pip install -e .")

GENE_DELIM = getattr(_shared, "GENE_JOIN_DELIM", ";")

# -----------------------------
# 1) Build demo Metascape xlsx
# -----------------------------
xlsx_path = os.path.join(OUTDIR, "metascape_result.demo.xlsx")
metascape_out = os.path.join(OUTDIR, "evidence_table.metascape.demo.tsv")

enrichment = pd.DataFrame([
    # Member row; Log(q-value) negative (log10(q))
    {"GroupID":"1_Member","Category":"Reactome Gene Sets","Term":"R-HSA-397014",
     "Description":"Muscle contraction",
     "LogP":-10.9658904,"Log(q-value)":-6.705,"InTerm_InList":"8/197",
     "Genes":"72,487,4624,4626","Symbols":"ACTG2,ATP2A1,MYH6,MYH8"},
    # Member row; Log(q-value) positive (-log10(q))
    {"GroupID":"1_Member","Category":"KEGG Pathway","Term":"hsa04261",
     "Description":"Adrenergic signaling in cardiomyocytes",
     "LogP":-5.009092925,"Log(q-value)":2.269,"InTerm_InList":"6/150",
     "Genes":"487,4624,4634,4635","Symbols":"ATP2A1,MYH6,MYL3,MYL4"},
    # Summary row (should be excluded by default)
    {"GroupID":"2_Summary","Category":"GO Biological Processes","Term":"GO:0003015",
     "Description":"heart process",
     "LogP":-5.465503832,"Log(q-value)":-2.492,"InTerm_InList":"4/108",
     "Genes":"4624,4634,4635,58498","Symbols":"MYH6,MYL3,MYL4,MYL7"},
    # Weird delimiters/brackets (should still parse)
    {"GroupID":"3_Member","Category":"KEGG Pathway","Term":"hsa04020",
     "Description":"Calcium signaling pathway",
     "LogP":-4.863429607,"Log(q-value)":-2.198,"InTerm_InList":"7/240",
     "Genes":"487|2248|5260","Symbols":"['ATP2A1','FGF3','PHKG1']"},
])

with pd.ExcelWriter(xlsx_path, engine="openpyxl") as w:
    pd.DataFrame({"dummy":[1]}).to_excel(w, sheet_name="Annotation", index=False)
    enrichment.to_excel(w, sheet_name="Enrichment", index=False)

# Run adapter
from llm_pathway_curator.adapters.metascape import (
    convert_metascape_table_to_evidence_tsv,
    MetascapeAdapterConfig,
)

cfg_m = MetascapeAdapterConfig(
    sheet_name="Enrichment",
    include_summary=False,
    prefer_symbols=True,
)
ev_m = convert_metascape_table_to_evidence_tsv(xlsx_path, metascape_out, config=cfg_m)

# -----------------------------
# 2) Build demo fgsea result TSV
# -----------------------------
fgsea_in = os.path.join(OUTDIR, "fgsea_results.demo.tsv")
fgsea_out = os.path.join(OUTDIR, "evidence_table.fgsea.demo.tsv")

fgsea_res = pd.DataFrame([
    {"pathway":"HALLMARK_P53_PATHWAY","NES":2.1,"padj":1e-4,"pval":1e-6,
     "leadingEdge":"1,2,3,4"},
    {"pathway":"HALLMARK_E2F_TARGETS","NES":-1.9,"padj":2e-3,"pval":5e-4,
     "leadingEdge":"c(10, 20, 30)"},
    {"pathway":"HALLMARK_OXIDATIVE_PHOSPHORYLATION","NES":1.5,"padj":5e-2,"pval":2e-2,
     "leadingEdge":"[100;200;300]"},
])
fgsea_res.to_csv(fgsea_in, sep="\t", index=False)

from llm_pathway_curator.adapters.fgsea import (
    convert_fgsea_table_to_evidence_tsv,
    FgseaAdapterConfig,
)

cfg_f = FgseaAdapterConfig(source_name="fgsea_demo")
ev_f = convert_fgsea_table_to_evidence_tsv(fgsea_in, fgsea_out, config=cfg_f)

# -----------------------------
# 3) Validate minimal contract
# -----------------------------
REQUIRED = ["term_id","term_name","source","stat","qval","direction","evidence_genes"]

def check(df: pd.DataFrame, name: str) -> None:
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise SystemExit(f"ERROR: {name} missing required columns: {missing}")

    if len(df) == 0:
        raise SystemExit(f"ERROR: {name} produced 0 rows")

    # qval should be numeric when present
    q = pd.to_numeric(df["qval"], errors="coerce")
    if q.isna().any():
        # allow NA if adapter kept NA, but at least one should be valid
        if q.notna().sum() == 0:
            raise SystemExit(f"ERROR: {name} qval all NA")
    else:
        bad = ~((q > 0) & (q <= 1))
        if bad.any():
            raise SystemExit(f"ERROR: {name} qval out of (0,1] at rows: {list(df.index[bad])[:5]}")

    # evidence_genes delimiter check (TSV string)
    eg = df["evidence_genes"].astype(str)
    has_multi = eg.str.contains(r"[;,\|]")
    if has_multi.any():
        # if any row has multiple genes, enforce canonical delimiter
        # (warn rather than hard-fail if your fgsea adapter still joins by ',')
        if eg.str.contains(",").any() and GENE_DELIM == ";":
            print(f"[WARN] {name}: evidence_genes contains ',' but canonical delimiter is '{GENE_DELIM}'.")
            print("       Consider joining via _shared.join_genes_tsv() for spec alignment.")

check(ev_m, "metascape")
check(ev_f, "fgsea")

# -----------------------------
# 4) Print summary
# -----------------------------
print("[OK] smoke adapters")
print("  metascape xlsx :", xlsx_path)
print("  metascape out  :", metascape_out, "n_terms=", len(ev_m))
print("  fgsea in       :", fgsea_in)
print("  fgsea out      :", fgsea_out, "n_terms=", len(ev_f))
print("  gene_delim(spec):", GENE_DELIM)

PY

echo ""
echo "[OK] wrote outputs under: ${OUTDIR}"
echo "  - metascape_result.demo.xlsx"
echo "  - evidence_table.metascape.demo.tsv"
echo "  - fgsea_results.demo.tsv"
echo "  - evidence_table.fgsea.demo.tsv"
