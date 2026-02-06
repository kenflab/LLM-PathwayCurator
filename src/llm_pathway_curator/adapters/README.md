# Adapters (Input → EvidenceTable)

This package provides adapters that normalize upstream enrichment outputs
into the **EvidenceTable** contract used by LLM-PathwayCurator.

Adapters are intentionally conservative:
they preserve evidence identity (term × genes), avoid destructive parsing,
and keep TSV round-trips stable.

## EvidenceTable contract (minimal)

Each row represents one enriched term.

Required columns:
- `term_id` : string (e.g., `GO:0002026`, `R-HSA-397014`, `HALLMARK_P53_PATHWAY`)
- `term_name` : string (human-readable description/name)
- `source` : string (adapter/source tag, e.g., `metascape`, `fgsea`)
- `stat` : float (monotone significance/effect metric; adapter-defined)
- `qval` : float in (0, 1] (FDR/q-value; may be NA if missing/invalid)
- `direction` : one of `{up, down, na}`
- `evidence_genes` : list[str] in-memory; TSV export joins by `;`

Optional provenance fields may be included (e.g., `category`, `group_id`, `pval`).

## Common parsing policy (spec-level)

Adapters delegate parsing and TSV round-trips to `llm_pathway_curator._shared`:
- NA handling: `_shared.is_na_scalar()` / `_shared.is_na_token()`
- Gene parsing: `_shared.parse_genes()` (supports `, ; |` and bracketed lists)
- TSV join: `_shared.join_genes_tsv()` (canonical delimiter `;`)
- Short hashes: `_shared.sha256_12hex()`

This prevents contract drift across layers (distill/modules/select/audit/report).

---
## fgsea adapter

### Input
A standard fgsea result table (TSV/CSV) with at least:
- `pathway`
- `leadingEdge`

Preferred stat columns:
- `NES` (preferred), else `ES`

q-values:
- `padj` is mapped to `qval`
- `pval` may be retained as `pval` (optional provenance)

term_id policy (recommended)
- `term_id == pathway` (paper-aligned; matches MSigDB set names)
- a stable short hash may be included in `term_id_h` for provenance

### Example (Python)
```python
from llm_pathway_curator.adapters.fgsea import convert_fgsea_table_to_evidence_tsv

convert_fgsea_table_to_evidence_tsv(
    "fgsea_res.tsv",
    "evidence_table.tsv",
)

```

---
## Metascape adapter

### Input
Metascape Excel exports typically contain two sheets:
- `Annotation`
- `Enrichment`  ← **used by this adapter (default)**

Expected columns in `Enrichment`:
`GroupID, Category, Term, Description, LogP, Log(q-value), InTerm_InList, Genes, Symbols`

### Summary rows
Metascape often contains `*_Summary` rows.
By default the adapter **excludes** these (`include_summary=False`) to avoid
double counting. Set `include_summary=True` only if you explicitly want them.

### q-value reconstruction
Metascape exports may encode `Log(q-value)` as either:
- `log10(q)`  (negative for q<1), or
- `-log10(q)` (positive)

The adapter infers the convention by sign and reconstructs:
- if value ≤ 0: `q = 10^(value)`
- if value > 0: `q = 10^(-value)`

Non-reconstructable values become NA (and may be dropped downstream).

### Evidence genes
By default, the adapter prefers `Symbols` over numeric `Genes`:
- `prefer_symbols=True` (recommended)
- if empty, it falls back to `Genes`

### Example (Python)
```python
from llm_pathway_curator.adapters.metascape import convert_metascape_table_to_evidence_tsv

convert_metascape_table_to_evidence_tsv(
    "metascape_result.xlsx",
    "evidence_table.tsv",
)

```

---

## Notes for reproducibility
- Adapters aim for deterministic outputs.
- Adapters should not reorder rows unless explicitly documented.
- TSV export joins `evidence_genes` with `;`. The parser accepts `,` and `|` as well.
- When comparing against paper-side scripts, compare gene sets as sets
(not raw delimiter strings).

---