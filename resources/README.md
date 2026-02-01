<!-- resources/README.md -->

# Resources (optional, reproducible)
This directory contains optional resources used for **display** and **context convenience**.
They do **not** change core auditing logic unless explicitly enabled.

# Gene ID maps (local, reproducible)
This directory contains scripts to build local ID→symbol mapping files for **display/interpretability** (e.g., `report.md` / `audit_log.tsv` outputs).  
These maps are **optional** and do **not** affect auditing logic.

## Entrez Gene ID → symbol (human)

Build from NCBI `gene_info`:

```bash
bash resources/gene_id_maps/build_entrez_id_map.sh
```

## Outputs:
- resources/gene_id_maps/id_map.tsv.gz
- resources/gene_id_maps/checksums.sha256

```python-repl
The mapping format is:
gene_id<TAB>gene_symbol
4033   STAT3
7062   TNF
...
```

## Use in Sample Card:
```yaml
extra:
  gene_id_map_tsv: "resources/gene_id_maps/id_map.tsv.gz"
```

## Notes:
- Default source is human-only (Homo_sapiens.gene_info.gz).
- To enforce tax_id==9606, filter by the first column:
```bash
awk -F'\t' 'BEGIN{OFS="\t"} $1=="9606" {print $2,$3}' resources/gene_id_maps/Homo_sapiens.gene_info
```



# Context lexicon (optional; disabled by default)
resources/context_lexicon.tsv defines single-token synonyms for context token expansion.
This helps match common naming variants (e.g., abbreviations). Keep it small and avoid generic terms.

##  Format (TSV):
- head<TAB>syn1,syn2,... (comma/semicolon/space separated synonyms are accepted)
- Lines starting with # are ignored
- Tokens are normalized (NFKC + lowercase); multi-word phrases are discouraged

## Enable in Sample Card:
```yaml
context_tokens_policy:
  lexicon_expand: true
```

## Tip:
- If the file is missing or malformed, the tool fails closed (no expansion).
- Prefer project-specific additions; users can safely append new lines.