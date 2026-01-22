<!-- resources/gene_id_maps/README.md -->

# Gene ID maps (local, reproducible)

This directory contains scripts to build local ID→symbol mapping files for **display/interpretability** (e.g., `report.md` / `audit_log.tsv` outputs).  
These maps are **optional** and do **not** affect auditing logic.

## Entrez Gene ID → symbol (human)

Build from NCBI `gene_info`:

```bash
bash resources/gene_id_maps/build_entrez_id_map.sh
```

Outputs:

resources/gene_id_maps/id_map.tsv.gz

resources/gene_id_maps/checksums.sha256

```python-repl
The mapping format is:
gene_id<TAB>gene_symbol
4033   STAT3
7062   TNF
...
```

## Using the map in the tool
Point your Sample Card to the mapping file:
```yaml
extra:
  gene_id_map_tsv: "resources/gene_id_maps/id_map.tsv.gz"
```

## Notes on correctness
- The default URL is human-only (Homo_sapiens.gene_info.gz).
- If you prefer to explicitly enforce tax_id==9606, replace the cut line with:
```bash
awk -F'\t' 'BEGIN{OFS="\t"} $1=="9606" {print $2,$3}' resources/gene_id_maps/Homo_sapiens.gene_info
```