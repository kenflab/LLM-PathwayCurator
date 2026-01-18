#!/usr/bin/env Rscript
# paper/scripts/fig2_fgsea_to_evidence_table.R

suppressPackageStartupMessages({
  library(data.table)
  library(fgsea)
  library(msigdbr)
})

ROOT <- normalizePath(file.path(dirname(sys.frame(1)$ofile), ".."), mustWork = TRUE)  # paper/
SD   <- file.path(ROOT, "source_data", "PANCAN_TP53_v1")
DER  <- file.path(SD, "derived")
OUT  <- file.path(SD, "evidence_tables")

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 1) {
  stop("Usage: Rscript paper/scripts/fig2_fgsea_to_evidence_table.R <CANCER>\nExample: Rscript ... HNSC", call. = FALSE)
}
CANCER <- toupper(args[[1]])

rank_path <- file.path(DER, "rankings", paste0(CANCER, ".deg_ranking.tsv"))
out_path  <- file.path(OUT, paste0(CANCER, ".evidence_table.tsv"))

if (!file.exists(rank_path)) stop(paste("missing:", rank_path), call. = FALSE)
dir.create(OUT, recursive = TRUE, showWarnings = FALSE)

rank <- fread(rank_path)
if (!all(c("gene", "score") %in% names(rank))) stop("deg_ranking.tsv must have columns: gene, score", call. = FALSE)

stats <- rank$score
names(stats) <- rank$gene
stats <- stats[!is.na(stats)]
stats <- sort(stats, decreasing = TRUE)

if (length(stats) < 1000) stop("ranking too small (<1000 genes) - check inputs", call. = FALSE)

# Hallmark sets (minimal, stable)
msig <- msigdbr(species = "Homo sapiens", category = "H")
pathways <- split(msig$gene_symbol, msig$gs_name)

set.seed(0)
res <- fgsea(pathways = pathways, stats = stats, nperm = 10000)
# columns include: pathway, pval, padj, ES, NES, size, leadingEdge

if (!("leadingEdge" %in% names(res))) stop("fgsea result missing leadingEdge", call. = FALSE)

# EvidenceTable v1
# term_id, term_name, source, stat, qval, direction, evidence_genes
dt <- data.table(
  term_id = res$pathway,
  term_name = res$pathway,
  source = "fgsea_msigdb_H",
  stat = res$NES,
  qval = res$padj,
  direction = ifelse(res$NES > 0, "up", ifelse(res$NES < 0, "down", "na")),
  evidence_genes = vapply(res$leadingEdge, function(x) paste(x, collapse = ","), character(1))
)

# keep finite
dt <- dt[is.finite(stat)]
dt <- dt[!is.na(qval)]
setorder(dt, qval, -abs(stat))

fwrite(dt, out_path, sep = "\t")
cat("[fgsea_to_evidence] OK\n")
cat("  cancer:", CANCER, "\n")
cat("  wrote:", out_path, "\n")
