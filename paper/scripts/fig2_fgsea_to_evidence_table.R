#!/usr/bin/env Rscript
# paper/scripts/fig2_fgsea_to_evidence_table.R

# =============================================================================
# fig2_fgsea_to_evidence_table.R
#
# Convert an fgsea run (Hallmark, msigdbr) into an EvidenceTable (v1).
#
# Parameters
# ----------
# CANCER : character(1)
#   TCGA cancer code (e.g., "HNSC"). Passed as a single CLI argument.
#
# Inputs
# ------
# derived/rankings/{CANCER}.deg_ranking.tsv
#   Required columns: gene, score
#   This benchmark expects ENTREZ IDs (as character strings) in `gene`.
#
# Outputs
# -------
# evidence_tables/{CANCER}.evidence_table.tsv
#   Columns:
#     term_id, term_name, source, stat, qval, direction, evidence_genes
#
# Dependencies
# ------------
# data.table, fgsea, msigdbr
#
# Reproducibility
# ---------------
# - Uses set.seed(0).
# - Adds an epsilon jitter to break ties deterministically.
#
# Failure modes
# -------------
# - Missing ranking file or required columns.
# - NA scores after numeric conversion.
# - Too-small ranking (< 1000 genes).
# - Zero overlap between stats genes and msigdbr gene sets.
# - fgsea returns 0 rows after minSize/maxSize filtering.
# =============================================================================


suppressPackageStartupMessages({
  library(data.table)
  library(fgsea)
  library(msigdbr)
})

die <- function(msg) stop(msg, call. = FALSE)

# Robust script directory (works under Rscript)
script_path <- NULL
for (a in commandArgs(trailingOnly = FALSE)) {
  if (startsWith(a, "--file=")) {
    script_path <- sub("^--file=", "", a)
    break
  }
}
if (is.null(script_path) || !nzchar(script_path)) {
  die("Cannot determine script path. Please run via: Rscript paper/scripts/fig2_fgsea_to_evidence_table.R <CANCER>")
}

SCRIPT_DIR <- dirname(normalizePath(script_path, mustWork = TRUE))
ROOT <- normalizePath(file.path(SCRIPT_DIR, ".."), mustWork = TRUE)  # paper/
SD   <- file.path(ROOT, "source_data", "PANCAN_TP53_v1")
DER  <- file.path(SD, "derived")
OUT  <- file.path(SD, "evidence_tables")

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 1) {
  die("Usage: Rscript paper/scripts/fig2_fgsea_to_evidence_table.R <CANCER>\nExample: Rscript paper/scripts/fig2_fgsea_to_evidence_table.R HNSC")
}
CANCER <- toupper(args[[1]])

rank_path <- file.path(DER, "rankings", paste0(CANCER, ".deg_ranking.tsv"))
out_path  <- file.path(OUT, paste0(CANCER, ".evidence_table.tsv"))

if (!file.exists(rank_path)) die(paste("missing:", rank_path))
dir.create(OUT, recursive = TRUE, showWarnings = FALSE)

cat("[fgsea_to_evidence] inputs\n")
cat("  cancer:", CANCER, "\n")
cat("  rank:", rank_path, "\n")
cat("  out :", out_path, "\n")
cat("  msigdbr_version:", as.character(packageVersion("msigdbr")), "\n")
cat("  fgsea_version  :", as.character(packageVersion("fgsea")), "\n")

rank <- fread(rank_path)
if (!all(c("gene", "score") %in% names(rank))) die("deg_ranking.tsv must have columns: gene, score")

# IMPORTANT: this benchmark uses Entrez IDs in the ranking (numeric strings)
# Ensure gene IDs are character strings
rank[, gene := as.character(gene)]
rank[, score := as.numeric(score)]
if (any(is.na(rank$score))) die("score contains NA after numeric conversion (check ranking file)")

stats <- rank$score
names(stats) <- rank$gene
stats <- stats[!is.na(stats)]
stats <- sort(stats, decreasing = TRUE)

cat("[fgsea_to_evidence] stats diagnostics\n")
cat("  stats_len:", length(stats), "\n")
cat("  stats_type:", typeof(stats), "\n")
cat("  stats_na:", sum(is.na(stats)), "\n")
cat("  stats_range:", paste(range(stats, na.rm = TRUE), collapse = ","), "\n")
cat("  gene_example:", paste(head(names(stats), 10), collapse = ","), "\n")
cat("  gene_id_type: entrez\n")

if (length(stats) < 1000) die("ranking too small (<1000 genes) - check inputs")

# Deterministic tie-breaking for fgsea
set.seed(0)
o <- order(stats, decreasing = TRUE)
eps <- 1e-12
stats[o] <- stats[o] + eps * seq_along(stats)

# Hallmark sets, matched by ENTREZ IDs
msig <- msigdbr(species = "Homo sapiens", collection = "H")

# Find Entrez ID column robustly across msigdbr versions
entrez_candidates <- c(
  "entrez_gene", "entrezgene", "entrez_gene_id", "entrez", "ncbi_gene", "ncbi_gene_id"
)
nms <- names(msig)
nms_l <- tolower(nms)

hit <- match(entrez_candidates, nms_l, nomatch = 0)
hit <- hit[hit != 0]

if (length(hit) == 0) {
  die(paste0(
    "msigdbr output missing an Entrez-like column. Available columns:\n",
    paste(nms, collapse = ", ")
  ))
}

entrez_col <- nms[hit[[1]]]
cat("[fgsea_to_evidence] msigdbr entrez column:", entrez_col, "\n")

msig[[entrez_col]] <- as.character(msig[[entrez_col]])
msig <- msig[!is.na(msig[[entrez_col]]) & nzchar(msig[[entrez_col]]), ]

pathways <- split(msig[[entrez_col]], msig$gs_name)


# overlap diagnostics (Entrez ↔ Entrez)
ov <- vapply(pathways, function(gs) sum(gs %in% names(stats)), integer(1))
cat("[fgsea_to_evidence] overlap diagnostics\n")
cat("  overlap_summary:", paste(
  "min", min(ov), "p25", as.integer(quantile(ov, 0.25)),
  "median", as.integer(median(ov)), "p75", as.integer(quantile(ov, 0.75)),
  "max", max(ov)
), "\n")
cat("  pathways_with_overlap>=15:", sum(ov >= 15), "/", length(ov), "\n")

if (max(ov) == 0) die("[fgsea_to_evidence] overlap is zero for all pathways (check gene ID type consistency)")

# Use fgseaMultilevel (recommended)
# Set minSize/maxSize explicitly for stability
res <- fgseaMultilevel(pathways = pathways, stats = stats, minSize = 15, maxSize = 500)

cat("[fgsea_to_evidence] fgsea diagnostics\n")
cat("  n_pathways:", length(pathways), "\n")
cat("  res_rows  :", nrow(res), "\n")

if (nrow(res) == 0) die("[fgsea_to_evidence] fgsea returned 0 rows after minSize/maxSize filtering (try minSize=10)")

if (!("leadingEdge" %in% names(res))) die("fgsea result missing leadingEdge")

# EvidenceTable v1
dt <- data.table(
  term_id = res$pathway,
  term_name = res$pathway,
  source = "fgsea_msigdb_H_entrez",
  stat = res$NES,
  qval = res$padj,
  direction = ifelse(res$NES > 0, "up", ifelse(res$NES < 0, "down", "na")),
  evidence_genes = vapply(res$leadingEdge, function(x) paste(x, collapse = ","), character(1))
)

dt <- dt[is.finite(stat)]
dt <- dt[!is.na(qval)]

if (nrow(dt) == 0) die("[fgsea_to_evidence] EvidenceTable has 0 rows after filtering (unexpected)")

dt[, stat_abs := abs(stat)]
setorder(dt, qval, -stat_abs)
dt[, stat_abs := NULL]

fwrite(dt, out_path, sep = "\t")
cat("[fgsea_to_evidence] OK\n")
cat("  wrote:", out_path, "\n")
cat("  n_terms:", nrow(dt), "\n")
