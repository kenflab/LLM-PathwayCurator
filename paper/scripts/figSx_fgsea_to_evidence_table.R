#!/usr/bin/env Rscript
# paper/scripts/figSx_fgsea_to_evidence_table.R

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
  die("Cannot determine script path. Please run via: Rscript paper/scripts/figSx_fgsea_to_evidence_table.R <CANCER> [--collection H] [--subcategory ...]")
}

SCRIPT_DIR <- dirname(normalizePath(script_path, mustWork = TRUE))
ROOT <- normalizePath(file.path(SCRIPT_DIR, ".."), mustWork = TRUE)  # paper/
SD   <- file.path(ROOT, "source_data", "PANCAN_TP53_v1")
DER  <- file.path(SD, "derived")
OUT  <- file.path(SD, "evidence_tables")

# -------- args --------
args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 1) {
  die("Usage: Rscript paper/scripts/figSx_fgsea_to_evidence_table.R <CANCER> [--collection H] [--subcategory CP:REACTOME] [--out-suffix ...]")
}
CANCER <- toupper(args[[1]])

# simple flag parser
get_flag <- function(flag, default = NA_character_) {
  hit <- which(args == flag)
  if (length(hit) == 0) return(default)
  if (hit[[1]] + 1 > length(args)) return(default)
  return(args[[hit[[1]] + 1]])
}

collection <- get_flag("--collection", "H")               # H / C2 / C5 ...
subcategory <- get_flag("--subcategory", NA_character_)   # e.g. CP:REACTOME, GO:BP
out_suffix <- get_flag("--out-suffix", NA_character_)     # optional; else auto from collection/subcategory

rank_path <- file.path(DER, "rankings", paste0(CANCER, ".deg_ranking.tsv"))
if (!file.exists(rank_path)) die(paste("missing:", rank_path))
dir.create(OUT, recursive = TRUE, showWarnings = FALSE)

# output naming
safe_sub <- function(x) {
  if (is.na(x) || !nzchar(x)) return("")
  gsub("[^A-Za-z0-9]+", "_", x)
}
if (is.na(out_suffix) || !nzchar(out_suffix)) {
  out_suffix <- paste0(collection, ifelse(!is.na(subcategory) && nzchar(subcategory), paste0("_", safe_sub(subcategory)), ""))
}
out_path <- file.path(OUT, paste0(CANCER, ".", out_suffix, ".evidence_table.tsv"))

source_tag <- paste0(
  "fgsea_msigdb_", collection,
  ifelse(!is.na(subcategory) && nzchar(subcategory), paste0("_", safe_sub(subcategory)), ""),
  "_entrez"
)

cat("[fgsea_to_evidence] inputs\n")
cat("  cancer:", CANCER, "\n")
cat("  rank:", rank_path, "\n")
cat("  out :", out_path, "\n")
cat("  collection:", collection, "\n")
cat("  subcategory:", ifelse(is.na(subcategory), "<none>", subcategory), "\n")
cat("  source_tag:", source_tag, "\n")
cat("  msigdbr_version:", as.character(packageVersion("msigdbr")), "\n")
cat("  fgsea_version  :", as.character(packageVersion("fgsea")), "\n")

rank <- fread(rank_path)
if (!all(c("gene", "score") %in% names(rank))) die("deg_ranking.tsv must have columns: gene, score")

# Entrez IDs expected
rank[, gene := as.character(gene)]
rank[, score := as.numeric(score)]
if (any(is.na(rank$score))) die("score contains NA after numeric conversion (check ranking file)")

stats <- rank$score
names(stats) <- rank$gene
stats <- stats[!is.na(stats)]
stats <- sort(stats, decreasing = TRUE)

if (length(stats) < 1000) die("ranking too small (<1000 genes) - check inputs")

# Deterministic tie-breaking for fgsea
set.seed(0)
o <- order(stats, decreasing = TRUE)
eps <- 1e-12
stats[o] <- stats[o] + eps * seq_along(stats)

# msigdbr gene sets
msig <- msigdbr(species = "Homo sapiens", category = collection)

# filter subcategory if requested (msigdbr uses `gs_subcat`)
if (!is.na(subcategory) && nzchar(subcategory)) {
  if (!("gs_subcat" %in% names(msig))) die("msigdbr output missing gs_subcat")
  msig <- msig[msig$gs_subcat == subcategory, ]
}

if (nrow(msig) == 0) {
  die(paste0("No gene sets after filtering: category=", collection, " subcategory=", subcategory))
}

# Find Entrez ID column robustly
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

# pathways: gs_name -> vector(entrez)
pathways <- split(msig[[entrez_col]], msig$gs_name)

# overlap diagnostics
ov <- vapply(pathways, function(gs) sum(gs %in% names(stats)), integer(1))
cat("[fgsea_to_evidence] overlap diagnostics\n")
cat("  overlap_summary:", paste(
  "min", min(ov), "p25", as.integer(quantile(ov, 0.25)),
  "median", as.integer(median(ov)), "p75", as.integer(quantile(ov, 0.75)),
  "max", max(ov)
), "\n")
cat("  pathways_with_overlap>=15:", sum(ov >= 15), "/", length(ov), "\n")
if (max(ov) == 0) die("[fgsea_to_evidence] overlap is zero for all pathways (check gene ID type consistency)")

# fgsea
res <- fgseaMultilevel(pathways = pathways, stats = stats, minSize = 15, maxSize = 500)
if (nrow(res) == 0) die("[fgsea_to_evidence] fgsea returned 0 rows after minSize/maxSize filtering (try minSize=10)")
if (!("leadingEdge" %in% names(res))) die("fgsea result missing leadingEdge")

# EvidenceTable v1
dt <- data.table(
  term_id = res$pathway,
  term_name = res$pathway,
  source = source_tag,
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
