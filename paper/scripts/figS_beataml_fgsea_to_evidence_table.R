#!/usr/bin/env Rscript
# paper/scripts/figS_beataml_fgsea_to_evidence_table.R

suppressPackageStartupMessages({
  library(data.table)
  library(fgsea)
  library(msigdbr)
})

die <- function(msg) stop(msg, call. = FALSE)

# Robust script directory
script_path <- NULL
for (a in commandArgs(trailingOnly = FALSE)) {
  if (startsWith(a, "--file=")) {
    script_path <- sub("^--file=", "", a)
    break
  }
}
if (is.null(script_path) || !nzchar(script_path)) {
  die("Cannot determine script path. Run via: Rscript paper/scripts/figS_beataml_fgsea_to_evidence_table.R")
}

SCRIPT_DIR <- dirname(normalizePath(script_path, mustWork = TRUE))
ROOT <- normalizePath(file.path(SCRIPT_DIR, ".."), mustWork = TRUE)  # paper/

SD   <- file.path(ROOT, "source_data", "BEATAML_TP53_v1")
DER  <- file.path(SD, "derived")
OUT  <- file.path(SD, "evidence_tables")

rank_path <- file.path(DER, "rankings", "BEATAML.deg_ranking.tsv")
out_path  <- file.path(OUT, "BEATAML.evidence_table.tsv")

if (!file.exists(rank_path)) die(paste("missing:", rank_path))
dir.create(OUT, recursive = TRUE, showWarnings = FALSE)

cat("[beataml_fgsea_to_evidence] inputs\n")
cat("  rank:", rank_path, "\n")
cat("  out :", out_path, "\n")
cat("  msigdbr_version:", as.character(packageVersion("msigdbr")), "\n")
cat("  fgsea_version  :", as.character(packageVersion("fgsea")), "\n")

# -------------------------
# Helpers
# -------------------------
norm_ensg <- function(x) {
  x <- as.character(x)
  x <- trimws(x)
  # ENSG00000123456.12 -> ENSG00000123456
  x <- sub("^((ENSG[0-9]+))(\\.[0-9]+)?$", "\\1", x, perl = TRUE)
  x
}

pick_col <- function(nms, cands) {
  nms_l <- tolower(nms)
  hit <- match(tolower(cands), nms_l, nomatch = 0)
  hit <- hit[hit != 0]
  if (length(hit) == 0) return(NULL)
  nms[hit[[1]]]
}

# -------------------------
# Ranking (BeatAML: Ensembl stable_id)
# -------------------------
rank <- fread(rank_path)
if (!all(c("gene", "score") %in% names(rank))) die("deg_ranking.tsv must have columns: gene, score")

rank[, gene := norm_ensg(gene)]
rank[, score := as.numeric(score)]
if (any(is.na(rank$score))) die("score contains NA after numeric conversion (check ranking file)")

# Ensembl-only contract for BeatAML
if (!any(grepl("^ENSG[0-9]+$", rank$gene))) {
  die("ranking gene IDs do not look like Ensembl (ENSG...). BeatAML expects stable_id (ENSG...).")
}
if (any(grepl("^\\d+$", rank$gene))) {
  die("ranking gene IDs include numeric-only IDs; BeatAML script expects Ensembl-only.")
}

stats <- rank$score
names(stats) <- rank$gene
stats <- stats[!is.na(stats)]
stats <- sort(stats, decreasing = TRUE)

cat("[beataml_fgsea_to_evidence] stats diagnostics\n")
cat("  stats_len:", length(stats), "\n")
cat("  stats_range:", paste(range(stats, na.rm = TRUE), collapse = ","), "\n")
cat("  gene_example:", paste(head(names(stats), 10), collapse = ","), "\n")
cat("  gene_id_type: ensembl\n")

if (length(stats) < 1000) die("ranking too small (<1000 genes) - check inputs")

# Deterministic tie-breaking for fgsea
set.seed(0)
o <- order(stats, decreasing = TRUE)
eps <- 1e-12
stats[o] <- stats[o] + eps * seq_along(stats)

# -------------------------
# Hallmark sets (msigdbr; Ensembl column required)
# -------------------------
msig <- msigdbr(species = "Homo sapiens", collection = "H")
nms <- names(msig)

ens_col <- pick_col(nms, c("ensembl_gene", "ensembl_gene_id", "ensembl"))
if (is.null(ens_col)) {
  die(paste0(
    "msigdbr output missing an Ensembl-like column. Available columns:\n",
    paste(nms, collapse = ", ")
  ))
}
cat("[beataml_fgsea_to_evidence] msigdbr gene column:", ens_col, "\n")

msig[[ens_col]] <- norm_ensg(msig[[ens_col]])
msig <- msig[!is.na(msig[[ens_col]]) & nzchar(msig[[ens_col]]), ]

pathways <- split(msig[[ens_col]], msig$gs_name)

# overlap diagnostics (Ensembl ↔ Ensembl)
ov <- vapply(pathways, function(gs) sum(gs %in% names(stats)), integer(1))
cat("[beataml_fgsea_to_evidence] overlap diagnostics\n")
cat("  pathways_with_overlap>=15:", sum(ov >= 15), "/", length(ov), "\n")
if (max(ov) == 0) die("overlap is zero for all pathways (gene ID type mismatch)")

# fgsea
res <- fgseaMultilevel(pathways = pathways, stats = stats, minSize = 15, maxSize = 500)
if (nrow(res) == 0) die("fgsea returned 0 rows after minSize/maxSize filtering (try minSize=10)")
if (!("leadingEdge" %in% names(res))) die("fgsea result missing leadingEdge")

# EvidenceTable v1 (BeatAML: Ensembl)
dt <- data.table(
  term_id = res$pathway,
  term_name = res$pathway,
  source = "fgsea_msigdb_H_ensembl",
  stat = res$NES,
  qval = res$padj,
  direction = ifelse(res$NES > 0, "up", ifelse(res$NES < 0, "down", "na")),
  evidence_genes = vapply(res$leadingEdge, function(x) paste(norm_ensg(x), collapse = ","), character(1))
)

dt <- dt[is.finite(stat)]
dt <- dt[!is.na(qval)]
if (nrow(dt) == 0) die("EvidenceTable has 0 rows after filtering (unexpected)")

dt[, stat_abs := abs(stat)]
setorder(dt, qval, -stat_abs)
dt[, stat_abs := NULL]

fwrite(dt, out_path, sep = "\t")
cat("[beataml_fgsea_to_evidence] OK\n")
cat("  wrote:", out_path, "\n")
cat("  n_terms:", nrow(dt), "\n")
