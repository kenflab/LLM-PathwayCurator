#!/usr/bin/env Rscript
# paper/scripts/figS4_beataml_deg_rank.R

suppressPackageStartupMessages({
  library(data.table)
  library(limma)
  library(edgeR)
})

die <- function(msg) stop(msg, call. = FALSE)

# -------------------------
# Robust script directory
# -------------------------
script_path <- NULL
for (a in commandArgs(trailingOnly = FALSE)) {
  if (startsWith(a, "--file=")) {
    script_path <- sub("^--file=", "", a)
    break
  }
}
if (is.null(script_path) || !nzchar(script_path)) {
  die("Cannot determine script path. Run via: Rscript paper/scripts/figS4_beataml_deg_rank.R")
}
SCRIPT_DIR <- dirname(normalizePath(script_path, mustWork = TRUE))
ROOT <- normalizePath(file.path(SCRIPT_DIR, ".."), mustWork = TRUE)  # paper/

# -------------------------
# Paths
# -------------------------
SD   <- file.path(ROOT, "source_data", "BEATAML_TP53_v1")
RAW  <- file.path(SD, "raw")
DER  <- file.path(SD, "derived")
OUT  <- file.path(DER, "rankings")

counts_path <- file.path(RAW, "beataml_waves1to4_counts_dbgap.txt")
groups_path <- file.path(DER, "groups", "BEATAML.groups.tsv")
out_path <- file.path(OUT, "BEATAML.deg_ranking.tsv")

# -------------------------
# Args
# -------------------------
args <- commandArgs(trailingOnly = TRUE)
include_clinical_only <- FALSE
if (length(args) >= 1) {
  if (any(args %in% c("--include_clinical_only"))) include_clinical_only <- TRUE
}

# -------------------------
# Input checks
# -------------------------
if (!file.exists(counts_path)) die(paste("missing:", counts_path))
if (!file.exists(groups_path)) die(paste("missing:", groups_path))
dir.create(OUT, recursive = TRUE, showWarnings = FALSE)

cat("[beataml_deg_rank] inputs\n")
cat("  counts:", counts_path, "\n")
cat("  groups:", groups_path, "\n")
cat("  include_clinical_only:", include_clinical_only, "\n")

# -------------------------
# Groups
# -------------------------
grp <- fread(groups_path)
if (!all(c("sample", "group") %in% names(grp))) die("groups.tsv must have columns: sample, group")

keep_groups <- c("TP53_wt_wes", "TP53_mut_wes")
if (include_clinical_only) keep_groups <- c(keep_groups, "TP53_clinical_only")
grp <- grp[group %in% keep_groups]
if (nrow(grp) < 20) die("too few samples after group filter; check groups.tsv")

# -------------------------
# Counts
# -------------------------
expr <- fread(counts_path)
if (ncol(expr) < 6) die("counts has too few columns")

gene_col <- if ("stable_id" %in% names(expr)) "stable_id" else names(expr)[1]
genes <- expr[[gene_col]]
if (anyDuplicated(genes) > 0) die(paste0("gene IDs are not unique in column: ", gene_col))

# sample columns = intersection(colnames(expr), grp$sample)
sample_cols <- intersect(names(expr), grp$sample)
cat("[beataml_deg_rank] detected sample cols:", length(sample_cols), "\n")
if (length(sample_cols) < 20) {
  die("too few matched sample columns in counts (<20). Check that counts colnames match groups$sample (e.g., BA####R).")
}

mat <- as.matrix(expr[, ..sample_cols])
rownames(mat) <- genes
mode(mat) <- "numeric"
if (any(!is.finite(mat))) die("counts matrix has non-numeric or non-finite values after conversion")

# align group order to sample_cols
grp2 <- grp[match(sample_cols, grp$sample)]
if (any(is.na(grp2$group))) die("group mapping has NA after matching")

group_factor <- grp2$group
if (include_clinical_only) group_factor[group_factor == "TP53_clinical_only"] <- "TP53_wt_wes"
group_factor <- factor(group_factor, levels = c("TP53_wt_wes", "TP53_mut_wes"))
if (any(is.na(group_factor))) die("unexpected group values after normalization")

# -------------------------
# edgeR + voom + limma
# -------------------------
dge <- DGEList(counts = mat)
keep <- filterByExpr(dge, group = group_factor)
dge <- dge[keep, , keep.lib.sizes = FALSE]
dge <- calcNormFactors(dge)

design <- model.matrix(~ 0 + group_factor)
colnames(design) <- levels(group_factor)

v <- voom(dge, design, plot = FALSE)
fit <- lmFit(v, design)
contr <- makeContrasts(TP53_mut_wes - TP53_wt_wes, levels = design)
fit2 <- contrasts.fit(fit, contr)
fit2 <- eBayes(fit2)

tt <- topTable(fit2, number = Inf, sort.by = "none")
if (!("t" %in% names(tt))) die("limma topTable missing 't' column")

rank <- data.table(gene = rownames(tt), score = tt$t)
rank <- rank[!is.na(score)]
setorder(rank, -score)

fwrite(rank, out_path, sep = "\t")

cat("[beataml_deg_rank] OK\n")
cat("  gene_col:", gene_col, "\n")
cat("  wrote:", out_path, "\n")
cat("  n_genes:", nrow(rank), "\n")
cat("  n_samples_used:", ncol(mat), "\n")
