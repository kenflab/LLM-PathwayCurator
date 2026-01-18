#!/usr/bin/env Rscript
# paper/scripts/fig2_deg_rank.R

suppressPackageStartupMessages({
  library(data.table)
  library(limma)
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
  die("Cannot determine script path. Please run via: Rscript paper/scripts/fig2_deg_rank.R <CANCER>")
}

SCRIPT_DIR <- dirname(normalizePath(script_path, mustWork = TRUE))
ROOT <- normalizePath(file.path(SCRIPT_DIR, ".."), mustWork = TRUE)  # paper/

SD   <- file.path(ROOT, "source_data", "PANCAN_TP53_v1")
RAW  <- file.path(SD, "raw")
DER  <- file.path(SD, "derived")
OUT  <- file.path(DER, "rankings")

# Prefer the fetched Xena file name first
expr_candidates <- c(
  file.path(RAW, "expression.xena.gz"),
  file.path(RAW, "expression.tsv.gz"),
  file.path(RAW, "expression.xena.tsv.gz")
)
expr_path <- expr_candidates[file.exists(expr_candidates)][1]
if (is.na(expr_path) || !nzchar(expr_path)) {
  die(paste("Missing expression file. Tried:\n", paste(expr_candidates, collapse = "\n")))
}

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 1) {
  die("Usage: Rscript paper/scripts/fig2_deg_rank.R <CANCER>\nExample: Rscript paper/scripts/fig2_deg_rank.R HNSC")
}
CANCER <- toupper(args[[1]])

groups_path <- file.path(DER, "groups", paste0(CANCER, ".groups.tsv"))
out_path <- file.path(OUT, paste0(CANCER, ".deg_ranking.tsv"))

if (!file.exists(groups_path)) die(paste("missing:", groups_path))

dir.create(OUT, recursive = TRUE, showWarnings = FALSE)

cat("[deg_rank] inputs\n")
cat("  cancer:", CANCER, "\n")
cat("  expr:", expr_path, "\n")
cat("  groups:", groups_path, "\n")

# expression: expected format
# gene \t sample1 \t sample2 ...
expr <- fread(expr_path)
if (ncol(expr) < 3) die("expression must be: gene + >=2 samples (wide matrix)")

gene_col <- names(expr)[1]
genes <- expr[[gene_col]]
mat <- as.matrix(expr[, -1, with = FALSE])
rownames(mat) <- genes

# groups
grp <- fread(groups_path)
if (!all(c("sample", "group") %in% names(grp))) die("groups.tsv must have columns: sample, group")

# match samples
samples <- intersect(colnames(mat), grp$sample)
cat("[deg_rank] matched samples:", length(samples), " / expr=", ncol(mat), " / groups=", nrow(grp), "\n")

if (length(samples) < 10) die("too few matched samples between expression and groups (<10)")

mat2 <- mat[, samples, drop = FALSE]

# QC: drop genes with zero variance across matched samples
vars <- apply(mat2, 1, var, na.rm = TRUE)
n0 <- sum(!is.finite(vars) | vars <= 0)
if (n0 > 0) {
  cat("[deg_rank] QC: dropping zero-variance genes:", n0, "\n")
  keep <- is.finite(vars) & (vars > 0)
  mat2 <- mat2[keep, , drop = FALSE]
}

grp2 <- grp[match(samples, grp$sample)]
if (any(is.na(grp2$group))) die("group mapping has NA after matching")

# design
group_factor <- factor(grp2$group, levels = c("TP53_wt", "TP53_mut"))
if (any(is.na(group_factor))) die("unexpected group values (expected TP53_wt/TP53_mut)")

design <- model.matrix(~ 0 + group_factor)
colnames(design) <- levels(group_factor)

# limma
fit <- lmFit(mat2, design)
contr <- makeContrasts(TP53_mut - TP53_wt, levels = design)
fit2 <- contrasts.fit(fit, contr)
fit2 <- eBayes(fit2)

tt <- topTable(fit2, number = Inf, sort.by = "none")
if (!("t" %in% names(tt))) die("limma topTable missing 't' column")

rank <- data.table(gene = rownames(tt), score = tt$t)
rank <- rank[!is.na(score)]
setorder(rank, -score)

fwrite(rank, out_path, sep = "\t")
cat("[deg_rank] OK\n")
cat("  wrote:", out_path, "\n")
cat("  n_genes:", nrow(rank), "\n")
