#!/usr/bin/env Rscript
# paper/scripts/fig2_deg_rank.R

# =============================================================================
# fig2_deg_rank.R
#
# Compute DEG ranking (TP53_mut vs TP53_wt) using limma t-statistic.
#
# Parameters
# ----------
# CANCER : character(1)
#   TCGA cancer code (e.g., "HNSC"). Passed as a single CLI argument.
#
# Inputs
# ------
# raw/expression.xena.gz  (or expression.tsv.gz / expression.xena.tsv.gz)
#   Wide matrix: first column = gene id, remaining columns = samples.
# derived/groups/{CANCER}.groups.tsv
#   Columns: sample, group, group_basis (UNKNOWN excluded)
#
# Outputs
# -------
# derived/rankings/{CANCER}.deg_ranking.tsv
#   Columns: gene, score, gene_id_type
#   score is limma moderated t-statistic for (TP53_mut - TP53_wt).
#
# Dependencies
# ------------
# data.table, limma
#
# Failure modes
# -------------
# - Missing expression or groups file.
# - Too few matched samples (< 10).
# - Unexpected group labels (not TP53_wt/TP53_mut).
# - Non-numeric expression matrix (would fail lmFit).
# =============================================================================


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
source(file.path(SCRIPT_DIR, "tcga_input_utils.R"))

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


args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 1) {
  die("Usage: Rscript paper/scripts/fig2_deg_rank.R <CANCER>\nExample: Rscript paper/scripts/fig2_deg_rank.R HNSC")
}
CANCER <- toupper(args[[1]])
expr_path <- option_value(args, "--expression", expr_path)
OUT <- option_value(args, "--outdir", OUT)
gene_map_path <- option_value(args, "--gene-map", file.path(ROOT, "..", "resources", "gene_id_maps", "id_map.tsv.gz"))

groups_path <- option_value(args, "--groups", file.path(DER, "groups", paste0(CANCER, ".groups.tsv")))
out_path <- file.path(OUT, paste0(CANCER, ".deg_ranking.tsv"))

if (is.na(expr_path) || !file.exists(expr_path)) die("Missing expression; use --expression")
if (!file.exists(groups_path)) die(paste("missing:", groups_path))
if (!file.exists(gene_map_path)) die(paste("missing gene map:", gene_map_path))
require_new_outputs(c(out_path, paste0(out_path, c(".gene_mapping.tsv", ".provenance.tsv"))))

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
genes <- as.character(expr[[gene_col]])
mat <- as.matrix(expr[, -1, with = FALSE])
prepared <- prepare_xena_expression(mat, genes, fread(gene_map_path, colClasses = "character"))
mat <- prepared$matrix

# UNKNOWN samples never enter the contrast. Reject stale two-column group files.
grp <- fread(groups_path)
grp2 <- validated_groups(as.data.frame(grp), colnames(mat))
samples <- grp2$sample
cat("[deg_rank] assessed matched samples:", length(samples), "\n")
mat2 <- mat[, samples, drop = FALSE]

# QC: drop genes with zero variance across matched samples
vars <- apply(mat2, 1, var, na.rm = TRUE)
n0 <- sum(!is.finite(vars) | vars <= 0)
if (n0 > 0) {
  cat("[deg_rank] QC: dropping zero-variance genes:", n0, "\n")
  keep <- is.finite(vars) & (vars > 0)
  mat2 <- mat2[keep, , drop = FALSE]
}

if (nrow(mat2) < 1000L) die("Too few variable genes after mapping/QC (<1000)")

# design
group_factor <- factor(grp2$group, levels = c("TP53_wt", "TP53_mut"))
if (any(is.na(group_factor))) die("unexpected group values (expected TP53_wt/TP53_mut)")

design <- model.matrix(~ 0 + group_factor)
colnames(design) <- levels(group_factor)

# limma
fit <- lmFit(mat2, design)
# Keep the original gene identity in an explicit annotation column.
fit$genes <- data.frame(gene_id = rownames(mat2))
contr <- makeContrasts(TP53_mut - TP53_wt, levels = design)
fit2 <- contrasts.fit(fit, contr)
fit2 <- eBayes(fit2)

rank <- extract_symbol_ranking(fit2, rownames(mat2))
fwrite(prepared$mapping, paste0(out_path, ".gene_mapping.tsv"), sep = "\t")
meta <- c(expression_md5 = unname(tools::md5sum(expr_path)),
          groups_md5 = unname(tools::md5sum(groups_path)),
          gene_map_md5 = unname(tools::md5sum(gene_map_path)),
          gene_id_type = "symbol", duplicate_policy = "mean_log_expression_before_fit",
          duplicate_rows_collapsed = prepared$duplicate_rows_collapsed,
          input_gene_rows = length(genes), mapped_genes = nrow(mat), variable_genes = nrow(mat2),
          non_estimable_genes = nrow(mat2) - nrow(rank),
          n_mut = sum(grp2$group == "TP53_mut"), n_wt = sum(grp2$group == "TP53_wt"),
          n_unknown_excluded = sum(grp$group == "TP53_unknown" & grp$sample %in% colnames(mat)),
          R = R.version.string, limma = as.character(packageVersion("limma")))
fwrite(data.frame(key = names(meta), value = unname(meta)), paste0(out_path, ".provenance.tsv"), sep = "\t")

fwrite(rank, out_path, sep = "\t")
cat("[deg_rank] OK\n")
cat("  wrote:", out_path, "\n")
cat("  n_genes:", nrow(rank), "\n")
