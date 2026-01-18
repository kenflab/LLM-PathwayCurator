#!/usr/bin/env Rscript
# paper/scripts/fig2_deg_rank.R
suppressPackageStartupMessages({
  library(data.table)
  library(limma)
})

ROOT <- normalizePath(file.path(dirname(sys.frame(1)$ofile), ".."), mustWork = TRUE)  # paper/
SD   <- file.path(ROOT, "source_data", "PANCAN_TP53_v1")
RAW  <- file.path(SD, "raw")
DER  <- file.path(SD, "derived")
OUT  <- file.path(DER, "rankings")

expr_path <- file.path(RAW, "expression.tsv.gz")

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 1) {
  stop("Usage: Rscript paper/scripts/fig2_deg_rank.R <CANCER>\nExample: Rscript ... HNSC", call. = FALSE)
}
CANCER <- toupper(args[[1]])

groups_path <- file.path(DER, "groups", paste0(CANCER, ".groups.tsv"))
out_path <- file.path(OUT, paste0(CANCER, ".deg_ranking.tsv"))

if (!file.exists(expr_path)) stop(paste("missing:", expr_path), call. = FALSE)
if (!file.exists(groups_path)) stop(paste("missing:", groups_path), call. = FALSE)

dir.create(OUT, recursive = TRUE, showWarnings = FALSE)

# expression: expected format
# gene \t sample1 \t sample2 ...
expr <- fread(expr_path)
if (ncol(expr) < 3) stop("expression.tsv.gz must be gene + >=2 samples", call. = FALSE)

gene_col <- names(expr)[1]
genes <- expr[[gene_col]]
mat <- as.matrix(expr[, -1, with = FALSE])
rownames(mat) <- genes

# groups
grp <- fread(groups_path)
if (!all(c("sample", "group") %in% names(grp))) stop("groups.tsv must have columns: sample, group", call. = FALSE)

# match samples
samples <- intersect(colnames(mat), grp$sample)
if (length(samples) < 10) stop("too few matched samples between expression and groups (<10)", call. = FALSE)

mat2 <- mat[, samples, drop = FALSE]
grp2 <- grp[match(samples, grp$sample)]
if (any(is.na(grp2$group))) stop("group mapping has NA after matching", call. = FALSE)

# design
group_factor <- factor(grp2$group, levels = c("TP53_wt", "TP53_mut"))
if (any(is.na(group_factor))) stop("unexpected group values (expected TP53_wt/TP53_mut)", call. = FALSE)

design <- model.matrix(~ 0 + group_factor)
colnames(design) <- levels(group_factor)

fit <- lmFit(mat2, design)
contr <- makeContrasts(TP53_mut - TP53_wt, levels = design)
fit2 <- contrasts.fit(fit, contr)
fit2 <- eBayes(fit2)

tt <- topTable(fit2, number = Inf, sort.by = "none")
# tt has rownames=genes, columns incl t, logFC, P.Value, adj.P.Val
if (!("t" %in% names(tt))) stop("limma topTable missing 't' column", call. = FALSE)

rank <- data.table(gene = rownames(tt), score = tt$t)
rank <- rank[!is.na(score)]
setorder(rank, -score)

fwrite(rank, out_path, sep = "\t")
cat("[deg_rank] OK\n")
cat("  cancer:", CANCER, "\n")
cat("  wrote:", out_path, "\n")
