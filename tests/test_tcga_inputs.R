# Run from the repository root: Rscript tests/test_tcga_inputs.R
suppressPackageStartupMessages(library(limma))
source("paper/scripts/tcga_input_utils.R")
must_fail <- function(expr, pattern) {
  err <- tryCatch({ force(expr); NULL }, error = identity)
  stopifnot(inherits(err, "error"), grepl(pattern, conditionMessage(err)))
}

set.seed(42)
ids <- c("TP53", "SLC35E2", "SLC35E2", "7157", "999999999", paste0("GENE", 1:100))
mat <- matrix(rnorm(length(ids) * 12), nrow = length(ids))
rownames(mat) <- ids
colnames(mat) <- paste0("S", 1:12)
design <- model.matrix(~ rep(c(0, 1), each = 6))

# Reproduce the actual failure with duplicated input IDs and real limma.
legacy <- topTable(eBayes(lmFit(mat, design)), coef = 2, number = Inf, sort.by = "none")
stopifnot(identical(rownames(legacy), as.character(seq_along(ids))))
stopifnot(identical(as.character(legacy$ID), ids))

mapping <- data.frame(gene_id = "7157", gene_symbol = "TP53")
prepared <- suppressWarnings(prepare_xena_expression(mat, ids, mapping))
fixed <- prepared$matrix
stopifnot(sum(rownames(fixed) == "SLC35E2") == 1L, sum(rownames(fixed) == "TP53") == 1L)
stopifnot(prepared$duplicate_rows_collapsed == 2L)
stopifnot(isTRUE(all.equal(unname(fixed["SLC35E2", ]), unname(colMeans(mat[2:3, ])))))
stopifnot(isTRUE(all.equal(unname(fixed["TP53", ]), unname(colMeans(mat[c(1, 4), ])))))
stopifnot(prepared$mapping$mapping_status[5] == "unmapped_entrez_excluded")
fit <- lmFit(fixed, design)
fit$genes <- data.frame(gene_id = rownames(fixed))
fit <- eBayes(fit[, 2, drop = FALSE])
ranking <- extract_symbol_ranking(fit, rownames(fixed))
stopifnot(setequal(ranking$gene, rownames(fixed)), all(ranking$gene_id_type == "symbol"))
stopifnot(isTRUE(all.equal(ranking$score, unname(fit$t[match(ranking$gene, rownames(fixed)), 1]))))
must_fail(extract_symbol_ranking(fit, rev(rownames(fixed))), "lost or reordered")
must_fail(prepare_xena_expression(mat, ids, rbind(mapping, data.frame(gene_id = "7157", gene_symbol = "OTHER"))), "Ambiguous")

# Old numeric row-number rankings cannot enter either TCGA fgsea entry point.
must_fail(symbol_stats(data.frame(gene = as.character(1:20), score = 1:20)), "Rebuild ranking")
must_fail(symbol_stats(data.frame(gene = as.character(1:20), score = 1:20, gene_id_type = "symbol")), "numeric row numbers")
must_fail(symbol_stats(rbind(ranking, ranking[1, ])), "unique")
bad <- ranking; bad$score[1] <- Inf
must_fail(symbol_stats(bad), "finite")
tied <- data.frame(gene = c("B", "A"), score = c(1, 1), gene_id_type = "symbol")
stopifnot(identical(symbol_stats(tied), c(A = 1, B = 1)))

# Gene-set membership must use symbols, even when an Entrez column also exists.
msig <- data.frame(gs_name = c("P", "P", "P"), gene_symbol = c("TP53", "TP53", "SLC35E2"), ncbi_gene = c("7157", "7157", "9906"))
sets <- symbol_pathways(msig)
stopifnot(setequal(sets$pathways$P, c("TP53", "SLC35E2")), length(sets$pathways$P) == 2L)
stopifnot(all(sets$pathways$P %in% names(symbol_stats(ranking))))

# Explicit sample alignment; an unassessed sample cannot become a comparator.
groups <- data.frame(sample = paste0("S", 1:13), group = c(rep("TP53_wt", 6), rep("TP53_mut", 6), "TP53_unknown"),
                     group_basis = c(rep("assessed_no_protein_altering_call", 6), rep("protein_altering_call", 6), "no_TP53_assessment"))
aligned <- validated_groups(groups, rev(groups$sample))
stopifnot(identical(aligned$sample, rev(groups$sample[1:12])), !any(aligned$group == "TP53_unknown"))
bad <- groups; bad$group_basis[1] <- "no_TP53_assessment"
must_fail(validated_groups(bad, groups$sample), "WT requires")
must_fail(validated_groups(groups[, c("sample", "group")], groups$sample), "Rebuild groups")
must_fail(validated_groups(groups[7:13, ], groups$sample), "UNKNOWN is not WT")
cat("TCGA input regression tests passed (real limma duplicate-ID reproduction included).\n")
