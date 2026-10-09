# Shared input contracts for the TCGA figure scripts. No network access on source().

option_value <- function(args, flag, default = NULL) {
  pos <- which(args == flag)
  if (!length(pos)) return(default)
  if (length(pos) != 1L || pos + 1L > length(args) || startsWith(args[pos + 1L], "--")) {
    stop(paste("Expected one value for", flag), call. = FALSE)
  }
  args[pos + 1L]
}

require_new_outputs <- function(paths) {
  existing <- paths[file.exists(paths)]
  if (length(existing)) stop(paste("Output already exists; use a new --outdir:",
                                  paste(existing, collapse = ", ")), call. = FALSE)
}

prepare_xena_expression <- function(mat, ids, gene_map) {
  # Xena EB++ uses mostly symbols, with a small number of numeric Entrez IDs.
  # Resolve numeric IDs only through an explicit mapping; never infer from row numbers.
  ids <- trimws(as.character(ids))
  if (length(ids) != nrow(mat) || anyNA(ids) || any(!nzchar(ids))) {
    stop("Missing expression gene IDs or row-count mismatch", call. = FALSE)
  }
  if (!all(c("gene_id", "gene_symbol") %in% names(gene_map))) {
    stop("Gene map requires gene_id and gene_symbol", call. = FALSE)
  }
  numeric_id <- grepl("^[0-9]+$", ids)
  symbols <- ids
  status <- rep("symbol_retained", length(ids))
  mapping <- unique(data.frame(gene_id = as.character(gene_map$gene_id),
                               gene_symbol = as.character(gene_map$gene_symbol)))
  for (id in unique(ids[numeric_id])) {
    hit <- unique(mapping$gene_symbol[!is.na(mapping$gene_id) & mapping$gene_id == id])
    hit <- hit[!is.na(hit) & nzchar(hit)]
    if (length(hit) > 1L) stop(paste("Ambiguous Entrez mapping:", id), call. = FALSE)
    ix <- numeric_id & ids == id
    if (length(hit)) {
      if (grepl("^[0-9]+$", hit)) stop("Gene map target must be a symbol", call. = FALSE)
      symbols[ix] <- hit
      status[ix] <- "entrez_mapped"
    } else {
      symbols[ix] <- NA_character_
      status[ix] <- "unmapped_entrez_excluded"
    }
  }
  audit <- data.frame(input_row = seq_along(ids), input_gene = ids,
                      gene_symbol = symbols, mapping_status = status)
  keep <- !is.na(symbols)
  if (!any(keep)) stop("No expression rows with resolved IDs", call. = FALSE)
  if (any(!keep)) warning(sum(!keep), " unmapped Entrez rows excluded; see gene_mapping.tsv",
                         call. = FALSE)
  mat <- mat[keep, , drop = FALSE]
  rownames(mat) <- symbols[keep]
  # Outcome-independent duplicate rule: mean log-expression per gene and sample,
  # before fitting. Do not choose the duplicate with the most significant t score.
  duplicate_rows <- sum(duplicated(rownames(mat)))
  mat <- limma::avereps(mat, ID = rownames(mat))
  stopifnot(!anyDuplicated(rownames(mat)))
  list(matrix = mat, mapping = audit, duplicate_rows_collapsed = duplicate_rows)
}

validated_groups <- function(groups, expression_samples) {
  required <- c("sample", "group", "group_basis")
  if (!all(required %in% names(groups))) {
    stop("Rebuild groups with fig2_make_groups.py: sample, group and group_basis required",
         call. = FALSE)
  }
  if (anyNA(groups$sample) || anyDuplicated(groups$sample) ||
      anyNA(expression_samples) || anyDuplicated(expression_samples)) {
    stop("Sample identifiers must be present and unique", call. = FALSE)
  }
  if (anyNA(groups$group) || any(!groups$group %in% c("TP53_mut", "TP53_wt", "TP53_unknown"))) {
    stop("Unexpected TP53 group", call. = FALSE)
  }
  if (anyNA(groups$group_basis) ||
      any(groups$group == "TP53_wt" & groups$group_basis != "assessed_no_protein_altering_call") ||
      any(groups$group == "TP53_mut" & groups$group_basis != "protein_altering_call")) {
    stop("WT requires documented TP53 assessment; MUT requires a qualifying call", call. = FALSE)
  }
  groups <- groups[match(expression_samples, groups$sample, nomatch = 0L), , drop = FALSE]
  groups <- groups[groups$group != "TP53_unknown", , drop = FALSE]
  counts <- table(factor(groups$group, levels = c("TP53_wt", "TP53_mut")))
  if (any(counts < 2L) || nrow(groups) < 10L) {
    stop("Need >=2 assessed samples per arm and >=10 total; UNKNOWN is not WT",
         call. = FALSE)
  }
  groups
}

extract_symbol_ranking <- function(fit, expected_ids) {
  tt <- limma::topTable(fit, number = Inf, sort.by = "none")
  if (!"gene_id" %in% names(tt) || !identical(as.character(tt$gene_id), expected_ids)) {
    stop("Gene annotation was lost or reordered during limma fitting", call. = FALSE)
  }
  rank <- data.frame(gene = as.character(tt$gene_id), score = tt$t,
                     gene_id_type = "symbol")
  if (anyDuplicated(rank$gene)) stop("Duplicate ranking IDs", call. = FALSE)
  keep <- is.finite(rank$score)
  if (any(!keep)) warning(sum(!keep), " non-estimable gene scores excluded", call. = FALSE)
  rank <- rank[keep, , drop = FALSE]
  rank[order(-rank$score, rank$gene, method = "radix"), , drop = FALSE]
}

symbol_stats <- function(rank) {
  if (!all(c("gene", "score", "gene_id_type") %in% names(rank))) {
    stop("Rebuild ranking: explicit gene_id_type is required; legacy row numbers are unsafe",
         call. = FALSE)
  }
  gene <- trimws(as.character(rank$gene))
  score <- suppressWarnings(as.numeric(rank$score))
  if (!nrow(rank) || anyNA(rank$gene_id_type) || any(rank$gene_id_type != "symbol") ||
      anyNA(gene) || any(!nzchar(gene)) || any(grepl("^[0-9]+$", gene))) {
    stop("Expected explicitly annotated gene symbols, not numeric row numbers", call. = FALSE)
  }
  if (anyDuplicated(gene) || any(!is.finite(score))) {
    stop("Ranking IDs must be unique and scores finite", call. = FALSE)
  }
  # Stable order for exact ties without modifying the fitted statistics.
  o <- order(-score, gene, method = "radix")
  setNames(score[o], gene[o])
}

symbol_pathways <- function(msig) {
  if (!all(c("gs_name", "gene_symbol") %in% names(msig))) {
    stop("msigdbr gene sets require gs_name and gene_symbol", call. = FALSE)
  }
  membership <- unique(data.frame(gs_name = as.character(msig$gs_name),
                                  gene_symbol = as.character(msig$gene_symbol)))
  membership <- membership[!is.na(membership$gs_name) & nzchar(membership$gs_name) &
                             !is.na(membership$gene_symbol) & nzchar(membership$gene_symbol), ]
  if (!nrow(membership)) stop("No gene-set memberships", call. = FALSE)
  membership <- membership[order(membership$gs_name, membership$gene_symbol), ]
  list(membership = membership, pathways = split(membership$gene_symbol, membership$gs_name))
}

load_msig <- function(collection, subcategory = NULL) {
  # msigdbr renamed category/subcategory and gs_subcat in recent versions.
  if ("collection" %in% names(formals(msigdbr::msigdbr))) {
    msig <- msigdbr::msigdbr(species = "Homo sapiens", collection = collection)
  } else {
    msig <- msigdbr::msigdbr(species = "Homo sapiens", category = collection)
  }
  if (!is.null(subcategory) && nzchar(subcategory)) {
    column <- intersect(c("gs_subcollection", "gs_subcat"), names(msig))[1L]
    if (is.na(column)) stop("msigdbr subcollection column missing", call. = FALSE)
    msig <- msig[!is.na(msig[[column]]) & msig[[column]] == subcategory, ]
  }
  msig
}

write_tcga_fgsea <- function(rank_path, out_path, collection, subcategory = NULL) {
  sidecars <- paste0(out_path, c(".memberships.tsv", ".provenance.tsv"))
  require_new_outputs(c(out_path, sidecars))
  rank <- data.table::fread(rank_path)
  stats <- symbol_stats(rank)
  if (length(stats) < 1000L) stop("Ranking too small (<1000 genes)", call. = FALSE)
  msig <- load_msig(collection, subcategory)
  sets <- symbol_pathways(msig)
  overlap <- vapply(sets$pathways, function(gs) sum(gs %in% names(stats)), integer(1))
  if (!any(overlap >= 15L & overlap <= 500L)) {
    stop("No gene sets with 15..500 matching symbols; check gene ID mapping", call. = FALSE)
  }
  cat("[fgsea] symbol overlap; eligible sets:", sum(overlap >= 15L & overlap <= 500L), "\n")
  set.seed(0)
  res <- fgsea::fgseaMultilevel(pathways = sets$pathways, stats = stats,
                               minSize = 15, maxSize = 500, nproc = 1)
  source_tag <- paste0("fgsea_msigdb_", collection,
                       if (!is.null(subcategory)) paste0("_", gsub("[^A-Za-z0-9]+", "_", subcategory)),
                       "_symbol")
  result <- data.frame(term_id = res$pathway, term_name = res$pathway, source = source_tag,
                       stat = res$NES, stat_kind = "NES", qval = res$padj, q_kind = "padj",
                       direction = ifelse(res$NES > 0, "up", ifelse(res$NES < 0, "down", "na")),
                       evidence_genes = vapply(res$leadingEdge, paste, character(1), collapse = ","),
                       gene_id_type = "symbol")
  result <- result[is.finite(result$stat) & is.finite(result$qval), ]
  if (!nrow(result)) stop("No finite fgsea results", call. = FALSE)
  result <- result[order(result$qval, -abs(result$stat), result$term_id), ]
  dir.create(dirname(out_path), recursive = TRUE, showWarnings = FALSE)
  data.table::fwrite(sets$membership, sidecars[1L], sep = "\t")
  versions <- if ("db_version" %in% names(msig)) paste(unique(msig$db_version), collapse = ",") else "unreported"
  provenance <- c(ranking_md5 = unname(tools::md5sum(rank_path)),
                   memberships_md5 = unname(tools::md5sum(sidecars[1L])),
                   gene_id_type = "symbol", msigdbr = as.character(utils::packageVersion("msigdbr")),
                   msigdb = versions, fgsea = as.character(utils::packageVersion("fgsea")),
                   R = R.version.string, seed = "0", minSize = "15", maxSize = "500",
                   nproc = "1", tie_policy = "score_then_gene_no_jitter",
                   ranking_genes = length(stats), genes_in_any_set = sum(names(stats) %in% sets$membership$gene_symbol))
  data.table::fwrite(data.frame(key = names(provenance), value = unname(provenance)), sidecars[2L], sep = "\t")
  data.table::fwrite(result, out_path, sep = "\t")
  cat("[fgsea] wrote:", out_path, "\n")
  invisible(result)
}
