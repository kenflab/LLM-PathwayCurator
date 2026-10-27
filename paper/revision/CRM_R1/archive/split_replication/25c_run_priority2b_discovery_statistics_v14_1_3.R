#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(data.table)
  library(fgsea)
  library(jsonlite)
  library(limma)
})

die <- function(message) stop(message, call. = FALSE)
`%||%` <- function(left, right) if (is.null(left)) right else left

parse_args <- function(values) {
  result <- list()
  index <- 1L
  while (index <= length(values)) {
    key <- values[[index]]
    if (!startsWith(key, "--") || index == length(values)) {
      die(paste("Malformed argument list near", key))
    }
    result[[substring(key, 3L)]] <- values[[index + 1L]]
    index <- index + 2L
  }
  result
}

sha256_file <- function(path) {
  shasum <- Sys.which("shasum")
  if (!nzchar(shasum)) die("shasum is required")
  output <- system2(shasum, c("-a", "256", path), stdout = TRUE, stderr = TRUE)
  status <- attr(output, "status")
  if (!is.null(status) && status != 0L) die(paste("shasum failed for", path))
  strsplit(output[[1L]], "[[:space:]]+")[[1L]][[1L]]
}

script_path <- NULL
for (value in commandArgs(trailingOnly = FALSE)) {
  if (startsWith(value, "--file=")) {
    script_path <- normalizePath(sub("^--file=", "", value), mustWork = TRUE)
    break
  }
}
if (is.null(script_path)) die("Cannot determine script path")

args <- parse_args(commandArgs(trailingOnly = TRUE))
required_args <- c("data-root", "expression", "work-root")
missing_args <- required_args[!vapply(required_args, function(x) nzchar(args[[x]] %||% ""), logical(1))]
if (length(missing_args) > 0L) {
  die(paste("Missing required arguments:", paste(missing_args, collapse = ", ")))
}

data_root <- normalizePath(args[["data-root"]], mustWork = TRUE)
expression_path <- normalizePath(args[["expression"]], mustWork = TRUE)
work_root <- normalizePath(args[["work-root"]], mustWork = FALSE)
priority2b_root <- file.path(data_root, "output", "priority2b")

prohibited <- c(
  file.path(priority2b_root, "discovery_stats_lock_v14_1_2"),
  file.path(priority2b_root, "discovery_stats_lock_v14_1_3"),
  file.path(priority2b_root, "membership_lock_v14_1_3"),
  file.path(priority2b_root, "validation_lock_v14_1_3"),
  file.path(priority2b_root, "final_v14_1_3"),
  file.path(priority2b_root, "membership_lock_v14_1_2"),
  file.path(priority2b_root, "validation_lock_v14_1_2"),
  file.path(priority2b_root, "final_v14_1_2"),
  file.path(priority2b_root, "membership_lock_v14_1"),
  file.path(priority2b_root, "validation_lock_v14_1"),
  file.path(priority2b_root, "final_v14_1")
)
present <- prohibited[file.exists(prohibited)]
if (length(present) > 0L) {
  die(paste("Downstream V14.1 output already exists:", paste(present, collapse = ", ")))
}

design_manifest_path <- file.path(priority2b_root, "design_lock_v14_1", "design_manifest.json")
hallmark_manifest_path <- file.path(
  priority2b_root, "hallmark_lock_v14_1_2", "hallmark_manifest.json"
)
numerical_manifest_path <- file.path(
  priority2b_root, "numerical_lock_v14_1_3", "fgsea_numerical_manifest.json"
)
if (!file.exists(design_manifest_path)) die(paste("Missing", design_manifest_path))
if (!file.exists(hallmark_manifest_path)) die(paste("Missing", hallmark_manifest_path))
if (!file.exists(numerical_manifest_path)) die(paste("Missing", numerical_manifest_path))

design_manifest <- fromJSON(design_manifest_path, simplifyVector = FALSE)
hallmark_manifest <- fromJSON(hallmark_manifest_path, simplifyVector = FALSE)
numerical_manifest <- fromJSON(numerical_manifest_path, simplifyVector = FALSE)
if (design_manifest$schema_version != "CRM_R1_P2B_DESIGN_v14_1") die("Design schema drift")
if (hallmark_manifest$schema_version != "CRM_R1_P2B_HALLMARK_v14_1_2") {
  die("Hallmark schema drift")
}
if (
  numerical_manifest$schema_version !=
    "CRM_R1_P2B_FGSEA_NUMERICAL_LOCK_v14_1_3"
) {
  die("Numerical-lock schema drift")
}
if (
  numerical_manifest$status !=
    "FGSEA_NUMERICAL_AMENDMENT_FROZEN_BEFORE_RESTART"
) {
  die("Numerical lock is not released")
}
if (as.integer(numerical_manifest$nPermSimple) != 10000L) {
  die("V14.1.3 requires nPermSimple=10000")
}

assignments_path <- design_manifest$outputs$assignments_private$path
gene_sets_path <- hallmark_manifest$outputs$gene_sets$path
protocol_path <- hallmark_manifest$outputs$protocol_frozen$path
amendment_path <- numerical_manifest$outputs$amendment_frozen$path
for (path in c(assignments_path, gene_sets_path, protocol_path, amendment_path)) {
  if (!file.exists(path)) die(paste("Missing locked input:", path))
}

protocol <- fromJSON(protocol_path, simplifyVector = FALSE)
if (protocol$status != "READY_TO_FREEZE_DISCOVERY") die("Discovery protocol is not approved")
if (protocol$schema_version != "CRM_R1_P2B_DISCOVERY_PROTOCOL_v14_1_2") {
  die("Discovery protocol schema drift")
}
if (protocol$analysis_boundary$validation_outcomes_read != FALSE) die("Validation leakage in protocol")
reference_versions <- protocol$reference_environment
observed_versions <- list(
  r = as.character(getRversion()),
  fgsea = as.character(packageVersion("fgsea")),
  limma = as.character(packageVersion("limma")),
  msigdbr = as.character(packageVersion("msigdbr"))
)
for (package_name in names(observed_versions)) {
  if (observed_versions[[package_name]] != reference_versions[[package_name]]) {
    die(paste(
      "Reference-environment drift for", package_name,
      ": expected", reference_versions[[package_name]],
      "observed", observed_versions[[package_name]]
    ))
  }
}

dir.create(work_root, recursive = TRUE, showWarnings = FALSE)
signature_path <- file.path(work_root, "discovery_run_signature.json")
expression_sha256 <- sha256_file(expression_path)
expected_expression_sha256 <- design_manifest$inputs$expression_matrix$sha256
if (!identical(expression_sha256, expected_expression_sha256)) {
  die("Expression matrix does not match the V14.1 design lock")
}
signature <- list(
  schema_version = "CRM_R1_P2B_DISCOVERY_RUN_SIGNATURE_v14_1_3",
  script_sha256 = sha256_file(script_path),
  design_manifest_sha256 = sha256_file(design_manifest_path),
  assignments_sha256 = sha256_file(assignments_path),
  hallmark_manifest_sha256 = sha256_file(hallmark_manifest_path),
  gene_sets_sha256 = sha256_file(gene_sets_path),
  protocol_sha256 = sha256_file(protocol_path),
  numerical_manifest_sha256 = sha256_file(numerical_manifest_path),
  numerical_amendment_sha256 = sha256_file(amendment_path),
  expression_sha256 = expression_sha256,
  nPermSimple = 10000L,
  expression_feature_policy = paste(
    "trim_labels; exclude_blank; exclude_all_numeric_unresolved;",
    "exclude_all_rows_for_every_duplicated_symbol"
  )
)
if (file.exists(signature_path)) {
  existing <- fromJSON(signature_path, simplifyVector = FALSE)
  if (!identical(existing, signature)) {
    die("Discovery work directory was created under a different input/code signature")
  }
} else {
  write_json(signature, signature_path, pretty = TRUE, auto_unbox = TRUE)
}

assignments <- fread(assignments_path, colClasses = "character")
required_assignment <- c("cohort_id", "split_id", "sample_id", "group_label", "partition")
if (!all(required_assignment %in% names(assignments))) die("Assignment schema drift")
assignments <- assignments[partition == "discovery"]
if (nrow(assignments) == 0L) die("No discovery assignments")

gene_sets <- fread(gene_sets_path, colClasses = "character")
if (!all(c("term_id", "gene_id") %in% names(gene_sets))) die("Hallmark schema drift")
if (any(!grepl("[[:alpha:]]", gene_sets$gene_id))) {
  die("V14.1.2 Hallmark lock contains a non-symbol gene ID")
}
pathways <- split(gene_sets$gene_id, gene_sets$term_id)
pathways <- lapply(pathways, unique)
pathways <- pathways[sort(names(pathways))]
if (length(pathways) != 50L) die("Expected exactly 50 frozen Hallmark pathways")

cat("[INFO] Reading PanCan expression matrix once\n")
expression <- fread(expression_path)
if (ncol(expression) < 3L) die("Expression matrix is too small")
gene_column <- names(expression)[[1L]]
genes <- trimws(as.character(expression[[gene_column]]))
blank_mask <- !nzchar(genes)
numeric_mask <- grepl("^[0-9]+$", genes)
duplicated_symbols <- sort(unique(genes[
  duplicated(genes) | duplicated(genes, fromLast = TRUE)
]))
duplicated_symbols <- duplicated_symbols[nzchar(duplicated_symbols)]
duplicated_mask <- genes %in% duplicated_symbols
retain_mask <- !(blank_mask | numeric_mask | duplicated_mask)
retained_genes <- genes[retain_mask]

if (length(retained_genes) < 1000L) {
  die("Fewer than 1000 unique gene-symbol features remain after pre-model QC")
}
if (anyDuplicated(retained_genes)) {
  die("Retained expression gene symbols are not unique")
}

exclusion_reason <- rep("", length(genes))
exclusion_reason[blank_mask] <- "BLANK_FEATURE_LABEL"
exclusion_reason[numeric_mask] <- "NUMERIC_UNRESOLVED_IDENTIFIER"
exclusion_reason[duplicated_mask & !blank_mask & !numeric_mask] <-
  "DUPLICATED_GENE_SYMBOL_EXCLUDE_ALL_ROWS"
excluded_rows <- data.table(
  expression_data_row = which(!retain_mask) + 1L,
  input_gene_id = genes[!retain_mask],
  action = "EXCLUDE",
  reason = exclusion_reason[!retain_mask]
)
feature_qc <- data.table(
  input_feature_rows = length(genes),
  retained_unique_gene_symbol_rows = length(retained_genes),
  excluded_rows = nrow(excluded_rows),
  excluded_blank_rows = sum(blank_mask),
  excluded_numeric_unresolved_rows = sum(numeric_mask),
  duplicated_gene_symbols = length(duplicated_symbols),
  excluded_duplicated_symbol_rows = sum(duplicated_mask),
  duplicate_policy = "exclude_all_rows_for_every_duplicated_symbol",
  selection_or_averaging_performed = FALSE
)
feature_qc_path <- file.path(work_root, "expression_feature_qc.tsv")
feature_exclusions_path <- file.path(work_root, "expression_feature_exclusions.tsv")
fwrite(feature_qc, feature_qc_path, sep = "\t", quote = FALSE)
fwrite(excluded_rows, feature_exclusions_path, sep = "\t", quote = FALSE)
cat(sprintf(
  paste0(
    "[INFO] Expression feature QC: input=%d; retained symbols=%d; ",
    "numeric excluded=%d; duplicated symbols=%d (%d rows excluded)\n"
  ),
  length(genes), length(retained_genes), sum(numeric_mask),
  length(duplicated_symbols), sum(duplicated_mask)
))
if ("SLC35E2" %in% duplicated_symbols) {
  cat("[INFO] Both SLC35E2 rows were excluded; no selection or averaging occurred\n")
}

expression <- expression[retain_mask]
genes <- retained_genes
matrix_values <- as.matrix(expression[, -1L, with = FALSE])
storage.mode(matrix_values) <- "double"
rownames(matrix_values) <- genes
rm(expression)
gc()

min_size <- as.integer(protocol$enrichment$min_size)
max_size <- as.integer(protocol$enrichment$max_size)
retained_gene_set <- unique(genes)
pathway_overlap <- data.table(
  term_id = names(pathways),
  frozen_gene_symbols = vapply(pathways, length, integer(1)),
  expression_overlap_gene_symbols = vapply(
    pathways,
    function(values) sum(unique(values) %in% retained_gene_set),
    integer(1)
  )
)
setorder(pathway_overlap, term_id)
if (nrow(pathway_overlap) != 50L) die("Hallmark overlap term-count drift")
if (any(pathway_overlap$expression_overlap_gene_symbols < min_size)) {
  failed <- pathway_overlap[expression_overlap_gene_symbols < min_size, term_id]
  die(paste(
    "Hallmark pathways below the frozen minSize after symbol matching:",
    paste(failed, collapse = ",")
  ))
}
fwrite(
  pathway_overlap,
  file.path(work_root, "hallmark_expression_overlap.tsv"),
  sep = "\t",
  quote = FALSE
)

normalize_sample <- function(value) {
  substring(gsub("\\.", "-", toupper(trimws(value))), 1L, 15L)
}
normalized_columns <- vapply(colnames(matrix_values), normalize_sample, character(1))
if (anyDuplicated(normalized_columns)) {
  duplicated_ids <- unique(normalized_columns[duplicated(normalized_columns) | duplicated(normalized_columns, fromLast = TRUE)])
  frozen_ids <- unique(assignments$sample_id)
  if (length(intersect(duplicated_ids, frozen_ids)) > 0L) {
    die("A frozen discovery sample maps to non-unique expression columns")
  }
}
column_by_sample <- setNames(colnames(matrix_values), normalized_columns)

cohorts <- sort(unique(assignments$cohort_id))
splits <- sort(unique(assignments$split_id))
expected_pairs <- unique(assignments[, .(cohort_id, split_id)])
if (nrow(expected_pairs) != 440L) die(paste("Expected 440 cohort/split pairs; observed", nrow(expected_pairs)))

base_seed <- as.integer(protocol$enrichment$deterministic_seed_base)
n_perm_simple <- as.integer(numerical_manifest$nPermSimple)
term_set <- sort(names(pathways))
job_files <- character()
computed <- 0L
skipped <- 0L

for (cohort_index in seq_along(cohorts)) {
  cohort_value <- cohorts[[cohort_index]]
  for (split_index in seq_along(splits)) {
    split_value <- splits[[split_index]]
    block <- assignments[
      cohort_id == cohort_value & split_id == split_value
    ]
    if (nrow(block) == 0L) next

    job_dir <- file.path(work_root, "jobs", cohort_value, split_value)
    dir.create(job_dir, recursive = TRUE, showWarnings = FALSE)
    output_path <- file.path(job_dir, "discovery_evidence.tsv")
    job_files <- c(job_files, output_path)

    if (file.exists(output_path)) {
      previous <- tryCatch(fread(output_path), error = function(error) NULL)
      if (!is.null(previous) && nrow(previous) == 50L &&
          identical(sort(as.character(previous$term_id)), term_set)) {
        skipped <- skipped + 1L
        next
      }
      die(paste("Incomplete existing job output; inspect before rerun:", output_path))
    }

    sample_ids <- block$sample_id
    raw_columns <- unname(column_by_sample[sample_ids])
    if (any(is.na(raw_columns))) {
      die(paste("Missing expression column for", cohort_value, split_value))
    }
    values <- matrix_values[, raw_columns, drop = FALSE]
    group <- factor(
      block$group_label[match(sample_ids, block$sample_id)],
      levels = c("TP53_WILD_TYPE", "TP53_MUTANT")
    )
    if (any(is.na(group))) die("Unexpected group label")
    if (any(table(group) < 2L)) {
      die(paste("Too few discovery samples for", cohort_value, split_value))
    }

    finite_rows <- rowSums(is.finite(values)) == ncol(values)
    row_sum <- rowSums(values, na.rm = TRUE)
    row_sq_sum <- rowSums(values * values, na.rm = TRUE)
    row_variance_numerator <- row_sq_sum - (row_sum * row_sum / ncol(values))
    keep <- finite_rows & is.finite(row_variance_numerator) & row_variance_numerator > 0
    values <- values[keep, , drop = FALSE]
    if (nrow(values) < 1000L) {
      die(paste("Fewer than 1000 variable genes for", cohort_value, split_value))
    }

    design <- model.matrix(~ 0 + group)
    colnames(design) <- levels(group)
    fit <- lmFit(values, design)
    contrast <- makeContrasts(TP53_MUTANT - TP53_WILD_TYPE, levels = design)
    fit <- eBayes(contrasts.fit(fit, contrast))
    ranking <- fit$t[, 1L]
    ranking <- ranking[is.finite(ranking)]
    ranking <- sort(ranking, decreasing = TRUE)
    ordering <- order(ranking, decreasing = TRUE)
    ranking[ordering] <- ranking[ordering] + 1e-12 * seq_along(ranking)

    analysis_seed <- base_seed + cohort_index * 1000L + split_index
    set.seed(analysis_seed)
    result <- fgseaMultilevel(
      pathways = pathways,
      stats = ranking,
      minSize = min_size,
      maxSize = max_size,
      scoreType = "std",
      nPermSimple = n_perm_simple
    )
    if (nrow(result) != 50L || !identical(sort(result$pathway), term_set)) {
      die(paste("fgsea candidate-pool drift for", cohort_value, split_value))
    }
    if (any(!is.finite(result$NES)) || any(is.na(result$padj))) {
      die(paste("Non-finite fgsea result for", cohort_value, split_value))
    }

    evidence <- data.table(
      cohort_id = cohort_value,
      split_id = split_value,
      term_id = result$pathway,
      term_name = result$pathway,
      source = "fgsea_msigdb_H_gene_symbol_v14_1_3_nPermSimple10000",
      stat = result$NES,
      pval = result$pval,
      qval = result$padj,
      direction = ifelse(result$NES > 0, "up", "down"),
      evidence_genes = vapply(result$leadingEdge, paste, collapse = ",", character(1)),
      pathway_size = result$size,
      n_discovery_mutant = as.integer(table(group)[["TP53_MUTANT"]]),
      n_discovery_wild_type = as.integer(table(group)[["TP53_WILD_TYPE"]]),
      analysis_seed = analysis_seed,
      nPermSimple = n_perm_simple
    )
    setorder(evidence, term_id)
    fwrite(evidence, output_path, sep = "\t", quote = FALSE)
    computed <- computed + 1L
    cat(sprintf(
      "[DONE] %s/%s (%d/440)\n",
      cohort_value, split_value, computed + skipped
    ))
  }
}

if (length(job_files) != 440L || any(!file.exists(job_files))) {
  die("Not all 440 discovery jobs completed")
}
combined <- rbindlist(lapply(job_files, fread), use.names = TRUE)
setorder(combined, cohort_id, split_id, term_id)
if (nrow(combined) != 22000L) die(paste("Expected 22000 rows; observed", nrow(combined)))
combined_path <- file.path(work_root, "discovery_statistics.tsv")
fwrite(combined, combined_path, sep = "\t", quote = FALSE)

session_path <- file.path(work_root, "R_sessionInfo.txt")
writeLines(capture.output(sessionInfo()), session_path)
summary <- data.table(
  cohort_split_pairs = 440L,
  terms_per_pair = 50L,
  rows = nrow(combined),
  computed_this_run = computed,
  resumed_jobs = skipped
)
fwrite(summary, file.path(work_root, "discovery_statistics_summary.tsv"), sep = "\t")

cat("[PASS] Completed all V14.1.3 discovery limma/fgsea jobs\n")
cat("[INFO] cohort/split pairs=440; rows=22000\n")
cat("[INFO] nPermSimple=10000 was applied uniformly to all jobs\n")
cat("[INFO] combined:", combined_path, "\n")
cat("[NEXT] Freeze discovery statistics with script 26c\n")
