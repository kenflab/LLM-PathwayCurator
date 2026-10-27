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

validate_record <- function(record, label) {
  path <- record$path %||% ""
  expected <- record$sha256 %||% ""
  if (!nzchar(path) || !file.exists(path)) die(paste("Missing", label, path))
  if (!identical(sha256_file(path), expected)) die(paste(label, "hash drift"))
  normalizePath(path, mustWork = TRUE)
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
required_args <- c("data-root", "work-root")
missing_args <- required_args[
  !vapply(required_args, function(x) nzchar(args[[x]] %||% ""), logical(1))
]
if (length(missing_args) > 0L) {
  die(paste("Missing required arguments:", paste(missing_args, collapse = ", ")))
}

data_root <- normalizePath(args[["data-root"]], mustWork = TRUE)
work_root <- normalizePath(args[["work-root"]], mustWork = FALSE)
priority2b_root <- file.path(data_root, "output", "priority2b")
validation_design_path <- file.path(
  priority2b_root,
  "validation_design_lock_v14_1_3",
  "validation_design_manifest.json"
)
if (!file.exists(validation_design_path)) {
  die(paste("Missing validation design lock:", validation_design_path))
}
digest_path <- sub("[.]json$", ".sha256", validation_design_path)
if (!file.exists(digest_path)) die("Missing validation-design manifest digest")
digest_tokens <- strsplit(readLines(digest_path, n = 1L), "[[:space:]]+")[[1L]]
digest_tokens <- digest_tokens[nzchar(digest_tokens)]
if (length(digest_tokens) < 1L || digest_tokens[[1L]] != sha256_file(validation_design_path)) {
  die("Validation-design manifest drift")
}

validation_design <- fromJSON(validation_design_path, simplifyVector = FALSE)
if (validation_design$schema_version != "CRM_R1_P2B_VALIDATION_DESIGN_v14_1_3") {
  die("Validation-design schema drift")
}
if (
  validation_design$status !=
    "METHOD_BLIND_VALIDATION_DESIGN_FROZEN_BEFORE_OUTCOMES"
) {
  die("Validation design is not released")
}
boundary <- validation_design$analysis_boundary
if (boundary$partition != "validation") die("Validation partition drift")
if (boundary$method_membership_rows_read_by_runner != FALSE) {
  die("Method-membership boundary drift")
}
if (boundary$discovery_statistics_read_by_runner != FALSE) {
  die("Discovery-statistics boundary drift")
}
if (boundary$human_ratings_read != FALSE) die("Human-rating boundary drift")

input_paths <- lapply(
  names(validation_design$inputs),
  function(label) validate_record(validation_design$inputs[[label]], label)
)
names(input_paths) <- names(validation_design$inputs)
protocol_path <- validate_record(
  validation_design$outputs$protocol_frozen,
  "frozen validation protocol"
)
runner_path <- validate_record(
  validation_design$outputs$runner_frozen,
  "frozen validation runner"
)
if (sha256_file(script_path) != sha256_file(runner_path)) {
  die("Only the frozen validation runner may be executed")
}

design_manifest <- fromJSON(input_paths$design_manifest, simplifyVector = FALSE)
hallmark_manifest <- fromJSON(input_paths$hallmark_manifest, simplifyVector = FALSE)
numerical_manifest <- fromJSON(input_paths$numerical_manifest, simplifyVector = FALSE)
protocol <- fromJSON(protocol_path, simplifyVector = FALSE)
if (design_manifest$schema_version != "CRM_R1_P2B_DESIGN_v14_1") {
  die("Design schema drift")
}
if (hallmark_manifest$schema_version != "CRM_R1_P2B_HALLMARK_v14_1_2") {
  die("Hallmark schema drift")
}
if (
  numerical_manifest$schema_version !=
    "CRM_R1_P2B_FGSEA_NUMERICAL_LOCK_v14_1_3"
) {
  die("Numerical-lock schema drift")
}
if (protocol$schema_version != "CRM_R1_P2B_VALIDATION_PROTOCOL_v14_1_3") {
  die("Validation-protocol schema drift")
}
if (protocol$status != "READY_TO_FREEZE_VALIDATION") {
  die("Validation protocol is not approved")
}
if (protocol$analysis_boundary$partition_used != "validation") {
  die("Protocol partition drift")
}
if (protocol$analysis_boundary$method_membership_rows_read_by_runner != FALSE) {
  die("Protocol permits membership leakage")
}
if (protocol$analysis_boundary$discovery_statistics_read_by_runner != FALSE) {
  die("Protocol permits discovery-statistics leakage")
}
if (protocol$analysis_boundary$human_ratings_read != FALSE) {
  die("Protocol permits human-rating leakage")
}
if (as.integer(protocol$enrichment$nPermSimple) != 10000L) {
  die("Validation requires nPermSimple=10000")
}
if (as.integer(numerical_manifest$nPermSimple) != 10000L) {
  die("Numerical-lock nPermSimple drift")
}

observed_versions <- list(
  r = as.character(getRversion()),
  fgsea = as.character(packageVersion("fgsea")),
  limma = as.character(packageVersion("limma"))
)
for (package_name in names(observed_versions)) {
  expected <- protocol$reference_environment[[package_name]]
  observed <- observed_versions[[package_name]]
  if (observed != expected) {
    die(paste(
      "Reference-environment drift for", package_name,
      ": expected", expected, "observed", observed
    ))
  }
}

assignments_path <- validate_record(
  design_manifest$outputs$assignments_private,
  "frozen split assignments"
)
gene_sets_path <- validate_record(
  hallmark_manifest$outputs$gene_sets,
  "frozen Hallmark gene sets"
)
expression_path <- input_paths$expression_matrix

prohibited <- c(
  file.path(priority2b_root, "validation_lock_v14_1_3"),
  file.path(priority2b_root, "final_v14_1_3")
)
present <- prohibited[file.exists(prohibited)]
if (length(present) > 0L) {
  die(paste("Downstream V14.1.3 output already exists:", paste(present, collapse = ", ")))
}

dir.create(work_root, recursive = TRUE, showWarnings = FALSE)
signature_path <- file.path(work_root, "validation_run_signature.json")
signature <- list(
  schema_version = "CRM_R1_P2B_VALIDATION_RUN_SIGNATURE_v14_1_3",
  script_sha256 = sha256_file(script_path),
  validation_design_manifest_sha256 = sha256_file(validation_design_path),
  design_manifest_sha256 = sha256_file(input_paths$design_manifest),
  assignments_sha256 = sha256_file(assignments_path),
  hallmark_manifest_sha256 = sha256_file(input_paths$hallmark_manifest),
  gene_sets_sha256 = sha256_file(gene_sets_path),
  numerical_manifest_sha256 = sha256_file(input_paths$numerical_manifest),
  membership_manifest_sha256 = sha256_file(input_paths$membership_manifest),
  expression_sha256 = sha256_file(expression_path),
  nPermSimple = 10000L,
  validation_seed_base = as.integer(protocol$enrichment$validation_seed_base),
  partition = "validation",
  method_membership_rows_read = FALSE,
  discovery_statistics_read = FALSE,
  human_ratings_read = FALSE,
  expression_feature_policy = paste(
    "trim_labels; exclude_blank; exclude_all_numeric_unresolved;",
    "exclude_all_rows_for_every_duplicated_symbol"
  )
)
if (file.exists(signature_path)) {
  existing <- fromJSON(signature_path, simplifyVector = FALSE)
  if (!identical(existing, signature)) {
    die("Validation work directory has a different input/code signature")
  }
} else {
  write_json(signature, signature_path, pretty = TRUE, auto_unbox = TRUE)
}

assignments <- fread(assignments_path, colClasses = "character")
required_assignment <- c(
  "cohort_id", "split_id", "sample_id", "group_label", "partition"
)
if (!all(required_assignment %in% names(assignments))) {
  die("Assignment schema drift")
}
assignments <- assignments[partition == "validation"]
if (nrow(assignments) == 0L) die("No validation assignments")

gene_sets <- fread(gene_sets_path, colClasses = "character")
if (!all(c("term_id", "gene_id") %in% names(gene_sets))) {
  die("Hallmark schema drift")
}
if (any(!grepl("[[:alpha:]]", gene_sets$gene_id))) {
  die("Hallmark lock contains a non-symbol gene ID")
}
pathways <- split(gene_sets$gene_id, gene_sets$term_id)
pathways <- lapply(pathways, unique)
pathways <- pathways[sort(names(pathways))]
if (length(pathways) != 50L) die("Expected exactly 50 Hallmark pathways")

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
  die("Fewer than 1000 unique gene symbols remain after pre-model QC")
}
if (anyDuplicated(retained_genes)) die("Retained gene symbols are not unique")

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
  failed <- pathway_overlap[
    expression_overlap_gene_symbols < min_size,
    term_id
  ]
  die(paste(
    "Hallmark pathways below minSize after symbol matching:",
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
  duplicated_ids <- unique(normalized_columns[
    duplicated(normalized_columns) |
      duplicated(normalized_columns, fromLast = TRUE)
  ])
  frozen_ids <- unique(assignments$sample_id)
  if (length(intersect(duplicated_ids, frozen_ids)) > 0L) {
    die("A frozen validation sample maps to non-unique expression columns")
  }
}
column_by_sample <- setNames(colnames(matrix_values), normalized_columns)

cohorts <- sort(unique(assignments$cohort_id))
splits <- sort(unique(assignments$split_id))
expected_pairs <- unique(assignments[, .(cohort_id, split_id)])
if (length(cohorts) != 22L) die("Expected 22 validation cohorts")
if (nrow(expected_pairs) != 440L) {
  die(paste("Expected 440 validation pairs; observed", nrow(expected_pairs)))
}

source_id <- "fgsea_msigdb_H_gene_symbol_v14_1_3_validation_nPermSimple10000"
base_seed <- as.integer(protocol$enrichment$validation_seed_base)
n_perm_simple <- as.integer(protocol$enrichment$nPermSimple)
term_set <- sort(names(pathways))
required_job_columns <- c(
  "cohort_id", "split_id", "claim_uid", "validation_direction",
  "validation_stat", "validation_p_value", "validation_q_value",
  "pathway_size", "n_validation_mutant", "n_validation_wild_type",
  "analysis_seed", "nPermSimple", "source"
)
job_files <- character()
computed <- 0L
skipped <- 0L

valid_job <- function(frame, cohort_value, split_value) {
  if (is.null(frame) || !all(required_job_columns %in% names(frame))) return(FALSE)
  if (nrow(frame) != 50L) return(FALSE)
  if (!identical(sort(as.character(frame$claim_uid)), term_set)) return(FALSE)
  if (!all(frame$cohort_id == cohort_value)) return(FALSE)
  if (!all(frame$split_id == split_value)) return(FALSE)
  if (!all(frame$source == source_id)) return(FALSE)
  if (!all(frame$nPermSimple == 10000L)) return(FALSE)
  if (!all(frame$validation_direction %in% c("UP", "DOWN"))) return(FALSE)
  numeric_columns <- c(
    "validation_stat", "validation_p_value", "validation_q_value"
  )
  if (!all(vapply(frame[, ..numeric_columns], function(x) all(is.finite(x)), logical(1)))) {
    return(FALSE)
  }
  if (!all(frame$validation_p_value >= 0 & frame$validation_p_value <= 1)) {
    return(FALSE)
  }
  if (!all(frame$validation_q_value >= 0 & frame$validation_q_value <= 1)) {
    return(FALSE)
  }
  TRUE
}

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
    output_path <- file.path(job_dir, "validation_statistics.tsv")
    job_files <- c(job_files, output_path)

    if (file.exists(output_path)) {
      previous <- tryCatch(fread(output_path), error = function(error) NULL)
      if (valid_job(previous, cohort_value, split_value)) {
        skipped <- skipped + 1L
        cat(sprintf("[SKIP] %s/%s (%d/440)\n", cohort_value, split_value, computed + skipped))
        next
      }
      die(paste("Invalid existing validation job; inspect before rerun:", output_path))
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
    group_counts <- table(group)
    if (any(group_counts < 2L)) {
      die(paste("Too few validation samples for", cohort_value, split_value))
    }

    finite_rows <- rowSums(is.finite(values)) == ncol(values)
    row_sum <- rowSums(values, na.rm = TRUE)
    row_sq_sum <- rowSums(values * values, na.rm = TRUE)
    variance_numerator <- row_sq_sum - (row_sum * row_sum / ncol(values))
    keep <- finite_rows & is.finite(variance_numerator) & variance_numerator > 0
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
      die(paste("Validation candidate-pool drift for", cohort_value, split_value))
    }
    if (
      any(!is.finite(result$NES)) || any(!is.finite(result$pval)) ||
        any(!is.finite(result$padj))
    ) {
      die(paste("Non-finite validation fgsea result for", cohort_value, split_value))
    }

    outcomes <- data.table(
      cohort_id = cohort_value,
      split_id = split_value,
      claim_uid = result$pathway,
      validation_direction = ifelse(result$NES > 0, "UP", "DOWN"),
      validation_stat = result$NES,
      validation_p_value = result$pval,
      validation_q_value = result$padj,
      pathway_size = result$size,
      n_validation_mutant = as.integer(group_counts[["TP53_MUTANT"]]),
      n_validation_wild_type = as.integer(group_counts[["TP53_WILD_TYPE"]]),
      analysis_seed = analysis_seed,
      nPermSimple = n_perm_simple,
      source = source_id
    )
    setorder(outcomes, claim_uid)
    fwrite(outcomes, output_path, sep = "\t", quote = FALSE)
    computed <- computed + 1L
    cat(sprintf(
      "[DONE] %s/%s (%d/440)\n",
      cohort_value, split_value, computed + skipped
    ))
  }
}

if (length(job_files) != 440L || any(!file.exists(job_files))) {
  die("Not all 440 validation jobs completed")
}
job_tables <- lapply(job_files, fread)
if (!all(vapply(seq_along(job_tables), function(index) {
  path_parts <- strsplit(job_files[[index]], .Platform$file.sep, fixed = TRUE)[[1L]]
  split_value <- path_parts[[length(path_parts) - 1L]]
  cohort_value <- path_parts[[length(path_parts) - 2L]]
  valid_job(job_tables[[index]], cohort_value, split_value)
}, logical(1)))) {
  die("At least one completed validation job failed final validation")
}
combined <- rbindlist(job_tables, use.names = TRUE)
setorder(combined, cohort_id, split_id, claim_uid)
if (nrow(combined) != 22000L) {
  die(paste("Expected 22000 validation rows; observed", nrow(combined)))
}
combined_path <- file.path(work_root, "heldout_validation_statistics.tsv")
fwrite(combined, combined_path, sep = "\t", quote = FALSE)

session_path <- file.path(work_root, "R_sessionInfo.txt")
writeLines(capture.output(sessionInfo()), session_path)
summary <- data.table(
  cohort_split_pairs = 440L,
  terms_per_pair = 50L,
  rows = nrow(combined),
  computed_this_run = computed,
  resumed_jobs = skipped,
  method_membership_rows_read = FALSE,
  discovery_statistics_read = FALSE,
  human_ratings_read = FALSE
)
fwrite(
  summary,
  file.path(work_root, "validation_statistics_summary.tsv"),
  sep = "\t",
  quote = FALSE
)

cat("[PASS] Completed all V14.1.3 method-blind held-out fgsea jobs\n")
cat("[INFO] cohort/split pairs=440; rows=22000; nPermSimple=10000\n")
cat("[INFO] No discovery statistics or method-membership rows were read\n")
cat("[INFO] combined:", combined_path, "\n")
cat("[NEXT] Freeze once with script 74e, then verify with 75c\n")
