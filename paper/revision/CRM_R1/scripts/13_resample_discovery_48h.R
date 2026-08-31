#!/usr/bin/env Rscript
# Priority 1 V4: exhaustive balanced delete-one-per-cell resampling at 48 h.

suppressPackageStartupMessages({
  library(data.table)
  library(edgeR)
  library(fgsea)
  library(jsonlite)
  library(limma)
})

die <- function(message) stop(message, call. = FALSE)
require_true <- function(condition, message) if (!isTRUE(condition)) die(message)

script_path <- NULL
for (arg in commandArgs(trailingOnly = FALSE)) {
  if (startsWith(arg, "--file=")) {
    script_path <- sub("^--file=", "", arg)
    break
  }
}
if (is.null(script_path) || !nzchar(script_path)) die("Cannot determine script path")
SCRIPT_PATH <- normalizePath(script_path, mustWork = TRUE)
CRM_DIR <- normalizePath(file.path(dirname(SCRIPT_PATH), ".."), mustWork = TRUE)
DEFAULT_CONFIG <- file.path(CRM_DIR, "config", "priority1_protocol.json")

parse_args <- function(values) {
  parsed <- list(config = DEFAULT_CONFIG, force = FALSE)
  index <- 1L
  while (index <= length(values)) {
    key <- values[[index]]
    if (identical(key, "--force")) {
      parsed$force <- TRUE
      index <- index + 1L
      next
    }
    require_true(startsWith(key, "--"), paste("Unexpected argument:", key))
    require_true(index < length(values), paste("Missing value for", key))
    name <- gsub("-", "_", substring(key, 3L), fixed = TRUE)
    parsed[[name]] <- values[[index + 1L]]
    index <- index + 2L
  }
  require_true(!is.null(parsed$data_root), "Required argument: --data-root")
  parsed
}

sha256_file <- function(path) {
  path <- normalizePath(path, mustWork = TRUE)
  sha256sum <- Sys.which("sha256sum")
  shasum <- Sys.which("shasum")
  if (nzchar(sha256sum)) {
    output <- system2(sha256sum, shQuote(path), stdout = TRUE)
  } else if (nzchar(shasum)) {
    output <- system2(shasum, c("-a", "256", shQuote(path)), stdout = TRUE)
  } else {
    die("Neither sha256sum nor shasum is available")
  }
  strsplit(output[[1L]], "[[:space:]]+")[[1L]][[1L]]
}

write_json_file <- function(value, path) {
  write_json(value, path, auto_unbox = TRUE, pretty = TRUE, null = "null")
  cat("\n", file = path, append = TRUE)
}

break_exact_ties <- function(statistics) {
  keys <- sprintf("%.17g", statistics)
  groups <- split(seq_along(statistics), keys)
  tied_groups <- groups[lengths(groups) > 1L]
  for (indices in tied_groups) {
    base_value <- statistics[[indices[[1L]]]]
    step <- max(1, abs(base_value)) * .Machine$double.eps * 8
    statistics[indices] <- statistics[indices] + rev(seq_along(indices)) * step
  }
  statistics
}

normalize_libraries <- function(dge) {
  if ("normLibSizes" %in% getNamespaceExports("edgeR")) {
    edgeR::normLibSizes(dge, method = "TMM")
  } else {
    edgeR::calcNormFactors(dge, method = "TMM")
  }
}

args <- parse_args(commandArgs(trailingOnly = TRUE))
data_root <- normalizePath(path.expand(args$data_root), mustWork = TRUE)
config_path <- normalizePath(path.expand(args$config), mustWork = TRUE)
config <- fromJSON(config_path, simplifyVector = FALSE)
primary <- config$primary_analysis
empirical <- config$empirical_stability
benchmark_id <- config$benchmark_id

require_true(config$status %in% c("DRAFT_NOT_FROZEN", "FROZEN"), paste(
  "Empirical resampling expects a draft or frozen Priority 1 protocol; observed:",
  config$status
))
if (identical(config$status, "FROZEN")) {
  require_true(
    isTRUE(all.equal(as.numeric(empirical$primary_tau), 0.8)),
    "Frozen Priority 1 protocol must retain primary_tau = 0.80"
  )
}
require_true(identical(as.integer(primary$discovery_time_h), 48L), "Discovery time must be 48 h")
require_true(identical(as.integer(empirical$expected_resamples), 81L), "Expected 81 resamples")
require_true(
  identical(empirical$method_id, "balanced_delete_one_per_factorial_cell"),
  "Unexpected empirical-resampling method"
)

benchmark_dir <- file.path(data_root, "output", "priority1", benchmark_id)
counts_path <- if (!is.null(args$counts)) {
  normalizePath(path.expand(args$counts), mustWork = TRUE)
} else {
  normalizePath(file.path(data_root, "input", config$dataset$expression_file), mustWork = TRUE)
}
metadata_path <- if (!is.null(args$metadata)) {
  normalizePath(path.expand(args$metadata), mustWork = TRUE)
} else {
  normalizePath(
    file.path(benchmark_dir, "preflight", "sample_metadata.normalized.tsv"),
    mustWork = TRUE
  )
}
universe_path <- if (!is.null(args$universe)) {
  normalizePath(path.expand(args$universe), mustWork = TRUE)
} else {
  normalizePath(
    file.path(benchmark_dir, "derived", "rankings", "discovery_48h_gene_universe.tsv"),
    mustWork = TRUE
  )
}
gene_sets_path <- if (!is.null(args$gene_sets)) {
  normalizePath(path.expand(args$gene_sets), mustWork = TRUE)
} else {
  normalizePath(
    file.path(benchmark_dir, "derived", "fgsea", "hallmark_gene_sets.tsv"),
    mustWork = TRUE
  )
}
baseline_fgsea_path <- if (!is.null(args$baseline_fgsea)) {
  normalizePath(path.expand(args$baseline_fgsea), mustWork = TRUE)
} else {
  normalizePath(
    file.path(benchmark_dir, "derived", "fgsea", "discovery_48h.tsv"),
    mustWork = TRUE
  )
}
out_dir <- if (!is.null(args$out_dir)) {
  path.expand(args$out_dir)
} else {
  file.path(benchmark_dir, "derived", "empirical_resampling_48h")
}
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
out_dir <- normalizePath(out_dir, mustWork = TRUE)

manifest_path <- file.path(out_dir, "resample_manifest.tsv")
fgsea_path <- file.path(out_dir, "fgsea_resamples.tsv")
qc_path <- file.path(out_dir, "resample_qc.tsv")
meta_path <- file.path(out_dir, "empirical_resampling_48h.run_meta.json")
session_path <- file.path(out_dir, "empirical_resampling_48h.session_info.txt")
outputs <- c(manifest_path, fgsea_path, qc_path, meta_path, session_path)
existing <- outputs[file.exists(outputs)]
require_true(isTRUE(args$force) || length(existing) == 0L, paste(
  "Outputs already exist; use --force:",
  paste(existing, collapse = ", ")
))

metadata <- fread(metadata_path)
required_metadata <- c("geo_accession", "genotype", "cell_state", "treatment", "time_h")
require_true(all(required_metadata %in% names(metadata)), "Normalized metadata columns are missing")
discovery <- metadata[
  cell_state == primary$cell_state & as.integer(time_h) == as.integer(primary$discovery_time_h)
]
require_true(nrow(discovery) == 12L, "Expected 12 ENDO discovery samples at 48 h")
discovery[, group := paste(genotype, treatment, sep = "_")]
group_levels <- unlist(empirical$factorial_cells, use.names = FALSE)
require_true(length(group_levels) == 4L, "Empirical protocol must define four factorial cells")
require_true(!anyDuplicated(group_levels), "Factorial-cell names must be unique")

samples_by_group <- lapply(group_levels, function(group_name) {
  sort(as.character(discovery[group == group_name, geo_accession]))
})
names(samples_by_group) <- group_levels
require_true(
  all(lengths(samples_by_group) == as.integer(empirical$original_replicates_per_cell)),
  "Expected three discovery replicates per factorial cell"
)

sample_ids <- as.character(discovery$geo_accession)
header_connection <- gzfile(counts_path, open = "rt")
header_line <- readLines(header_connection, n = 1L)
close(header_connection)
header <- strsplit(header_line, "\t", fixed = TRUE)[[1L]]
require_true(identical(header[[1L]], "GeneID"), "Count matrix first column must be GeneID")
require_true(all(sample_ids %in% header), "Discovery sample IDs are missing from count matrix")
require_true(nzchar(Sys.which("gzip")), "gzip command is required to stream the count matrix")

counts <- fread(
  cmd = paste("gzip -dc", shQuote(counts_path)),
  select = c("GeneID", sample_ids),
  check.names = FALSE,
  showProgress = FALSE
)
counts[, GeneID := as.character(GeneID)]
require_true(!anyDuplicated(counts$GeneID), "Count matrix GeneID values must be unique")

universe <- fread(universe_path)
require_true("gene" %in% names(universe), "Frozen universe must contain a gene column")
universe_genes <- as.character(universe$gene)
require_true(length(universe_genes) >= 1000L, "Frozen universe contains fewer than 1,000 genes")
require_true(!anyDuplicated(universe_genes), "Frozen universe genes must be unique")
gene_index <- match(universe_genes, counts$GeneID)
require_true(!anyNA(gene_index), "Frozen-universe genes are missing from the count matrix")

count_matrix <- as.matrix(counts[gene_index, ..sample_ids])
storage.mode(count_matrix) <- "numeric"
rownames(count_matrix) <- universe_genes
require_true(all(is.finite(count_matrix)), "Count matrix contains non-finite values")
require_true(all(count_matrix >= 0 & count_matrix == floor(count_matrix)), "Counts must be integers")

gene_sets <- fread(gene_sets_path)
require_true(all(c("pathway", "gene") %in% names(gene_sets)), "Invalid Hallmark snapshot")
gene_sets[, `:=`(pathway = as.character(pathway), gene = as.character(gene))]
gene_sets <- unique(gene_sets[gene %in% universe_genes, .(pathway, gene)])
pathways <- split(gene_sets$gene, gene_sets$pathway)
pathways <- lapply(pathways, unique)

baseline_fgsea <- fread(baseline_fgsea_path)
require_true("pathway" %in% names(baseline_fgsea), "Baseline fgsea is missing pathway")
baseline_pathways <- sort(unique(as.character(baseline_fgsea$pathway)))
require_true(
  length(baseline_pathways) == as.integer(primary$candidate_pool_size),
  "Baseline pathway count differs from candidate_pool_size"
)
require_true(all(baseline_pathways %in% names(pathways)), "Baseline pathways missing from snapshot")
pathways <- pathways[baseline_pathways]

grid_args <- c(samples_by_group, list(KEEP.OUT.ATTRS = FALSE, stringsAsFactors = FALSE))
manifest <- as.data.table(do.call(expand.grid, grid_args))
setnames(manifest, paste0("omit_", group_levels))
manifest[, resample_index := seq_len(.N)]
manifest[, replicate_id := sprintf("balanced_delete1_%03d", resample_index)]
manifest[, retained_samples := vapply(seq_len(.N), function(index) {
  omitted <- as.character(unlist(manifest[index, paste0("omit_", group_levels), with = FALSE]))
  paste(sample_ids[!sample_ids %in% omitted], collapse = ",")
}, character(1L))]
setcolorder(
  manifest,
  c("resample_index", "replicate_id", paste0("omit_", group_levels), "retained_samples")
)
require_true(nrow(manifest) == as.integer(empirical$expected_resamples), "Resample count is not 81")
require_true(!anyDuplicated(manifest$replicate_id), "Resample IDs must be unique")

contrast_string <- "(WT_MMS - WT_UT) - (TP53_KO_MMS - TP53_KO_UT)"
require_true(identical(primary$contrast, contrast_string), "Unexpected interaction contrast")
min_size <- as.integer(primary$fgsea_min_size)
max_size <- as.integer(primary$fgsea_max_size)
seed_base <- as.integer(primary$fgsea_seed)

fgsea_results <- vector("list", nrow(manifest))
qc_results <- vector("list", nrow(manifest))

for (index in seq_len(nrow(manifest))) {
  omitted <- as.character(unlist(
    manifest[index, paste0("omit_", group_levels), with = FALSE],
    use.names = FALSE
  ))
  retained <- sample_ids[!sample_ids %in% omitted]
  retained_metadata <- discovery[match(retained, geo_accession)]
  retained_group <- factor(retained_metadata$group, levels = group_levels)
  retained_counts <- table(retained_group)
  require_true(
    all(retained_counts == as.integer(empirical$retained_replicates_per_cell)),
    paste("Unbalanced retained design in resample", index)
  )

  design <- model.matrix(~0 + retained_group)
  colnames(design) <- group_levels
  rownames(design) <- retained
  require_true(qr(design)$rank == length(group_levels), "Resample design matrix is not full rank")
  contrast <- makeContrasts(
    (WT_MMS - WT_UT) - (TP53_KO_MMS - TP53_KO_UT),
    levels = design
  )
  colnames(contrast) <- "WT_MMS_response_minus_TP53_KO_MMS_response"

  dge <- DGEList(counts = count_matrix[, retained, drop = FALSE])
  dge <- normalize_libraries(dge)
  voom_fit <- voom(dge, design, plot = FALSE)
  fit <- lmFit(voom_fit, design)
  fit <- contrasts.fit(fit, contrast)
  fit <- eBayes(fit)
  results <- topTable(fit, number = Inf, sort.by = "none", adjust.method = "BH")
  require_true("t" %in% names(results), "limma output is missing moderated t-statistics")

  statistics <- as.numeric(results[universe_genes, "t"])
  names(statistics) <- universe_genes
  require_true(all(is.finite(statistics)), paste("Non-finite ranking in resample", index))
  exact_tie_values <- sum(duplicated(statistics))
  statistics <- break_exact_ties(statistics)
  statistics <- sort(statistics, decreasing = TRUE)

  resample_seed <- seed_base + index
  RNGkind("Mersenne-Twister", "Inversion", "Rejection")
  set.seed(resample_seed)
  enrichment <- as.data.table(fgseaMultilevel(
    pathways = pathways,
    stats = statistics,
    minSize = min_size,
    maxSize = max_size,
    nproc = 1
  ))
  require_true("leadingEdge" %in% names(enrichment), "fgsea output is missing leadingEdge")
  require_true(
    identical(sort(as.character(enrichment$pathway)), baseline_pathways),
    paste("Pathway membership drift in resample", index)
  )
  enrichment[, leadingEdge := vapply(
    leadingEdge,
    function(genes) paste(as.character(genes), collapse = ","),
    character(1L)
  )]
  enrichment[, `:=`(
    replicate_id = manifest$replicate_id[[index]],
    resample_index = index,
    ranking_statistic = "limma_moderated_t_interaction",
    seed = resample_seed
  )]
  enrichment[, abs_NES_order := abs(NES)]
  setorderv(enrichment, c("padj", "abs_NES_order", "pathway"), c(1L, -1L, 1L))
  enrichment[, abs_NES_order := NULL]
  fgsea_results[[index]] <- enrichment

  qc_results[[index]] <- data.table(
    resample_index = index,
    replicate_id = manifest$replicate_id[[index]],
    retained_samples = length(retained),
    retained_per_factorial_cell = min(as.integer(retained_counts)),
    ranked_genes = length(statistics),
    exact_tied_values_before_breaking = exact_tie_values,
    tested_pathways = nrow(enrichment),
    fgsea_seed = resample_seed,
    min_TMM_norm_factor = min(dge$samples$norm.factors),
    max_TMM_norm_factor = max(dge$samples$norm.factors)
  )
}

fgsea_long <- rbindlist(fgsea_results, use.names = TRUE, fill = TRUE)
qc <- rbindlist(qc_results, use.names = TRUE, fill = TRUE)
setorderv(fgsea_long, c("resample_index", "padj", "pathway"), c(1L, 1L, 1L))
setorder(qc, resample_index)
require_true(
  nrow(fgsea_long) == nrow(manifest) * length(baseline_pathways),
  "Unexpected long fgsea row count"
)

fwrite(manifest, manifest_path, sep = "\t", na = "NA")
fwrite(fgsea_long, fgsea_path, sep = "\t", na = "NA")
fwrite(qc, qc_path, sep = "\t", na = "NA")
capture.output(sessionInfo(), file = session_path)

package_names <- c("data.table", "edgeR", "fgsea", "jsonlite", "limma")
package_versions <- as.list(vapply(
  package_names,
  function(package) as.character(packageVersion(package)),
  character(1L)
))
metadata_output <- list(
  benchmark_id = benchmark_id,
  protocol_version = config$protocol_version,
  protocol_status = config$status,
  analysis_scope = "ENDO 48 h discovery-only balanced empirical resampling",
  held_out_expression_columns_loaded = FALSE,
  held_out_expression_outcomes_calculated = FALSE,
  method = empirical$method_id,
  design = list(
    factorial_cells = group_levels,
    original_replicates_per_cell = empirical$original_replicates_per_cell,
    deleted_replicates_per_cell = empirical$deleted_replicates_per_cell,
    retained_replicates_per_cell = empirical$retained_replicates_per_cell,
    resamples = nrow(manifest)
  ),
  analysis = list(
    frozen_gene_universe = TRUE,
    normalization = empirical$normalization,
    model = empirical$model,
    contrast = primary$contrast,
    ranking_statistic = primary$fgsea_ranking_statistic,
    collection = primary$gene_set_collection,
    fgsea_algorithm = primary$fgsea_algorithm,
    fgsea_min_size = min_size,
    fgsea_max_size = max_size,
    fgsea_seed_rule = empirical$fgsea_seed_rule
  ),
  dimensions = list(
    discovery_samples_loaded = length(sample_ids),
    frozen_genes = length(universe_genes),
    resamples = nrow(manifest),
    pathways_per_resample = length(baseline_pathways),
    fgsea_rows = nrow(fgsea_long)
  ),
  inputs = list(
    counts = list(path = counts_path, sha256 = sha256_file(counts_path)),
    normalized_metadata = list(path = metadata_path, sha256 = sha256_file(metadata_path)),
    frozen_gene_universe = list(path = universe_path, sha256 = sha256_file(universe_path)),
    hallmark_gene_sets = list(path = gene_sets_path, sha256 = sha256_file(gene_sets_path)),
    baseline_fgsea = list(path = baseline_fgsea_path, sha256 = sha256_file(baseline_fgsea_path)),
    config = list(path = config_path, sha256 = sha256_file(config_path))
  ),
  outputs = list(
    resample_manifest = list(path = manifest_path, sha256 = sha256_file(manifest_path)),
    fgsea_resamples = list(path = fgsea_path, sha256 = sha256_file(fgsea_path)),
    resample_qc = list(path = qc_path, sha256 = sha256_file(qc_path)),
    session_info = list(path = session_path, sha256 = sha256_file(session_path))
  ),
  software = list(R = R.version.string, packages = package_versions),
  script = list(path = SCRIPT_PATH, sha256 = sha256_file(SCRIPT_PATH))
)
write_json_file(metadata_output, meta_path)

cat("[PASS] Priority 1 V4 empirical 48 h resampling\n")
cat("[INFO] Balanced resamples:", nrow(manifest), "\n")
cat("[INFO] Samples retained per resample: 8 (2 per factorial cell)\n")
cat("[INFO] Pathways per resample:", length(baseline_pathways), "\n")
cat("[INFO] Held-out expression columns loaded: false\n")
cat("[INFO] Wrote:", fgsea_path, "\n")
