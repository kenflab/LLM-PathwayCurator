#!/usr/bin/env Rscript
# Priority 1: one-time ENDO 72 h held-out interaction and frozen Hallmark fgsea.

suppressPackageStartupMessages({
  library(data.table)
  library(edgeR)
  library(fgsea)
  library(jsonlite)
  library(limma)
  library(org.Hs.eg.db)
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
REPO_ROOT <- normalizePath(file.path(CRM_DIR, "..", "..", ".."), mustWork = TRUE)
DEFAULT_CONFIG <- file.path(CRM_DIR, "config", "priority1_protocol.json")
DEFAULT_FREEZE_CHECK <- file.path(dirname(SCRIPT_PATH), "17_check_priority1_freeze.py")

parse_args <- function(values) {
  parsed <- list(config = DEFAULT_CONFIG, freeze_check = DEFAULT_FREEZE_CHECK)
  index <- 1L
  while (index <= length(values)) {
    key <- values[[index]]
    require_true(!identical(key, "--force"), "72 h validation has no --force option")
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

run_freeze_gate <- function(python, checker, data_root, config_path) {
  gate_args <- c(
    shQuote(checker),
    "--data-root", shQuote(data_root),
    "--config", shQuote(config_path)
  )
  output <- system2(python, gate_args, stdout = TRUE, stderr = TRUE)
  status <- attr(output, "status")
  if (is.null(status)) status <- 0L
  cat(paste(output, collapse = "\n"), "\n")
  require_true(status == 0L, "Priority 1 freeze checker failed")
  require_true(any(grepl("^\\[GO\\] Held-out 72 h analysis", output)), paste(
    "Freeze checker did not release the held-out endpoint; observed:",
    paste(output, collapse = " | ")
  ))
  output
}

git_value <- function(arguments) {
  output <- system2("git", c("-C", shQuote(REPO_ROOT), arguments), stdout = TRUE, stderr = TRUE)
  status <- attr(output, "status")
  if (is.null(status)) status <- 0L
  require_true(status == 0L, paste("Git command failed:", paste(arguments, collapse = " ")))
  trimws(paste(output, collapse = "\n"))
}

args <- parse_args(commandArgs(trailingOnly = TRUE))
data_root <- normalizePath(path.expand(args$data_root), mustWork = TRUE)
config_path <- normalizePath(path.expand(args$config), mustWork = TRUE)
freeze_check_path <- normalizePath(path.expand(args$freeze_check), mustWork = TRUE)
config <- fromJSON(config_path, simplifyVector = FALSE)
primary <- config$primary_analysis
benchmark_id <- config$benchmark_id

require_true(identical(config$status, "FROZEN"), "Priority 1 protocol must be FROZEN")
require_true(identical(config$protocol_version, "CRM_R1_PRIORITY1_v5"), "Protocol drift")
require_true(identical(as.integer(primary$validation_time_h), 72L), "Validation time must be 72 h")
require_true(identical(as.integer(primary$discovery_time_h), 48L), "Discovery time must be 48 h")
require_true(
  identical(
    primary$contrast,
    "(WT_MMS - WT_UT) - (TP53_KO_MMS - TP53_KO_UT)"
  ),
  "Unexpected interaction contrast in protocol"
)

tracked_status <- git_value(c("status", "--porcelain", "--untracked-files=no"))
require_true(!nzchar(tracked_status), paste(
  "Commit V6 and leave tracked files clean before releasing 72 h; observed:",
  tracked_status
))
code_commit <- git_value(c("rev-parse", "HEAD"))
code_branch <- git_value(c("branch", "--show-current"))

python_bin <- if (!is.null(args$python)) path.expand(args$python) else Sys.which("python")
require_true(nzchar(python_bin), "python is required to run the freeze checker")

# This must be the first operation that can release held-out biological calculations.
freeze_gate_output <- run_freeze_gate(
  python_bin,
  freeze_check_path,
  data_root,
  config_path
)

benchmark_dir <- file.path(data_root, "output", "priority1", benchmark_id)
validation_dir <- file.path(benchmark_dir, "validation")
dir.create(validation_dir, recursive = TRUE, showWarnings = FALSE)
validation_dir <- normalizePath(validation_dir, mustWork = TRUE)

final_paths <- c(
  ranking = file.path(validation_dir, "ranking_72h.tsv"),
  design = file.path(validation_dir, "design_72h.tsv"),
  pathways = file.path(validation_dir, "pathway_statistics_72h.tsv"),
  session = file.path(validation_dir, "validation_72h.session_info.txt"),
  metadata = file.path(validation_dir, "validation_72h.run_meta.json")
)
existing <- final_paths[file.exists(final_paths)]
require_true(length(existing) == 0L, paste(
  "Held-out outputs are immutable and may be written only once; existing:",
  paste(existing, collapse = ", ")
))

staging_dir <- file.path(validation_dir, ".priority1_validation_staging")
require_true(!file.exists(staging_dir), paste("Remove stale staging directory:", staging_dir))
dir.create(staging_dir, recursive = FALSE, showWarnings = FALSE)
require_true(dir.exists(staging_dir), "Could not create validation staging directory")
on.exit(unlink(staging_dir, recursive = TRUE, force = TRUE), add = TRUE)
staging_paths <- file.path(staging_dir, basename(final_paths))
names(staging_paths) <- names(final_paths)

counts_path <- if (!is.null(args$counts)) {
  normalizePath(path.expand(args$counts), mustWork = TRUE)
} else {
  normalizePath(
    file.path(data_root, "input", config$dataset$expression_file),
    mustWork = TRUE
  )
}
metadata_path <- if (!is.null(args$metadata)) {
  normalizePath(path.expand(args$metadata), mustWork = TRUE)
} else {
  normalizePath(
    file.path(benchmark_dir, "preflight", "sample_metadata.normalized.tsv"),
    mustWork = TRUE
  )
}
universe_path <- normalizePath(
  file.path(benchmark_dir, "derived", "rankings", "discovery_48h_gene_universe.tsv"),
  mustWork = TRUE
)
gene_sets_path <- normalizePath(
  file.path(benchmark_dir, "derived", "fgsea", "hallmark_gene_sets.tsv"),
  mustWork = TRUE
)
baseline_fgsea_path <- normalizePath(
  file.path(benchmark_dir, "derived", "fgsea", "discovery_48h.tsv"),
  mustWork = TRUE
)
freeze_manifest_path <- normalizePath(
  file.path(benchmark_dir, "metrics", "priority1_freeze_manifest.json"),
  mustWork = TRUE
)
freeze_sidecar_path <- normalizePath(
  file.path(benchmark_dir, "metrics", "priority1_freeze_manifest.sha256"),
  mustWork = TRUE
)

metadata <- fread(metadata_path)
required_metadata <- c("geo_accession", "genotype", "cell_state", "treatment", "time_h")
require_true(all(required_metadata %in% names(metadata)), "Normalized metadata columns are missing")
validation <- metadata[
  cell_state == primary$cell_state & as.integer(time_h) == as.integer(primary$validation_time_h)
]
require_true(nrow(validation) == 12L, "Expected 12 ENDO validation samples at 72 h")
validation[, group := paste(genotype, treatment, sep = "_")]
group_levels <- c("WT_UT", "WT_MMS", "TP53_KO_UT", "TP53_KO_MMS")
cell_counts <- table(factor(validation$group, levels = group_levels))
require_true(all(cell_counts == 3L), paste(
  "Expected three validation replicates per factorial cell; observed:",
  paste(names(cell_counts), cell_counts, sep = "=", collapse = ", ")
))

sample_ids <- validation$geo_accession
header_connection <- gzfile(counts_path, open = "rt")
header_line <- readLines(header_connection, n = 1L)
close(header_connection)
header <- strsplit(header_line, "\t", fixed = TRUE)[[1L]]
require_true(identical(header[[1L]], "GeneID"), "Count matrix first column must be GeneID")
require_true(all(sample_ids %in% header), "Validation sample IDs are missing from count matrix")
require_true(nzchar(Sys.which("gzip")), "gzip command is required to stream the count matrix")

# Only GeneID and the twelve frozen-context ENDO 72 h columns are loaded.
counts <- fread(
  cmd = paste("gzip -dc", shQuote(counts_path)),
  select = c("GeneID", sample_ids),
  check.names = FALSE,
  showProgress = FALSE
)
setcolorder(counts, c("GeneID", sample_ids))
genes <- as.character(counts$GeneID)
require_true(!anyNA(genes) && all(nzchar(genes)), "GeneID contains missing values")
require_true(!anyDuplicated(genes), "GeneID values must be unique")
require_true(all(grepl("^[0-9]+$", genes)), "GeneID values must be numeric Entrez IDs")

universe <- fread(universe_path, colClasses = list(character = "gene"))
require_true("gene" %in% names(universe), "Frozen universe requires a gene column")
frozen_genes <- as.character(universe$gene)
require_true(!anyNA(frozen_genes) && all(nzchar(frozen_genes)), "Frozen universe has missing genes")
require_true(!anyDuplicated(frozen_genes), "Frozen universe genes must be unique")
match_index <- match(frozen_genes, genes)
require_true(!anyNA(match_index), "Frozen discovery genes are missing from the count matrix")

count_matrix <- as.matrix(counts[match_index, ..sample_ids])
storage.mode(count_matrix) <- "numeric"
rownames(count_matrix) <- frozen_genes
require_true(all(is.finite(count_matrix)), "Count matrix contains non-finite values")
require_true(all(count_matrix >= 0 & count_matrix == floor(count_matrix)), "Counts must be integers")
require_true(nrow(count_matrix) == nrow(universe), "Frozen universe size drift")

group <- factor(validation$group, levels = group_levels)
design <- model.matrix(~0 + group)
colnames(design) <- group_levels
rownames(design) <- sample_ids
contrast <- makeContrasts(
  (WT_MMS - WT_UT) - (TP53_KO_MMS - TP53_KO_UT),
  levels = design
)
colnames(contrast) <- "WT_MMS_response_minus_TP53_KO_MMS_response"

frozen_universe_raw_library_size <- colSums(count_matrix)
dge <- DGEList(counts = count_matrix)
dge <- calcNormFactors(dge, method = "TMM")
voom_fit <- voom(dge, design, plot = FALSE)
fit <- lmFit(voom_fit, design)
fit <- contrasts.fit(fit, contrast)
fit <- eBayes(fit)
results <- topTable(fit, number = Inf, sort.by = "none", adjust.method = "BH")
require_true("t" %in% names(results), "limma output is missing moderated t-statistics")
require_true(all(is.finite(results$t)), paste(
  "Frozen 72 h universe produced non-finite moderated t-statistics:",
  sum(!is.finite(results$t))
))

entrez_ids <- rownames(results)
symbols <- AnnotationDbi::mapIds(
  org.Hs.eg.db,
  keys = entrez_ids,
  keytype = "ENTREZID",
  column = "SYMBOL",
  multiVals = "first"
)
ranking <- data.table(
  gene = entrez_ids,
  gene_symbol = unname(symbols[entrez_ids]),
  score = results$t,
  logFC = results$logFC,
  AveExpr = results$AveExpr,
  P.Value = results$P.Value,
  adj.P.Val = results$adj.P.Val,
  B = results$B
)
setorder(ranking, -score, gene)
require_true(nrow(ranking) == length(frozen_genes), "72 h ranking/universe size drift")

design_output <- data.table(
  sample = sample_ids,
  genotype = validation$genotype,
  treatment = validation$treatment,
  group = as.character(group),
  time_h = as.integer(primary$validation_time_h),
  frozen_universe_raw_library_size = as.numeric(
    frozen_universe_raw_library_size[sample_ids]
  ),
  frozen_universe_library_size = as.numeric(dge$samples$lib.size),
  TMM_norm_factor = as.numeric(dge$samples$norm.factors),
  effective_library_size = as.numeric(dge$samples$lib.size * dge$samples$norm.factors)
)

gene_sets <- fread(gene_sets_path, colClasses = list(character = c("pathway", "gene")))
require_true(all(c("pathway", "gene") %in% names(gene_sets)), "Frozen Hallmark snapshot is invalid")
require_true(!anyNA(gene_sets$pathway) && !anyNA(gene_sets$gene), "Hallmark snapshot has NA")
pathways <- split(as.character(gene_sets$gene), as.character(gene_sets$pathway))
pathways <- lapply(pathways, unique)

statistics <- ranking$score
names(statistics) <- ranking$gene
raw_tie_count <- sum(duplicated(statistics))
statistics <- break_exact_ties(statistics)
names(statistics) <- ranking$gene
statistics <- sort(statistics, decreasing = TRUE)

seed <- as.integer(primary$fgsea_seed)
min_size <- as.integer(primary$fgsea_min_size)
max_size <- as.integer(primary$fgsea_max_size)
RNGkind("Mersenne-Twister", "Inversion", "Rejection")
set.seed(seed)
pathway_results <- fgseaMultilevel(
  pathways = pathways,
  stats = statistics,
  minSize = min_size,
  maxSize = max_size,
  nproc = 1
)
pathway_results <- as.data.table(pathway_results)
require_true(nrow(pathway_results) > 0L, "72 h fgseaMultilevel returned no pathways")
require_true("leadingEdge" %in% names(pathway_results), "72 h fgsea lacks leadingEdge")

baseline_fgsea <- fread(baseline_fgsea_path)
require_true("pathway" %in% names(baseline_fgsea), "Baseline fgsea lacks pathway")
candidate_pathways <- unique(as.character(baseline_fgsea$pathway))
require_true(length(candidate_pathways) == as.integer(primary$candidate_pool_size), paste(
  "Frozen baseline candidate count drift:",
  length(candidate_pathways)
))
require_true(
  setequal(as.character(pathway_results$pathway), candidate_pathways),
  "72 h fgsea pathway set differs from the frozen 48 h candidate pool"
)

pathway_results[, leadingEdge := vapply(
  leadingEdge,
  function(values) paste(as.character(values), collapse = ","),
  character(1L)
)]
db_versions <- if ("db_version" %in% names(gene_sets)) {
  unique(as.character(gene_sets$db_version))
} else {
  "not_reported_in_frozen_snapshot"
}
db_versions <- db_versions[!is.na(db_versions) & nzchar(db_versions)]
require_true(length(db_versions) == 1L, "Frozen Hallmark snapshot has multiple DB versions")
pathway_results[, `:=`(
  collection = "H",
  db_version = db_versions[[1L]],
  ranking_statistic = "limma_moderated_t_interaction",
  seed = seed,
  cell_state = primary$cell_state,
  time_h = as.integer(primary$validation_time_h),
  contrast = primary$contrast
)]
pathway_results[, abs_NES_order := abs(NES)]
setorderv(pathway_results, c("padj", "abs_NES_order", "pathway"), c(1L, -1L, 1L))
pathway_results[, abs_NES_order := NULL]

fwrite(ranking, staging_paths[["ranking"]], sep = "\t", na = "NA")
fwrite(design_output, staging_paths[["design"]], sep = "\t", na = "NA")
fwrite(pathway_results, staging_paths[["pathways"]], sep = "\t", na = "NA")
capture.output(sessionInfo(), file = staging_paths[["session"]])

package_names <- c("data.table", "edgeR", "fgsea", "jsonlite", "limma", "org.Hs.eg.db")
package_versions <- as.list(vapply(
  package_names,
  function(package) as.character(packageVersion(package)),
  character(1L)
))
metadata_output <- list(
  benchmark_id = benchmark_id,
  protocol_version = config$protocol_version,
  protocol_status = config$status,
  analysis_scope = "one-time held-out ENDO 72 h temporal replication",
  validation_endpoint_calculated = TRUE,
  samples_loaded = list(cell_state = primary$cell_state, time_h = 72L, n = ncol(count_matrix)),
  discovery_expression_columns_loaded = FALSE,
  gene_universe = "frozen full-discovery 48 h filter; no 72 h refiltering",
  normalization = "edgeR TMM recalculated using 72 h samples only",
  model = "voom-limma four-cell design fitted using 72 h samples only",
  contrast = primary$contrast,
  ranking_statistic = "limma moderated t-statistic",
  fgsea = list(
    algorithm = "fgsea::fgseaMultilevel",
    min_size = min_size,
    max_size = max_size,
    seed = seed,
    nproc = 1L,
    exact_tied_values_before_deterministic_breaking = raw_tie_count
  ),
  dimensions = list(
    raw_input_gene_rows = nrow(counts),
    frozen_universe_genes = nrow(count_matrix),
    validation_samples = ncol(count_matrix),
    tested_pathways = nrow(pathway_results)
  ),
  freeze_gate = list(
    checker = list(path = freeze_check_path, sha256 = sha256_file(freeze_check_path)),
    output = as.list(freeze_gate_output),
    manifest = list(path = freeze_manifest_path, sha256 = sha256_file(freeze_manifest_path)),
    sidecar = list(path = freeze_sidecar_path, sha256 = sha256_file(freeze_sidecar_path))
  ),
  code = list(branch = code_branch, commit = code_commit, tracked_worktree_clean = TRUE),
  inputs = list(
    counts = list(path = counts_path, sha256 = sha256_file(counts_path)),
    normalized_metadata = list(path = metadata_path, sha256 = sha256_file(metadata_path)),
    frozen_gene_universe = list(path = universe_path, sha256 = sha256_file(universe_path)),
    frozen_hallmark_snapshot = list(path = gene_sets_path, sha256 = sha256_file(gene_sets_path)),
    frozen_baseline_fgsea = list(
      path = baseline_fgsea_path,
      sha256 = sha256_file(baseline_fgsea_path)
    ),
    config = list(path = config_path, sha256 = sha256_file(config_path))
  ),
  outputs = list(
    ranking_72h = list(
      path = final_paths[["ranking"]],
      sha256 = sha256_file(staging_paths[["ranking"]])
    ),
    design_72h = list(
      path = final_paths[["design"]],
      sha256 = sha256_file(staging_paths[["design"]])
    ),
    pathway_statistics_72h = list(
      path = final_paths[["pathways"]],
      sha256 = sha256_file(staging_paths[["pathways"]])
    ),
    session_info = list(
      path = final_paths[["session"]],
      sha256 = sha256_file(staging_paths[["session"]])
    )
  ),
  software = list(R = R.version.string, packages = package_versions),
  script = list(path = SCRIPT_PATH, sha256 = sha256_file(SCRIPT_PATH))
)
write_json_file(metadata_output, staging_paths[["metadata"]])

for (name in names(final_paths)) {
  require_true(file.rename(staging_paths[[name]], final_paths[[name]]), paste(
    "Failed to publish held-out output:",
    final_paths[[name]]
  ))
}

cat("[PASS] Priority 1 one-time held-out ENDO 72 h analysis\n")
cat("[INFO] Freeze gate passed before expression outcomes were calculated\n")
cat("[INFO] Samples used:", ncol(count_matrix), "(ENDO, 72 h only)\n")
cat("[INFO] Frozen genes modeled:", nrow(count_matrix), "\n")
cat("[INFO] Frozen Hallmark pathways tested:", nrow(pathway_results), "\n")
cat("[INFO] Wrote:", final_paths[["pathways"]], "\n")
cat("[NEXT] Run 19_evaluate_replication.py once; do not alter frozen inputs or endpoints.\n")
