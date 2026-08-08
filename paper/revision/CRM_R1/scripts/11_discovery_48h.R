#!/usr/bin/env Rscript
# Priority 1: 48 h ENDO discovery interaction using edgeR, voom, and limma.

suppressPackageStartupMessages({
  library(data.table)
  library(edgeR)
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

args <- parse_args(commandArgs(trailingOnly = TRUE))
data_root <- normalizePath(path.expand(args$data_root), mustWork = TRUE)
config_path <- normalizePath(path.expand(args$config), mustWork = TRUE)
config <- fromJSON(config_path, simplifyVector = FALSE)
primary <- config$primary_analysis
benchmark_id <- config$benchmark_id

require_true(identical(as.integer(primary$discovery_time_h), 48L), "Discovery time must be 48 h")
require_true(
  identical(
    primary$contrast,
    "(WT_MMS - WT_UT) - (TP53_KO_MMS - TP53_KO_UT)"
  ),
  "Unexpected interaction contrast in protocol"
)

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
    file.path(
      data_root,
      "output",
      "priority1",
      benchmark_id,
      "preflight",
      "sample_metadata.normalized.tsv"
    ),
    mustWork = TRUE
  )
}
out_dir <- if (!is.null(args$out_dir)) {
  path.expand(args$out_dir)
} else {
  file.path(data_root, "output", "priority1", benchmark_id, "derived", "rankings")
}
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
out_dir <- normalizePath(out_dir, mustWork = TRUE)

ranking_path <- file.path(out_dir, "discovery_48h.tsv")
universe_path <- file.path(out_dir, "discovery_48h_gene_universe.tsv")
design_path <- file.path(out_dir, "discovery_48h_design.tsv")
meta_path <- file.path(out_dir, "discovery_48h.run_meta.json")
session_path <- file.path(out_dir, "discovery_48h.session_info.txt")
outputs <- c(ranking_path, universe_path, design_path, meta_path, session_path)
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
group_levels <- c("WT_UT", "WT_MMS", "TP53_KO_UT", "TP53_KO_MMS")
cell_counts <- table(factor(discovery$group, levels = group_levels))
require_true(all(cell_counts == 3L), paste(
  "Expected three replicates per factorial cell; observed:",
  paste(names(cell_counts), cell_counts, sep = "=", collapse = ", ")
))

sample_ids <- discovery$geo_accession
header_connection <- gzfile(counts_path, open = "rt")
header_line <- readLines(header_connection, n = 1L)
close(header_connection)
header <- strsplit(header_line, "\t", fixed = TRUE)[[1L]]
require_true(identical(header[[1L]], "GeneID"), "Count matrix first column must be GeneID")
require_true(all(sample_ids %in% header), "Discovery sample IDs are missing from count matrix")
require_true(nzchar(Sys.which("gzip")), "gzip command is required to stream the count matrix")

# Only GeneID and the twelve prespecified 48 h discovery columns are loaded.
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

count_matrix <- as.matrix(counts[, ..sample_ids])
storage.mode(count_matrix) <- "numeric"
rownames(count_matrix) <- genes
require_true(all(is.finite(count_matrix)), "Count matrix contains non-finite values")
require_true(all(count_matrix >= 0 & count_matrix == floor(count_matrix)), "Counts must be integers")

group <- factor(discovery$group, levels = group_levels)
design <- model.matrix(~0 + group)
colnames(design) <- group_levels
rownames(design) <- sample_ids
contrast <- makeContrasts(
  (WT_MMS - WT_UT) - (TP53_KO_MMS - TP53_KO_UT),
  levels = design
)
colnames(contrast) <- "WT_MMS_response_minus_TP53_KO_MMS_response"

raw_library_size <- colSums(count_matrix)
dge <- DGEList(counts = count_matrix)
keep <- filterByExpr(dge, group = group)
require_true(sum(keep) >= 1000L, "Fewer than 1,000 genes passed filterByExpr")
dge <- dge[keep, , keep.lib.sizes = FALSE]
dge <- calcNormFactors(dge, method = "TMM")

voom_fit <- voom(dge, design, plot = FALSE)
fit <- lmFit(voom_fit, design)
fit <- contrasts.fit(fit, contrast)
fit <- eBayes(fit)
results <- topTable(fit, number = Inf, sort.by = "none", adjust.method = "BH")
require_true("t" %in% names(results), "limma output is missing moderated t-statistics")

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
ranking <- ranking[is.finite(score)]
setorder(ranking, -score, gene)
require_true(nrow(ranking) == sum(keep), "Ranking and filtered universe sizes differ")

universe <- ranking[, .(gene, gene_symbol)]
design_output <- data.table(
  sample = sample_ids,
  genotype = discovery$genotype,
  treatment = discovery$treatment,
  group = as.character(group),
  raw_library_size = as.numeric(raw_library_size[sample_ids]),
  filtered_library_size = as.numeric(dge$samples$lib.size),
  TMM_norm_factor = as.numeric(dge$samples$norm.factors),
  effective_library_size = as.numeric(dge$samples$lib.size * dge$samples$norm.factors)
)

fwrite(ranking, ranking_path, sep = "\t", na = "NA")
fwrite(universe, universe_path, sep = "\t", na = "NA")
fwrite(design_output, design_path, sep = "\t", na = "NA")
capture.output(sessionInfo(), file = session_path)

package_names <- c("data.table", "edgeR", "jsonlite", "limma", "org.Hs.eg.db")
package_versions <- as.list(vapply(
  package_names,
  function(package) as.character(packageVersion(package)),
  character(1L)
))
metadata_output <- list(
  benchmark_id = benchmark_id,
  protocol_version = config$protocol_version,
  protocol_status = config$status,
  analysis_scope = "ENDO 48 h discovery only",
  held_out_expression_columns_loaded = FALSE,
  held_out_expression_outcomes_calculated = FALSE,
  contrast = primary$contrast,
  positive_direction = primary$positive_direction,
  filter = "edgeR::filterByExpr(group = four-level genotype-treatment factor)",
  normalization = "edgeR TMM",
  model = "voom-limma four-cell design with prespecified interaction contrast",
  ranking_statistic = "limma moderated t-statistic",
  dimensions = list(
    input_gene_rows = nrow(counts),
    discovery_samples = ncol(count_matrix),
    filtered_genes = sum(keep)
  ),
  inputs = list(
    counts = list(path = counts_path, sha256 = sha256_file(counts_path)),
    normalized_metadata = list(path = metadata_path, sha256 = sha256_file(metadata_path)),
    config = list(path = config_path, sha256 = sha256_file(config_path))
  ),
  outputs = list(
    ranking = list(path = ranking_path, sha256 = sha256_file(ranking_path)),
    gene_universe = list(path = universe_path, sha256 = sha256_file(universe_path)),
    design = list(path = design_path, sha256 = sha256_file(design_path)),
    session_info = list(path = session_path, sha256 = sha256_file(session_path))
  ),
  software = list(R = R.version.string, packages = package_versions),
  script = list(path = SCRIPT_PATH, sha256 = sha256_file(SCRIPT_PATH))
)
write_json_file(metadata_output, meta_path)

cat("[PASS] Priority 1 48 h discovery ranking\n")
cat("[INFO] Samples used:", ncol(count_matrix), "(ENDO, 48 h only)\n")
cat("[INFO] Genes retained:", sum(keep), "of", nrow(counts), "\n")
cat("[INFO] Held-out expression columns loaded: false\n")
cat("[INFO] Wrote:", ranking_path, "\n")
