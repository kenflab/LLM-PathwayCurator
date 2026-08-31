#!/usr/bin/env Rscript
# Priority 1: Hallmark fgseaMultilevel on the prespecified 48 h ranking.

suppressPackageStartupMessages({
  library(data.table)
  library(fgsea)
  library(jsonlite)
  library(msigdbr)
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

pick_column <- function(columns, candidates) {
  match_index <- match(tolower(candidates), tolower(columns), nomatch = 0L)
  match_index <- match_index[match_index != 0L]
  if (length(match_index) == 0L) return(NULL)
  columns[[match_index[[1L]]]]
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

args <- parse_args(commandArgs(trailingOnly = TRUE))
data_root <- normalizePath(path.expand(args$data_root), mustWork = TRUE)
config_path <- normalizePath(path.expand(args$config), mustWork = TRUE)
config <- fromJSON(config_path, simplifyVector = FALSE)
primary <- config$primary_analysis
benchmark_id <- config$benchmark_id
seed <- if (!is.null(args$seed)) as.integer(args$seed) else as.integer(primary$fgsea_seed)
min_size <- as.integer(primary$fgsea_min_size)
max_size <- as.integer(primary$fgsea_max_size)
require_true(is.finite(seed), "--seed must be an integer")
require_true(min_size >= 1L && max_size >= min_size, "Invalid fgsea size limits in protocol")
require_true(identical(as.integer(primary$discovery_time_h), 48L), "Discovery time must be 48 h")

ranking_path <- if (!is.null(args$ranking)) {
  normalizePath(path.expand(args$ranking), mustWork = TRUE)
} else {
  normalizePath(
    file.path(
      data_root,
      "output",
      "priority1",
      benchmark_id,
      "derived",
      "rankings",
      "discovery_48h.tsv"
    ),
    mustWork = TRUE
  )
}
out_dir <- if (!is.null(args$out_dir)) {
  path.expand(args$out_dir)
} else {
  file.path(data_root, "output", "priority1", benchmark_id, "derived", "fgsea")
}
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
out_dir <- normalizePath(out_dir, mustWork = TRUE)

fgsea_path <- file.path(out_dir, "discovery_48h.tsv")
gene_sets_path <- file.path(out_dir, "hallmark_gene_sets.tsv")
overlap_path <- file.path(out_dir, "hallmark_overlap.tsv")
meta_path <- file.path(out_dir, "discovery_48h.run_meta.json")
session_path <- file.path(out_dir, "discovery_48h.session_info.txt")
outputs <- c(fgsea_path, gene_sets_path, overlap_path, meta_path, session_path)
existing <- outputs[file.exists(outputs)]
require_true(isTRUE(args$force) || length(existing) == 0L, paste(
  "Outputs already exist; use --force:",
  paste(existing, collapse = ", ")
))

ranking <- fread(ranking_path)
require_true(all(c("gene", "score") %in% names(ranking)), "Ranking needs gene and score columns")
ranking[, gene := as.character(gene)]
ranking[, score := as.numeric(score)]
require_true(!anyNA(ranking$gene) && all(nzchar(ranking$gene)), "Ranking has missing genes")
require_true(!anyDuplicated(ranking$gene), "Ranking gene IDs must be unique")
require_true(all(grepl("^[0-9]+$", ranking$gene)), "Ranking must use Entrez IDs")
require_true(all(is.finite(ranking$score)), "Ranking scores must be finite")
require_true(nrow(ranking) >= 1000L, "Ranking contains fewer than 1,000 genes")

setorderv(ranking, c("score", "gene"), c(-1L, 1L))
raw_statistics <- ranking$score
names(raw_statistics) <- ranking$gene
exact_tie_values <- sum(duplicated(raw_statistics))
statistics <- break_exact_ties(raw_statistics)
names(statistics) <- names(raw_statistics)
statistics <- sort(statistics, decreasing = TRUE)

msig <- as.data.table(msigdbr(species = "Homo sapiens", collection = "H"))
entrez_column <- pick_column(
  names(msig),
  c("ncbi_gene", "ncbi_gene_id", "entrez_gene", "entrezgene", "entrez_gene_id")
)
require_true(!is.null(entrez_column), paste(
  "msigdbr output has no Entrez-like column; available:",
  paste(names(msig), collapse = ", ")
))
require_true("gs_name" %in% names(msig), "msigdbr output is missing gs_name")
msig[, gene := as.character(get(entrez_column))]
msig <- msig[!is.na(gene) & nzchar(gene) & grepl("^[0-9]+$", gene)]

db_versions <- if ("db_version" %in% names(msig)) unique(as.character(msig$db_version)) else NA_character_
db_versions <- db_versions[!is.na(db_versions) & nzchar(db_versions)]
require_true(length(db_versions) <= 1L, "Multiple MSigDB versions detected in one Hallmark export")
db_version <- if (length(db_versions) == 1L) db_versions[[1L]] else "not_reported_by_msigdbr"

gene_sets <- unique(msig[, .(pathway = as.character(gs_name), gene)])
setorderv(gene_sets, c("pathway", "gene"), c(1L, 1L))
gene_sets[, `:=`(
  collection = "H",
  db_version = db_version,
  in_discovery_universe = gene %in% names(statistics)
)]
pathways <- split(gene_sets$gene, gene_sets$pathway)
pathways <- lapply(pathways, unique)
overlap <- data.table(
  pathway = names(pathways),
  genes_in_msigdb_set = lengths(pathways),
  genes_in_discovery_universe = vapply(
    pathways,
    function(genes) sum(genes %in% names(statistics)),
    integer(1L)
  )
)
setorder(overlap, pathway)
require_true(max(overlap$genes_in_discovery_universe) > 0L, "No Hallmark genes overlap ranking")
require_true(
  sum(overlap$genes_in_discovery_universe >= min_size) > 0L,
  paste("No Hallmark pathway meets minSize =", min_size)
)

RNGkind("Mersenne-Twister", "Inversion", "Rejection")
set.seed(seed)
result <- fgseaMultilevel(
  pathways = pathways,
  stats = statistics,
  minSize = min_size,
  maxSize = max_size,
  nproc = 1
)
result <- as.data.table(result)
require_true(nrow(result) > 0L, "fgseaMultilevel returned no pathways")
require_true("leadingEdge" %in% names(result), "fgsea output is missing leadingEdge")
result[, leadingEdge := vapply(
  leadingEdge,
  function(genes) paste(as.character(genes), collapse = ","),
  character(1L)
)]
result[, `:=`(
  collection = "H",
  db_version = db_version,
  ranking_statistic = "limma_moderated_t_interaction",
  seed = seed
)]
result[, abs_NES_order := abs(NES)]
setorderv(result, c("padj", "abs_NES_order", "pathway"), c(1L, -1L, 1L))
result[, abs_NES_order := NULL]

fwrite(result, fgsea_path, sep = "\t", na = "NA")
fwrite(gene_sets, gene_sets_path, sep = "\t", na = "NA")
fwrite(overlap, overlap_path, sep = "\t", na = "NA")
capture.output(sessionInfo(), file = session_path)

package_names <- c("data.table", "fgsea", "jsonlite", "msigdbr")
package_versions <- as.list(vapply(
  package_names,
  function(package) as.character(packageVersion(package)),
  character(1L)
))
metadata_output <- list(
  benchmark_id = benchmark_id,
  protocol_version = config$protocol_version,
  protocol_status = config$status,
  analysis_scope = "Hallmark enrichment from ENDO 48 h discovery ranking only",
  held_out_expression_outcomes_calculated = FALSE,
  collection = "MSigDB Hallmark (H)",
  msigdb_version = db_version,
  gene_id_type = "NCBI Gene/Entrez",
  algorithm = "fgsea::fgseaMultilevel",
  parameters = list(minSize = min_size, maxSize = max_size, nproc = 1L, seed = seed),
  ranking = list(
    statistic = "limma moderated t-statistic for the prespecified interaction",
    genes = length(statistics),
    exact_tied_values_before_deterministic_breaking = exact_tie_values
  ),
  dimensions = list(
    hallmark_pathways_in_snapshot = length(pathways),
    tested_pathways = nrow(result),
    pathways_meeting_min_size = sum(
      overlap$genes_in_discovery_universe >= min_size
    )
  ),
  inputs = list(
    ranking = list(path = ranking_path, sha256 = sha256_file(ranking_path)),
    config = list(path = config_path, sha256 = sha256_file(config_path))
  ),
  outputs = list(
    fgsea = list(path = fgsea_path, sha256 = sha256_file(fgsea_path)),
    hallmark_gene_sets = list(path = gene_sets_path, sha256 = sha256_file(gene_sets_path)),
    overlap = list(path = overlap_path, sha256 = sha256_file(overlap_path)),
    session_info = list(path = session_path, sha256 = sha256_file(session_path))
  ),
  software = list(R = R.version.string, packages = package_versions),
  script = list(path = SCRIPT_PATH, sha256 = sha256_file(SCRIPT_PATH))
)
write_json_file(metadata_output, meta_path)

cat("[PASS] Priority 1 Hallmark fgsea at 48 h\n")
cat("[INFO] MSigDB version:", db_version, "\n")
cat("[INFO] Tested pathways:", nrow(result), "\n")
cat("[INFO] Held-out expression outcomes calculated: false\n")
cat("[INFO] Wrote:", fgsea_path, "\n")
