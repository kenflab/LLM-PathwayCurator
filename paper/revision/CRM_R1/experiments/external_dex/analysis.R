#!/usr/bin/env Rscript
# Fixed statistical dex case; this file never calls a language model.
require_true <- function(ok, message) if (!isTRUE(ok)) stop(message, call. = FALSE)
single_symbol <- function(x) {
  x <- trimws(as.character(x))
  !is.na(x) & nzchar(x) & grepl("^[[:alnum:]][[:alnum:]_.-]*$", x)
}
write_tsv <- function(x, file) {
  require_true(!file.exists(file), paste("Output already exists:", file))
  # Preserve round-trip precision for numeric source data and reporting templates.
  for (name in names(x)) if (is.numeric(x[[name]])) {
    x[[name]] <- vapply(x[[name]], function(z) if (is.na(z)) "NA" else formatC(z, format = "g", digits = 17L), character(1))
  }
  write.table(x, file, sep = "\t", row.names = FALSE, quote = FALSE, na = "NA")
}
raw_probe_order <- function(file) {
  lines <- readLines(file, warn = FALSE)
  idx <- which(startsWith(lines, "FEATURES\t"))
  require_true(length(idx) == 1L, "Agilent FEATURES header is not unique")
  header <- strsplit(lines[[idx]], "\t", fixed = TRUE)[[1L]]
  required <- c("ProbeName", "ControlType", "gMedianSignal", "gBGMedianSignal", "gIsWellAboveBG")
  require_true(all(required %in% header), "Raw Agilent fields missing from FEATURES header")
  after <- lines[seq.int(idx + 1L, length(lines))]
  rows <- after[startsWith(after, "DATA\t")]
  require_true(length(rows) > 1000L, "Unexpected number of raw Agilent features")
  fields <- strsplit(rows, "\t", fixed = TRUE)
  pi <- match("ProbeName", header)
  ci <- match("ControlType", header)
  require_true(all(lengths(fields) >= max(pi, ci)), "Truncated raw feature row")
  data.frame(probe = vapply(fields, `[[`, character(1), pi), control = vapply(fields, `[[`, character(1), ci), stringsAsFactors = FALSE)
}

main <- function(job_file) {
  suppressPackageStartupMessages({
    library(airway)
    library(SummarizedExperiment)
    library(edgeR)
    library(limma)
    library(fgsea)
    library(jsonlite)
  })
  Sys.setlocale("LC_COLLATE", "C")
  options(digits = 17, warn = 1)
  job <- fromJSON(job_file)
  protocol <- fromJSON(job$protocol)
  out <- job$outdir
  require_true(dir.exists(out), "Output directory missing")
  capture.output(sessionInfo(), file = file.path(out, "session_info.txt"))
  for (name in names(job$R_runtime$packages)) {
    require_true(as.character(packageVersion(name)) == job$R_runtime$packages[[name]], "R package identity changed before analysis")
  }
  membership <- read.delim(job$hallmark, stringsAsFactors = FALSE, check.names = FALSE)
  terms <- sort(unique(membership$term_id))
  require_true(length(terms) == 50L && !anyDuplicated(membership), "Wrong fixed Hallmark census")
  pathways <- split(membership$gene_symbol, membership$term_id)

  # Only the integer-count airway object is loaded, never FPKM/Cuffdiff or gse.
  environment <- new.env()
  data("airway", package = "airway", envir = environment)
  se <- environment$airway
  require_true(inherits(se, "RangedSummarizedExperiment"), "airway count object unavailable")
  expected <- protocol$discovery$samples
  md <- as.data.frame(colData(se))
  require_true(all(c("SampleName", "cell", "dex", "albut") %in% names(md)), "airway sample annotation missing")
  require_true(ncol(se) == 8L && setequal(colnames(se), expected$run), "airway sample census changed")
  se <- se[, match(expected$run, colnames(se))]
  md <- as.data.frame(colData(se))
  require_true(identical(as.character(md$SampleName), expected$geo_accession), "airway run-to-GSM mapping changed")
  require_true(identical(as.character(md$cell), expected$donor), "airway donor mapping changed")
  require_true(identical(as.character(md$dex), ifelse(expected$treatment == "dex", "trt", "untrt")), "airway treatment mapping changed")
  require_true(all(as.character(md$albut) == "untrt"), "Albuterol sample entered the dex contrast")
  annotation <- as.data.frame(rowData(se))
  require_true("gene_name" %in% names(annotation), "airway gene_name snapshot missing; no live annotation fallback")
  require_true(!anyDuplicated(rownames(se)), "Duplicated Ensembl gene IDs")
  counts <- as.matrix(assay(se, "counts"))
  require_true(all(is.finite(counts) & counts >= 0 & counts == floor(counts)), "airway assay is not an integer-count matrix")
  symbols <- trimws(as.character(annotation$gene_name))
  mapped <- single_symbol(symbols)
  mapping_record <- data.frame(ensembl_id = rownames(se), gene_symbol = symbols, retained_single_symbol = mapped)
  write_tsv(mapping_record, file.path(out, "discovery_gene_mapping.tsv"))
  counts <- rowsum(counts[mapped, , drop = FALSE], symbols[mapped], reorder = TRUE)
  counts <- counts[order(rownames(counts)), , drop = FALSE]
  keep <- rowSums(counts >= protocol$discovery$filter$minimum_count) >= protocol$discovery$filter$minimum_samples
  counts <- counts[keep, , drop = FALSE]
  require_true(nrow(counts) >= 1000L, "Fewer than 1,000 discovery genes passed the frozen count filter")
  discovery_md <- expected
  discovery_md$donor <- factor(discovery_md$donor)
  discovery_md$treatment <- factor(discovery_md$treatment, levels = c("control", "dex"))
  require_true(all(table(discovery_md$donor, discovery_md$treatment) == 1L), "Discovery is not four paired donor units")
  write_tsv(expected, file.path(out, "discovery_samples.tsv"))

  # Verify spot ordering independently; never assign first-array annotation to
  # a differently ordered subsequent array.
  validation_md <- protocol$validation$samples
  ids <- validation_md$geo_accession
  files <- vapply(ids, function(id) job$raw_files[[id]], character(1))
  orders <- lapply(files, raw_probe_order)
  require_true(all(vapply(orders, identical, logical(1), orders[[1L]])), "Agilent probe order/control annotation differs across public arrays")
  raw <- read.maimages(files, source = "agilent", green.only = TRUE, other.columns = "gIsWellAboveBG", verbose = FALSE)
  require_true(inherits(raw, "EListRaw") && all(c("ProbeName", "ControlType") %in% names(raw$genes)), "Expected unlogged Agilent EListRaw")
  require_true(identical(as.character(raw$genes$ProbeName), orders[[1L]]$probe), "limma/raw probe identity mismatch")
  require_true(all(is.finite(raw$E)) && all(is.finite(raw$Eb)), "Missing raw foreground/background intensity")
  detection <- raw$other$gIsWellAboveBG
  require_true(is.matrix(detection) && identical(dim(detection), dim(raw$E)) && all(is.finite(detection)), "Raw detection flag matrix unavailable")
  normalized <- normalizeBetweenArrays(backgroundCorrect(raw, method = "normexp", offset = 50), method = "quantile")
  require_true(inherits(normalized, "EList") && !inherits(normalized, "EListRaw"), "Expected log2-normalized Agilent EList")
  colnames(normalized$E) <- ids
  platform <- read.delim(job$platform_mapping, stringsAsFactors = FALSE, check.names = FALSE)
  index <- match(as.character(normalized$genes$ProbeName), platform$probe_id)
  val_symbols <- trimws(platform$gene_symbol[index])
  val_control <- suppressWarnings(as.numeric(platform$control_type[index]))
  ok <- !is.na(index) & single_symbol(val_symbols) & !is.na(val_control) & val_control == 0 & normalized$genes$ControlType == 0 & rowSums(detection > 0) >= 3 & rowSums(!is.finite(normalized$E)) == 0
  ok[is.na(ok)] <- FALSE
  require_true(sum(ok) >= 1000L, "Fewer than 1,000 mapped/detected non-control probes; no normalization fallback")
  write_tsv(data.frame(probe_id = normalized$genes$ProbeName, gene_symbol = val_symbols, retained = ok), file.path(out, "validation_probe_mapping.tsv"))
  expression <- normalized$E[ok, , drop = FALSE]
  groups <- split(seq_len(nrow(expression)), val_symbols[ok])
  expression <- t(vapply(groups, function(i) apply(expression[i, , drop = FALSE], 2L, median), numeric(length(ids))))
  colnames(expression) <- ids
  expression <- expression[apply(expression, 1L, var) > 0, , drop = FALSE]
  require_true(all(is.finite(expression)), "Non-finite gene-level microarray values")
  write_tsv(validation_md, file.path(out, "validation_samples.tsv"))
  write_tsv(data.frame(sample = ids, normalized_log2_median = apply(normalized$E, 2L, median), retained_gene_median = apply(expression, 2L, median), retained_gene_sd = apply(expression, 2L, sd)), file.path(out, "validation_normalization_qc.tsv"))
  write_tsv(data.frame(gene_symbol = rownames(counts)), file.path(out, "discovery_filtered_gene_universe.tsv"))
  write_tsv(data.frame(gene_symbol = rownames(expression)), file.path(out, "validation_measured_gene_universe.tsv"))

  # This universe is written before any differential-expression or pathway fit.
  universe <- sort(intersect(rownames(counts), rownames(expression)))
  require_true(length(universe) >= 1000L, "Fewer than 1,000 common measured genes")
  write_tsv(data.frame(gene_symbol = universe), file.path(out, "common_gene_universe.tsv"))
  paths <- lapply(pathways[terms], intersect, universe)
  size <- lengths(paths)
  write_tsv(data.frame(term_id = terms, original_set_size = lengths(pathways[terms]), measured_set_size = size, common_universe_size = length(universe)), file.path(out, "pathway_measured_coverage.tsv"))

  fit_discovery <- function(selected) {
    meta <- droplevels(discovery_md[selected, , drop = FALSE])
    design <- model.matrix(~donor + treatment, meta)
    require_true(qr(design)$rank == ncol(design) && nrow(design) > ncol(design), "Singular paired discovery design")
    dge <- calcNormFactors(DGEList(counts = counts[, selected, drop = FALSE]), method = "TMM")
    fit <- eBayes(lmFit(voom(dge, design, plot = FALSE), design), trend = FALSE, robust = FALSE)
    k <- match("treatmentdex", colnames(design))
    require_true(!is.na(k), "Discovery contrast not dex minus control")
    data.frame(gene_symbol = rownames(counts), score = fit$t[, k], logFC = fit$coefficients[, k], pval = fit$p.value[, k], q = p.adjust(fit$p.value[, k], "BH"))
  }
  fit_validation <- function(treated) {
    selected <- validation_md$treatment %in% c("control", treated)
    meta <- validation_md[selected, , drop = FALSE]
    meta$treatment <- factor(ifelse(meta$treatment == "control", "control", "dex"), levels = c("control", "dex"))
    require_true(identical(as.integer(table(meta$treatment)), c(4L, 3L)), "Validation culture-group size changed")
    design <- model.matrix(~treatment, meta)
    fit <- eBayes(lmFit(expression[, selected, drop = FALSE], design), trend = TRUE, robust = FALSE)
    k <- match("treatmentdex", colnames(design))
    data.frame(gene_symbol = rownames(expression), score = fit$t[, k], logFC = fit$coefficients[, k], pval = fit$p.value[, k], q = p.adjust(fit$p.value[, k], "BH"))
  }
  enrich <- function(ranking, name) {
    write_tsv(ranking, file.path(out, paste0(name, ".gene_rank.tsv")))
    statistics <- ranking$score[match(universe, ranking$gene_symbol)]
    names(statistics) <- universe
    eligible <- size >= protocol$enrichment$minSize & size <= protocol$enrichment$maxSize
    result <- data.frame(term_id = terms, pval = NA_real_, q = NA_real_, ES = NA_real_, NES = NA_real_, size = size, leading_genes = "", log2err = NA_real_, status = ifelse(eligible, "NUMERICAL_FAILURE", "OUTSIDE_SIZE_LIMITS"), stringsAsFactors = FALSE)
    if (any(!is.finite(statistics))) {
      result$status[eligible] <- "INCOMPLETE_FIXED_UNIVERSE"
    } else if (any(eligible)) {
      statistics <- statistics[order(-statistics, names(statistics), method = "radix")]
      set.seed(protocol$enrichment$seed)
      fg <- fgseaMultilevel(pathways = paths[eligible], stats = statistics,
        minSize = protocol$enrichment$minSize, maxSize = protocol$enrichment$maxSize,
        eps = protocol$enrichment$eps, sampleSize = protocol$enrichment$sampleSize,
        nPermSimple = protocol$enrichment$nPermSimple, nproc = protocol$enrichment$nproc,
        gseaParam = protocol$enrichment$gseaParam, scoreType = protocol$enrichment$scoreType)
      require_true(!anyDuplicated(fg$pathway) && all(fg$pathway %in% terms), "Unexpected fgsea term output")
      ii <- match(fg$pathway, result$term_id)
      result$pval[ii] <- fg$pval
      result$ES[ii] <- fg$ES
      result$NES[ii] <- fg$NES
      result$log2err[ii] <- fg$log2err
      result$leading_genes[ii] <- vapply(fg$leadingEdge, paste, character(1), collapse = ";")
      good <- is.finite(result$pval) & result$pval >= 0 & result$pval <= 1 & is.finite(result$NES)
      # n=50 retains the locked multiplicity census, including non-estimable sets.
      result$q[good] <- p.adjust(result$pval[good], method = "BH", n = 50L)
      result$status[good] <- "ESTIMABLE"
    }
    write_tsv(result, file.path(out, paste0(name, ".tsv")))
    cat("[R08]", name, "retained", nrow(result), "terms; estimable", sum(result$status == "ESTIMABLE"), "\n")
  }
  enrich(fit_discovery(seq_len(8L)), "discovery")
  for (donor in sort(levels(discovery_md$donor))) {
    enrich(fit_discovery(which(discovery_md$donor != donor)), paste0("donor_loo_", donor))
  }
  enrich(fit_validation("dex24"), "validation_24h")
  enrich(fit_validation("dex4"), "validation_4h_secondary")
  write_json(list(schema = "CRM_R1_EXTERNAL_DEX_R_STATS_R08", discovery_filtered_genes = nrow(counts), validation_measured_genes = nrow(expression), common_genes = length(universe), validation_retained_probes = sum(ok), donor_folds = 4L, model_calls = 0L, expression_outcomes_loaded = TRUE), file.path(out, "R_STATISTICS_COMPLETE.json"), auto_unbox = TRUE, pretty = TRUE)
}

args <- commandArgs(trailingOnly = TRUE)
require_true(length(args) == 1L && file.exists(args[[1L]]), "Supply the Python coordinator's JOB.private.json")
main(args[[1L]])
