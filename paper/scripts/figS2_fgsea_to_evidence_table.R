#!/usr/bin/env Rscript
# TCGA fgsea inputs must be rebuilt by the corrected fig2_deg_rank.R.
suppressPackageStartupMessages({
  library(data.table)
  library(fgsea)
  library(msigdbr)
})
script_arg <- grep("^--file=", commandArgs(FALSE), value = TRUE)
if (length(script_arg) != 1L) stop("Run this file with Rscript", call. = FALSE)
SCRIPT_DIR <- dirname(normalizePath(sub("^--file=", "", script_arg)))
source(file.path(SCRIPT_DIR, "tcga_input_utils.R"))
SD <- file.path(SCRIPT_DIR, "..", "source_data", "PANCAN_TP53_v1")
args <- commandArgs(TRUE)
if (!length(args) || startsWith(args[1L], "--")) {
  stop("Usage: Rscript SCRIPT.R CANCER [--rank PATH] [--outdir DIR]", call. = FALSE)
}
CANCER <- toupper(args[1L])
rank_path <- option_value(args, "--rank", file.path(SD, "derived", "rankings", paste0(CANCER, ".deg_ranking.tsv")))
outdir <- option_value(args, "--outdir", file.path(SD, "evidence_tables"))
collection <- option_value(args, "--collection", "H")
subcategory <- option_value(args, "--subcategory")
suffix <- paste0(collection, if (!is.null(subcategory)) paste0("_", gsub("[^A-Za-z0-9]+", "_", subcategory)))
suffix <- option_value(args, "--out-suffix", suffix)
out_path <- file.path(outdir, paste0(CANCER, ".", suffix, ".evidence_table.tsv"))
write_tcga_fgsea(rank_path, out_path, collection, subcategory)
