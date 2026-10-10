args <- commandArgs(trailingOnly = TRUE)
stopifnot(length(args) == 3L)
packages <- c("data.table", "R.utils", "limma", "fgsea", "msigdbr")
available <- vapply(packages, requireNamespace, logical(1), quietly = TRUE)
if (!all(available)) stop("Missing R packages: ", paste(packages[!available], collapse = ", "))
source(file.path(args[1], "paper/scripts/tcga_input_utils.R"))
reference <- data.table::fread(args[2], colClasses = "character")
current <- symbol_pathways(load_msig("H"))$membership
keys <- function(x) paste(x$gs_name, x$gene_symbol, sep = "\t")
if (!setequal(keys(reference), keys(current))) {
  stop("Installed msigdbr gene sets differ from this bundle's MSigDB 2026.1.Hs Hallmark snapshot. Update msigdbr (reference run: 26.1.1) or report the mismatch before comparing results.")
}
meta <- data.frame(package = packages, version = vapply(packages, function(p) as.character(packageVersion(p)), character(1)))
data.table::fwrite(meta, args[3], sep = "\t")
print(meta)
cat("Hallmark membership snapshot matched.\n")
sessionInfo()
