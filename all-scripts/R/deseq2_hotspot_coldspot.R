#!/usr/bin/env Rscript
# Pseudobulk DESeq2, hotspot vs coldspot (paper Fig 6b).
#
# EXTRACTED (not reconstructed) from `v_pipeline/dstream_loop.ipynb`, the
# `run_deseq2()` Python function (~line 2102 of the nbconvert'ed script) and its
# embedded R call (~lines 2111-2121), invoked there via rpy2 rather than as a
# standalone R script. The original notebook cell is titled "PSEUDOBULKING HOTSPOTS
# AND COLDSPOTS PER SAMPLE (AND USING DESEQ2)". The design formula (~condition),
# contrast (hotspot vs coldspot), and shrinkage estimator (ashr) below are copied
# verbatim from that cell and match the description in `FigS7_methods.txt`'s
# "Hotspot-versus-coldspot differential expression" section exactly. A second,
# near-identical `run_deseq2()`/DESeq2 call exists later in the same notebook
# (~line 2355, "PSEUDOBULKING SAMPLES and DEseq2") for a different sample-level
# comparison -- not extracted here since it isn't the hotspot-vs-coldspot contrast
# this figure needs; flagged for whoever builds that other analysis's release script.
#
# Input convention (preserved from the original): a genes x samples raw pseudobulk
# count matrix, where each sample column is named "<sample_id>__<condition>" (e.g.
# "Brain2g_O1_hotspot", "Brain2g_O1_coldspot") -- exactly how the original's
# `pseudobulk_adata()` helper named pseudobulk columns. `condition` is parsed by
# splitting each column name on "__" and taking the second element, matching the
# original's `[x.split("__")[1] for x in sample_groups]` exactly. This script does
# not reproduce the upstream pseudobulking step itself (that lives in
# `stage/hotspots.py` / the Gi* + pseudobulk-aggregation code, not in this R file) --
# it starts from an already-pseudobulked counts matrix.

suppressPackageStartupMessages({
  library(optparse)
  library(DESeq2)
})

option_list <- list(
  make_option("--counts", type = "character", default = NULL,
              help = "REQUIRED. CSV of raw pseudobulk counts, genes as rows (first column = gene id), sample columns named '<sample_id>__<condition>' with condition in {hotspot, coldspot}."),
  make_option("--out", type = "character", default = NULL,
              help = "REQUIRED. Output CSV path for the DESeq2 results table.")
)

opt <- parse_args(OptionParser(option_list = option_list))

if (is.null(opt$counts) || is.null(opt$out)) {
  stop("Both --counts and --out are required.")
}

counts_df <- read.csv(opt$counts, row.names = 1, check.names = FALSE)
counts <- as.matrix(round(counts_df))
storage.mode(counts) <- "integer"

sample_groups <- colnames(counts)
condition <- sapply(strsplit(sample_groups, "__"), `[`, 2)
if (any(is.na(condition))) {
  stop("Could not parse a condition from every column name. Expected '<sample_id>__<condition>', got: ",
       paste(sample_groups[is.na(condition)], collapse = ", "))
}
if (!all(condition %in% c("hotspot", "coldspot"))) {
  stop("Expected condition values to be exactly 'hotspot' or 'coldspot', got: ",
       paste(unique(condition), collapse = ", "))
}

n_hot <- sum(condition == "hotspot")
n_cold <- sum(condition == "coldspot")
if (n_hot < 2 || n_cold < 2) {
  stop("Not enough replicates for a DESeq2 comparison (need >=2 per condition): ",
       "n_hotspot=", n_hot, ", n_coldspot=", n_cold,
       " -- matches the original notebook's own skip condition for under-replicated tissues.")
}

design_df <- data.frame(sample = sample_groups, condition = condition, row.names = sample_groups)

# --- The following block is the extracted R logic, unchanged from the original
# rpy2-embedded cell (only variable-passing mechanics differ: direct R objects here
# instead of ro.globalenv[...] <- pandas2ri.py2rpy(...)). ---
dds <- DESeqDataSetFromMatrix(countData = counts, colData = design_df, design = ~condition)
dds <- DESeq(dds)
res <- results(dds, contrast = c("condition", "hotspot", "coldspot"))
res <- lfcShrink(dds, coef = "condition_hotspot_vs_coldspot", res = res, type = "ashr")
res_df <- as.data.frame(res)
res_df$gene <- rownames(res_df)

# Column renaming preserved from the original run_deseq2()'s Python-side rename.
out_df <- res_df[, c("gene", "log2FoldChange", "pvalue", "padj")]
colnames(out_df) <- c("gene", "logfoldchange", "pval", "pval_adj")

write.csv(out_df, opt$out, row.names = FALSE)
cat("Wrote", nrow(out_df), "genes (", sum(!is.na(out_df$pval_adj) & out_df$pval_adj < 0.05),
    "significant at FDR<0.05 ) to", opt$out, "\n")
