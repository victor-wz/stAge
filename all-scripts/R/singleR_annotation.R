#!/usr/bin/env Rscript
# SingleR-based reference cell-type annotation for spatial transcriptomics h5ad files.
#
# Cleaned/parameterized version of the original `reference_cell_annot.R`. That script
# was a hand-edited workspace: it contained FIVE alternative "prepare reference object"
# blocks (matched to five different reference input formats), of which only one was
# active at a time by manually commenting/uncommenting, followed by one live query-
# annotation loop (SingleR call + save), followed by a large commented-out earlier
# version of that same loop, followed by an UNRELATED exploratory tail section
# ("UMAP plotting without singleR reference annotation", Spleen dataset, no SingleR
# call at all) appended at the bottom.
#
# This version preserves the reference-loading logic for every format exactly (each is
# now a `load_reference_<format>()` function, selected via --reference-format instead of
# hand-editing), and the live SingleR query-annotation loop exactly. It DROPS the
# unrelated UMAP-only tail section (it performed no SingleR annotation and used a
# different, hard-coded dataset/path convention -- Spleen data via a
# `vvicente/scripts/d_pipeline` path -- unrelated to the cell-type reference-annotation
# task this script is named for). This is an editorial exclusion for a "clean, runnable
# script", not a behavior change to the annotation logic -- flagged here rather than
# silently dropped; the original tail section can be recovered from
# `reference_cell_annot.R` lines 563-648 if it's still needed for something else.
#
# KNOWN UNRESOLVED PATH (flagged, not guessed): the original hard-coded
# `ipynb_dir <- "vvicente/scripts/old_pipeline"` for most reference folders. That
# directory does not exist anywhere in the current repository (confirmed during the
# Phase 1 audit). Rather than silently substitute a guessed replacement, `--ref-root`
# below is a REQUIRED argument with no default -- the caller must supply the correct
# current location of that reference data.

suppressPackageStartupMessages({
  library(optparse)
  library(dplyr)
  library(Matrix)
  library(SeuratObject)
  library(Seurat)
  library(SingleR)
  library(SingleCellExperiment)
  library(scuttle)
  library(scran)
  library(scater)
  library(readr)
  library(zellkonverter)
})

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
# NOTE: none of the other R scripts in the audited codebase used `optparse` (they all
# hard-coded paths as top-of-script variables), so there was no existing convention to
# follow. `optparse` is used here since this script needs more than 1-2 positional args.

option_list <- list(
  make_option("--ref-root", type = "character", default = NULL,
              help = "REQUIRED. Root directory containing the reference dataset folder (was hard-coded to the now-missing 'vvicente/scripts/old_pipeline' for most formats, or 'vvicente/scripts/v_pipeline' for the ovary_fixed case -- see script header)."),
  make_option("--ref-folder", type = "character", default = NULL,
              help = "REQUIRED. Subfolder of --ref-root holding the reference data (e.g. 'ovary_sharif', 'ct_reference/brain', 'ct_reference/ovary_fixed')."),
  make_option("--reference-format", type = "character", default = "mtx_tsv",
              help = "One of: mtx_tsv, rds, txt_csv, txt_txt, h5ad [default %default]. Selects which of the original script's 5 reference-loading blocks to run."),
  make_option("--query-dir", type = "character", default = NULL,
              help = "REQUIRED. Directory containing query *.h5ad files to annotate."),
  make_option("--query-tissue-pattern", type = "character", default = NULL,
              help = "REQUIRED. Pattern used to list query h5ad files, e.g. 'ovary' or 'Aging_Ovary'."),
  make_option("--exclude-pattern", type = "character", default = NULL,
              help = "Optional substring; query files containing it are excluded (original script used this to drop '*LR*' files)."),
  make_option("--out-dir", type = "character", default = NULL,
              help = "REQUIRED. Directory to write the annotated Seurat-object RDS list to."),
  make_option("--working-dir", type = "character", default = NULL,
              help = "If set, setwd() to this path first (original hard-coded '/home/vvicente/spatial_aging').")
)

opt <- parse_args(OptionParser(option_list = option_list))

required <- c("ref_root", "ref_folder", "query_dir", "query_tissue_pattern", "out_dir")
missing_args <- required[sapply(required, function(a) is.null(opt[[a]]))]
if (length(missing_args) > 0) {
  stop("Missing required arguments: ", paste0("--", gsub("_", "-", missing_args), collapse = ", "))
}

if (!is.null(opt$working_dir)) {
  setwd(opt$working_dir)
  message("Working directory: ", getwd())
}

dir.create(opt$out_dir, recursive = TRUE, showWarnings = FALSE)

# ---------------------------------------------------------------------------
# Reference-loading, one function per format found in the original script.
# Each returns a Seurat object with a `$CellType` metadata column, matching what
# the original script built by hand for each format before the live SingleR loop.
# ---------------------------------------------------------------------------

load_reference_mtx_tsv <- function(ref_root, ref_folder) {
  # Original block: "PREPARE REFERENCE OBJECT (IF STORED AS .mtx and .tsv)"
  mtx_files <- list.files(path = file.path(ref_root, ref_folder), pattern = "\\.mtx$", full.names = TRUE)
  mtx_path <- mtx_files[1]
  sample_prefix <- tools::file_path_sans_ext(basename(mtx_path))

  count_matrix <- readMM(mtx_path)
  barcodes <- read_tsv(file.path(ref_root, ref_folder, paste0(sample_prefix, "_barcodes.tsv")), col_names = FALSE)
  genes <- read_tsv(file.path(ref_root, ref_folder, paste0(sample_prefix, "_genes.tsv")), col_names = FALSE)

  rownames(count_matrix) <- make.unique(genes$X2)
  colnames(count_matrix) <- barcodes$X1

  df <- readxl::read_excel(file.path(ref_root, ref_folder, "SuppTableCells.xlsx"))
  df_sorted <- df %>%
    filter(Barcode %in% colnames(count_matrix)) %>%
    arrange(match(Barcode, colnames(count_matrix)))
  metadata_df <- df_sorted[, c("Barcode", "CellType")]

  CreateSeuratObject(counts = count_matrix, meta.data = metadata_df,
                      project = "reference", min.cells = 3, min.features = 200)
}

load_reference_rds <- function(ref_root, ref_folder) {
  # Original block: "PREPARE REFERENCE OBJECT (IF STORED AS .rds)"
  rds_files <- list.files(path = file.path(ref_root, ref_folder), pattern = "\\.RDS$", full.names = TRUE)
  wat_ref <- readRDS(rds_files[1])
  wat_ref$CellType <- wat_ref$cluster.names
  wat_ref
}

load_reference_txt_csv <- function(ref_root, ref_folder) {
  # Original block: "PREPARE REFERENCE OBJECT (IF STORED AS metadata.csv and counts.txt)"
  txt_files <- list.files(path = file.path(ref_root, ref_folder), pattern = "\\.txt$", full.names = TRUE)
  txt_ref <- read.delim(txt_files[1])
  csv_files <- list.files(path = file.path(ref_root, ref_folder), pattern = "\\.csv$", full.names = TRUE)
  df <- read_csv(csv_files[1])

  df_sorted <- df %>%
    filter(nGene %in% colnames(txt_ref)) %>%
    arrange(match(nGene, colnames(txt_ref)))
  metadata_df <- df_sorted[, c("nGene", "CellType")]

  CreateSeuratObject(counts = txt_ref, meta.data = metadata_df,
                      project = "reference", min.cells = 3, min.features = 200)
}

load_reference_txt_txt <- function(ref_root, ref_folder) {
  # Original block: "If counts and metadata are both TXT"
  txt_files <- list.files(path = file.path(ref_root, ref_folder), pattern = "\\X.txt$", full.names = TRUE)
  txt_ref <- read.delim(txt_files[1], row.names = 1)

  meta_files <- list.files(path = file.path(ref_root, ref_folder), pattern = "\\Etc.txt$", full.names = TRUE)
  df <- read.delim(meta_files[1], skip = 1)
  df$Barcode <- sub(".*_(\\w{16})$", "\\1", df$TYPE)

  barcodes_in_counts <- sub(".*_", "", colnames(txt_ref))
  colnames(txt_ref) <- barcodes_in_counts

  df_sorted <- df %>%
    filter(Barcode %in% colnames(txt_ref)) %>%
    distinct(Barcode, .keep_all = TRUE) %>%
    arrange(match(Barcode, colnames(txt_ref)))
  stopifnot(ncol(txt_ref) == nrow(df_sorted))

  metadata_df <- df_sorted %>%
    select(Barcode, CellType = group.2) %>%
    tibble::column_to_rownames("Barcode")

  CreateSeuratObject(counts = txt_ref, meta.data = metadata_df,
                      project = "reference", min.cells = 3, min.features = 200)
}

load_reference_h5ad <- function(ref_root, ref_folder) {
  # Original block: "PREPARE REFERENCE OBJECT (IF STORED AS .h5ad)"
  suppressPackageStartupMessages(library(SeuratDisk))
  h5ad_files <- list.files(path = file.path(ref_root, ref_folder), pattern = "\\Li_Anndata.h5ad$", full.names = TRUE)
  Convert(h5ad_files[1], dest = "h5seurat", overwrite = TRUE)
  h5s <- sub("\\.h5ad$", ".h5seurat", h5ad_files[1])
  wat_ref <- LoadH5Seurat(h5s, assays = "RNA", active.ident = FALSE)
  wat_ref$CellType <- wat_ref$cell_type
  wat_ref
}

load_reference <- function(format, ref_root, ref_folder) {
  switch(format,
    mtx_tsv = load_reference_mtx_tsv(ref_root, ref_folder),
    rds     = load_reference_rds(ref_root, ref_folder),
    txt_csv = load_reference_txt_csv(ref_root, ref_folder),
    txt_txt = load_reference_txt_txt(ref_root, ref_folder),
    h5ad    = load_reference_h5ad(ref_root, ref_folder),
    stop("Unknown --reference-format: ", format,
         " (expected one of mtx_tsv, rds, txt_csv, txt_txt, h5ad)")
  )
}

# ---------------------------------------------------------------------------
# Live query-annotation loop -- preserved exactly from the original script's
# uncommented final version (the "ovary"/ct_reference/ovary_fixed section).
# ---------------------------------------------------------------------------

wat_ref <- load_reference(opt$reference_format, opt$ref_root, opt$ref_folder)

h5ad_files <- list.files(path = opt$query_dir,
                          pattern = paste0(opt$query_tissue_pattern, ".*\\.h5ad$"),
                          full.names = TRUE)
if (!is.null(opt$exclude_pattern)) {
  h5ad_files <- h5ad_files[!grepl(opt$exclude_pattern, h5ad_files)]
}
if (length(h5ad_files) == 0) {
  stop("No query h5ad files matched --query-tissue-pattern='", opt$query_tissue_pattern,
       "' under --query-dir='", opt$query_dir, "'")
}

seurat_list <- list()

for (file in h5ad_files) {
  cat("Processing file:", file, "\n")
  file_prefix <- gsub("\\.h5ad", "", basename(file))

  sce <- readH5AD(file)

  if ("raw_count" %in% assayNames(sce)) {
    assayNames(sce)[assayNames(sce) == "raw_count"] <- "X"
  }
  gene_names <- rownames(sce)
  if (is.null(gene_names) || anyDuplicated(gene_names) > 0) {
    rownames(sce) <- make.unique(if (is.null(gene_names)) paste0("Gene", seq_len(nrow(sce))) else gene_names)
  }
  rownames(sce) <- sub("-ENSMUSG[0-9]+", "", rownames(sce))

  mat <- assay(sce, "X")
  mat_agg <- rowsum(mat, group = rownames(sce))
  sce_clean <- SingleCellExperiment(assays = list(X = mat_agg), colData = colData(sce))
  sizeFactors(sce_clean) <- NULL

  seuratobj <- as.Seurat(sce_clean, counts = "X", data = "X")

  gene_names <- rownames(wat_ref)
  common_names <- intersect(gene_names, rownames(seuratobj))
  wat_ref_subset <- wat_ref[common_names, ]
  seuratobj <- seuratobj[common_names, ]

  wat_ref_experiment <- as.SingleCellExperiment(wat_ref_subset)
  wat_ref_experiment$Cell.type <- colData(wat_ref_experiment)$CellType

  DefaultAssay(seuratobj) <- "originalexp"
  seurat_experiment <- as.SingleCellExperiment(seuratobj)

  wat_ref_experiment <- logNormCounts(wat_ref_experiment)

  lib_sizes <- colSums(assay(seurat_experiment, "counts"))
  seurat_experiment <- seurat_experiment[, lib_sizes > 0]
  seurat_experiment <- computeSumFactors(seurat_experiment)
  seurat_experiment <- logNormCounts(seurat_experiment)

  results <- SingleR(test = seurat_experiment, ref = wat_ref_experiment,
                      labels = wat_ref_experiment$Cell.type)

  seuratobj$Cell.type_SingleR <- results$labels
  seurat_list[[file_prefix]] <- seuratobj
}

out_rds <- file.path(opt$out_dir, paste0(opt$query_tissue_pattern, "_singleR_backup.rds"))
saveRDS(seurat_list, file = out_rds)
cat("Wrote", length(seurat_list), "annotated objects to", out_rds, "\n")
