#!/usr/bin/env Rscript
# Seurat/SpatialExperiment RDS -> per-sample .h5ad conversion utility.
#
# CORRECTION TO AN EARLIER PHASE-1 AUDIT NOTE: the original `rds_to_h5ad.R` was
# characterized as "mostly commented-out example code for an alzheimer_ dataset."
# On a full read, that is only true of a ~50-line block at the very top of the file
# (preserved below as a comment, per the original audit request, since it's a usage
# example). The REST of the original file (~370 lines) is LIVE, uncommented,
# executable code implementing FOUR separate dataset conversions:
#   1. an AD mouse dataset (Condition x Age x SAMPLE subsetting)
#   2. an AD HUMAN dataset (Diagnosis x Age x SampleID subsetting)
#   3. a dataset the original literally labelled "LIVER BREAST CANCER" (per-sample
#      subsetting of an object named `liver.integrated`)
#   4. a human dentate-gyrus ("hbrain"/DG) SpatialExperiment dataset (age x sex x
#      sample x diagnosis subsetting)
# This means the codebase DOES contain real, executable RDS->h5ad conversion code
# bearing on the human AD analysis (paper Fig 7e) and the breast-cancer analysis
# (Fig 7f-h) -- flag this correction when reconciling against the Phase 1 inventory's
# "no code found" gap list for those two figures. This script is still only the
# *data-conversion* step, not the downstream statistical analysis (OLS residual-age
# for Fig 7e, tumor-vs-non-tumor comparison for Fig 7f-h) -- that analysis code was
# still not located anywhere in the audited tree.
#
# This version: preserves all 4 conversion blocks' subsetting/export logic exactly,
# turns each into a `convert_<dataset_type>()` function selected via --dataset-type,
# and replaces hard-coded paths with CLI args. One inconsistency in the original is
# normalized rather than silently worked around: 3 of the 4 blocks load their input
# via `readRDS(rds_files[1])` found by listing a folder, but the "LIVER BREAST CANCER"
# block instead referenced `liver.integrated` as an object assumed already present in
# an interactive R session, with no load call at all. All 4 now load via
# `readRDS(opt$input_rds)` uniformly -- this only standardizes *how the input Seurat/
# SpatialExperiment object is obtained* so the script is runnable non-interactively;
# no subsetting, filtering, or export logic was changed for any of the 4 datasets.
#
# A second, earlier draft of the human DG/hbrain conversion (operating directly on
# the SpatialExperiment via case-insensitive `colData` matching, without going
# through `as.Seurat` first) was present in the original as a large commented-out
# block preceding the live version used here. It is not reproduced in this cleaned
# script (the live version below supersedes it per the original file's own
# structure), but is noted here so that trace isn't lost: see the original
# `rds_to_h5ad.R`, the block starting "## ---------- Split SpatialExperiment by
# (age, sex, sample_id, diagnosis) ... ----------".

suppressPackageStartupMessages({
  library(optparse)
  library(Seurat)
  library(SeuratDisk)
  library(SpatialExperiment)
  library(SingleCellExperiment)
})

option_list <- list(
  make_option("--dataset-type", type = "character", default = NULL,
              help = "REQUIRED. One of: ad_mouse, ad_human, breast_cancer_liver, human_dg_hbrain."),
  make_option("--input-rds", type = "character", default = NULL,
              help = "REQUIRED. Path to the source .rds (Seurat or SpatialExperiment object)."),
  make_option("--out-dir", type = "character", default = NULL,
              help = "REQUIRED. Directory to write per-sample .h5ad files into."),
  make_option("--working-dir", type = "character", default = NULL,
              help = "If set, setwd() to this path first (original hard-coded '/home/vvicente/spatial_aging').")
)

opt <- parse_args(OptionParser(option_list = option_list))

required <- c("dataset_type", "input_rds", "out_dir")
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
# 1. AD MOUSE dataset -- preserved exactly from the original's live "FOR AD MOUSE
#    DATASET" block.
# ---------------------------------------------------------------------------
convert_ad_mouse <- function(seurat_obj, out_dir) {
  for (cond in unique(seurat_obj$Condition)) {
    for (age in unique(seurat_obj$Age)) {
      for (samp in unique(seurat_obj$SAMPLE)) {
        cells_here <- colnames(seurat_obj)[
          seurat_obj$Condition == cond & seurat_obj$Age == age & seurat_obj$SAMPLE == samp
        ]
        if (!length(cells_here)) next

        sub <- subset(seurat_obj, cells = cells_here)
        sub_name <- paste("alz", samp, cond, age, sep = "_")
        message(">>> Writing: ", sub_name)

        h5s <- file.path(out_dir, paste0(sub_name, ".h5seurat"))
        SaveH5Seurat(sub, filename = h5s, overwrite = TRUE,
                     counts = TRUE, data = TRUE, scale.data = TRUE, features = NULL)
        Convert(h5s, dest = "h5ad", assay = "Spatial", overwrite = TRUE, layers = TRUE)
        unlink(h5s)
      }
    }
  }
}

# ---------------------------------------------------------------------------
# 2. AD HUMAN dataset -- preserved exactly from the original's live "FOR AD HUMAN
#    DATASET" block. Relevant to paper Fig 7e (human AD residual-age OLS) --
#    conversion only, the OLS analysis itself was not located in the audited tree.
# ---------------------------------------------------------------------------
convert_ad_human <- function(seurat_obj, out_dir) {
  for (cond in unique(seurat_obj$Diagnosis)) {
    for (age in unique(seurat_obj$Age)) {
      for (samp in unique(seurat_obj$SampleID)) {
        cells_here <- colnames(seurat_obj)[
          seurat_obj$Diagnosis == cond & seurat_obj$Age == age & seurat_obj$SampleID == samp
        ]
        if (!length(cells_here)) next

        sub <- subset(seurat_obj, cells = cells_here)
        sub_name <- paste("alz", samp, cond, age, sep = "_")
        message(">>> Writing: ", sub_name)

        h5s <- file.path(out_dir, paste0(sub_name, ".h5seurat"))
        SaveH5Seurat(sub, filename = h5s, overwrite = TRUE,
                     counts = TRUE, data = TRUE, scale.data = TRUE, features = NULL)
        Convert(h5s, dest = "h5ad", assay = "Spatial", overwrite = TRUE, layers = TRUE)
        unlink(h5s)
      }
    }
  }
}

# ---------------------------------------------------------------------------
# 3. "LIVER BREAST CANCER" dataset (original's own label) -- preserved exactly from
#    the original's live block, per-sample subsetting only (no Condition/Age/etc.
#    factors in this one). Relevant to paper Fig 7f-h (breast cancer tumor vs
#    non-tumor) -- conversion only, the tumor-vs-non-tumor analysis itself was not
#    located in the audited tree.
# ---------------------------------------------------------------------------
convert_breast_cancer_liver <- function(seurat_obj, out_dir) {
  for (samp in unique(seurat_obj$sample)) {
    cells_here <- colnames(seurat_obj)[seurat_obj$sample == samp]
    if (!length(cells_here)) next

    sub <- subset(seurat_obj, cells = cells_here)
    name <- paste0("lcancer_", samp)

    sub@graphs <- list()
    for (a in names(sub@assays)) {
      if (inherits(sub[[a]], "SCTAssay")) {
        sub[[a]]@SCTModel.list <- list()
      }
    }

    h5s <- file.path(out_dir, paste0(name, ".h5seurat"))
    SaveH5Seurat(sub, filename = h5s, overwrite = TRUE, images = TRUE)
    Convert(h5s, dest = "h5ad", overwrite = TRUE)
    unlink(h5s)
    message("Wrote ", name, ".h5ad")
  }
}

# ---------------------------------------------------------------------------
# 4. Human DG / hbrain SpatialExperiment dataset -- preserved exactly from the
#    original's live block (converts SpatialExperiment -> SCE -> Seurat, then loops
#    over age x sex x sample x diagnosis with case-insensitive column-name lookup).
# ---------------------------------------------------------------------------
convert_human_dg_hbrain <- function(spe, out_dir) {
  stopifnot(inherits(spe, "SpatialExperiment"))
  colnames(spe) <- make.unique(colnames(spe))

  assay_name <- if ("counts" %in% SummarizedExperiment::assayNames(spe)) "counts" else SummarizedExperiment::assayNames(spe)[1]
  sce <- as(spe, "SingleCellExperiment")
  seurat_obj <- as.Seurat(sce, counts = assay_name, data = NULL)  # creates assay 'RNA'

  age_band <- function(a) {
    a_num <- suppressWarnings(as.numeric(a))
    ifelse(is.na(a_num), "NA",
           ifelse(a_num < 2, "infant",
                  ifelse(a_num >= 12 & a_num < 20, "teen",
                         ifelse(a_num < 65, "adult", "elderly"))))
  }
  sanitize <- function(x) {
    x <- ifelse(is.na(x) | x == "", "NA", as.character(x))
    x <- gsub("[^A-Za-z0-9._-]+", "_", x)
    x <- gsub("[.]{2,}", ".", x)
    sub("[.]$", "", x)
  }
  grab <- function(obj, key) {
    nms <- colnames(obj[[]])
    hit <- nms[match(tolower(key), tolower(nms))]
    if (is.na(hit)) stop("Missing metadata column: ", key, "\nAvailable: ", paste(nms, collapse = ", "))
    obj[[hit, drop = TRUE]]
  }

  age_vec <- grab(seurat_obj, "age")
  sex_vec <- grab(seurat_obj, "sex")
  sample_vec <- grab(seurat_obj, "sample_id")
  diagnosis_vec <- grab(seurat_obj, "diagnosis")

  for (age in unique(age_vec)) {
    for (sex in unique(sex_vec)) {
      for (samp in unique(sample_vec)) {
        for (dx in unique(diagnosis_vec)) {
          cells_here <- colnames(seurat_obj)[
            age_vec == age & sex_vec == sex & sample_vec == samp & diagnosis_vec == dx
          ]
          if (!length(cells_here)) next

          sub <- subset(seurat_obj, cells = cells_here)
          band <- age_band(age)
          sub_name <- paste0("age_", sanitize(age), "__sex_", sanitize(sex),
                              "__sample_", sanitize(samp), "__dx_", sanitize(dx), "_", band)
          message(">>> Writing: ", sub_name)

          h5s <- file.path(out_dir, paste0(sub_name, ".h5seurat"))
          SaveH5Seurat(sub, filename = h5s, overwrite = TRUE,
                       counts = TRUE, data = TRUE, scale.data = TRUE, features = NULL)
          # NOTE: preserved verbatim from the original -- the assay name passed to
          # Convert() here is "originalexp", not "RNA", despite as.Seurat() (above)
          # creating an assay named "RNA". This looks like a latent mismatch in the
          # original script (flagged, not fixed): if it actually ran successfully,
          # either as.Seurat's default assay name differs from what its own
          # documentation suggests in this Seurat/SpatialExperiment version, or this
          # call silently used the wrong assay. Not independently verified here.
          Convert(h5s, dest = "h5ad", assay = "originalexp", overwrite = TRUE, layers = TRUE)
          unlink(h5s)
        }
      }
    }
  }
}

# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------
seurat_obj <- readRDS(opt$input_rds)

switch(opt$dataset_type,
  ad_mouse            = convert_ad_mouse(seurat_obj, opt$out_dir),
  ad_human            = convert_ad_human(seurat_obj, opt$out_dir),
  breast_cancer_liver = convert_breast_cancer_liver(seurat_obj, opt$out_dir),
  human_dg_hbrain     = convert_human_dg_hbrain(seurat_obj, opt$out_dir),
  stop("Unknown --dataset-type: ", opt$dataset_type,
       " (expected one of ad_mouse, ad_human, breast_cancer_liver, human_dg_hbrain)")
)

cat("Done.\n")

# ---------------------------------------------------------------------------
# Original usage example (preserved verbatim as a comment, per audit request --
# this early block was fully commented out in the original file and is the only
# part of it that looked like a plain "example" rather than live code; kept here
# so that trace isn't lost even though the logic now lives in convert_ad_mouse()
# above with different variable names):
#
# rawdata_dir <- "vvicente/stomics_datasets/notion2"
# ipynb_dir <- "vvicente/scripts/v_pipeline"
# folder <- "alzheimer_"
# out_dir <- "as_h5ad/alzheimer_"
#
# library(Seurat)
# library(SeuratDisk)
#
# rds_files <- list.files(path = file.path(rawdata_dir, folder), pattern = "\\.rds$", full.names = TRUE)
# seurat_obj <- readRDS(rds_files[1])
#
# out_dir <- file.path(rawdata_dir, "as_h5ad/alzheimer_")
# dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
#
# for (cond in unique(seurat_obj$Condition)) {
#   for (age in unique(seurat_obj$Age)) {
#     cells_here <- WhichCells(seurat_obj, expression = Condition == cond & Age == age)
#     if (!length(cells_here)) next
#     sub <- subset(seurat_obj, cells = cells_here)
#     sub_name <- paste("alz", cond, age, sep = "_")
#     h5s <- file.path(out_dir, paste0(sub_name, ".h5seurat"))
#     SaveH5Seurat(sub, filename = h5s, overwrite = TRUE)
#     Convert(h5s, dest = "h5ad", overwrite = TRUE)
#     unlink(h5s)
#   }
# }
