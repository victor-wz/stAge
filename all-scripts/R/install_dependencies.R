# Install R dependencies for the stAge-release R/ and analyses/*.R scripts.
#
# Package list assembled from an explicit `grep` of every `library()`/`require()`
# call across R/*.R and analyses/*.R in this release (not guessed) -- see the
# comment block below each install call for which script(s) need it.
#
# Run once: `Rscript R/install_dependencies.R`
# Then, to produce a real version-pinned lockfile (recommended before publishing,
# not fabricated here -- see README.md and environment.yml for why):
#   install.packages("renv")
#   renv::init()
#   renv::snapshot()

if (!requireNamespace("BiocManager", quietly = TRUE)) {
  install.packages("BiocManager")
}

cran_pkgs <- c(
  "optparse",     # CLI arg parsing, used in every R/*.R script in this release
  "dplyr",        # singleR_annotation.R
  "readr",        # singleR_annotation.R
  "Matrix",       # singleR_annotation.R
  "ggplot2",      # fig5d_lung_ad_mem.R (plots)
  "tidyr"         # fig5d_lung_ad_mem.R (data reshaping, if used)
)

bioc_pkgs <- c(
  "DESeq2",             # R/deseq2_hotspot_coldspot.R (Fig 6b)
  "ashr",               # R/deseq2_hotspot_coldspot.R -- lfcShrink(..., type="ashr")
  "SingleR",            # R/singleR_annotation.R (Fig S11)
  "SingleCellExperiment",
  "scuttle",
  "scran",
  "scater",
  "zellkonverter",      # R/singleR_annotation.R -- AnnData <-> SCE bridge
  "SpatialExperiment",  # R/rds_to_h5ad.R
  "Seurat",             # R/rds_to_h5ad.R, R/singleR_annotation.R
  "SeuratObject",
  "SeuratDisk"          # R/rds_to_h5ad.R -- Seurat -> h5ad export path
)

cran_missing <- cran_pkgs[!sapply(cran_pkgs, requireNamespace, quietly = TRUE)]
if (length(cran_missing) > 0) install.packages(cran_missing)

bioc_missing <- bioc_pkgs[!sapply(bioc_pkgs, requireNamespace, quietly = TRUE)]
if (length(bioc_missing) > 0) BiocManager::install(bioc_missing, update = FALSE, ask = FALSE)

# metafor (Fig 5d meta-regression) and SeuratDisk (GitHub-only, not on CRAN/Bioc)
if (!requireNamespace("metafor", quietly = TRUE)) install.packages("metafor")
if (!requireNamespace("SeuratDisk", quietly = TRUE)) {
  if (!requireNamespace("remotes", quietly = TRUE)) install.packages("remotes")
  remotes::install_github("mojaveazure/seurat-disk")
}

cat("Done. Verify with: Rscript -e 'sessionInfo()' and consider renv::snapshot() (see header).\n")
