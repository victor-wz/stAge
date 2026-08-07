# stAge: transcriptomic-age clocks applied to spatial transcriptomics

Code release accompanying the stAge paper. This repository predicts transcriptomic
age (tAge) from spatial transcriptomics data using bulk-trained elastic-net clocks
applied to pseudobulk "metapixel" aggregates, and reproduces every downstream
statistical analysis and figure in the paper.

**Status of this release:** built by a systematic audit and refactor of the
original research codebase (documented in full in [`INVENTORY.md`](INVENTORY.md)).
Most figures are fully reproduced; a handful of gaps and open questions remain —
see [Known gaps and open questions](#known-gaps-and-open-questions) below before
assuming full coverage. This is refactored code, not reanalyzed code: every
statistical method, parameter, and default was preserved exactly from the
original notebooks/scripts. Where the original had a bug, it's fixed only when
doing so was required to make the code *runnable at all* (documented case by
case in each file), never to change a reported result.

## Repository layout

```
stage/          reusable package -- metapixel clustering, resolution search,
                preprocessing, clock prediction, hotspot calling, composition
                analysis, meta-analysis, GSEA, region annotation, plotting
R/              shared R utilities (SingleR annotation, DESeq2, RDS->h5ad, R deps)
analyses/       one script per paper figure, importing from stage/ and R/ --
                see INVENTORY.md for the source notebook each was extracted from
data/           download.py -- pulls public GEO/GSE datasets only; never commit data
examples/       quickstart.ipynb -- worked end-to-end example on one small dataset
environment.yml conda environment (Python side)
INVENTORY.md    full audit: what every file in the original codebase did, where
                it went in this release, every duplicate/superseded-version
                decision, and every analysis with no located source code
```

## Install

```bash
conda env create -f environment.yml
conda activate stage-release
Rscript R/install_dependencies.R
```

`environment.yml`'s R-side comment explains why this release does not ship a
version-pinned `renv.lock` — one should be generated with `renv::snapshot()`
against a real, executed run before final publication rather than fabricated
here.

You will also need, outside this repository (not distributed here — see the
paper's data/model availability statement):
- **The trained clock models** (`tAge_clocks/`, e.g.
  `EN differential models 4.6/EN_Chronoage_Mouse_All_WT_{scaleddiff,yugenediff}.pkl`).
- **An NCBI `*.gene_info` reference file** (e.g. `Mus_musculus.gene_info`,
  `Homo_sapiens.gene_info`) for gene-symbol/Ensembl → Entrez ID mapping — these
  are standard NCBI files, downloadable from the
  [NCBI Gene FTP site](https://ftp.ncbi.nlm.nih.gov/gene/DATA/GENE_INFO/).

Both paths are passed explicitly as function arguments / CLI flags throughout
`stage/` and `analyses/` — nothing is hard-coded to a specific machine's
directory layout (a change made in this release; the original research code had
several absolute, machine-specific paths, documented per-file in `INVENTORY.md`).

## Quickstart

```bash
python data/download.py --dataset brain3g   # pulls GSE212903 (4 samples)
jupyter notebook examples/quickstart.ipynb
```

Walks through the core pipeline end to end: metapixel clustering → gene
filtering → normalization → elastic-net clock prediction → propagation to
pixel level → a spatial tAge map, on the smallest dataset encountered during
the audit.

**Clock folder note (found during Phase 3 parity checking):** `stage.pipeline`'s
functions default `clock_folder`/`clock_dir` to `EN differential models 4.6`,
but not every dataset in the original pipeline was actually run against that
folder — Brain 2g's real published predictions, for example, were generated
using `tAge_clocks/tms_clocks/` instead (confirmed by reproducing them exactly
only after switching folders; see `INVENTORY.md` and the parity-check results
below). There is no single correct default — check which clock folder was used
for the specific dataset/figure you're reproducing before trusting the
parameter default.

## Reproducing a specific figure

Every file under `analyses/` corresponds to one figure number in the paper and
is independently runnable (`python analyses/fig6a_hotspot_calling.py --help`,
or the R equivalent). Each imports its statistical/computational logic from
`stage/` (or calls into `R/`) rather than duplicating it — if you're modifying
a method, change it once in `stage/`, not in every figure script that uses it.

A handful of `analyses/*.py` files are honest placeholders rather than full
reproductions — every one says so explicitly in its own module docstring
(`# TODO(gap):` or `# CAVEAT (unconfirmed mapping):`), and every one is also
listed below.

## Known gaps and open questions

Full detail, including exactly what *was* found and why it's not enough, lives
in [`INVENTORY.md`](INVENTORY.md) under "Analyses in the paper with NO code
found". Summary:

| Figure | Status |
|---|---|
| Fig 6f (cell-type abundance) | No analysis code located anywhere in the audited codebase; `analyses/fig6f_celltype_abundance.py` is a stub. |
| Fig 4e-g (whole-mouse LPS mixed-effects) | Plotting/stats code is real (`stage.plotting`); the data-loading step (`preds_per_file` for the actual LPS cohort) is a stub pointing at the likely source notebook. |
| Fig 5d (lung/AD mixed-effects) | Lung half fully reproduced (`analyses/fig5d_lung_ad_mem.R`); AD half is unimplemented, no source code found. |
| Fig 7e (human AD residual-age OLS) | Stub; a real Seurat RDS→h5ad conversion step for the AD-human dataset exists (`R/rds_to_h5ad.R`), but the OLS analysis itself was not found. |
| Fig 7f-h (breast cancer tumor vs. non-tumor) | Stub, same situation as Fig 7e (a "LIVER BREAST CANCER" conversion block exists in `R/rds_to_h5ad.R`, the comparison analysis does not). |
| Fig 7i-k (maternal-fetal interface) | The distance-gradient statistics (LOWESS/Fisher-z) are real and ported. The compartment-annotation step is intentionally unimplemented — the original marker-panel classifier is being replaced with a cell-type-based approach that doesn't exist yet (author decision, see `INVENTORY.md`). |
| Figs 2, 3, S4, S5 (clock prediction pipeline) | The underlying pipeline (`stage.pipeline`) is fully real; what specifically distinguishes each of these four figure panels from one another was not established by the audit. Four structurally-identical placeholder scripts exist pending author input. |
| Figs 6e, 7d, 7h, S9, S10 (module-specific clocks) | Real module-clock logic ported for the mouse-tissue panels (6e, S9, S10); the human-cohort panels (7d, 7h) are thinner, inferred placeholders — the mouse/human split itself is this release's inference, not confirmed. |
| Figs 6c vs. S7, and 6d vs. S8 | Both pairs share one underlying `stage.meta_analysis` / `stage.gsea` computation; what distinguishes the main-text panel from the supplementary panel in each pair was not established. |
| Fig S11 downstream (deconvolution proportions) | `R/singleR_annotation.R` produces the underlying cell-type calls; whether/where those get turned into the actual Fig S11 figure was not confirmed. |
| Fig 4d (SenMayo senescence) | The gene list is a confirmed SenMayo-core subset (Saul et al. 2022) and the scoring/correlation logic is real, but it's only confirmed to run on three injury-model cohorts (spinal cord crush, MI, bone fracture) — not confirmed to be the actual whole-mouse natural-aging cohort the figure implies. |

## A note on duplicated/consolidated code

Several statistical routines (Getis-Ord Gi* hotspot classification, in
particular) existed as 3-4 independently-drifted copies across the original
research notebooks. This release consolidates each into one function in
`stage/`, after confirming (and where necessary, asking the paper's author to
confirm) that all copies used identical parameters. See `INVENTORY.md`'s
"Duplicate/near-duplicate pairs" section for the full reasoning behind every
such consolidation.

## Citing

See the main paper. This release's own provenance (which original file every
piece of code came from) is documented in `INVENTORY.md` for methods-section
verification.
