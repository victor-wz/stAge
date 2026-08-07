# Fig S11 — SingleR deconvolution

This figure is **not** a standalone Python analysis script (unlike the other figures
in `analyses/`) — its only located source code is the R annotation step, and no
downstream code that turns those calls into the published deconvolution-proportion
figure itself was found anywhere in the audited codebase.

## What exists: `R/singleR_annotation.R`

Produces per-spot cell-type calls via `SingleR::SingleR(test=..., ref=..., labels=...)`,
writing an annotated Seurat-object list to `--out-dir/<query-tissue-pattern>_singleR_backup.rds`
(one `Cell.type_SingleR` column per query h5ad file). See that script's own header
comment for the full account of how it was cleaned up from the original
`reference_cell_annot.R` (five alternate reference-loading blocks consolidated into
one script with a `--reference-format` switch; one unrelated exploratory tail section
dropped, recoverable from the original if needed).

**Required arguments, none defaulted:**
- `--ref-root`, `--ref-folder` — location of the reference dataset.
- `--reference-format` — one of `mtx_tsv`, `rds`, `txt_csv`, `txt_txt`, `h5ad`.
- `--query-dir`, `--query-tissue-pattern` — which h5ad files to annotate.
- `--out-dir`.

**Unresolved (flagged, not guessed) input path:** the original script hard-coded
`ipynb_dir <- "vvicente/scripts/old_pipeline"` as the reference root for most formats.
That directory does not exist anywhere in the current repository (confirmed during the
Phase 1 audit). `--ref-root` has **no default** for this reason — the correct current
location of that reference data needs to be supplied by whoever runs this script, and
ideally confirmed with the author and permanently recorded here once known.

## What's missing: the deconvolution-proportion figure itself

Searched `celltype_hotspot_tAge.ipynb` and `hotspot_senescence_stAge.ipynb` (the two
notebooks most likely to consume `Cell.type_SingleR`) for any code that computes or
plots cell-type deconvolution *proportions* as a standalone figure (e.g. a stacked bar
chart of estimated cell-type fractions per sample/region). Neither contains this —
`Cell.type_SingleR` calls are consumed only as categorical per-spot labels, used to
build cell-type x hotspot/coldspot pseudobulks (feeding Fig 6a and the composition
analyses in Fig S12), never as a proportions figure in their own right.

**TODO(gap, confirmed absent):** the actual Fig S11 deconvolution-proportion plot
(whatever its exact form — a stacked bar/box plot of per-sample or per-region
cell-type composition, most plausibly built from the same `obsm['composition']`
fraction vectors already computed in `stage/composition.py`'s pipeline for Fig S12)
was not located anywhere in the audited codebase. If it exists, it wasn't found by any
of the four Phase 1 inventory passes or this Phase 2 build; if it doesn't exist yet, it
would need to be written fresh from the composition fraction vectors that
`R/singleR_annotation.R`'s output feeds into upstream. Flagging for the author rather
than fabricating a plotting script for an analysis whose original form is unknown.

## Related, not the same figure

`stage/composition.py`'s `obsm['composition']` fraction vectors (used for Fig S12's
Oaxaca-Blinder decomposition) are downstream of `Cell.type_SingleR` for the Brain
2g/3g/25 datasets, so any future Fig S11 script would likely reuse that same
composition-fraction computation rather than deriving proportions independently —
worth checking with the author whether Fig S11 and Fig 6f (cell-type abundance, also
flagged as a gap in `INVENTORY.md`) are in fact the same analysis under two figure
numbers, or genuinely distinct.
