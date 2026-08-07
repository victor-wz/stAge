"""Fig 6f: cell-type abundance.

TODO(gap) -- confirmed absent from the audited codebase. `celltype_hotspot_tAge.ipynb` was
the most likely candidate and was explicitly checked during the code audit: it builds
cell-type x hotspot/coldspot pseudobulks and re-runs the clock on them, but never computes
or plots cell-type ABUNDANCE (proportions/counts) itself. No other file was found that does.
See stAge-release/INVENTORY.md, "Analyses in the paper with NO code found", item 3.

`stage.R.singleR_annotation` (R/singleR_annotation.R) produces the SingleR cell-type calls
this figure would need as an input. `stage.composition`'s `.obsm['composition']` fraction
matrices (built for the Fig S12 Oaxaca-Blinder decomposition -- see
`stage.composition.residualize_composition_and_reclock`/`oaxaca_blinder_decomposition`) are a
plausible starting point if this needs to be authored fresh, since they already contain
per-metapixel cell-type fractions aligned to a `ct_universe` -- but no abundance-statistics
methodology (e.g. which comparison, which test, hotspot/coldspot vs. age-group vs. region) was
located anywhere, so nothing is invented here. Per the author's explicit instruction, a stub is
acceptable for confirmed gaps rather than guessing at unverified methodology.
"""

from __future__ import annotations


def compute_celltype_abundance(*args, **kwargs):
    """STUB -- no source implementation found for this analysis.

    Not implemented. See module docstring and stAge-release/INVENTORY.md gap list item 3.
    """
    raise NotImplementedError(
        "compute_celltype_abundance has no source implementation -- see this function's "
        "docstring and INVENTORY.md gap list item 3 (Cell-type abundance, Fig 6f)."
    )
