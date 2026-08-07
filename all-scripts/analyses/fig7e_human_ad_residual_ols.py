"""Fig 7e -- Human AD residual-age OLS.

STATUS: gap stub. The residual-age OLS analysis itself was not located anywhere in
the audited codebase (see stAge-release/INVENTORY.md, "Analyses in the paper with
NO code found", item 6).

What DOES exist: stage-release/R/rds_to_h5ad.R's `convert_ad_human()` block is real,
live Seurat-RDS-to-h5ad conversion code for a human AD dataset (Diagnosis x Age x
SampleID subsetting) -- this is the data-preparation starting point for this figure,
not the analysis itself. Run it via:

    Rscript stage-release/R/rds_to_h5ad.R --dataset-type ad_human \\
        --input-rds <path-to-source.rds> --out-dir <h5ad-output-dir>

before this script can be written for real.

Do NOT fill in a fabricated OLS implementation here -- the paper's actual residual-age
OLS specification (which covariates, which residualization target, which age proxy)
was not confirmed against any source in this codebase. This stub exists only to mark
where that code belongs once the author supplies or locates it.
"""

from __future__ import annotations

import pandas as pd


def residual_age_ols(adata, *args, **kwargs) -> pd.DataFrame:
    """STUB -- human AD residual-age OLS (Fig 7e).

    TODO(gap): no source implementation found. Expected to take an AnnData/DataFrame
    of per-sample or per-spot tAge predictions (produced by the `ad_human` conversion
    in R/rds_to_h5ad.R, run through stage.pipeline, e.g. `full_nonoverlap_mp_pipeline`)
    plus AD-diagnosis/covariate metadata, and fit an OLS model of residual age (tAge
    minus some age-associated baseline) against diagnosis/covariates. Exact model
    specification unconfirmed -- see INVENTORY.md gap list item 6.
    """
    raise NotImplementedError(
        "residual_age_ols has no source implementation -- see this function's "
        "docstring and stAge-release/INVENTORY.md gap list item 6 (Human AD "
        "residual-age OLS)."
    )
