"""Fig 7f-h -- Breast cancer tumor vs. non-tumor tAge comparison (one figure, panels f-h).

STATUS: gap stub. The tumor-vs-non-tumor comparison analysis itself was not located
anywhere in the audited codebase (see stAge-release/INVENTORY.md, "Analyses in the
paper with NO code found", item 7).

What DOES exist: stage-release/R/rds_to_h5ad.R's `convert_breast_cancer_liver()`
block is real, live Seurat-RDS-to-h5ad conversion code for a dataset the original
codebase itself labelled "LIVER BREAST CANCER" (per-sample subsetting, no
tumor/non-tumor factor present in the conversion step itself) -- this is the
data-preparation starting point for this figure, not the analysis itself. Run it via:

    Rscript stage-release/R/rds_to_h5ad.R --dataset-type breast_cancer_liver \\
        --input-rds <path-to-source.rds> --out-dir <h5ad-output-dir>

before this script can be written for real. Note the conversion step has no
tumor/non-tumor label -- that classification (however it's derived, e.g. from a
metadata column not exercised by the conversion function, or from spatial/marker-gene
annotation) is itself unlocated and would need to be established before a
tumor-vs-non-tumor comparison could be written.

Do NOT fill in a fabricated tumor-vs-non-tumor test here -- the paper's actual
comparison method (which statistic, which tumor annotation) was not confirmed against
any source in this codebase.
"""

from __future__ import annotations

import pandas as pd


def tumor_vs_nontumor_tage(adata, *args, **kwargs) -> pd.DataFrame:
    """STUB -- breast cancer tumor vs. non-tumor tAge comparison (Fig 7f-h).

    TODO(gap): no source implementation found. Expected to take per-spot/per-metapixel
    tAge predictions from the "LIVER BREAST CANCER" dataset (produced via
    R/rds_to_h5ad.R --dataset-type breast_cancer_liver, then stage.pipeline) plus a
    tumor/non-tumor region label, and compare tAge between the two groups. Neither the
    tumor/non-tumor labeling method nor the comparison statistic is confirmed -- see
    INVENTORY.md gap list item 7.
    """
    raise NotImplementedError(
        "tumor_vs_nontumor_tage has no source implementation -- see this function's "
        "docstring and stAge-release/INVENTORY.md gap list item 7 (breast cancer "
        "tumor vs non-tumor)."
    )
