#!/usr/bin/env python
"""Fig 6b — Pseudobulk DESeq2, hotspot vs coldspot.

Python driver: loads per-sample tAge predictions, classifies hotspot/normal/coldspot
via Getis-Ord Gi* (stage.hotspots, author-confirmed canonical parameters: k=8, 999
perms, BH-FDR<0.05, |z|>1), pseudobulks raw counts by (sample, aging_type), writes one
CSV per tissue in the `<sample_id>__<condition>` column format that
`R/deseq2_hotspot_coldspot.R` expects, and (optionally) invokes that script directly.

Source (EXTRACTED, not reconstructed): v_pipeline/dstream_loop.ipynb, the cell titled
"PSEUDOBULKING HOTSPOTS AND COLDSPOTS PER SAMPLE (AND USING DESEQ2)" (~lines 2042-2188
of the nbconvert'ed script) -- `pseudobulk_adata()`, the main tissue loop, and the
n_hot/n_cold >= 2 replicate-count gate are all transcribed from there; the R portion of
`run_deseq2()` was moved into `R/deseq2_hotspot_coldspot.R` (see that file's header).

IMPORTANT discrepancy, flagged not silently resolved (found while extracting this
code): `FigS7_methods.txt`'s "Hotspot-versus-coldspot differential expression" section
states pseudobulks were built from spot data "normalized to 10,000 total counts per
spot and log1p-transformed, then pseudobulked ... by summing". The actual source code
in this cell does NOT do that -- `pseudobulk_adata()` sums `adata.X` directly and casts
the result to int (`.astype(int)`), which only makes sense as a DESeq2 input if `X`
already holds raw integer counts at this point (consistent with a note elsewhere in the
codebase, in celltype_hotspot_tAge.ipynb, that `adata.X` holds "the original spot-level
raw counts" after the main pipeline's `propagate_into_pixel_level` step). Summing
log1p-normalized floats and casting to int would destroy the data (nearly everything
would truncate to 0), so that reading of the methods text cannot be what the code
actually does. This script follows the CODE (raw-count summation, matching what DESeq2
expects) rather than the methods-text description, per "preserve behavior exactly" --
but the mismatch between the methods write-up and the code is real and should be
reconciled by the author (either the methods text needs correcting, or there's a
normalize/log1p step applied upstream of this cell, in a different notebook, that this
extraction didn't capture).
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
from scipy.sparse import issparse

from stage.hotspots import classify_hotspots, K_KNN, N_PERMS, FDR_ALPHA, GI_Z_THRESH

TAGE_COL = 'tAge_SM'
AGING_TYPE_COL = 'aging_type'


def default_tissue_key(fname: str) -> str:
    """Default tissue-grouping key, transcribed verbatim from the original notebook's
    `k.split('_')[2]` (its own comment says "Adjust if needed" -- this is fragile and
    filename-convention-specific, not a robust general rule; pass --tissue-key-regex to
    override for a different filename convention)."""
    parts = fname.replace('.h5ad', '').split('_')
    return parts[2] if len(parts) > 2 else fname


def pseudobulk_by_sample_and_aging_type(combined_adata: ad.AnnData) -> pd.DataFrame:
    """genes x (sample__condition) raw-count pseudobulk matrix. Verbatim port of the
    source `pseudobulk_adata(adata, groupby='aging_type')` -- sums `adata.X` (raw
    counts) directly, no normalization step (see module docstring)."""
    X = combined_adata.X
    X = X.toarray() if issparse(X) else np.asarray(X)
    df = pd.DataFrame(X, index=combined_adata.obs_names, columns=combined_adata.var_names)
    meta = combined_adata.obs[[AGING_TYPE_COL, 'sample']].copy()
    meta['sample_group'] = meta['sample'].astype(str) + '__' + meta[AGING_TYPE_COL].astype(str)
    return df.groupby(meta['sample_group']).sum().T  # genes x samples


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--preds-dir', required=True, help="Directory of per-sample h5ad files with tAge_SM + obsm['spatial'] + raw counts in .X")
    p.add_argument('--file-pattern', default='pred', help="Only files starting with this string are loaded [default: %(default)s]")
    p.add_argument('--out-dir', required=True)
    p.add_argument('--deseq2-script', default=None,
                    help="Path to R/deseq2_hotspot_coldspot.R. If given, this script also invokes "
                         "`Rscript <deseq2-script> --counts <per-tissue CSV> --out <results CSV>` per tissue. "
                         "If omitted, only the pseudobulk count CSVs are written.")
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    preds_per_file = {}
    for fname in sorted(os.listdir(args.preds_dir)):
        if not fname.startswith(args.file_pattern) or not fname.endswith('.h5ad'):
            continue
        adata = sc.read_h5ad(os.path.join(args.preds_dir, fname))
        if TAGE_COL not in adata.obs.columns or 'spatial' not in adata.obsm:
            continue
        preds_per_file[fname] = adata

    unique_tissues = pd.Series([default_tissue_key(k) for k in preds_per_file]).unique()
    print("Sample sets that will be analysed:", *unique_tissues, sep="\n  - ")

    for tissue in unique_tissues:
        print(f"\nAnalyzing {tissue} ...")
        preds_per_tissue = {k: a for k, a in preds_per_file.items() if default_tissue_key(k) == tissue}

        adatas = []
        for k, adata in preds_per_tissue.items():
            adata = adata.copy()
            classify_hotspots(adata, value_col=TAGE_COL, k=K_KNN, n_perms=N_PERMS,
                               fdr_alpha=FDR_ALPHA, z_thresh=GI_Z_THRESH, out_col=AGING_TYPE_COL)
            adata.var_names_make_unique()
            adata.obs_names = [f"{k}_{i}" for i in adata.obs_names]
            adata.obs['sample'] = k
            adatas.append(adata)

        if not adatas:
            print(f"  no samples for {tissue}, skipping.")
            continue

        combined_adata = ad.concat(adatas, label='sample', index_unique=None)
        pb_counts = pseudobulk_by_sample_and_aging_type(combined_adata).astype(int)

        # Keep only hotspot/coldspot columns -- R script expects exactly these two conditions.
        keep_cols = [c for c in pb_counts.columns if c.endswith('__hotspot') or c.endswith('__coldspot')]
        pb_counts = pb_counts[keep_cols]

        n_hot = sum(c.endswith('__hotspot') for c in pb_counts.columns)
        n_cold = sum(c.endswith('__coldspot') for c in pb_counts.columns)
        if n_hot < 2 or n_cold < 2:
            print(f"  not enough replicates (n_hotspot={n_hot}, n_coldspot={n_cold}). Skipping (matches original's own skip gate).")
            continue

        counts_path = os.path.join(args.out_dir, f'{tissue}_hotspot_coldspot_pseudobulk_counts.csv')
        pb_counts.to_csv(counts_path)
        print(f"  wrote {counts_path}  ({pb_counts.shape[0]} genes x {pb_counts.shape[1]} samples)")

        if args.deseq2_script:
            results_path = os.path.join(args.out_dir, f'{tissue}_hotspot_vs_coldspot_deseq2.csv')
            cmd = ['Rscript', args.deseq2_script, '--counts', counts_path, '--out', results_path]
            print(f"  running: {' '.join(cmd)}")
            subprocess.run(cmd, check=True)


if __name__ == '__main__':
    main()
