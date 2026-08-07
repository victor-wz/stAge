#!/usr/bin/env python
"""Fig 1c -- Aggregation-strategy benchmark: metaspots (non-overlapping Leiden
metapixels) vs. k=5/10/20-NN spatial smoothing, compared on spatial tAge maps,
per-unit count distribution, and gene dropout fraction (Panels A-C only).

Source: v_pipeline/suppfig_smoothing_comparison.py -- confirmed by Phase 1/2 audit
to already be a clean standalone script covering exactly Panels A-C (the full
notebook, suppfig_smoothing_comparison.ipynb, has additional ablation panels --
Transcriptomic-Leiden / Grid / Anatomical strategies vs. SpatialGroup -- not
reproduced here, matching the original two-file split documented in CLAUDE.md).
That source script already imported cleanly from st_utils (no inline duplicate
pipeline logic) -- ported 1:1 here onto the equivalent stage.* modules.

IMPORTANT SIGNATURE NOTE: the original script calls `get_scaled_counts(df,
clock_model, 'symbol')` -- a 3-positional-arg call that was correct under
st_utils.py's ORIGINAL signature `get_scaled_counts(df, clock_model,
original_ids='symbol')` (no `ncbi_reference_path` parameter at all; it silently
fell back to a hard-coded default path inside `preprocess_counts`). The
refactored `stage.preprocessing.get_scaled_counts` now REQUIRES
`ncbi_reference_path` as an explicit positional/keyword argument (3rd position,
before `original_ids`) -- ported below with `ncbi_reference_path` threaded
through explicitly (via `--ncbi-reference-path`), NOT reproducing the original
3-arg call verbatim, since doing so would now silently bind 'symbol' into the
path slot instead. This is a required consequence of removing the hard-coded
absolute path per this release's "replace absolute paths with CLI args" rule,
not a statistical/behavioral change.

Every other computation (metaspot construction, spatial-neighbor-graph
smoothing via a `k`-NN self-loop adjacency matmul, filter_genes/no-filter
policy per method, dropout-fraction accounting, the 3-panel figure) is
preserved exactly from the source script.
"""

from __future__ import annotations

import argparse
import math
import os
import uuid
import warnings

warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from scipy.sparse import issparse, csr_matrix, eye as speye
import scanpy as sc
# dask/dask.dataframe must be imported before squidpy on some dask/dask-expr version
# pairings, or squidpy's transitive `spatialdata` -> `dask.dataframe` import raises
# NotImplementedError -- see stage/metapixels.py's module docstring for the full
# explanation (same fix applied here since this script also imports squidpy directly).
import dask
import dask.dataframe as dd
import squidpy as sq
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import TwoSlopeNorm
import seaborn as sns

from stage.metapixels import non_overlapping_MPs
from stage.preprocessing import filter_genes, get_scaled_counts
from stage.clock import final_clock_preparation, predict_age, propagate_into_pixel_level

YOUNG_COLOR = '#4C72B0'
OLD_COLOR = '#C44E52'


def _run_clock(ad, clock_model, ncbi_reference_path):
    """Run scaled-diff clock on an AnnData; return it with tAge_SM in obs."""
    X = ad.X.toarray() if issparse(ad.X) else np.asarray(ad.X)
    df = pd.DataFrame(X.T, index=ad.var_names, columns=ad.obs.index)
    df.index.name = 'geneID'

    counts_scaled = get_scaled_counts(df, clock_model, ncbi_reference_path, 'symbol')
    preprocessed = final_clock_preparation(counts_scaled, clock_model, diff_suffix='young')
    preds = predict_age(preprocessed, clock_model)

    ad.obs['tAge_SM'] = (
        pd.Series(preds, index=preprocessed.columns)
        .loc[ad.obs.index] * 48
    )
    return ad


def run_clock_metaspot(merged, clock_model, ncbi_reference_path):
    """
    Metaspot pipeline: apply filter_genes (>=10 counts in >=20% of metaspots),
    then run the clock. Returns (adata_with_tAge_SM, dropout_fraction).
    Matches integrated_stAge.ipynb / full_nonoverlap_mp_pipeline (lower_res=True).
    """
    n_vars_pre = merged.n_vars
    ad = filter_genes(merged)
    dropout_frac = 1.0 - ad.n_vars / n_vars_pre
    return _run_clock(ad, clock_model, ncbi_reference_path), dropout_frac


def run_clock_smooth(merged, clock_model, ncbi_reference_path):
    """
    Smoothing pipeline: skip filter_genes (too aggressive for spot-level pseudobulks),
    run the clock directly on all genes.
    Matches smoothing_integrated_stAge.ipynb which comments out filter_genes.

    For dropout reporting only, also compute what fraction would have been removed.
    Returns (adata_with_tAge_SM, dropout_fraction_if_filter_had_been_applied).
    """
    mat = merged.X
    if not issparse(mat):
        mat = csr_matrix(mat)
    threshold = math.ceil(0.2 * merged.n_obs)
    gene_counts = np.array((mat >= 10).sum(axis=0)).flatten()
    n_passing = int((gene_counts >= threshold).sum())
    dropout_frac = 1.0 - n_passing / merged.n_vars

    ad = merged.copy()
    return _run_clock(ad, clock_model, ncbi_reference_path), dropout_frac


def smooth_aggregate(adata, k, age_group):
    """
    Aggregate raw counts by summing each spot together with its k spatial neighbors.
    Returns a spot-level AnnData with unique obs_names embedding 'young'/'old'.

    Spatial graph is built fresh here for each k, so repeated calls with different k
    on the same adata are safe (each call overwrites obsp in adata, but only reads
    the result immediately and does not depend on a prior graph).
    """
    sq.gr.spatial_neighbors(adata, coord_type='generic', n_neighs=k)

    X = adata.X.toarray() if issparse(adata.X) else np.asarray(adata.X, dtype=float)
    conn = adata.obsp['spatial_connectivities']

    # Binary adjacency + self-loop so each spot contributes its own counts
    adj_with_self = (conn > 0).astype(float) + speye(conn.shape[0], format='csr')
    X_agg = np.asarray(adj_with_self @ X)  # n_spots x n_genes

    obs_names = [f"{age_group}.pixel.{i}.{uuid.uuid4()}" for i in range(adata.n_obs)]
    smoothed = sc.AnnData(
        X=X_agg,
        obs=pd.DataFrame(index=obs_names),
        obsm={'spatial': adata.obsm['spatial'].copy()},
        var=adata.var.copy(),
    )
    smoothed.obs['cumulative_coverage'] = X_agg.sum(axis=1)
    return smoothed


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--raw-dir', required=True,
                    help="Directory of raw per-sample .h5ad files (e.g. spatial_aging/data/immunoglobulin)")
    p.add_argument('--tissue-prefix', default='Hippocampus',
                    help="Filename prefix selecting which tissue's Young/Old pair to compare (default: Hippocampus, matching the source script)")
    p.add_argument('--clock-path', required=True,
                    help="Path to EN_Chronoage_Mouse_All_WT_scaleddiff.pkl (or equivalent scaled-diff clock)")
    p.add_argument('--ncbi-reference-path', required=True,
                    help="Path to a *.gene_info reference file (e.g. Mus_musculus.gene_info)")
    p.add_argument('--out-dir', required=True)
    p.add_argument('--leiden-res', type=float, default=2)
    p.add_argument('--mp-threshold', type=float, default=1_000)
    p.add_argument('--smoothing-ks', type=int, nargs='+', default=[5, 10, 20])
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    out_prefix = os.path.join(args.out_dir, 'fig1c_aggregation_benchmark')

    sns.set_style('ticks')
    plt.rcParams.update({
        'font.size': 11, 'axes.labelsize': 12,
        'xtick.labelsize': 10, 'ytick.labelsize': 10, 'axes.linewidth': 0.8,
    })

    # ─── Select representative samples ───────────────────────────────────────
    tissue_files = sorted(
        f for f in os.listdir(args.raw_dir)
        if f.startswith(args.tissue_prefix) and f.endswith('.h5ad')
    )
    young_fname = next(f for f in tissue_files if '_Y_' in f)
    old_fname = next(f for f in tissue_files if '_O_' in f)
    print(f'Young sample : {young_fname}')
    print(f'Old   sample : {old_fname}')

    print('\nLoading H5AD files...')
    adata_y = sc.read(os.path.join(args.raw_dir, young_fname))
    adata_o = sc.read(os.path.join(args.raw_dir, old_fname))
    print(f'  Young: {adata_y.n_obs} spots  |  Old: {adata_o.n_obs} spots')

    clock_model = joblib.load(args.clock_path)
    print(f'Clock: {os.path.basename(args.clock_path)}')

    # ─── 1. METASPOT APPROACH ─────────────────────────────────────────────────
    print(f'\n=== Metaspot approach (Leiden res={args.leiden_res}) ===')

    mp_y = non_overlapping_MPs(adata_y, age_group='young', n_neighs=20, resolution=args.leiden_res)
    mp_o = non_overlapping_MPs(adata_o, age_group='old', n_neighs=20, resolution=args.leiden_res)

    mp_y = mp_y[mp_y.obs['cumulative_coverage'] >= args.mp_threshold].copy()
    mp_o = mp_o[mp_o.obs['cumulative_coverage'] >= args.mp_threshold].copy()
    mp_y.obs['File'] = young_fname
    mp_o.obs['File'] = old_fname
    print(f'  Metaspots — young: {mp_y.n_obs}  old: {mp_o.n_obs}')

    merged_mp = sc.concat([mp_y, mp_o], join='outer')
    merged_mp.obsm['spatial'] = np.vstack([mp_y.obsm['spatial'], mp_o.obsm['spatial']])
    merged_mp.var_names_make_unique()
    merged_mp.obs_names_make_unique()

    ad_mp, mp_dropout = run_clock_metaspot(merged_mp, clock_model, args.ncbi_reference_path)

    mp_y_pred = ad_mp[ad_mp.obs['File'] == young_fname].copy()
    mp_o_pred = ad_mp[ad_mp.obs['File'] == old_fname].copy()
    print(f'  tAge_SM — young μ={mp_y_pred.obs.tAge_SM.mean():.2f}  '
          f'old μ={mp_o_pred.obs.tAge_SM.mean():.2f}')
    print(f'  Gene dropout: {mp_dropout:.1%}')

    mp_y_spatial = propagate_into_pixel_level(mp_y_pred, adata_y, 'young', obs_to_propagate=['tAge_SM'])
    mp_o_spatial = propagate_into_pixel_level(mp_o_pred, adata_o, 'old', obs_to_propagate=['tAge_SM'])
    print(f'  Propagated to spots — young: {mp_y_spatial.n_obs}  old: {mp_o_spatial.n_obs}')

    # ─── 2. SMOOTHING APPROACHES ──────────────────────────────────────────────
    smooth_results = {}

    for k in args.smoothing_ks:
        print(f'\n=== Smoothing k={k} ===')

        sm_y = smooth_aggregate(adata_y, k, 'young')
        sm_o = smooth_aggregate(adata_o, k, 'old')
        sm_y.obs['File'] = young_fname
        sm_o.obs['File'] = old_fname

        merged_sm = sc.concat([sm_y, sm_o], join='outer')
        merged_sm.obsm['spatial'] = np.vstack([sm_y.obsm['spatial'], sm_o.obsm['spatial']])
        merged_sm.var_names_make_unique()
        merged_sm.obs_names_make_unique()

        ad_sm, sm_dropout = run_clock_smooth(merged_sm, clock_model, args.ncbi_reference_path)

        sm_y_pred = ad_sm[ad_sm.obs['File'] == young_fname].copy()
        sm_o_pred = ad_sm[ad_sm.obs['File'] == old_fname].copy()
        print(f'  tAge_SM — young μ={sm_y_pred.obs.tAge_SM.mean():.2f}  '
              f'old μ={sm_o_pred.obs.tAge_SM.mean():.2f}')
        print(f'  Gene dropout: {sm_dropout:.1%}')

        smooth_results[k] = dict(young=sm_y_pred, old=sm_o_pred, dropout=sm_dropout)

    # ─── 3. Summary stats for plots B and C ──────────────────────────────────
    method_labels = ['Metaspot'] + [f'k={k}' for k in args.smoothing_ks]

    counts_per_unit = {
        'Metaspot': np.concatenate([
            mp_y_pred.obs['cumulative_coverage'].values,
            mp_o_pred.obs['cumulative_coverage'].values,
        ])
    }
    for k in args.smoothing_ks:
        counts_per_unit[f'k={k}'] = np.concatenate([
            smooth_results[k]['young'].obs['cumulative_coverage'].values,
            smooth_results[k]['old'].obs['cumulative_coverage'].values,
        ])

    mean_counts = {lbl: counts_per_unit[lbl].mean() for lbl in method_labels}

    dropouts = {'Metaspot': mp_dropout}
    for k in args.smoothing_ks:
        dropouts[f'k={k}'] = smooth_results[k]['dropout']

    # ─── 4. FIGURE ─────────────────────────────────────────────────────────────
    print('\nGenerating figure...')

    all_tage = np.concatenate([
        mp_y_pred.obs.tAge_SM.values,
        mp_o_pred.obs.tAge_SM.values,
        *[smooth_results[k]['young'].obs.tAge_SM.values for k in args.smoothing_ks],
        *[smooth_results[k]['old'].obs.tAge_SM.values for k in args.smoothing_ks],
    ])
    vlim = max(abs(np.nanpercentile(all_tage, 2)), abs(np.nanpercentile(all_tage, 98)))
    norm = TwoSlopeNorm(vmin=-vlim, vcenter=0, vmax=vlim)
    cmap = plt.cm.coolwarm

    fig = plt.figure(figsize=(14, 12))
    fig.subplots_adjust(left=0.06, right=0.97, top=0.94, bottom=0.07, hspace=0.50, wspace=0.30)

    outer = gridspec.GridSpec(3, 1, figure=fig, hspace=0.50, height_ratios=[2.8, 1.3, 1.3])

    # ── Panel A: spatial tAge maps ──────────────────────────────────────────
    n_cols = 1 + len(args.smoothing_ks)
    gs_a = gridspec.GridSpecFromSubplotSpec(
        2, n_cols + 1, subplot_spec=outer[0], wspace=0.06, hspace=0.12,
        width_ratios=[1] * n_cols + [0.05],
    )

    col_specs = [('Metaspot', mp_y_spatial, mp_o_spatial)]
    for k in args.smoothing_ks:
        col_specs.append((f'k={k}', smooth_results[k]['young'], smooth_results[k]['old']))

    row_labels = ['Young', 'Old']
    row_colors = [YOUNG_COLOR, OLD_COLOR]

    for col_i, (method_name, adata_y_plot, adata_o_plot) in enumerate(col_specs):
        pt_size = 3
        for row_i, (ad_plot, row_label, row_color) in enumerate(
            zip([adata_y_plot, adata_o_plot], row_labels, row_colors)
        ):
            ax = fig.add_subplot(gs_a[row_i, col_i])
            coords = ad_plot.obsm['spatial']
            vals = ad_plot.obs['tAge_SM'].values

            ax.scatter(coords[:, 0], coords[:, 1], c=vals, cmap=cmap, norm=norm,
                       s=pt_size, linewidths=0, rasterized=True)
            ax.invert_yaxis()
            ax.set_aspect('equal')
            ax.axis('off')

            if row_i == 0:
                ax.set_title(f'{method_name}\n{mean_counts[method_name]:.0f} cts/unit', fontsize=10, pad=4)
            if col_i == 0:
                ax.set_ylabel(row_label, fontsize=10, color=row_color, fontweight='bold',
                              rotation=0, labelpad=38, va='center')

    cax = fig.add_subplot(gs_a[:, -1])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax, label='tAge_SM')
    cbar.ax.tick_params(labelsize=8)

    fig.text(0.005, 0.965, 'A', fontsize=14, fontweight='bold', transform=fig.transFigure, va='top')

    # ── Panel B: count distribution per pseudobulk unit ────────────────────
    ax_b = fig.add_subplot(outer[1])

    df_counts = pd.DataFrame({
        'Method': np.repeat(method_labels, [len(counts_per_unit[m]) for m in method_labels]),
        'log10_counts': np.log10(1 + np.concatenate([counts_per_unit[m] for m in method_labels])),
    })

    palette_b = {lbl: '#888888' for lbl in method_labels}
    palette_b['Metaspot'] = '#2d6a4f'

    sns.violinplot(data=df_counts, x='Method', y='log10_counts', order=method_labels,
                    palette=palette_b, inner='box', cut=0, linewidth=0.8, ax=ax_b)
    ax_b.set_xlabel('Aggregation method')
    ax_b.set_ylabel('log₁₀(total counts per unit)')
    ax_b.set_title('B — Count distribution per pseudobulk unit', fontsize=11, loc='left', pad=4)
    sns.despine(ax=ax_b)

    fig.text(0.005, 0.60, 'B', fontsize=14, fontweight='bold', transform=fig.transFigure, va='top')

    # ── Panel C: gene dropout fraction ──────────────────────────────────────
    ax_c = fig.add_subplot(outer[2])

    dropout_vals = [dropouts[m] for m in method_labels]
    bar_colors = ['#2d6a4f'] + ['#888888'] * len(args.smoothing_ks)

    bars = ax_c.bar(method_labels, dropout_vals, color=bar_colors, edgecolor='black',
                     linewidth=0.7, width=0.6)
    ax_c.axhline(mp_dropout, color='#2d6a4f', linestyle='--', linewidth=1.2,
                 label=f'Metaspot baseline ({mp_dropout:.1%})')
    for bar, val in zip(bars, dropout_vals):
        ax_c.text(bar.get_x() + bar.get_width() / 2, val + 0.003, f'{val:.1%}',
                   ha='center', va='bottom', fontsize=9)

    ax_c.set_xlabel('Aggregation method')
    ax_c.set_ylabel('Gene dropout fraction')
    ax_c.set_ylim(0, max(dropout_vals) * 1.20)
    ax_c.set_title('C — Gene dropout fraction vs aggregation method', fontsize=11, loc='left', pad=4)
    ax_c.legend(fontsize=9, frameon=False)
    sns.despine(ax=ax_c)

    fig.text(0.005, 0.32, 'C', fontsize=14, fontweight='bold', transform=fig.transFigure, va='top')

    # ─── Save ────────────────────────────────────────────────────────────────
    for ext in ('pdf', 'png'):
        path = f'{out_prefix}.{ext}'
        fig.savefig(path, dpi=300, bbox_inches='tight')
        print(f'Saved: {path}')

    plt.close(fig)
    print('\nDone.')


if __name__ == '__main__':
    main()
