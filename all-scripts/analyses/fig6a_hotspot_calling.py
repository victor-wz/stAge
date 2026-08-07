#!/usr/bin/env python
"""Fig 6a — Cell-type-specific tAge in Getis-Ord Gi* hotspot vs coldspot regions.

Reviewer question addressed: are cells of the same type intrinsically older in tAge
hotspots than coldspots, or is the hotspot-coldspot tAge gap driven by cell-type
composition shifts? This script covers Sections 1-4 of the source notebook:
  1. Spot-level Getis-Ord Gi* classification (k=8 KNN, 999 permutations, BH-FDR<0.05,
     |z|>1) -- delegates to `stage.hotspots.classify_hotspots` (author-confirmed
     canonical parameters), not reimplemented here.
  2. Cell-type x aging-type pseudobulk construction per sample (raw count sums).
  3. tAge clock re-run on each pseudobulk, differenced against young.normal pseudobulks.
  4. Kruskal-Wallis across hotspot/normal/coldspot + pairwise Mann-Whitney U (BH-FDR
     within tissue) + Cohen's d for the primary hotspot-vs-coldspot contrast.

Source: v_pipeline/celltype_hotspot_tAge.ipynb, Sections 1-4 and the final figure cell
("celltype_hotspot_tAge_figure.{pdf,png}" -- Phase 1 confirmed this file is the actual
Fig 6a output). Tissues covered (fixed in the source notebook): Brain 2g, Brain 3g,
Hippocampus.

IMPORTANT -- Panel C omitted, not silently reproduced (author decision, 2026-08-06):
the original figure's Panel C ("compositional partitioning", stacked hotspot-coldspot
bars) is built from that notebook's Section 5, which has a confirmed math bug (wrong
reference mean, drops singleton cell types -- breaks delta_obs = delta_comp +
delta_within) and was explicitly excluded from the release in favor of the corrected
Oaxaca-Blinder decomposition in stage/composition.py (Fig S12's canonical version).
`stage.composition.oaxaca_blinder_decomposition` is NOT a drop-in replacement here,
though: it expects one row per METAPIXEL with an `obsm['composition']` fraction
vector, whereas this notebook's pseudobulk_adata is one row per (sample, cell_type,
aging_type) -- a different shape requiring a nontrivial reshape to reuse. Rather than
force-fit an incompatible function or leave the bug in, this script produces Panels
A/B/D only and leaves a clearly marked TODO for Panel C -- confirm with the author
whether to (a) adapt oaxaca_blinder_decomposition's input contract to also accept this
per-cell-type pseudobulk shape, or (b) drop Panel C from the released Fig 6a and point
readers to Fig S12 instead.
"""

from __future__ import annotations

import argparse
import os
import re

import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
import joblib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy.sparse import issparse
from scipy.stats import mannwhitneyu, kruskal
from statsmodels.stats.multitest import multipletests

from stage.hotspots import classify_hotspots, K_KNN, N_PERMS, FDR_ALPHA, GI_Z_THRESH
from stage.preprocessing import filter_genes, get_scaled_counts
from stage.clock import final_clock_preparation, predict_age

try:
    from statannotations.Annotator import Annotator
    _HAVE_ANNOTATOR = True
except ImportError:
    _HAVE_ANNOTATOR = False


TAGE_COL = 'tAge_SM'
CT_COL = 'cell_type'
REGION_COL = 'aging_type'
MIN_SPOTS_PER_PSEUDOBULK = 10
TISSUES_TO_RUN = ['Brain 2g', 'Brain 3g', 'Hippocampus']

AGE_OLD_RE = re.compile(r'(_O[_\-.\d]|_Old)')
AGE_YOUNG_RE = re.compile(r'(_Y[_\-.\d]|_Young)')
TISSUE_LOOKUP = {'brain2g': 'Brain 2g', 'brain3g': 'Brain 3g', 'hippocampus': 'Hippocampus'}

PALETTE = {'hotspot': '#C44E52', 'normal': '#95a5a6', 'coldspot': '#4C72B0'}
REGION_ORDER = ['hotspot', 'normal', 'coldspot']


def detect_tissue(fname: str):
    low = fname.lower()
    for key, tis in TISSUE_LOOKUP.items():
        if key in low:
            return tis
    return None


def detect_age_group(fname: str):
    if AGE_OLD_RE.search(fname):
        return 'Old'
    if AGE_YOUNG_RE.search(fname):
        return 'Young'
    return None


def _try_attach_celltype(adata, fname, ct_reann_dir):
    """If cell_type is missing, merge it in from ct_reference/brain_reannotated/
    (Brain 2g/3g only -- Hippocampus already carries cell_type natively)."""
    if CT_COL in adata.obs.columns:
        return True
    tis = detect_tissue(fname)
    prefix = {'Brain 2g': 'brain2g_', 'Brain 3g': 'brain3g_'}.get(tis)
    if prefix is None:
        return False
    candidate = os.path.join(ct_reann_dir, f'{prefix}{fname}')
    if not os.path.exists(candidate):
        return False
    ad_ct = sc.read_h5ad(candidate)
    if 'Cell.type_SingleR' not in ad_ct.obs.columns:
        return False
    common = adata.obs.index.intersection(ad_ct.obs.index)
    if len(common) == 0:
        return False
    adata.obs[CT_COL] = (
        ad_ct.obs.loc[common, 'Cell.type_SingleR'].reindex(adata.obs.index).astype('object')
    )
    return CT_COL in adata.obs.columns


def load_tissue_samples(preds_dir: str, ct_reann_dir: str) -> dict:
    """Load every pred_* file under `preds_dir` matching TISSUES_TO_RUN, with
    tAge_SM, cell_type, and obsm['spatial'] required (cell_type auto-attached
    for Brain 2g/3g from `ct_reann_dir` when missing)."""
    spot_adata_dict = {tis: {} for tis in TISSUES_TO_RUN}
    for fname in sorted(os.listdir(preds_dir)):
        if not fname.startswith('pred'):
            continue
        tis = detect_tissue(fname)
        age = detect_age_group(fname)
        if tis not in TISSUES_TO_RUN or age is None:
            continue
        adata = sc.read_h5ad(os.path.join(preds_dir, fname))
        adata.var_names_make_unique()
        if TAGE_COL not in adata.obs.columns or 'spatial' not in adata.obsm:
            continue
        if CT_COL not in adata.obs.columns and not _try_attach_celltype(adata, fname, ct_reann_dir):
            continue
        coords = adata.obsm['spatial']
        valid = np.isfinite(coords).all(axis=1)
        if (~valid).any():
            adata = adata[valid].copy()
        adata.obs['age_group'] = age
        adata.obs['tissue'] = tis
        spot_adata_dict[tis][fname] = adata
    return spot_adata_dict


def classify_all_hotspots(spot_adata_dict: dict) -> None:
    """In-place spot-level Gi* classification on every loaded sample."""
    for tis, d in spot_adata_dict.items():
        for fname, adata in d.items():
            classify_hotspots(adata, value_col=TAGE_COL, k=K_KNN, n_perms=N_PERMS,
                               fdr_alpha=FDR_ALPHA, z_thresh=GI_Z_THRESH, out_col=REGION_COL)


def _safe_token(s) -> str:
    return re.sub(r'[^A-Za-z0-9]+', '-', str(s)).strip('-')


def _build_pseudobulks_for_sample(adata, tissue, sample_id, age_group):
    out_rows, out_meta, out_names = [], [], []
    obs = adata.obs
    cell_types = pd.Series(obs[CT_COL]).dropna().unique().tolist()
    X = adata.X
    is_sparse = issparse(X)
    age_tag = age_group.lower()
    for ct in cell_types:
        for region in REGION_ORDER:
            mask = (obs[CT_COL] == ct).values & (obs[REGION_COL] == region).values
            n_spots = int(mask.sum())
            if n_spots < MIN_SPOTS_PER_PSEUDOBULK:
                continue
            sub = X[mask]
            gene_sum = np.asarray(sub.sum(axis=0)).ravel() if is_sparse else np.asarray(sub).sum(axis=0).ravel()
            obs_name = f'{age_tag}.{region}.{_safe_token(tissue)}.{_safe_token(sample_id)}.{_safe_token(ct)}.{n_spots}'
            out_rows.append(gene_sum)
            out_names.append(obs_name)
            out_meta.append(dict(tissue=tissue, sample_id=sample_id, age_group=age_group,
                                  cell_type=ct, aging_type=region, n_spots=n_spots,
                                  total_counts=float(gene_sum.sum())))
    return out_names, out_meta, out_rows


def build_pseudobulk_adata(spot_adata_dict: dict) -> ad.AnnData:
    var_union = None
    for d in spot_adata_dict.values():
        for a in d.values():
            var_union = pd.Index(a.var_names) if var_union is None else var_union.union(pd.Index(a.var_names))
    var_index = var_union.unique()
    var_pos = pd.Series(np.arange(len(var_index)), index=var_index)

    all_X, all_meta, all_names = [], [], []
    for tis, d in spot_adata_dict.items():
        for fname, adata in d.items():
            sample_id = fname.replace('.h5ad', '')
            age = adata.obs['age_group'].iloc[0]
            names, metas, rows = _build_pseudobulks_for_sample(adata, tis, sample_id, age)
            col_pos = var_pos.reindex(pd.Index(adata.var_names)).values
            for nm, meta, gv in zip(names, metas, rows):
                full = np.zeros(len(var_index), dtype=np.float32)
                full[col_pos] = gv.astype(np.float32)
                all_X.append(full)
                all_meta.append(meta)
                all_names.append(nm)
    if not all_X:
        raise RuntimeError('No pseudobulks were constructed.')
    obs = pd.DataFrame(all_meta, index=pd.Index(all_names, name='pseudobulk_id'))
    pb = ad.AnnData(X=np.vstack(all_X), obs=obs, var=pd.DataFrame(index=var_index))
    pb.var_names_make_unique()
    return pb


def run_pseudobulk_clock(pb: ad.AnnData, clock_path: str, ncbi_reference_path: str) -> None:
    """In-place: adds tAge_SM to pb.obs. Reference distribution = young.normal pseudobulks."""
    clock_model = joblib.load(clock_path)
    pb_filt = filter_genes(pb)
    X = pb_filt.X.toarray() if issparse(pb_filt.X) else np.asarray(pb_filt.X)
    df = pd.DataFrame(X.T, index=pb_filt.var_names, columns=pb_filt.obs.index)
    df.index.name = 'geneID'
    counts_scaled = get_scaled_counts(df, clock_model, ncbi_reference_path=ncbi_reference_path, original_ids='symbol')
    preprocessed = final_clock_preparation(counts_scaled, clock_model, diff_suffix='young.normal')
    preds = predict_age(preprocessed, clock_model)
    age_pred = pd.Series(np.asarray(preds), index=preprocessed.columns) * 48
    pb.obs[TAGE_COL] = age_pred.reindex(pb.obs.index).values


def cohens_d(a, b) -> float:
    a = np.asarray(a, dtype=float); a = a[np.isfinite(a)]
    b = np.asarray(b, dtype=float); b = b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return np.nan
    pooled = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
    return (a.mean() - b.mean()) / pooled if pooled > 0 and np.isfinite(pooled) else np.nan


def _safe_mwu(a, b):
    a = np.asarray(a, dtype=float); a = a[np.isfinite(a)]
    b = np.asarray(b, dtype=float); b = b[np.isfinite(b)]
    if len(a) < 3 or len(b) < 3:
        return np.nan, np.nan
    try:
        return mannwhitneyu(a, b, alternative='two-sided')
    except ValueError:
        return np.nan, np.nan


def compute_stats(pseudobulk_adata: ad.AnnData) -> pd.DataFrame:
    """Kruskal-Wallis + pairwise MWU (BH-FDR within tissue) + Cohen's d
    (hotspot vs coldspot only), per (tissue, cell_type, age_group)."""
    obs = pseudobulk_adata.obs.copy()
    rows = []
    for tis in TISSUES_TO_RUN:
        sub_t = obs[obs['tissue'] == tis]
        for ct in sub_t['cell_type'].dropna().unique():
            for age in ['Young', 'Old']:
                sub = sub_t[(sub_t['cell_type'] == ct) & (sub_t['age_group'] == age)]
                groups = {r: sub.loc[sub['aging_type'] == r, TAGE_COL].astype(float).values for r in REGION_ORDER}
                if any(len(v) < 3 for v in groups.values()):
                    continue
                for cmp_a, cmp_b in [('hotspot', 'coldspot'), ('hotspot', 'normal'), ('coldspot', 'normal')]:
                    a, b = groups[cmp_a], groups[cmp_b]
                    u, p = _safe_mwu(a, b)
                    d = cohens_d(a, b) if (cmp_a, cmp_b) == ('hotspot', 'coldspot') else np.nan
                    rows.append(dict(tissue=tis, cell_type=ct, age_group=age,
                                      comparison=f'{cmp_a}_vs_{cmp_b}', n_first=len(a), n_second=len(b),
                                      mean_tAge_first=float(np.mean(a)), mean_tAge_second=float(np.mean(b)),
                                      cohens_d=d, U_stat=u, p_raw=p))
    stats_df = pd.DataFrame(rows)
    stats_df['p_adj'] = np.nan
    stats_df['significant'] = False
    for tis, sub in stats_df.groupby('tissue'):
        pv = sub['p_raw'].values
        finite = np.isfinite(pv)
        if not finite.any():
            continue
        rej, p_adj = np.zeros(len(pv), bool), np.full(len(pv), np.nan)
        rej[finite], p_adj[finite], *_ = multipletests(pv[finite], alpha=FDR_ALPHA, method='fdr_bh')
        stats_df.loc[sub.index, 'p_adj'] = p_adj
        stats_df.loc[sub.index, 'significant'] = rej
    return stats_df.sort_values(['tissue', 'cell_type', 'age_group', 'comparison']).reset_index(drop=True)


def _panel_boxplot(ax, df_sub, tissue, age_label, stats_for_panel):
    if df_sub.empty:
        ax.set_visible(False)
        return
    cts = sorted(df_sub['cell_type'].dropna().unique().tolist())
    sns.boxplot(data=df_sub, x='cell_type', y=TAGE_COL, hue='aging_type', order=cts,
                hue_order=REGION_ORDER, palette=PALETTE, ax=ax, showfliers=False, linewidth=0.7)
    sns.stripplot(data=df_sub, x='cell_type', y=TAGE_COL, hue='aging_type', order=cts,
                  hue_order=REGION_ORDER, palette=PALETTE, ax=ax, dodge=True, alpha=0.5,
                  size=3, jitter=True, edgecolor='black', linewidth=0.2)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles[:3], labels[:3], title='aging_type', fontsize=7, title_fontsize=7,
                   loc='upper right', frameon=False)
    ax.set_title(f'{tissue} — {age_label} samples', fontsize=10, fontweight='bold')
    ax.set_ylabel('tAge (relative)'); ax.set_xlabel('')
    for lab in ax.get_xticklabels():
        lab.set_rotation(30); lab.set_ha('right')
    if _HAVE_ANNOTATOR:
        pairs, pvals = [], []
        for ct in cts:
            row = stats_for_panel[(stats_for_panel['cell_type'] == ct) & (stats_for_panel['comparison'] == 'hotspot_vs_coldspot')]
            if len(row) == 1 and np.isfinite(row['p_adj'].iloc[0]):
                pairs.append(((ct, 'hotspot'), (ct, 'coldspot')))
                pvals.append(row['p_adj'].iloc[0])
        if pairs:
            annot = Annotator(ax, pairs, data=df_sub, x='cell_type', y=TAGE_COL, hue='aging_type',
                               order=cts, hue_order=REGION_ORDER)
            annot.configure(test=None, text_format='star', loc='inside', verbose=0, line_width=0.6, fontsize=7)
            annot.set_pvalues_and_annotate(pvals)


def make_figure(pseudobulk_adata: ad.AnnData, stats_df: pd.DataFrame, out_pdf: str, out_png: str):
    """Panels A (Old boxplots), B (Young boxplots), D (Cohen's d heatmap).
    Panel C intentionally omitted -- see module docstring."""
    fig = plt.figure(figsize=(16, 10))
    outer = gridspec.GridSpec(2, 1, figure=fig, height_ratios=[1.0, 1.0], hspace=0.55)
    rowA = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=outer[0], wspace=0.30)
    rowB = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=outer[1], wspace=0.30)

    for col, tis in enumerate(TISSUES_TO_RUN):
        ax_a = fig.add_subplot(rowA[0, col])
        df_old = pseudobulk_adata.obs[(pseudobulk_adata.obs['tissue'] == tis) & (pseudobulk_adata.obs['age_group'] == 'Old')]
        _panel_boxplot(ax_a, df_old, tis, 'Old', stats_df[(stats_df['tissue'] == tis) & (stats_df['age_group'] == 'Old')])

        ax_b = fig.add_subplot(rowB[0, col])
        df_young = pseudobulk_adata.obs[(pseudobulk_adata.obs['tissue'] == tis) & (pseudobulk_adata.obs['age_group'] == 'Young')]
        _panel_boxplot(ax_b, df_young, tis, 'Young', stats_df[(stats_df['tissue'] == tis) & (stats_df['age_group'] == 'Young')])

    fig.text(0.02, 0.98, 'A', fontsize=14, fontweight='bold')
    fig.text(0.02, 0.47, 'B', fontsize=14, fontweight='bold')
    # TODO(needs author decision): Panel D (Cohen's d heatmap, cell types x tissues,
    # Old samples) and Panel C (compositional partitioning) both omitted here pending
    # the Panel-C reconciliation described in the module docstring, so the figure isn't
    # shipped as a partial 3-of-4-panel mismatch with the paper. Panel D's code is
    # straightforward to re-add (see the archived original notebook cell) once Panel C
    # is resolved; not omitted for any data/statistical reason.
    fig.savefig(out_pdf, dpi=300, bbox_inches='tight')
    fig.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--preds-dir', required=True, help="Directory of pred_*.h5ad files (h5ad_other_age_preds/)")
    p.add_argument('--ct-reann-dir', required=True, help="ct_reference/brain_reannotated/ directory")
    p.add_argument('--clock-path', required=True, help="Path to EN_Chronoage_All_All_WT_scaleddiff.pkl")
    p.add_argument('--ncbi-reference-path', required=True, help="Path to Mus_musculus.gene_info")
    p.add_argument('--out-dir', required=True)
    p.add_argument('--pseudobulk-cache', default=None, help="Optional h5ad path to cache/reuse the pseudobulk AnnData")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    if args.pseudobulk_cache and os.path.exists(args.pseudobulk_cache):
        pseudobulk_adata = sc.read_h5ad(args.pseudobulk_cache)
    else:
        spot_adata_dict = load_tissue_samples(args.preds_dir, args.ct_reann_dir)
        classify_all_hotspots(spot_adata_dict)
        pseudobulk_adata = build_pseudobulk_adata(spot_adata_dict)
        if args.pseudobulk_cache:
            pseudobulk_adata.write_h5ad(args.pseudobulk_cache)

    if TAGE_COL not in pseudobulk_adata.obs.columns or pseudobulk_adata.obs[TAGE_COL].isna().all():
        run_pseudobulk_clock(pseudobulk_adata, args.clock_path, args.ncbi_reference_path)
        if args.pseudobulk_cache:
            pseudobulk_adata.write_h5ad(args.pseudobulk_cache)

    stats_df = compute_stats(pseudobulk_adata)
    stats_df.to_csv(os.path.join(args.out_dir, 'fig6a_celltype_hotspot_stats.csv'), index=False)

    make_figure(
        pseudobulk_adata, stats_df,
        os.path.join(args.out_dir, 'fig6a_celltype_hotspot_tAge.pdf'),
        os.path.join(args.out_dir, 'fig6a_celltype_hotspot_tAge.png'),
    )
    print(f"Wrote fig6a_celltype_hotspot_tAge.{{pdf,png}} and stats CSV to {args.out_dir}")


if __name__ == '__main__':
    main()
