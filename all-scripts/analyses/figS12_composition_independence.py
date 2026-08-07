#!/usr/bin/env python
"""Fig S12 -- Spatial transcriptomic age is not just cell-type composition.
Four-panel figure across 7 independent mouse datasets (Brain 2g, Brain 3g,
Brain 25, Hippocampus, Liver, Spinal cord, Intestine):

  A. OLS residualization -- regress cell-type composition + sample identity out
     of every gene at the metapixel level, re-run the tAge clock on the
     residual matrix; Gi*-defined hotspots remain older than coldspots in
     every dataset.
  B. Exact symmetric (Reimers/Cotton) Oaxaca-Blinder decomposition of the
     hotspot-minus-coldspot tAge gap into compositional vs. intrinsic terms.
  C. A single cell type (neurons) across 7 anatomical brain regions, Brain 25
     (Old) only -- same composition, different tAge by region.
  D. The same cell types (neurons, oligodendrocytes), hotspot vs. coldspot,
     Brain 25 (Old) only.

Source: v_pipeline/spatial_tage_beyond_composition.ipynb -- this is the
canonical, author-approved version of this analysis (per stAge-release/
INVENTORY.md's confirmed decision), NOT celltype_hotspot_tAge.ipynb Section 5,
which has a confirmed reference-mean bug and is excluded from this release.

NUMBERING NOTE: the co-located `FigureS11.md` labels this analysis "Figure
S11" -- per the author's confirmed 2026-08-06 decision (see stAge-release/
INVENTORY.md), the paper's authoritative numbering is **Fig S12**, used
throughout this file and its outputs.

CONSOLIDATED ONTO stage/ (mechanical, not a behavior change): Panels A and B's
core statistics now call `stage.composition.residualize_composition_and_reclock`
/ `oaxaca_blinder_decomposition` (verified line-for-line identical to this
notebook's own `run_composition_residual_clock`/`oaxaca_blinder_decomposition`
during the port); the source's own spot-level Gi* classification (`_classify_gi`,
used by the metapixel builder for Panels A/B and fresh for Panel D) now calls
`stage.hotspots.classify_hotspots` (verified identical: same k=8/999-perm/
BH-FDR<0.05/z>1 parameters, same row-standardized KNN construction, same
seed=0 default); Panel C's region annotation now calls
`stage.region_annotation.annotate_brain_regions` (verified identical
normalize/log1p/HVG/PCA/neighbors/Leiden/marker-z-score parameters and the
sparse-cluster-indicator-matmul memory-safety rewrite already documented in
FigureS11.md Sec. 5). `stage.preprocessing.get_scaled_counts` now requires an
explicit `ncbi_reference_path` argument (the source's 3-positional-arg
`get_scaled_counts(df, clock, 'symbol')` calls matched an older st_utils
signature with no such parameter) -- threaded through explicitly here via
`--ncbi-reference-path`.

SIMPLIFICATION (flagged, not silent): Panels C and D's plotted/statistical
results are Brain-25-only in the source notebook (documented explicitly in
FigureS11.md: "though not used in Panels C/D's plotted results, which are
Brain 25 only"). The source notebook nonetheless loads an external cache
(`region_pseudobulk_celltype.h5ad` / `pseudobulk_celltype_hotspot.h5ad`,
built by the sibling notebook `celltype_hotspot_tAge.ipynb`) as a Brain 2g/3g
"schema reference" before concatenating Brain 25 onto it. Since that schema
reference plays no role in the actual figure content, this script builds
Brain 25's region/hotspot pseudobulks directly and does NOT reproduce the
external-cache dependency -- avoiding a hard dependency on a second notebook's
multi-GB cache for output this script never uses. If Panels C/D need to be
extended to include Brain 2g/3g in a future revision, that cache-loading logic
would need to be reintroduced; it is deliberately omitted here.

EXCLUDED: a commented-out earlier "Panel A — Old only" figure variant
(iterative-editing residue, superseded by the live Young|Old nested version)
is not reproduced.
"""

from __future__ import annotations

import argparse
import collections
import gc
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
from scipy.stats import mannwhitneyu, kruskal, ttest_1samp

from stage.preprocessing import filter_genes, get_scaled_counts
from stage.clock import final_clock_preparation, predict_age
from stage.hotspots import classify_hotspots
from stage.composition import residualize_composition_and_reclock, oaxaca_blinder_decomposition
from stage.region_annotation import annotate_brain_regions, REGION_MARKERS

TAGE_COL = 'tAge_SM'
CT_COL = 'cell_type'
MIN_SPOTS_PER_MP = 10
MIN_SPOTS_PER_PSEUDOBULK = 10

TISSUE_ORDER = ['Brain 2g', 'Brain 3g', 'Brain 25', 'Hippocampus', 'Liver', 'Spinalcord', 'Intestine']
BRAIN_TISSUES = ['Brain 25']

HOTSPOT_COLOR, COLDSPOT_COLOR, NORMAL_COLOR = '#C44E52', '#4C72B0', '#B0B0B0'
AGING_PALETTE = {'coldspot': COLDSPOT_COLOR, 'normal': NORMAL_COLOR, 'hotspot': HOTSPOT_COLOR}
REGION_COLORS = {
    'Isocortex': '#4C72B0', 'Hippocampus': '#55A868', 'Fiber_tracts': '#C44E52',
    'Thalamus': '#8172B2', 'Hypothalamus': '#CCB974', 'Striatum_CNu': '#64B5CD',
    'OLF_CTX': '#A9A9A9', 'Unknown': '#EBEBEB',
}
TISSUE_SHORT = {'Brain 2g': 'Brain 2g', 'Brain 3g': 'Brain 3g', 'Brain 25': 'Brain 25',
                'Hippocampus': 'Hippo.', 'Liver': 'Liver', 'Spinalcord': 'Spinal cord',
                'Intestine': 'Intestine'}

AGE_OLD_RE = re.compile(r'(_O[_\-.\d]|_Old)')
AGE_YOUNG_RE = re.compile(r'(_Y[_\-.\d]|_Young)')


def _safe_token(s):
    return re.sub(r'[^A-Za-z0-9]+', '-', str(s)).strip('-')


def detect_age_group(fname):
    if AGE_OLD_RE.search(fname):
        return 'Old'
    if AGE_YOUNG_RE.search(fname):
        return 'Young'
    return None


def attach_celltype(adata, fname, ct_source, ct_reann_dir, ct_prefix=None):
    """Cell type either native (obs[CT_COL] already present, e.g. Hippocampus/Liver/
    Spinalcord/Intestine) or via SingleR reannotation file lookup (Brain 2g/3g/25)."""
    if CT_COL in adata.obs.columns:
        return True
    if ct_source != 'singler':
        return False
    cand = os.path.join(ct_reann_dir, f'{ct_prefix}{fname}')
    if not os.path.exists(cand):
        return False
    adct = sc.read_h5ad(cand)
    if 'Cell.type_SingleR' not in adct.obs.columns:
        return False
    common = adata.obs.index.intersection(adct.obs.index)
    if len(common) == 0:
        return False
    adata.obs[CT_COL] = (adct.obs.loc[common, 'Cell.type_SingleR']
                          .reindex(adata.obs.index).astype('object'))
    return True


def cohens_d(a, b):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return np.nan
    pooled = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
    return (a.mean() - b.mean()) / pooled if pooled else np.nan


def p_to_stars(p):
    if not np.isfinite(p):
        return 'ns'
    return '***' if p < 1e-3 else ('**' if p < 1e-2 else ('*' if p < 5e-2 else 'ns'))


# --------------------------------------------------------------------------- #
# Per-tissue file selection registry
# --------------------------------------------------------------------------- #

def select_brain_files(preds_dir, prefix):
    out = []
    for fname in sorted(os.listdir(preds_dir)):
        if not fname.startswith(f'pred_tms_{prefix}_'):
            continue
        age = detect_age_group(fname)
        if age is None:
            continue
        out.append((fname, age))
    return out


def select_brain25_files(preds_dir):
    """De-duplicate repeat-imaging rounds of the same physical section: keep round2 > round1 > base."""
    base_re = re.compile(r'^pred_tms_brain25_([OY]\d[A-Z]{2})(?:_round(\d))?\.h5ad$')
    groups = collections.defaultdict(dict)
    for f in sorted(os.listdir(preds_dir)):
        m = base_re.match(f)
        if not m:
            continue
        base_id, rnd = m.group(1), m.group(2)
        rnd = int(rnd) if rnd else 0
        groups[base_id][rnd] = f
    out = []
    for base_id, rounds in sorted(groups.items()):
        fname = rounds[max(rounds.keys())]
        age = 'Old' if base_id.startswith('O') else 'Young'
        out.append((fname, age))
    return out


def select_hippocampus_files(preds_dir):
    out = []
    for fname in sorted(os.listdir(preds_dir)):
        if not fname.startswith('pred_tms_Hippocampus_Hippocampus_'):
            continue
        age = detect_age_group(fname)
        if age is None:
            continue
        out.append((fname, age))
    return out


def select_native_tissue_files(preds_dir, tissue):
    out = []
    prefix = f'pred_tms_{tissue}_{tissue}_'
    for fname in sorted(os.listdir(preds_dir)):
        if not fname.startswith(prefix):
            continue
        age = detect_age_group(fname)
        if age is None:
            continue
        out.append((fname, age))
    return out


def build_tissue_registry(preds_dir):
    return {
        'Brain 2g': dict(selector=lambda: select_brain_files(preds_dir, 'brain2g'), ct_source='singler', ct_prefix='brain2g_'),
        'Brain 3g': dict(selector=lambda: select_brain_files(preds_dir, 'brain3g'), ct_source='singler', ct_prefix='brain3g_'),
        'Brain 25': dict(selector=lambda: select_brain25_files(preds_dir), ct_source='singler', ct_prefix='brain25_'),
        'Hippocampus': dict(selector=lambda: select_hippocampus_files(preds_dir), ct_source='native', ct_prefix=None),
        'Liver': dict(selector=lambda: select_native_tissue_files(preds_dir, 'Liver'), ct_source='native', ct_prefix=None),
        'Spinalcord': dict(selector=lambda: select_native_tissue_files(preds_dir, 'Spinalcord'), ct_source='native', ct_prefix=None),
        'Intestine': dict(selector=lambda: select_native_tissue_files(preds_dir, 'Intestine'), ct_source='native', ct_prefix=None),
    }


# --------------------------------------------------------------------------- #
# Metapixel tables (Panels A & B) -- one per dataset
# --------------------------------------------------------------------------- #

def build_metapixel_adata_for_tissue(preds_dir, ct_reann_dir, tissue, file_age_list, ct_source, ct_prefix):
    spot_dict = {}
    for fname, age in file_age_list:
        a = sc.read_h5ad(os.path.join(preds_dir, fname))
        a.var_names_make_unique()
        if TAGE_COL not in a.obs or 'metapixel' not in a.obs or 'spatial' not in a.obsm:
            print(f'  skip (missing fields): {fname}')
            continue
        if not attach_celltype(a, fname, ct_source, ct_reann_dir, ct_prefix):
            print(f'  skip (no cell type): {fname}')
            continue
        a.obs['age_group'] = age
        a.obs['tissue'] = tissue
        a.obs['sample_id'] = fname.replace('.h5ad', '')
        classify_hotspots(a, value_col=TAGE_COL)  # spot-level Gi* -> a.obs['aging_type']
        spot_dict[fname] = a
        gi_counts = a.obs['aging_type'].value_counts().to_dict()
        print(f'  loaded {age:5s} {tissue:12s} {fname}  spots={a.n_obs} '
              f'mps={a.obs["metapixel"].nunique()} cts={a.obs[CT_COL].nunique()}  Gi*={gi_counts}')
    if not spot_dict:
        return None

    var_union = None
    for a in spot_dict.values():
        vn = pd.Index(a.var_names)
        var_union = vn if var_union is None else var_union.union(vn)
    var_union = var_union.unique()
    var_pos = pd.Series(np.arange(len(var_union)), index=var_union)
    ct_universe = sorted(set().union(*[set(a.obs[CT_COL].dropna().unique()) for a in spot_dict.values()]))
    print(f'  [{tissue}] gene union: {len(var_union)} | cell types ({len(ct_universe)}): {ct_universe}')

    all_X, all_comp, all_cen, all_meta, all_names = [], [], [], [], []
    for fname, a in spot_dict.items():
        obs = a.obs
        X = a.X
        is_sp = issparse(X)
        col_pos = np.array([var_pos[g] for g in a.var_names], dtype=int)
        coords = np.asarray(a.obsm['spatial'], dtype=float)
        age = obs['age_group'].iloc[0]
        tis = obs['tissue'].iloc[0]
        sid = obs['sample_id'].iloc[0]
        for mp in obs['metapixel'].dropna().unique():
            mask = (obs['metapixel'] == mp).values & obs[CT_COL].notna().values
            n = int(mask.sum())
            if n < MIN_SPOTS_PER_MP:
                continue
            sub = X[mask]
            gsum = (np.asarray(sub.sum(axis=0)).ravel() if is_sp else np.asarray(sub).sum(axis=0).ravel())
            full = np.zeros(len(var_union), dtype=np.float32)
            full[col_pos] = gsum.astype(np.float32)
            cts = obs[CT_COL].values[mask]
            comp = np.array([np.mean(cts == c) for c in ct_universe], dtype=np.float32)
            mp_aging_type = pd.Series(obs['aging_type'].values[mask]).value_counts().idxmax()
            all_X.append(full)
            all_comp.append(comp)
            all_cen.append(coords[mask].mean(0))
            all_names.append(f'{age.lower()}.{_safe_token(tis)}.{_safe_token(sid)}.mp{_safe_token(mp)}')
            all_meta.append({'tissue': tis, 'sample_id': sid, 'age_group': age,
                              'metapixel': str(mp), 'n_spots': n,
                              TAGE_COL: float(obs[TAGE_COL].values[mask][0]),
                              'total_counts': float(gsum.sum()), 'aging_type': mp_aging_type})
    mp_ad = ad.AnnData(X=np.vstack(all_X),
                        obs=pd.DataFrame(all_meta, index=pd.Index(all_names, name='mp_id')),
                        var=pd.DataFrame(index=pd.Index(var_union)))
    mp_ad.obs['aging_type'] = pd.Categorical(mp_ad.obs['aging_type'], categories=['coldspot', 'normal', 'hotspot'])
    mp_ad.obsm['composition'] = np.vstack(all_comp)
    mp_ad.obsm['spatial'] = np.vstack(all_cen)
    mp_ad.uns['ct_universe'] = ct_universe
    mp_ad.var_names_make_unique()
    return mp_ad


# --------------------------------------------------------------------------- #
# Panel C: Brain 25 region pseudobulks (fresh only, see module docstring)
# --------------------------------------------------------------------------- #

def build_brain25_region_pseudobulks(preds_dir, ct_reann_dir, clock, ncbi_reference_path):
    file_age_list = select_brain25_files(preds_dir)
    spot_dict = {}
    for fname, age in file_age_list:
        a = sc.read_h5ad(os.path.join(preds_dir, fname))
        a.var_names_make_unique()
        if TAGE_COL not in a.obs or 'spatial' not in a.obsm:
            continue
        if not attach_celltype(a, fname, 'singler', ct_reann_dir, 'brain25_'):
            continue
        a.obs['age_group'] = age
        a.obs['tissue'] = 'Brain 25'
        a.obs['sample_id'] = fname.replace('.h5ad', '')
        spot_dict[fname] = a
    print(f'  {len(spot_dict)} Brain 25 samples loaded, {sum(a.n_obs for a in spot_dict.values())} spots total')

    raw_adata = sc.concat(list(spot_dict.values()), keys=list(spot_dict.keys()), label='sample', join='outer')
    raw_adata.var_names_make_unique()
    raw_adata.obs_names_make_unique()
    del spot_dict
    gc.collect()
    print(f'  concatenated: {raw_adata.shape}')

    region_labels = annotate_brain_regions(raw_adata, batch_key='sample', region_markers=REGION_MARKERS)
    raw_adata.obs['region_auto'] = region_labels.reindex(raw_adata.obs_names).values
    print('  region counts:', raw_adata.obs['region_auto'].value_counts().to_dict())

    var_union = pd.Index(raw_adata.var_names)
    raw_X = raw_adata.X
    is_sparse = issparse(raw_X)
    obs = raw_adata.obs
    all_X, all_meta, all_names = [], [], []
    for sample_id_full, idx in obs.groupby('sample').groups.items():
        rows_pos = obs.index.get_indexer(idx)
        sample_obs = obs.iloc[rows_pos]
        age = sample_obs['age_group'].iloc[0]
        age_tag = age.lower()
        sample_id = sample_id_full.replace('.h5ad', '')
        for ct in sample_obs[CT_COL].dropna().unique().tolist():
            for region in [r for r in sample_obs['region_auto'].dropna().unique() if r != 'Unknown']:
                mask = (sample_obs[CT_COL] == ct).values & (sample_obs['region_auto'] == region).values
                n_spots = int(mask.sum())
                if n_spots < MIN_SPOTS_PER_PSEUDOBULK:
                    continue
                gene_sum = np.asarray(raw_X[rows_pos[mask]].sum(axis=0)).ravel() if is_sparse else \
                    np.asarray(raw_X[rows_pos[mask]]).sum(axis=0).ravel()
                obs_name = f'{age_tag}.{_safe_token(region)}.brain-25.{_safe_token(sample_id)}.{_safe_token(ct)}.{n_spots}'
                all_X.append(gene_sum.astype(np.float32))
                all_names.append(obs_name)
                all_meta.append({'tissue': 'Brain 25', 'sample_id': sample_id, 'age_group': age,
                                  'cell_type': ct, 'region': region, 'n_spots': n_spots,
                                  'total_counts': float(gene_sum.sum())})
    pb = ad.AnnData(X=np.vstack(all_X),
                     obs=pd.DataFrame(all_meta, index=pd.Index(all_names, name='pseudobulk_id')),
                     var=pd.DataFrame(index=var_union))
    pb.var_names_make_unique()
    del raw_adata, raw_X
    gc.collect()
    print(f'  brain25 region pseudobulks: {pb.shape}')

    pb_filt = filter_genes(pb)
    X = pb_filt.X.toarray() if issparse(pb_filt.X) else np.asarray(pb_filt.X)
    df = pd.DataFrame(X.T, index=pb_filt.var_names, columns=pb_filt.obs.index)
    df.index.name = 'geneID'
    counts_scaled = get_scaled_counts(df, clock, ncbi_reference_path, 'symbol')
    n_ref = counts_scaled.columns.str.contains('young.Isocortex').sum()
    diff_sfx = 'young.Isocortex' if n_ref > 0 else 'young'
    preprocessed = final_clock_preparation(counts_scaled, clock, diff_suffix=diff_sfx)
    preds = predict_age(preprocessed, clock)
    pb.obs[TAGE_COL] = pd.Series(np.asarray(preds) * 48, index=preprocessed.columns).reindex(pb.obs.index).values
    return pb.obs[['tissue', 'sample_id', 'age_group', 'cell_type', 'region', 'n_spots', 'total_counts', TAGE_COL]].copy()


# --------------------------------------------------------------------------- #
# Panel D: Brain 25 hotspot pseudobulks (fresh only, see module docstring)
# --------------------------------------------------------------------------- #

def build_brain25_hotspot_pseudobulks(preds_dir, ct_reann_dir, clock, ncbi_reference_path):
    file_age_list = select_brain25_files(preds_dir)
    all_X, all_meta, all_names = [], [], []
    var_union = None
    spot_adatas = []
    for fname, age in file_age_list:
        a = sc.read_h5ad(os.path.join(preds_dir, fname))
        a.var_names_make_unique()
        if TAGE_COL not in a.obs or 'spatial' not in a.obsm:
            continue
        if not attach_celltype(a, fname, 'singler', ct_reann_dir, 'brain25_'):
            continue
        a.obs['age_group'] = age
        a.obs['tissue'] = 'Brain 25'
        a.obs['sample_id'] = fname.replace('.h5ad', '')
        classify_hotspots(a, value_col=TAGE_COL)
        spot_adatas.append(a)
        vn = pd.Index(a.var_names)
        var_union = vn if var_union is None else var_union.union(vn)
    var_union = var_union.unique()
    var_pos = pd.Series(np.arange(len(var_union)), index=var_union)
    print(f'  {len(spot_adatas)} Brain 25 samples Gi*-classified')

    for a in spot_adatas:
        obs = a.obs
        X = a.X
        is_sp = issparse(X)
        col_pos = np.array([var_pos[g] for g in a.var_names], dtype=int)
        age_tag = obs['age_group'].iloc[0].lower()
        sid = obs['sample_id'].iloc[0]
        for ct in pd.Series(obs[CT_COL]).dropna().unique().tolist():
            for region in ['hotspot', 'normal', 'coldspot']:
                mask = (obs[CT_COL] == ct).values & (obs['aging_type'] == region).values
                n_spots = int(mask.sum())
                if n_spots < MIN_SPOTS_PER_PSEUDOBULK:
                    continue
                sub = X[mask]
                gsum = (np.asarray(sub.sum(axis=0)).ravel() if is_sp else np.asarray(sub).sum(axis=0).ravel())
                full = np.zeros(len(var_union), dtype=np.float32)
                full[col_pos] = gsum.astype(np.float32)
                obs_name = f'{age_tag}.{region}.brain-25.{_safe_token(sid)}.{_safe_token(ct)}.{n_spots}'
                all_X.append(full)
                all_names.append(obs_name)
                all_meta.append({'tissue': 'Brain 25', 'sample_id': sid, 'age_group': obs['age_group'].iloc[0],
                                  'cell_type': ct, 'aging_type': region, 'n_spots': n_spots,
                                  'total_counts': float(gsum.sum())})
    pb = ad.AnnData(X=np.vstack(all_X),
                     obs=pd.DataFrame(all_meta, index=pd.Index(all_names, name='pseudobulk_id')),
                     var=pd.DataFrame(index=pd.Index(var_union)))
    pb.var_names_make_unique()
    print(f'  brain25 hotspot pseudobulks: {pb.shape}')

    pb_filt = filter_genes(pb)
    X = pb_filt.X.toarray() if issparse(pb_filt.X) else np.asarray(pb_filt.X)
    df = pd.DataFrame(X.T, index=pb_filt.var_names, columns=pb_filt.obs.index)
    df.index.name = 'geneID'
    counts_scaled = get_scaled_counts(df, clock, ncbi_reference_path, 'symbol')
    preprocessed = final_clock_preparation(counts_scaled, clock, diff_suffix='young.normal')
    preds = predict_age(preprocessed, clock)
    pb.obs[TAGE_COL] = pd.Series(np.asarray(preds) * 48, index=preprocessed.columns).reindex(pb.obs.index).values
    return pb.obs[['tissue', 'sample_id', 'age_group', 'cell_type', 'aging_type',
                    'n_spots', 'total_counts', TAGE_COL]].copy()


# --------------------------------------------------------------------------- #
# Figure assembly
# --------------------------------------------------------------------------- #

def make_figure(panelA_long, panelA_stats, panelB_summary, cC, order_regions, panelC_kw,
                 hpb_obs, panelD_stats, panel_c_ct, panel_d_cts, out_dir):
    fig = plt.figure(figsize=(10, 20))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.55, wspace=0.28)
    order3 = ['coldspot', 'normal', 'hotspot']
    tissue_labels = [TISSUE_SHORT[t] for t in TISSUE_ORDER]

    # ---- Panel A ----
    gsA = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[0, 0], wspace=0.12)
    xposA = np.arange(len(TISSUE_ORDER))
    for col, age in enumerate(['Young', 'Old']):
        axA = fig.add_subplot(gsA[0, col])
        dfA = panelA_long[(panelA_long['age_group'] == age) &
                           (panelA_long['aging_type'].isin(['coldspot', 'hotspot']))].copy()
        dfA['aging_type'] = dfA['aging_type'].astype(str)
        sns.boxplot(data=dfA, x='tissue', y='tAge_residual', hue='aging_type', order=TISSUE_ORDER,
                    hue_order=['coldspot', 'hotspot'], palette=AGING_PALETTE, showfliers=False,
                    width=0.6, linewidth=0.7, ax=axA)
        if age == 'Old':
            sns.stripplot(data=dfA, x='tissue', y='tAge_residual', hue='aging_type', order=TISSUE_ORDER,
                          hue_order=['coldspot', 'hotspot'], dodge=True, color='0.2', size=2,
                          alpha=0.35, ax=axA, legend=False)
        axA.axhline(0, color='0.6', lw=0.6, ls='--', zorder=0)
        axA.set_xticks(xposA)
        axA.set_xticklabels(tissue_labels, rotation=35, ha='right', fontsize=8)
        axA.set_xlabel('')
        axA.set_title(age, fontsize=9, fontweight='bold')
        for ti, tissue in enumerate(TISSUE_ORDER):
            st = panelA_stats.get((tissue, age))
            if st is None:
                continue
            ylim = axA.get_ylim()
            axA.text(ti, ylim[1] - 0.04 * (ylim[1] - ylim[0]), p_to_stars(st['p']),
                     ha='center', va='top', fontsize=8)
        if col == 0:
            axA.set_ylabel('Residual tAge\n(composition removed, months)')
            axA.legend(title='', frameon=False, fontsize=7, loc='lower right')
        else:
            axA.set_ylabel('')
            axA.get_legend().remove()
        sns.despine(ax=axA)
    fig.text(0.06, 0.93, 'A', fontsize=13, fontweight='bold')
    fig.text(0.16, 0.905, 'Hotspot vs coldspot residual tAge, per dataset', fontsize=9)

    # ---- Panel B ----
    axB = fig.add_subplot(gs[0, 1])
    seg = 'pct_comp_mean'
    age_colors = {'Young': 'lightblue', 'Old': 'lightcoral'}
    yposB = np.arange(len(TISSUE_ORDER))
    w = 0.36
    for ai, age in enumerate(['Young', 'Old']):
        yoff = yposB + (ai - 0.5) * w
        vals = np.nan_to_num(np.array([
            panelB_summary.loc[(panelB_summary['tissue'] == t) & (panelB_summary['age_group'] == age), seg]
            .pipe(lambda s: s.values[0] if len(s) else np.nan)
            for t in TISSUE_ORDER
        ])) / 100.0
        axB.barh(yoff, vals, w, color=age_colors[age], edgecolor='white', lw=0.6, label=age)
        for ti, v in enumerate(vals):
            axB.text(v + 0.007, yoff[ti], f'{v * 100:.0f}%', ha='left', va='center', fontsize=9, color='0.3')
    axB.set_yticks(yposB)
    axB.set_yticklabels(tissue_labels, fontsize=9)
    axB.set_xlabel('Fraction of hotspot-coldspot ΔtAge')
    axB.set_xlim(0, max(0.01, np.nanmax(panelB_summary['pct_comp_mean'].values) / 100.0) * 1.6)
    axB.invert_yaxis()
    axB.legend(frameon=False, fontsize=9, loc='upper center', bbox_to_anchor=(0.5, -0.18), ncol=2)
    sns.despine(ax=axB)
    fig.text(0.53, 0.93, 'B', fontsize=13, fontweight='bold')
    fig.text(0.57, 0.905, 'Cell-type composition explains only a small fraction of the tAge gap, per dataset', fontsize=9)

    # ---- Panel C ----
    axC = fig.add_subplot(gs[1, 0])
    sC = cC[(cC['age_group'] == 'Old') & (cC['region'].isin(order_regions))].copy()
    pal = [REGION_COLORS.get(r, '#999999') for r in order_regions]
    sns.boxplot(data=sC, x='region', y=TAGE_COL, order=order_regions, palette=pal,
                showfliers=False, width=0.65, linewidth=0.7, ax=axC)
    sns.stripplot(data=sC, x='region', y=TAGE_COL, order=order_regions, color='0.2',
                  size=2.8, alpha=0.7, ax=axC)
    axC.set_xlabel('')
    axC.set_ylabel('tAge  (months)')
    axC.set_xticklabels([r.replace('_', ' ') for r in order_regions], rotation=35, ha='right', fontsize=8)
    kw_old = panelC_kw.get('Old', np.nan)
    sns.despine(ax=axC)
    fig.text(0.06, 0.47, 'C', fontsize=13, fontweight='bold')
    fig.text(0.20, 0.445,
             f'{panel_c_ct.replace("_Lin", "")} tAge across brain regions — Brain 25 (Old), KW p={kw_old:.1e}',
             fontsize=8.5)

    # ---- Panel D ----
    axD = fig.add_subplot(gs[1, 1])
    sD = hpb_obs[hpb_obs['cell_type'].isin(panel_d_cts) & hpb_obs['aging_type'].isin(order3)].copy()
    sD['cell_type'] = sD['cell_type'].str.replace('_Lin', '', regex=False)
    sns.boxplot(data=sD, x='cell_type', y=TAGE_COL, hue='aging_type', hue_order=order3,
                palette=AGING_PALETTE, showfliers=False, width=0.7, linewidth=0.7, ax=axD)
    sns.stripplot(data=sD, x='cell_type', y=TAGE_COL, hue='aging_type', hue_order=order3,
                  dodge=True, color='0.2', size=2.5, alpha=0.6, ax=axD, legend=False)
    axD.set_xlabel('')
    axD.set_ylabel('tAge  (months)')
    handles, lbls = axD.get_legend_handles_labels()
    axD.legend(handles[:3], lbls[:3], title='', frameon=False, fontsize=8, loc='upper left')
    dtxt = []
    for ct in panel_d_cts:
        st = panelD_stats.get(ct)
        if st:
            dtxt.append(f"{ct.replace('_Lin', '')}: d={st['d']:.1f} {p_to_stars(st['p'])}")
    if dtxt:
        axD.text(0.98, 0.02, '   '.join(dtxt), transform=axD.transAxes, ha='right', va='bottom',
                 fontsize=7.5, style='italic')
    sns.despine(ax=axD)
    fig.text(0.53, 0.47, 'D', fontsize=13, fontweight='bold')
    fig.text(0.57, 0.445, 'Hotspot vs coldspot tAge — Brain 25 (Old)', fontsize=8.5)

    fig_pdf = os.path.join(out_dir, 'figS12_composition_independence.pdf')
    fig_png = os.path.join(out_dir, 'figS12_composition_independence.png')
    fig.savefig(fig_pdf, bbox_inches='tight')
    fig.savefig(fig_png, dpi=300, bbox_inches='tight')
    print('saved:', fig_pdf)
    print('saved:', fig_png)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--preds-dir', required=True, help="Directory of pred_tms_*.h5ad metapixel/spot prediction files")
    p.add_argument('--ct-reann-dir', required=True, help="Directory of SingleR reannotation files (ct_reference/brain_reannotated)")
    p.add_argument('--clock-path', required=True, help="Path to EN_Chronoage_All_All_WT_scaleddiff.pkl")
    p.add_argument('--ncbi-reference-path', required=True)
    p.add_argument('--out-dir', required=True)
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    sns.set_style('ticks')
    plt.rcParams.update({
        'font.family': 'sans-serif', 'font.size': 9,
        'axes.labelsize': 10, 'axes.titlesize': 11,
        'xtick.labelsize': 8, 'ytick.labelsize': 8, 'axes.linewidth': 0.8,
    })

    clock = joblib.load(args.clock_path)
    tissue_registry = build_tissue_registry(args.preds_dir)

    # ---- Metapixel tables (Panels A & B), one per dataset ----
    mp_dict = {}
    for tissue in TISSUE_ORDER:
        cfg = tissue_registry[tissue]
        print(f'\n=== {tissue} ===')
        file_age_list = cfg['selector']()
        print(f'  {len(file_age_list)} files selected')
        mp_ad = build_metapixel_adata_for_tissue(args.preds_dir, args.ct_reann_dir, tissue, file_age_list,
                                                    cfg['ct_source'], cfg['ct_prefix'])
        if mp_ad is None:
            print(f'  FAILED: no metapixels built for {tissue}')
            continue
        mp_dict[tissue] = mp_ad

    # ---- Panel A ----
    panelA_frames = []
    panelA_stats = {}
    for tissue, mp_ad in mp_dict.items():
        ct_universe = list(mp_ad.uns['ct_universe'])
        print(f'\n=== Panel A: {tissue} ===')
        mpf = residualize_composition_and_reclock(mp_ad, ct_universe, clock, args.ncbi_reference_path)
        mp_dict[tissue] = mpf

        r_reclock = np.corrcoef(mpf.obs['tAge_SM'], mpf.obs['tAge_reclock'])[0, 1]
        r_resid = np.corrcoef(mpf.obs['tAge_SM'], mpf.obs['tAge_residual'])[0, 1]
        print(f'  corr(tAge_SM, tAge_reclock)={r_reclock:.3f}   corr(tAge_SM, tAge_residual)={r_resid:.3f}')

        df = mpf.obs[['tissue', 'sample_id', 'age_group', TAGE_COL, 'tAge_residual', 'aging_type']].copy()
        panelA_frames.append(df)

        for age in ['Young', 'Old']:
            sub = df[df['age_group'] == age]
            hot = sub.loc[sub['aging_type'] == 'hotspot', 'tAge_residual']
            cold = sub.loc[sub['aging_type'] == 'coldspot', 'tAge_residual']
            hot0 = sub.loc[sub['aging_type'] == 'hotspot', TAGE_COL]
            cold0 = sub.loc[sub['aging_type'] == 'coldspot', TAGE_COL]
            if len(hot) < 3 or len(cold) < 3:
                print(f'  [{age}] too few hot/cold ({len(hot)}/{len(cold)})')
                continue
            u, pval = mannwhitneyu(hot, cold, alternative='two-sided')
            panelA_stats[(tissue, age)] = dict(n_hot=len(hot), n_cold=len(cold), p=pval,
                                                d_resid=cohens_d(hot, cold), d_orig=cohens_d(hot0, cold0))
            print(f'  [{age}] residual hot={hot.mean():.2f} cold={cold.mean():.2f}  '
                  f"d_orig={panelA_stats[(tissue, age)]['d_orig']:.2f} -> "
                  f"d_residual={panelA_stats[(tissue, age)]['d_resid']:.2f}  p={pval:.1e}")

    panelA_long = pd.concat(panelA_frames, ignore_index=True)
    panelA_long.to_csv(os.path.join(args.out_dir, 'figS12_residual_hotcold_multidataset.csv'), index=False)
    print('\npanelA_long:', panelA_long.shape)

    # ---- Panel B ----
    panelB_persample_frames = []
    for tissue, mpf in mp_dict.items():
        ct_universe = list(mpf.uns['ct_universe'])
        print(f'\n=== Panel B: {tissue} ===')
        res = oaxaca_blinder_decomposition(mpf, ct_universe)
        res.insert(0, 'tissue', tissue)
        panelB_persample_frames.append(res)
        print(f'  max |reconstruction residual| = {res["residual"].abs().max():.2e} months '
              f'(should be ~0; confirms delta_obs = delta_comp + delta_within)')

    panelB_persample = pd.concat(panelB_persample_frames, ignore_index=True)
    panelB_persample.to_csv(os.path.join(args.out_dir, 'figS12_oaxaca_blinder_persample.csv'), index=False)

    panelB_rows = []
    for (tissue, age), g in panelB_persample.groupby(['tissue', 'age_group']):
        n = len(g)
        t_p = ttest_1samp(g['delta_within'], 0).pvalue if n >= 2 else np.nan
        panelB_rows.append(dict(
            tissue=tissue, age_group=age, n_samples=n,
            delta_obs_mean=g['delta_obs'].mean(),
            delta_comp_mean=g['delta_comp'].mean(), delta_comp_sem=g['delta_comp'].sem(),
            delta_within_mean=g['delta_within'].mean(), delta_within_sem=g['delta_within'].sem(),
            pct_comp_mean=g['pct_compositional'].mean(), pct_within_mean=g['pct_intrinsic'].mean(),
            ttest_p_within=t_p,
        ))
    panelB_summary = pd.DataFrame(panelB_rows)
    panelB_summary.to_csv(os.path.join(args.out_dir, 'figS12_oaxaca_blinder_summary.csv'), index=False)
    print('\npanelB_summary:')
    print(panelB_summary.round(3).to_string(index=False))

    # ---- Panel C ----
    print('\n=== Panel C: Brain 25 region pseudobulks ===')
    region_pb = build_brain25_region_pseudobulks(args.preds_dir, args.ct_reann_dir, clock, args.ncbi_reference_path)
    region_pb.to_csv(os.path.join(args.out_dir, 'figS12_region_pseudobulk_brain25.csv'), index=False)

    panel_c_ct = 'NEURON_Lin'
    cC = region_pb[(region_pb['tissue'].isin(BRAIN_TISSUES)) & (region_pb['cell_type'] == panel_c_ct)].copy()
    order_regions = (cC[cC['age_group'] == 'Old'].groupby('region')[TAGE_COL].median().sort_values().index.tolist())
    panelC_kw = {}
    for age in ['Young', 'Old']:
        s = cC[cC['age_group'] == age]
        grp = [s.loc[s['region'] == r, TAGE_COL].values for r in order_regions if (s['region'] == r).sum() >= 2]
        if len(grp) >= 3:
            H, pval = kruskal(*grp)
            panelC_kw[age] = pval
            print(f'  {panel_c_ct} {age}: KW p={pval:.2e} across {len(grp)} regions (n={len(s)})')

    # ---- Panel D ----
    print('\n=== Panel D: Brain 25 hotspot pseudobulks ===')
    hpb_all = build_brain25_hotspot_pseudobulks(args.preds_dir, args.ct_reann_dir, clock, args.ncbi_reference_path)
    hpb_all.to_csv(os.path.join(args.out_dir, 'figS12_hotspot_pseudobulk_brain25.csv'), index=False)

    panel_d_cts = ['NEURON_Lin', 'OLG_Lin']
    hpb_obs = hpb_all[hpb_all['tissue'].isin(BRAIN_TISSUES) & (hpb_all['age_group'] == 'Old')].copy()
    panelD_stats = {}
    for ct in panel_d_cts:
        s = hpb_obs[hpb_obs['cell_type'] == ct]
        g = {r: s.loc[s['aging_type'] == r, TAGE_COL].values for r in ['hotspot', 'normal', 'coldspot']}
        if len(g['hotspot']) >= 2 and len(g['coldspot']) >= 2:
            _, pval = mannwhitneyu(g['hotspot'], g['coldspot'])
            panelD_stats[ct] = dict(p=pval, d=cohens_d(g['hotspot'], g['coldspot']),
                                     n_hot=len(g['hotspot']), n_cold=len(g['coldspot']))
            print(f"  {ct} Old: hot(n={len(g['hotspot'])})={g['hotspot'].mean():.1f}  "
                  f"cold(n={len(g['coldspot'])})={g['coldspot'].mean():.1f}  "
                  f"d={panelD_stats[ct]['d']:.2f} p={pval:.2e}")

    # ---- Figure ----
    make_figure(panelA_long, panelA_stats, panelB_summary, cC, order_regions, panelC_kw,
                hpb_obs, panelD_stats, panel_c_ct, panel_d_cts, args.out_dir)


if __name__ == '__main__':
    main()
