#!/usr/bin/env python
"""Fig S6c -- Does the hotspot-vs-coldspot transcriptional signature converge
across tissues with age? Pairwise between-tissue Spearman correlation of the
hotspot-vs-coldspot DESeq2 log2FC signature, computed separately within Young
and within Old animals across 9 tissues (36 matched pairs), tested via paired
Wilcoxon signed-rank (primary) and paired t-test (secondary).

Source: v_pipeline/hotspot_signature_correlation_by_age.ipynb -- ported
near-verbatim (identical statistics/formulas), restructured into functions with
CLI-parameterized I/O in place of the original's hard-coded
`/home/vvicente/spatial_aging/...` + `os.chdir` convention.

This notebook does NOT compute DESeq2 or run Getis-Ord Gi* itself -- it is
purely downstream of cached per-tissue hotspot-vs-coldspot differential-
expression tables (`per_tissue_tables_de_{YOUNG,OLD}_{brains,other}_Gi>1_
DEseq2.pkl.gz`), produced upstream by `dstream_loop.ipynb` (Fig 6b's DESeq2
pipeline, ported to `stage-release/R/deseq2_hotspot_coldspot.R` +
`analyses/fig6b_deseq2_hotspot_coldspot.py`). Point `--intermediate-dir` at
wherever those four cache files were written; this script does not regenerate
them.

NUMBERING NOTE: the source repo's own `FigS7.md` labels this analysis "Figure
S7 Panel C" -- per the author's confirmed 2026-08-06 decision (see
stAge-release/INVENTORY.md), the paper's authoritative numbering is **Fig
S6c**, used throughout this file and its outputs.
"""

from __future__ import annotations

import argparse
import gzip
import itertools
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy.stats import spearmanr, wilcoxon, ttest_rel

YOUNG_COLOR, OLD_COLOR = '#4C72B0', '#C44E52'
AGE_PALETTE = {'Young': YOUNG_COLOR, 'Old': OLD_COLOR}
TISSUE_LABELS = {'brain25': 'Brain 25', 'brain2g': 'Brain 2g', 'brain3g': 'Brain 3g', 'ovary': 'Ovary'}


def lbl(t):
    return TISSUE_LABELS.get(t, t)


def load_pkl(intermediate_dir: Path, fname: str):
    with gzip.open(intermediate_dir / fname, 'rb') as fh:
        return pickle.load(fh)


def load_de_tables(intermediate_dir: Path):
    young = {**load_pkl(intermediate_dir, 'per_tissue_tables_de_YOUNG_brains_Gi>1_DEseq2.pkl.gz'),
             **load_pkl(intermediate_dir, 'per_tissue_tables_de_YOUNG_other_Gi>1_DEseq2.pkl.gz')}
    old = {**load_pkl(intermediate_dir, 'per_tissue_tables_de_OLD_brains_Gi>1_DEseq2.pkl.gz'),
           **load_pkl(intermediate_dir, 'per_tissue_tables_de_OLD_other_Gi>1_DEseq2.pkl.gz')}
    assert set(young) == set(old), 'Young/Old tissue sets must match for a paired comparison'
    tissues = sorted(young.keys())
    print(f'{len(tissues)} tissues, matched Young/Old:', [lbl(t) for t in tissues])
    return young, old, tissues


def pairwise_corr_fillna(dfs, tissues, beta_col='Coef_interaction'):
    """Primary method: genome-wide gene union, missing/untested genes filled with 0 --
    matches the correlation-heatmap convention used in dstream_loop.ipynb."""
    beta_mat = pd.concat({t: dfs[t][beta_col] for t in tissues}, axis=1).fillna(0).astype(float)
    beta_mat.columns = tissues
    mat = beta_mat.corr(method='spearman')
    rows = []
    for t1, t2 in itertools.combinations(tissues, 2):
        rows.append(dict(tissue1=lbl(t1), tissue2=lbl(t2), n_genes=len(beta_mat), rho=mat.loc[t1, t2]))
    mat = mat.rename(index=lbl, columns=lbl)
    return mat.astype(float), pd.DataFrame(rows)


def pairwise_corr_intersection(dfs, tissues, beta_col='Coef_interaction'):
    """Robustness check: restrict each pairwise correlation to genes actually tested
    (non-NaN DESeq2 p-value) in BOTH tissues of the pair."""
    rows = []
    for t1, t2 in itertools.combinations(tissues, 2):
        df1, df2 = dfs[t1], dfs[t2]
        common = df1.index[df1['pval'].notna()].intersection(df2.index[df2['pval'].notna()])
        b1 = df1.loc[common, beta_col].astype(float)
        b2 = df2.loc[common, beta_col].astype(float)
        r, p = spearmanr(b1, b2)
        rows.append(dict(tissue1=lbl(t1), tissue2=lbl(t2), n_genes=len(common), rho=r, p=p))
    return pd.DataFrame(rows)


def make_figure(young_pairs, old_pairs, merged, w_p, n_increase, n_pairs, out_dir):
    plt.rcParams.update({
        'font.size': 12, 'axes.labelsize': 13, 'axes.titlesize': 13,
        'xtick.labelsize': 11.5, 'ytick.labelsize': 11.5,
    })

    fig = plt.figure(figsize=(7.8, 3.6))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 0.92], wspace=0.4)

    axC = fig.add_subplot(gs[0, 0])
    long_df = pd.concat([
        young_pairs.assign(age_group='Young')[['tissue1', 'tissue2', 'rho', 'age_group']],
        old_pairs.assign(age_group='Old')[['tissue1', 'tissue2', 'rho', 'age_group']],
    ])
    long_df['pair'] = long_df['tissue1'] + '__' + long_df['tissue2']
    for pair, g in long_df.groupby('pair'):
        g = g.set_index('age_group').loc[['Young', 'Old']]
        axC.plot([0, 1], g['rho'].values, color='#999999', lw=0.7, alpha=0.5, zorder=1)
    for age, x in [('Young', 0), ('Old', 1)]:
        vals = long_df[long_df.age_group == age]['rho'].values
        axC.scatter(np.full(len(vals), x), vals, color=AGE_PALETTE[age], s=32, zorder=2,
                    edgecolor='white', linewidth=0.5, alpha=0.9)
        bp = axC.boxplot(vals, positions=[x], widths=0.32, showfliers=False, patch_artist=True,
                          zorder=3, manage_ticks=False)
        for box in bp['boxes']:
            box.set(facecolor=AGE_PALETTE[age], alpha=0.35, edgecolor='black', linewidth=1.0)
        for med in bp['medians']:
            med.set(color='black', linewidth=1.4)
        for whisk in bp['whiskers']:
            whisk.set(linewidth=1.0)
        for cap in bp['caps']:
            cap.set(linewidth=1.0)
    axC.set_xticks([0, 1])
    axC.set_xticklabels(['Young', 'Old'])
    axC.set_xlim(-0.45, 1.45)
    axC.axhline(0, color='black', lw=0.7, ls=':', zorder=0)
    axC.set_ylabel('Pairwise Spearman ρ of\nhotspot-vs-coldspot DE signature')
    stars = '***' if w_p < 1e-3 else ('**' if w_p < 1e-2 else ('*' if w_p < 5e-2 else 'ns'))
    axC.set_title(f'A   Matched tissue-pairs (n={n_pairs})\npaired Wilcoxon p={w_p:.1e} {stars}',
                  loc='left', fontsize=12)
    sns.despine(ax=axC)

    axD = fig.add_subplot(gs[0, 1])
    colors_d = [OLD_COLOR if r else YOUNG_COLOR for r in (merged.rho_old > merged.rho_young)]
    axD.scatter(merged.rho_young, merged.rho_old, c=colors_d, s=36, edgecolor='black', linewidth=0.4, alpha=0.85)
    lims = [min(merged.rho_young.min(), merged.rho_old.min()) - 0.05,
            max(merged.rho_young.max(), merged.rho_old.max()) + 0.05]
    axD.plot(lims, lims, color='black', lw=0.9, ls='--', zorder=0)
    axD.set_xlim(lims)
    axD.set_ylim(lims)
    axD.set_xlabel('ρ (Young)')
    axD.set_ylabel('ρ (Old)')
    axD.set_title(f'B   {n_increase}/{n_pairs} pairs increase\nwith age', loc='left', fontsize=12)
    sns.despine(ax=axD)

    fig_pdf = os.path.join(out_dir, 'figS6c_hotspot_signature_convergence.pdf')
    fig_png = os.path.join(out_dir, 'figS6c_hotspot_signature_convergence.png')
    fig.savefig(fig_pdf, bbox_inches='tight')
    fig.savefig(fig_png, dpi=300, bbox_inches='tight')
    print('saved:', fig_pdf, '/', fig_png)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--intermediate-dir', required=True,
                    help="Directory containing per_tissue_tables_de_{YOUNG,OLD}_{brains,other}_Gi>1_DEseq2.pkl.gz "
                         "(produced upstream by the Fig 6b DESeq2 pipeline)")
    p.add_argument('--out-dir', required=True)
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    sns.set_style('ticks')
    plt.rcParams.update({
        'font.family': 'sans-serif', 'font.size': 9,
        'axes.labelsize': 10, 'axes.titlesize': 11,
        'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5, 'axes.linewidth': 0.8,
        'pdf.fonttype': 42, 'ps.fonttype': 42,
    })

    print('== 1. Load cached per-tissue DE tables ==')
    young, old, tissues = load_de_tables(Path(args.intermediate_dir))

    sanity_rows = []
    for age, d_ in [('Young', young), ('Old', old)]:
        for t, df in d_.items():
            sanity_rows.append(dict(age_group=age, tissue=lbl(t),
                                     n_genes_tested=int(df['pval'].notna().sum()),
                                     n_sig_padj_lt_0p05=int((df['pval_adj'] < 0.05).sum())))
    print(pd.DataFrame(sanity_rows).pivot(index='tissue', columns='age_group', values='n_sig_padj_lt_0p05'))

    print('\n== 2. Pairwise between-tissue correlation (genome-wide union, fillna(0)) ==')
    young_pairs_mat, young_pairs = pairwise_corr_fillna(young, tissues)
    old_pairs_mat, old_pairs = pairwise_corr_fillna(old, tissues)
    print(f'Young: mean pairwise rho = {young_pairs.rho.mean():+.3f}, median = {young_pairs.rho.median():+.3f}')
    print(f'Old:   mean pairwise rho = {old_pairs.rho.mean():+.3f}, median = {old_pairs.rho.median():+.3f}')

    merged = young_pairs.merge(old_pairs, on=['tissue1', 'tissue2'], suffixes=('_young', '_old'))
    merged['delta'] = merged['rho_old'] - merged['rho_young']

    print('\n== 3. Paired significance test ==')
    w_stat, w_p = wilcoxon(merged.rho_old, merged.rho_young)
    t_stat, t_p = ttest_rel(merged.rho_old, merged.rho_young)
    n_increase = int((merged.rho_old > merged.rho_young).sum())
    n_pairs = len(merged)
    print(f'Paired Wilcoxon signed-rank test (Old vs Young, n={n_pairs} matched tissue-pairs): '
          f'W={w_stat:.1f}, p={w_p:.2e}')
    print(f'Paired t-test: t={t_stat:.3f}, p={t_p:.2e}')
    print(f'Tissue-pairs with higher correlation in Old: {n_increase}/{n_pairs}')

    robust_pairs_y = pairwise_corr_intersection(young, tissues)
    robust_pairs_o = pairwise_corr_intersection(old, tissues)
    w_stat_r, w_p_r = wilcoxon(robust_pairs_o.rho, robust_pairs_y.rho)
    print(f'\nRobustness check (tested-gene intersection instead of genome-wide fillna(0) union): '
          f'Young mean rho={robust_pairs_y.rho.mean():+.3f}, Old mean rho={robust_pairs_o.rho.mean():+.3f}, '
          f'paired Wilcoxon p={w_p_r:.2e} '
          f'(n increased = {int((robust_pairs_o.rho > robust_pairs_y.rho).sum())}/{n_pairs})')

    pairs_csv = os.path.join(args.out_dir, 'figS6c_hotspot_signature_correlation_pairs.csv')
    merged.round(4).to_csv(pairs_csv, index=False)
    print('\nsaved:', pairs_csv)

    print('\n== 4. Figure ==')
    make_figure(young_pairs, old_pairs, merged, w_p, n_increase, n_pairs, args.out_dir)

    tissue_list_str = ', '.join(lbl(t) for t in tissues)
    brain_set = ["Brain 2g", "Brain 3g", "Brain 25"]
    brain_pairs = old_pairs[old_pairs.tissue1.isin(brain_set) & old_pairs.tissue2.isin(brain_set)]
    print(f"""
== Summary ==
{len(tissues)} tissues ({tissue_list_str}), n={n_pairs} matched tissue pairs.
Young mean rho={young_pairs.rho.mean():+.3f}, Old mean rho={old_pairs.rho.mean():+.3f}.
{n_increase}/{n_pairs} pairs higher in Old. Paired Wilcoxon p={w_p:.2e} (t-test p={t_p:.2e}).
Brain-only pairwise Old rho range: {brain_pairs.rho.min():.2f}-{brain_pairs.rho.max():.2f}.
""")


if __name__ == '__main__':
    main()
