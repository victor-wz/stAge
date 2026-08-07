#!/usr/bin/env python
"""Fig S6a,b -- Spatial tAge variance does not increase uniformly with age:
within-animal spatial variance of tAge_SM across 8 tissues (Brain 2g, Brain 3g,
Brain 25, Hippocampus, Spinal cord, Liver, Spleen, Intestine), tested per-tissue
via Brown-Forsythe (Levene, median-centered) + sample-level Mann-Whitney U, with
effect size = log2(variance ratio Old/Young) and 95% CI from a cluster
(animal-level) bootstrap; cross-tissue generalization via Wilcoxon signed-rank
on the 8 per-tissue log2 ratios; and a direct test of the reviewer-proposed
"variance tracks the mean" mechanism via Spearman correlation.

Source: v_pipeline/spatial_tage_variance_by_age.ipynb -- ported near-verbatim
(same statistics, same formulas, same sample-selection/exclusion logic), only
restructured into functions with CLI-parameterized I/O paths in place of the
original's hard-coded `/home/vvicente/spatial_aging/...` + `os.chdir` convention.

NUMBERING NOTE: the source repo's own `FigS7.md`/`FigS7_methods.txt` label this
analysis "Figure S7 Panels A/B" -- per the author's confirmed 2026-08-06
decision (see stAge-release/INVENTORY.md), the paper's authoritative numbering
for this content is **Fig S6a,b**, not S7; used throughout this file and its
outputs.

Operates entirely on already-predicted `tAge_SM` values cached in per-sample
`.h5ad` prediction files (`h5ad_other_age_preds/LR_pred_tms_*`) -- does not
invoke any stage.pipeline/stage.hotspots machinery, since no clock prediction
or Gi* classification happens in this analysis; it is pure downstream statistics.
"""

from __future__ import annotations

import argparse
import collections
import os
import re
import warnings

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=UserWarning)

import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import stats
import statsmodels.formula.api as smf

TAGE_COL = 'tAge_SM'
TISSUE_ORDER = ['Brain 2g', 'Brain 3g', 'Brain 25', 'Hippocampus', 'Spinalcord', 'Liver', 'Spleen', 'Intestine']
YOUNG_COLOR, OLD_COLOR = '#4C72B0', '#C44E52'
AGE_PALETTE = {'Young': YOUNG_COLOR, 'Old': OLD_COLOR}

AGE_OLD_RE = re.compile(r'(_O[_\-.\d]|_Old)')
AGE_YOUNG_RE = re.compile(r'(_Y[_\-.\d]|_Young)')


def p_to_stars(p):
    if not np.isfinite(p):
        return 'ns'
    return '***' if p < 1e-3 else ('**' if p < 1e-2 else ('*' if p < 5e-2 else 'ns'))


def detect_age_group(fname):
    if AGE_OLD_RE.search(fname):
        return 'Old'
    if AGE_YOUNG_RE.search(fname):
        return 'Young'
    return None


def select_simple(preds_dir, prefix, exclude_tokens=('_reg_',)):
    out = []
    for fname in sorted(os.listdir(preds_dir)):
        if not fname.startswith(prefix):
            continue
        if any(tok in fname for tok in exclude_tokens):
            continue
        age = detect_age_group(fname)
        if age is None:
            continue
        out.append((fname, age))
    return out


def select_brain25(preds_dir):
    """De-duplicate repeat-imaging rounds of the same physical section: keep round2 > round1 > base."""
    base_re = re.compile(r'^LR_pred_tms_brain25_([OY]\d[A-Z]{2})(?:_round(\d))?\.h5ad$')
    groups = collections.defaultdict(dict)
    for f in sorted(os.listdir(preds_dir)):
        if '_reg_' in f:
            continue
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


def build_tissue_registry(preds_dir):
    return {
        'Brain 2g': lambda: select_simple(preds_dir, 'LR_pred_tms_brain2g_'),
        'Brain 3g': lambda: select_simple(preds_dir, 'LR_pred_tms_brain3g_'),
        'Brain 25': lambda: select_brain25(preds_dir),
        'Hippocampus': lambda: select_simple(preds_dir, 'LR_pred_tms_Hippocampus_', exclude_tokens=('_reg_', 'M-')),
        'Spinalcord': lambda: select_simple(preds_dir, 'LR_pred_tms_Spinalcord_'),
        'Liver': lambda: select_simple(preds_dir, 'LR_pred_tms_Liver_'),
        'Spleen': lambda: select_simple(preds_dir, 'LR_pred_tms_Spleen_'),
        'Intestine': lambda: select_simple(preds_dir, 'LR_pred_tms_Intestine_'),
    }


def load_sample_and_mp_tables(preds_dir):
    file_registry = {tissue: fn() for tissue, fn in build_tissue_registry(preds_dir).items()}

    sample_rows, mp_rows = [], []
    for tissue in TISSUE_ORDER:
        for fname, age in file_registry[tissue]:
            a = sc.read_h5ad(os.path.join(preds_dir, fname))
            if TAGE_COL not in a.obs.columns:
                print('  MISSING', TAGE_COL, tissue, fname)
                continue
            vals = a.obs[TAGE_COL].astype(float).values
            vals = vals[np.isfinite(vals)]
            if len(vals) < 5:
                print('  too few metapixels, skipped:', tissue, fname, len(vals))
                continue
            sample_id = fname.replace('.h5ad', '')
            sample_rows.append(dict(
                tissue=tissue, sample_id=sample_id, age_group=age, n_mp=len(vals),
                mean_tAge=vals.mean(), var_tAge=vals.var(ddof=1), std_tAge=vals.std(ddof=1),
                median_tAge=np.median(vals),
                iqr_tAge=np.percentile(vals, 75) - np.percentile(vals, 25),
                mad_tAge=np.median(np.abs(vals - np.median(vals))),
            ))
            centered = vals - vals.mean()
            for v, c in zip(vals, centered):
                mp_rows.append(dict(tissue=tissue, sample_id=sample_id, age_group=age,
                                     tAge_SM=v, centered_tAge=c))

    sample_df = pd.DataFrame(sample_rows)
    sample_df['tissue'] = pd.Categorical(sample_df['tissue'], categories=TISSUE_ORDER, ordered=True)
    mp_df = pd.DataFrame(mp_rows)
    mp_df['tissue'] = pd.Categorical(mp_df['tissue'], categories=TISSUE_ORDER, ordered=True)
    return sample_df, mp_df, file_registry


def bootstrap_var_ratio_ci(mp_df, tissue, rng, n_boot):
    sub = mp_df[mp_df.tissue == tissue]
    old_groups = {sid: g['tAge_SM'].values for sid, g in sub[sub.age_group == 'Old'].groupby('sample_id')}
    young_groups = {sid: g['tAge_SM'].values for sid, g in sub[sub.age_group == 'Young'].groupby('sample_id')}
    old_samples, young_samples = list(old_groups), list(young_groups)
    obs_old = np.concatenate(list(old_groups.values()))
    obs_young = np.concatenate(list(young_groups.values()))
    obs_ratio = obs_old.var(ddof=1) / obs_young.var(ddof=1)
    ratios = np.empty(n_boot)
    for i in range(n_boot):
        bo = rng.choice(old_samples, size=len(old_samples), replace=True)
        by = rng.choice(young_samples, size=len(young_samples), replace=True)
        pooled_old = np.concatenate([old_groups[s] for s in bo])
        pooled_young = np.concatenate([young_groups[s] for s in by])
        ratios[i] = pooled_old.var(ddof=1) / pooled_young.var(ddof=1)
    lo, hi = np.nanpercentile(ratios, [2.5, 97.5])
    return obs_ratio, lo, hi


def compute_tissue_stats(sample_df, mp_df, n_boot, seed):
    rng = np.random.default_rng(seed)
    results = []
    for tissue in TISSUE_ORDER:
        sub = mp_df[mp_df.tissue == tissue]
        old_vals = sub[sub.age_group == 'Old']['tAge_SM'].values
        young_vals = sub[sub.age_group == 'Young']['tAge_SM'].values
        ssub = sample_df[sample_df.tissue == tissue]
        n_old_s = (ssub.age_group == 'Old').sum()
        n_young_s = (ssub.age_group == 'Young').sum()

        lev_stat, lev_p = stats.levene(old_vals, young_vals, center='median')

        old_sv = ssub[ssub.age_group == 'Old']['var_tAge'].values
        young_sv = ssub[ssub.age_group == 'Young']['var_tAge'].values
        try:
            mwu_stat, mwu_p = stats.mannwhitneyu(old_sv, young_sv, alternative='two-sided')
        except ValueError:
            mwu_stat, mwu_p = np.nan, np.nan

        obs_ratio, ci_lo, ci_hi = bootstrap_var_ratio_ci(mp_df, tissue, rng, n_boot)

        results.append(dict(
            tissue=tissue, n_old_samples=n_old_s, n_young_samples=n_young_s,
            n_old_mp=len(old_vals), n_young_mp=len(young_vals),
            var_old=old_vals.var(ddof=1), var_young=young_vals.var(ddof=1),
            var_ratio_old_young=obs_ratio, log2_var_ratio=np.log2(obs_ratio),
            log2_ratio_ci_lo=np.log2(ci_lo), log2_ratio_ci_hi=np.log2(ci_hi),
            levene_stat=lev_stat, levene_p=lev_p, mwu_sample_p=mwu_p,
        ))
    return pd.DataFrame(results)


def cross_tissue_generalization(tissue_stats_df, mp_df):
    log_ratios = tissue_stats_df['log2_var_ratio'].values
    w_stat, w_p = stats.wilcoxon(log_ratios)
    n_increase = int((log_ratios > 0).sum())
    median_log_ratio = float(np.median(log_ratios))
    print(f'Across-tissue Wilcoxon signed-rank on log2(variance ratio, Old/Young): '
          f'W={w_stat:.2f}, p={w_p:.3f}')
    print(f'Tissues with Old > Young variance: {n_increase} / {len(log_ratios)}  '
          f'(median log2 ratio = {median_log_ratio:+.3f}, i.e. {2 ** median_log_ratio:.2f}x)')

    tissue_sd = mp_df.groupby('tissue', observed=True)['tAge_SM'].transform('std')
    mp_df = mp_df.copy()
    mp_df['scaled_abs_centered'] = mp_df['centered_tAge'].abs() / tissue_sd
    samp_scaled = (mp_df.groupby(['tissue', 'sample_id', 'age_group'], observed=True)['scaled_abs_centered']
                   .mean().reset_index())

    old_s = samp_scaled[samp_scaled.age_group == 'Old']['scaled_abs_centered']
    young_s = samp_scaled[samp_scaled.age_group == 'Young']['scaled_abs_centered']
    pooled_mwu_stat, pooled_mwu_p = stats.mannwhitneyu(old_s, young_s, alternative='two-sided')
    print(f'\nPooled (8 tissues, scale-normalized) sample-level Mann-Whitney U: '
          f'p={pooled_mwu_p:.3f}  (Old mean={old_s.mean():.3f}, Young mean={young_s.mean():.3f}, n={len(samp_scaled)} samples)')

    mixed = smf.mixedlm('scaled_abs_centered ~ C(age_group)', samp_scaled, groups=samp_scaled['tissue']).fit(reml=True)
    print('\nSample-level mixed-effects model (scale-normalized |deviation from sample mean|, '
          'random intercept per tissue):')
    print(mixed.summary().tables[1])

    return w_p, n_increase, median_log_ratio, pooled_mwu_p, samp_scaled


def mean_variance_relationship(sample_df):
    corr_rows = []
    for tissue in TISSUE_ORDER:
        g = sample_df[sample_df.tissue == tissue]
        r, p = stats.spearmanr(g['mean_tAge'], g['std_tAge'])
        corr_rows.append(dict(tissue=tissue, n=len(g), spearman_r=r, p=p))
    meanvar_df = pd.DataFrame(corr_rows)

    sample_df = sample_df.copy()
    sample_df['mean_c'] = sample_df.groupby('tissue', observed=True)['mean_tAge'].transform(lambda x: x - x.mean())
    sample_df['std_c'] = sample_df.groupby('tissue', observed=True)['std_tAge'].transform(lambda x: x - x.mean())
    r_within, p_within = stats.spearmanr(sample_df['mean_c'], sample_df['std_c'])
    r_pooled, p_pooled = stats.spearmanr(sample_df['mean_tAge'], sample_df['std_tAge'])

    print(f'Pooled (all {len(sample_df)} samples, raw units) mean vs SD: r={r_pooled:+.2f}, p={p_pooled:.3f}')
    print(f'Within-tissue (mean-centered per tissue) mean vs SD: r={r_within:+.2f}, p={p_within:.2e}')
    return meanvar_df, r_within, p_within


def make_figure1(sample_df, tissue_stats_df, w_p, median_log_ratio, out_dir):
    fig1 = plt.figure(figsize=(11, 4.3))
    gs = gridspec.GridSpec(1, 2, width_ratios=[2.1, 1], wspace=0.55)

    axA = fig1.add_subplot(gs[0, 0])
    sns.boxplot(data=sample_df, x='tissue', y='std_tAge', hue='age_group', order=TISSUE_ORDER,
                hue_order=['Young', 'Old'], palette=AGE_PALETTE, ax=axA,
                showfliers=False, width=0.6, linewidth=0.9,
                boxprops=dict(alpha=0.55), whiskerprops=dict(alpha=0.8), capprops=dict(alpha=0.8))
    sns.stripplot(data=sample_df, x='tissue', y='std_tAge', hue='age_group', order=TISSUE_ORDER,
                  hue_order=['Young', 'Old'], palette=AGE_PALETTE, ax=axA, dodge=True,
                  size=3.4, linewidth=0.3, edgecolor='white', alpha=0.9, legend=False)
    axA.set_ylabel('Within-sample SD of tAge$_{SM}$ (months)')
    axA.set_xlabel('')
    axA.set_xticks(range(len(TISSUE_ORDER)))
    axA.set_xticklabels(TISSUE_ORDER, rotation=32, ha='right')
    handles, labels = axA.get_legend_handles_labels()
    axA.legend(handles[:2], labels[:2], frameon=False, loc='upper left', fontsize=7.5)
    ymax = sample_df['std_tAge'].max()
    for i, t in enumerate(TISSUE_ORDER):
        row = tissue_stats_df[tissue_stats_df.tissue == t].iloc[0]
        stars = p_to_stars(row['levene_p'])
        y = sample_df[sample_df.tissue == t]['std_tAge'].max() * 1.08
        axA.text(i, y, stars, ha='center', va='bottom', fontsize=8)
    axA.set_ylim(0, ymax * 1.28)
    axA.set_title('A   Spatial tAge spread per animal', loc='left', fontweight='bold', fontsize=10)
    sns.despine(ax=axA)

    axB = fig1.add_subplot(gs[0, 1])
    fdf = tissue_stats_df.set_index('tissue').loc[TISSUE_ORDER].reset_index()
    ypos = np.arange(len(fdf))[::-1]
    colors = [OLD_COLOR if r > 0 else YOUNG_COLOR for r in fdf['log2_var_ratio']]
    axB.hlines(ypos, fdf['log2_ratio_ci_lo'], fdf['log2_ratio_ci_hi'], color='#888888', lw=1.2, zorder=1)
    axB.scatter(fdf['log2_var_ratio'], ypos, c=colors, s=34, zorder=2, edgecolor='black', linewidth=0.4)
    axB.axvline(0, color='black', lw=0.8, ls='--', zorder=0)
    axB.set_yticks(ypos)
    axB.set_yticklabels(fdf['tissue'], fontsize=8)
    axB.set_xlabel(r'$\log_2$(variance ratio, Old / Young)')
    axB.set_title('B   Effect size', loc='left', fontweight='bold', fontsize=10)
    axB.scatter([median_log_ratio], [-1.3], marker='D', color='black', s=40, zorder=3, clip_on=False)
    axB.text(median_log_ratio, -1.9, f'median\nWilcoxon p={w_p:.2f}', ha='center', va='top', fontsize=6.8)
    axB.set_ylim(-2.6, len(fdf) - 0.4)
    sns.despine(ax=axB)

    fig1_pdf = os.path.join(out_dir, 'figS6ab_variance_by_age_fig1.pdf')
    fig1_png = os.path.join(out_dir, 'figS6ab_variance_by_age_fig1.png')
    fig1.savefig(fig1_pdf, bbox_inches='tight')
    fig1.savefig(fig1_png, dpi=300, bbox_inches='tight')
    print('saved:', fig1_pdf, '/', fig1_png)
    plt.close(fig1)


def make_figure2(sample_df, meanvar_df, samp_scaled, pooled_mwu_p, out_dir):
    fig2 = plt.figure(figsize=(11.5, 5.6))
    gs2 = gridspec.GridSpec(2, 5, height_ratios=[1, 1], wspace=0.55, hspace=0.75)

    axes = []
    for i, tissue in enumerate(TISSUE_ORDER):
        r_, c_ = divmod(i, 4)
        ax = fig2.add_subplot(gs2[r_, c_])
        axes.append(ax)
        sub = sample_df[sample_df.tissue == tissue]
        for age, color in AGE_PALETTE.items():
            g = sub[sub.age_group == age]
            ax.scatter(g['mean_tAge'], g['std_tAge'], color=color, s=20, edgecolor='white',
                       linewidth=0.4, alpha=0.9, label=age, zorder=2)
        row = meanvar_df[meanvar_df.tissue == tissue].iloc[0]
        ax.set_title(f"{tissue}\nr={row['spearman_r']:+.2f}, p={row['p']:.2f}", fontsize=8)
        ax.set_xlabel('mean tAge$_{SM}$', fontsize=7.5)
        if c_ == 0:
            ax.set_ylabel('SD tAge$_{SM}$', fontsize=7.5)
        ax.tick_params(labelsize=6.8)
        sns.despine(ax=ax)
    axes[0].legend(fontsize=6.5, frameon=False, loc='upper left', handletextpad=0.2)

    axC = fig2.add_subplot(gs2[:, 4])
    sns.violinplot(data=samp_scaled, x='age_group', y='scaled_abs_centered', order=['Young', 'Old'],
                    palette=AGE_PALETTE, ax=axC, inner=None, cut=0, linewidth=0.8, alpha=0.55)
    sns.stripplot(data=samp_scaled, x='age_group', y='scaled_abs_centered', order=['Young', 'Old'],
                  palette=AGE_PALETTE, ax=axC, size=3.5, edgecolor='white', linewidth=0.3,
                  alpha=0.85, jitter=0.15)
    axC.set_title(f'Pooled (8 tissues)\nsample-level MWU p={pooled_mwu_p:.2f}', fontsize=8)
    axC.set_ylabel('Scale-normalized |deviation\nfrom sample mean tAge$_{SM}$|', fontsize=7.5)
    axC.set_xlabel('')
    axC.tick_params(labelsize=7.5)
    sns.despine(ax=axC)

    fig2.text(0.01, 1.03, 'A', fontweight='bold', fontsize=13)
    fig2.text(0.795, 1.03, 'B', fontweight='bold', fontsize=13)
    fig2.text(0.06, 1.03, 'Within-sample mean tAge vs. spatial spread, per tissue', fontsize=9.5, fontweight='bold')

    fig2_pdf = os.path.join(out_dir, 'figS6ab_variance_by_age_fig2.pdf')
    fig2_png = os.path.join(out_dir, 'figS6ab_variance_by_age_fig2.png')
    fig2.savefig(fig2_pdf, bbox_inches='tight')
    fig2.savefig(fig2_png, dpi=300, bbox_inches='tight')
    print('saved:', fig2_pdf, '/', fig2_png)
    plt.close(fig2)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--preds-dir', required=True,
                    help="Directory of LR_pred_tms_*.h5ad metapixel-level prediction files")
    p.add_argument('--out-dir', required=True)
    p.add_argument('--n-boot', type=int, default=10000)
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    sns.set_style('ticks')
    plt.rcParams.update({
        'font.family': 'sans-serif', 'font.size': 9,
        'axes.labelsize': 10, 'axes.titlesize': 11,
        'xtick.labelsize': 8, 'ytick.labelsize': 8, 'axes.linewidth': 0.8,
        'pdf.fonttype': 42, 'ps.fonttype': 42,
    })

    print('== 1. Sample selection & loading ==')
    sample_df, mp_df, file_registry = load_sample_and_mp_tables(args.preds_dir)
    print(f'{len(sample_df)} samples, {len(mp_df)} metapixels total loaded.')

    sample_csv = os.path.join(args.out_dir, 'figS6ab_variance_by_age_persample.csv')
    mp_csv = os.path.join(args.out_dir, 'figS6ab_variance_by_age_metapixel.csv')
    sample_df.to_csv(sample_csv, index=False)
    mp_df.to_csv(mp_csv, index=False)
    print('saved:', sample_csv)
    print('saved:', mp_csv)

    print('\n== 3. Per-tissue Brown-Forsythe + bootstrap effect size ==')
    tissue_stats_df = compute_tissue_stats(sample_df, mp_df, args.n_boot, args.seed)
    tissue_stats_csv = os.path.join(args.out_dir, 'figS6ab_variance_by_age_tissue_stats.csv')
    tissue_stats_df.round(4).to_csv(tissue_stats_csv, index=False)
    print('saved:', tissue_stats_csv)
    print(tissue_stats_df.round(4).to_string(index=False))

    print('\n== 4. Cross-tissue generalization ==')
    w_p, n_increase, median_log_ratio, pooled_mwu_p, samp_scaled = cross_tissue_generalization(tissue_stats_df, mp_df)

    print('\n== 5. Mean-variance relationship ==')
    meanvar_df, r_within, p_within = mean_variance_relationship(sample_df)
    print(meanvar_df.round(3).to_string(index=False))

    print('\n== 6-7. Figures ==')
    make_figure1(sample_df, tissue_stats_df, w_p, median_log_ratio, args.out_dir)
    make_figure2(sample_df, meanvar_df, samp_scaled, pooled_mwu_p, args.out_dir)

    n_sig_increase = int(((tissue_stats_df.levene_p < 0.05) & (tissue_stats_df.log2_var_ratio > 0)).sum())
    n_sig_decrease = int(((tissue_stats_df.levene_p < 0.05) & (tissue_stats_df.log2_var_ratio < 0)).sum())
    n_ns = int((tissue_stats_df.levene_p >= 0.05).sum())
    print(f'\n== Summary == {n_sig_increase}/{len(tissue_stats_df)} tissues sig. higher variance in Old, '
          f'{n_sig_decrease}/{len(tissue_stats_df)} sig. lower, {n_ns}/{len(tissue_stats_df)} n.s.; '
          f'cross-tissue Wilcoxon p={w_p:.3f}.')


if __name__ == '__main__':
    main()
