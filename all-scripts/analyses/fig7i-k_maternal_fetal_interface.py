"""Fig 7i-k -- Maternal-fetal interface (MFI): compartment tAge + distance gradient
(LOWESS, paired Fisher-z, mixed-effects interaction test).

STATUS: partially real, partially stub -- read both halves of this file's docstring
before using it.

REAL (ported from source): `distance_gradient_stats()` below is a faithful port of
v_pipeline/stage_res_pred.ipynb's MFI distance-analysis section (~lines 4432-4560 of
the nbconvert'd script, the final/most-developed of several near-duplicate iterations
in that notebook -- same "keep only the last of several near-identical re-executions"
situation documented for celltype_hotspot_tAge.ipynb's Region Analysis section in
INVENTORY.md). It runs three tests comparing the tAge-vs-distance-to-MFI gradient
between fetal and maternal compartments:
  1. A mixed-effects interaction model (tAge_SM ~ compartment * abs_dist, random
     slope+intercept per sample) -- tests whether the two gradients have different
     slopes.
  2. A paired Fisher-z test (per-sample Spearman r of dist-vs-tAge, z-transformed,
     paired t-test fetal vs. maternal across samples).
  3. A near-vs-far (per-sample median split) Mann-Whitney U + Cohen's d contrast,
     within each compartment.
Plus the 2-panel figure (signed-distance LOWESS crossover plot, per-sample paired
correlation dumbbell plot) from the same source cells.

`distance_gradient_stats()` takes an ALREADY compartment-labeled DataFrame as input --
it does not compute the maternal/fetal split itself. Expected columns: `sample`
(sample/animal ID), `compartment` (values indicating fetal vs. maternal, case-
insensitive substring match on "fet" is used to normalize labels, matching the
source's own normalization step), `dist_norm` (signed, per-sample-normalized distance
to the maternal-fetal interface -- negative on the fetal side, positive on the
maternal side, 0 = at the interface; this is a normalized version of `dist_to_MFI`,
itself a nearest-neighbor distance to the interface computed upstream in the source
notebook), `tAge_SM` (propagated per-spot/metapixel tAge prediction).

STUB (genuine gap, author-confirmed 2026-08-06 -- see INVENTORY.md gap list item 8):
the compartment-labeling step that PRODUCES the `compartment`/`dist_norm` columns
`distance_gradient_stats()` consumes is NOT reused from this codebase. The only
compartment classifier found anywhere in the audited tree is the marker-gene-panel
classifier in v_pipeline/stage_res_pred-Copy1.ipynb (`FETAL_PANELS`/`MATERNAL_PANELS`,
z-scored `fetal_delta` splitting) -- the author explicitly confirmed this is NOT the
intended approach: maternal-fetal compartments will instead be annotated by cell type.
That cell-type-based annotation method does not yet exist in this codebase and must be
written fresh (or supplied by the author) before this figure can be reproduced
end-to-end. `annotate_compartment_by_celltype()` below is a placeholder marking where
it belongs -- do not treat it as a working implementation.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch
from scipy import stats
import statsmodels.formula.api as smf
from statsmodels.nonparametric.smoothers_lowess import lowess


def annotate_compartment_by_celltype(adata, *args, **kwargs):
    """STUB -- cell-type-based maternal/fetal compartment annotation.

    TODO(gap): no source implementation. Per the author (2026-08-06), compartments
    are to be annotated by cell type rather than the marker-gene-panel z-score
    classifier previously used in stage_res_pred-Copy1.ipynb (which is explicitly NOT
    to be carried into this release). This function should ultimately produce a
    per-obs `compartment` label (fetal/maternal) and a `dist_norm` signed, per-sample-
    normalized distance-to-interface column, matching the schema `distance_gradient_
    stats()` below expects. See INVENTORY.md gap list item 8.
    """
    raise NotImplementedError(
        "annotate_compartment_by_celltype has no source implementation -- see this "
        "function's docstring and stAge-release/INVENTORY.md gap list item 8 "
        "(maternal-fetal interface compartment annotation)."
    )


def distance_gradient_stats(
    df: pd.DataFrame,
    sample_col: str = 'sample',
    compartment_col: str = 'compartment',
    dist_col: str = 'dist_norm',
    tage_col: str = 'tAge_SM',
    lowess_frac: float = 0.5,
    min_n_per_compartment: int = 5,
    make_figure: bool = True,
):
    """Real, ported analysis (see module docstring). Requires `df` to already carry
    compartment + normalized signed distance-to-MFI columns (see module docstring for
    exact schema and the still-unimplemented step that must produce them).

    Returns a dict with keys:
      - 'interaction_model': fitted statsmodels MixedLMResults
        (tAge ~ C(compartment, Treatment('Fetal')) * abs_dist, random slope+intercept
        per sample).
      - 'fetal_slope', 'maternal_slope', 'interaction_p': floats from the model above.
      - 'paired_fisher_z': dict with 'rdf' (per-sample fetal/maternal Spearman r and
        z), 't_stat', 'p_value', 'mean_r_fetal', 'mean_r_maternal' (paired Fisher-z
        test of dist-vs-tAge correlation, fetal vs. maternal, paired within sample).
      - 'near_far': dict keyed by compartment label, each a dict with 'cohens_d' and
        'p_value' (Mann-Whitney U) comparing tAge in the near-half vs. far-half of
        that compartment's per-sample distance distribution (median split).
      - 'figure': the matplotlib Figure (only if `make_figure=True`).
    """
    d = df.copy()
    d['comp'] = np.where(
        d[compartment_col].astype(str).str.lower().str.contains('fet'),
        'Fetal', 'Maternal',
    )
    d['abs_dist'] = d[dist_col].abs()

    # ── 1. Interaction model: does the tAge-vs-distance slope differ by compartment? ──
    md = smf.mixedlm(
        f"{tage_col} ~ C(comp, Treatment('Fetal')) * abs_dist",
        d, groups=d[sample_col], re_formula="~abs_dist",
    )
    mdf = md.fit(method='lbfgs')

    p, pv = mdf.params, mdf.pvalues
    fetal_key = 'abs_dist'
    inter_key = [k for k in p.index if k.endswith(':abs_dist')][0]
    fetal_slope = p[fetal_key]
    maternal_slope = p[fetal_key] + p[inter_key]
    interaction_p = pv[inter_key]

    # ── 2. Paired Fisher-z: per-sample fetal vs. maternal dist-tAge correlation ──
    rows = []
    for s, g in d.groupby(sample_col):
        rec = {'sample': s}
        ok = True
        for c in ['Fetal', 'Maternal']:
            gc = g[g['comp'] == c]
            if len(gc) < min_n_per_compartment:
                ok = False
                break
            r, _ = stats.spearmanr(gc['abs_dist'], gc[tage_col])
            rec[f'{c}_r'] = r
            rec[f'{c}_z'] = np.arctanh(np.clip(r, -0.999, 0.999))
        if ok:
            rows.append(rec)
    rdf = pd.DataFrame(rows)

    dz = rdf['Maternal_z'] - rdf['Fetal_z']
    t_stat, p_paired = stats.ttest_1samp(dz, 0)
    mean_r_fetal = np.tanh(rdf['Fetal_z'].mean())
    mean_r_maternal = np.tanh(rdf['Maternal_z'].mean())

    # ── 3. Near-vs-far zone contrast, within compartment ──
    d['zone'] = np.where(
        d['abs_dist'] <= d.groupby(sample_col)['abs_dist'].transform('median'),
        'Near MFI', 'Far from MFI',
    )

    def _cohens_d(a, b):
        na, nb = len(a), len(b)
        sp = np.sqrt(((na - 1) * a.std(ddof=1) ** 2 + (nb - 1) * b.std(ddof=1) ** 2) / (na + nb - 2))
        return (a.mean() - b.mean()) / sp if sp > 0 else np.nan

    near_far = {}
    for c in ['Fetal', 'Maternal']:
        near = d[(d['comp'] == c) & (d['zone'] == 'Near MFI')][tage_col]
        far = d[(d['comp'] == c) & (d['zone'] == 'Far from MFI')][tage_col]
        u, pmw = stats.mannwhitneyu(near, far, alternative='two-sided')
        near_far[c] = {'cohens_d': _cohens_d(near, far), 'p_value': pmw}

    result = dict(
        interaction_model=mdf,
        fetal_slope=fetal_slope, maternal_slope=maternal_slope, interaction_p=interaction_p,
        paired_fisher_z=dict(
            rdf=rdf, t_stat=t_stat, p_value=p_paired,
            mean_r_fetal=mean_r_fetal, mean_r_maternal=mean_r_maternal,
        ),
        near_far=near_far,
    )

    if make_figure:
        col = {'Fetal': 'lightblue', 'Maternal': 'orange'}
        fig, axes = plt.subplots(1, 2, figsize=(13, 6))

        ax = axes[0]
        for c in ['Fetal', 'Maternal']:
            gc = d[d['comp'] == c]
            ax.scatter(gc[dist_col], gc[tage_col], s=6, alpha=0.08, color=col[c])
            lw = lowess(gc[tage_col], gc[dist_col], frac=lowess_frac, return_sorted=True)
            ax.plot(lw[:, 0], lw[:, 1], color=col[c], lw=3, label=c)
        ax.axvline(0, color='black', lw=1, ls=':')
        ax.set_title(f'tAge across the MFI\ninteraction p = {interaction_p:.1e}', fontsize=14)
        ax.set_xlabel('Normalized signed distance to MFI\n← fetal    |    maternal →', fontsize=13)
        ax.set_ylabel('Relative chronological tAge', fontsize=13)
        ax.legend(handles=[Patch(color=col[c], label=c) for c in ['Fetal', 'Maternal']], fontsize=12)
        sns.despine(ax=ax)

        ax2 = axes[1]
        rng = np.random.default_rng(0)
        xpos = {'Fetal': 0.0, 'Maternal': 1.0}
        jit = 0.07
        xf = rng.normal(xpos['Fetal'], jit, len(rdf))
        xm = rng.normal(xpos['Maternal'], jit, len(rdf))
        for xi, yi, xj, yj in zip(xf, rdf['Fetal_r'], xm, rdf['Maternal_r']):
            ax2.plot([xi, xj], [yi, yj], color='gray', alpha=0.3, lw=0.8, zorder=1)
        ax2.scatter(xf, rdf['Fetal_r'], color=col['Fetal'], s=60, zorder=3, edgecolor='white', linewidth=0.7)
        ax2.scatter(xm, rdf['Maternal_r'], color=col['Maternal'], s=60, zorder=3, edgecolor='white', linewidth=0.7)
        for c, xp in xpos.items():
            mr = np.tanh(rdf[f'{c}_z'].mean())
            ax2.hlines(mr, xp - 0.22, xp + 0.22, color='black', lw=2.5, zorder=4)
        ax2.axhline(0, color='black', lw=1, ls=':')
        ax2.set_xlim(-0.55, 1.55)
        ax2.set_xticks([0, 1])
        ax2.set_xticklabels(['Fetal', 'Maternal'], fontsize=12)
        ax2.set_ylabel('Spearman r  (dist-to-MFI vs tAge)', fontsize=12)
        ax2.set_title(f'Per-sample gradients\npaired p = {p_paired:.1e}', fontsize=14)
        sns.despine(ax=ax2)

        plt.tight_layout()
        result['figure'] = fig

    return result
