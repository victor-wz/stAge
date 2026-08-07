"""Fig 4d: SenMayo senescence score vs. transcriptomic age (tAge), Spearman correlation.

CAVEAT (unconfirmed mapping): no notebook was found in the audited codebase that computes
this on the whole-mouse natural-aging cohort implied by the paper's figure list. The closest
real, working code is `v_pipeline/hotspot_senescence_stAge.ipynb`, which runs this exact
score-then-correlate procedure but across THREE injury/disease datasets (spinal cord crush,
myocardial infarction "heartbreak", bone fracture) rather than the natural-aging cohort. The
gene list below IS explicitly labeled by that notebook as "SenMayo core, mouse symbols" citing
Saul et al. 2022 Nat Aging (the SenMayo source paper) -- so the gene-set identity is a curated
core subset of the real SenMayo signature, not an arbitrary/unrelated list. What's unconfirmed
is only whether this injury-model notebook (vs. some other, unlocated notebook run on the
natural-aging cohort) is actually the source of the published Fig 4d panel. Confirm with the
author before treating this as a verified reproduction.

Source: v_pipeline/hotspot_senescence_stAge.ipynb ("Compute senescence score at metaspot
level" and "Spearman correlation: tAge_SM vs senescence_score" cells). Ported as real,
callable functions -- no statistical logic changed.
"""

from __future__ import annotations

from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd
import scanpy as sc
from scipy.stats import spearmanr

# SenMayo core signature (mouse symbols), curated subset per Saul et al. 2022 Nat Aging,
# as used in v_pipeline/hotspot_senescence_stAge.ipynb. Pass a different `gene_list` to
# `compute_senescence_score` to substitute the full/official SenMayo gene set if confirmed.
SENMAYO_CORE_GENES = [
    # SASP cytokines / chemokines
    'Il6', 'Il1a', 'Il1b', 'Il8', 'Tnf', 'Tgfb1',
    'Cxcl1', 'Cxcl2', 'Cxcl10', 'Cxcl12',
    'Ccl2', 'Ccl5', 'Ccl7',
    # Matrix remodelling
    'Mmp3', 'Mmp9', 'Mmp13', 'Serpine1', 'Timp1',
    # Growth factors
    'Vegfa', 'Igfbp3', 'Igfbp4', 'Igfbp5', 'Igfbp7', 'Hgf',
    # Cell cycle arrest
    'Cdkn1a', 'Cdkn2a', 'Tp53',
    # DNA damage response
    'H2afx', 'Hmgb1',
    # Anti-apoptotic
    'Bcl2', 'Bcl2l1',
    # Other SASP
    'Csf1', 'Csf2', 'Fn1', 'Mif', 'Spp1', 'Lif',
]


def compute_senescence_score(
    adata,
    gene_list: Iterable[str] = SENMAYO_CORE_GENES,
    score_name: str = 'senescence_score',
    random_state: int = 42,
):
    """Score each obs (metaspot/spot) for the senescence gene signature via `sc.tl.score_genes`.

    Operates on a normalized+log1p COPY internally (`sc.pp.normalize_total(target_sum=1e4)` +
    `sc.pp.log1p`) so the caller's `adata.X` is left untouched; writes `adata.obs[score_name]`
    in place on the ORIGINAL `adata` (matching the source notebook, which keeps raw X intact
    on `adata_ms` while scoring a separate `adata_score` copy). Returns `adata` for chaining.

    Prints how many of `gene_list` were actually found in `adata.var_names` (genes missing
    from the panel are silently dropped by `sc.tl.score_genes`, matching source behavior).
    """
    gene_list = list(gene_list)
    available = [g for g in gene_list if g in adata.var_names]
    missing = [g for g in gene_list if g not in adata.var_names]
    print(f'Senescence genes found: {len(available)}/{len(gene_list)}')
    if missing:
        print(f'  Missing: {missing}')

    scored = adata.copy()
    sc.pp.normalize_total(scored, target_sum=1e4)
    sc.pp.log1p(scored)
    sc.tl.score_genes(scored, gene_list=available, score_name=score_name, random_state=random_state)
    adata.obs[score_name] = scored.obs[score_name].values
    return adata


def senescence_tage_correlation(
    adata,
    tage_col: str = 'tAge_SM',
    score_col: str = 'senescence_score',
    group_col: Optional[str] = 'group',
    sample_col: Optional[str] = 'sample_id',
    min_n: int = 5,
) -> pd.DataFrame:
    """Spearman correlation of tAge vs. senescence score: overall, per group, per sample.

    Returns a tidy DataFrame with one row per (level, group[, sample_id]) giving rho, p, n.
    `level='overall'` uses every obs with finite values in both columns; `level='group'` and
    `level='sample'` require at least `min_n` finite pairs (matches source notebook's `>= 5`
    cutoff) or are skipped.
    """
    x_all = adata.obs[tage_col].values.astype(float)
    y_all = adata.obs[score_col].values.astype(float)
    mask_all = np.isfinite(x_all) & np.isfinite(y_all)
    rho_all, p_all = spearmanr(x_all[mask_all], y_all[mask_all])
    rows = [dict(level='overall', group=None, sample_id=None, rho=rho_all, pvalue=p_all, n=int(mask_all.sum()))]
    print(f'Overall  Spearman r = {rho_all:.3f},  p = {p_all:.2e}  (n={mask_all.sum()})')

    if group_col is not None and group_col in adata.obs:
        for grp in adata.obs[group_col].unique():
            sub = adata.obs[adata.obs[group_col] == grp]
            xg, yg = sub[tage_col].values.astype(float), sub[score_col].values.astype(float)
            mk = np.isfinite(xg) & np.isfinite(yg)
            if mk.sum() < min_n:
                continue
            r, p = spearmanr(xg[mk], yg[mk])
            rows.append(dict(level='group', group=grp, sample_id=None, rho=r, pvalue=p, n=int(mk.sum())))

    if sample_col is not None and sample_col in adata.obs:
        for sid in adata.obs[sample_col].unique():
            sub = adata.obs[adata.obs[sample_col] == sid]
            xs, ys = sub[tage_col].values.astype(float), sub[score_col].values.astype(float)
            mk = np.isfinite(xs) & np.isfinite(ys)
            if mk.sum() < min_n:
                continue
            r, p = spearmanr(xs[mk], ys[mk])
            grp = sub[group_col].iloc[0] if group_col is not None and group_col in sub else None
            rows.append(dict(level='sample', group=grp, sample_id=sid, rho=r, pvalue=p, n=int(mk.sum())))

    return pd.DataFrame(rows)


def run_fig4d(preds_mp: Dict[str, "sc.AnnData"], gene_list: Iterable[str] = SENMAYO_CORE_GENES) -> pd.DataFrame:
    """End-to-end: concat per-file metaspot predictions, score senescence, correlate vs. tAge_SM.

    `preds_mp`: dict of filename -> metaspot-level AnnData with `tAge_SM` in `.obs` (e.g. the
    output of `stage.pipeline.full_nonoverlap_mp_pipeline`), each already carrying whatever
    `group`/`sample_id` obs columns the caller wants correlations broken out by.

    TODO(needs confirmation): the natural-aging-cohort data-loading step feeding `preds_mp`
    is not wired up here -- the caller must build it (see module docstring for the dataset
    ambiguity this reflects).
    """
    adata_ms = sc.concat(list(preds_mp.values()), join='inner')
    adata_ms.obs_names_make_unique()
    compute_senescence_score(adata_ms, gene_list=gene_list)
    return senescence_tage_correlation(adata_ms)
