"""Composition-independence analysis: OLS residualization of cell-type composition
out of gene expression, and exact symmetric (Reimers/Cotton) Oaxaca-Blinder
decomposition of a hotspot-minus-coldspot tAge gap into compositional vs. intrinsic
components.

Canonical source (per author decision): v_pipeline/spatial_tage_beyond_composition.ipynb
Panels A and B. This replaces celltype_hotspot_tAge.ipynb Section 5's decomposition,
which has a confirmed reference-mean bug (pools in "normal" spots outside the
hotspot/coldspot contrast, and silently drops cell types unique to one side) that
breaks the delta_obs = delta_comp + delta_within identity. Do not use that version.

Maps to Fig S12 (Composition-independence: OLS residualization, Oaxaca-Blinder
decomposition, per-cell-type/region pseudobulks).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.sparse import issparse

from .preprocessing import filter_genes, get_scaled_counts
from .clock import final_clock_preparation, predict_age


def residualize_composition_and_reclock(
    mp_adata,
    ct_universe: list[str],
    clock,
    ncbi_reference_path: str,
    sample_col: str = 'sample_id',
    diff_suffix: str = 'young',
    tage_col: str = 'tAge_SM',
):
    """Panel A: regress cell-type composition + sample identity out of every gene at
    the metapixel level, then re-run the tAge clock on the residual expression matrix.

    `mp_adata` must have `.obsm['composition']` (metapixels x len(ct_universe) fraction
    matrix) and `.obs[sample_col]`. `ncbi_reference_path` is required -- passed straight
    through to `stage.preprocessing.get_scaled_counts` (see that module's docstring for
    why this is now an explicit parameter rather than a hard-coded default). Returns the
    gene-filtered AnnData with two new `.obs` columns added:
      - `tAge_residual`: clock re-run on the composition/sample-residualized matrix.
      - `tAge_reclock`: clock re-run on the ORIGINAL (composition-inclusive) scaled
        matrix, as a sanity check that this re-derivation pipeline reproduces the
        already-propagated `tage_col` value (expect high but not perfect correlation
        — see the source notebook's per-dataset corr(tAge_SM, tAge_reclock) table).

    Design matrix: [intercept | composition fractions (last cell type dropped, since
    fractions sum to 1) | sample-ID dummies (first level dropped)]. Solved for all
    genes simultaneously via `numpy.linalg.lstsq`. The gene's original mean is added
    back to the residual so downstream clock preprocessing sees a realistic scale
    (a pure mean-zero residual would break the differential/young-baseline subtraction
    step in `final_clock_preparation`).
    """
    mpf = filter_genes(mp_adata)
    X = mpf.X.toarray() if issparse(mpf.X) else np.asarray(mpf.X)
    gene_by_mp = pd.DataFrame(X.T, index=mpf.var_names, columns=mpf.obs.index)
    gene_by_mp.index.name = 'geneID'
    counts_scaled = get_scaled_counts(gene_by_mp, clock, ncbi_reference_path, 'symbol')  # entrez x metapixels
    mp_ids = counts_scaled.columns
    obs = mpf.obs.loc[mp_ids]

    comp = pd.DataFrame(mpf.obsm['composition'], index=mpf.obs.index, columns=ct_universe).loc[mp_ids].values
    sample_dummies = pd.get_dummies(obs[sample_col], drop_first=True).astype(float).values
    X_design = np.hstack([np.ones((len(mp_ids), 1)), comp[:, :-1], sample_dummies])

    Y = counts_scaled.values.T.astype(float)  # metapixels x genes
    beta, *_ = np.linalg.lstsq(X_design, Y, rcond=None)
    residual = (Y - X_design @ beta) + Y.mean(axis=0, keepdims=True)
    residual_df = pd.DataFrame(residual.T, index=counts_scaled.index, columns=mp_ids)

    prepped = final_clock_preparation(residual_df, clock, diff_suffix=diff_suffix)
    mpf.obs['tAge_residual'] = (
        pd.Series(np.asarray(predict_age(prepped, clock)) * 48, index=prepped.columns)
        .reindex(mpf.obs.index).values
    )

    prepped0 = final_clock_preparation(counts_scaled.copy(), clock, diff_suffix=diff_suffix)
    mpf.obs['tAge_reclock'] = (
        pd.Series(np.asarray(predict_age(prepped0, clock)) * 48, index=prepped0.columns)
        .reindex(mpf.obs.index).values
    )
    return mpf


def oaxaca_blinder_decomposition(
    mp_adata,
    ct_universe: list[str],
    sample_col: str = 'sample_id',
    aging_type_col: str = 'aging_type',
    tage_col: str = 'tAge_SM',
    n_spots_col: str = 'n_spots',
    age_group_col: str = 'age_group',
) -> pd.DataFrame:
    """Panel B: per-sample, exact symmetric (Reimers/Cotton) two-fold Oaxaca-Blinder
    decomposition of the hotspot-minus-coldspot tAge gap into a compositional term
    (different cell-type mix) and an intrinsic term (same cell type, different tAge).

    Per-cell-type spot counts are reconstructed exactly from each metapixel's
    composition fraction x n_spots (mathematically identical to pooling raw spot-level
    data, since every spot in a metapixel inherits that metapixel's single propagated
    tAge value). Cell types present in only one of hotspot/coldspot for a sample are
    assigned their ENTIRE contribution to `delta_comp` (no within-type comparison is
    possible without data on both sides) — this is what makes the decomposition
    reconstruct `delta_obs` exactly (residual ~1e-7, floating-point noise), unlike the
    superseded celltype_hotspot_tAge.ipynb Section 5 version which silently dropped
    these instead of assigning them.

    Returns one row per sample with delta_obs, delta_comp, delta_within, residual
    (should be ~0 — verify this after running on real data, not fabricated here),
    pct_compositional, pct_intrinsic, and hotspot/coldspot counts.
    """
    rows = []
    comp_all = pd.DataFrame(mp_adata.obsm['composition'], index=mp_adata.obs.index, columns=ct_universe)
    obs = mp_adata.obs.join(comp_all)

    for sample_id, sdf in obs.groupby(sample_col):
        hot = sdf[sdf[aging_type_col] == 'hotspot']
        cold = sdf[sdf[aging_type_col] == 'coldspot']
        if len(hot) == 0 or len(cold) == 0:
            continue

        n_hot_spots = hot[n_spots_col].sum()
        n_cold_spots = cold[n_spots_col].sum()
        delta_obs = (
            np.average(hot[tage_col], weights=hot[n_spots_col])
            - np.average(cold[tage_col], weights=cold[n_spots_col])
        )
        delta_comp = 0.0
        delta_within = 0.0
        for c in ct_universe:
            w_hot_c = hot[c] * hot[n_spots_col]
            w_cold_c = cold[c] * cold[n_spots_col]
            n_hot_c, n_cold_c = w_hot_c.sum(), w_cold_c.sum()
            if n_hot_c <= 0 and n_cold_c <= 0:
                continue
            p_hot_c = n_hot_c / n_hot_spots if n_hot_spots > 0 else 0.0
            p_cold_c = n_cold_c / n_cold_spots if n_cold_spots > 0 else 0.0
            if n_hot_c > 0 and n_cold_c > 0:
                mean_hot_c = (hot[tage_col] * w_hot_c).sum() / n_hot_c
                mean_cold_c = (cold[tage_col] * w_cold_c).sum() / n_cold_c
                delta_comp += (p_hot_c - p_cold_c) * (mean_hot_c + mean_cold_c) / 2.0
                delta_within += (p_hot_c + p_cold_c) / 2.0 * (mean_hot_c - mean_cold_c)
            elif n_hot_c > 0:
                # cell type present only in this sample's hotspot metapixels: entire
                # contribution is compositional (no within-type comparison possible).
                delta_comp += p_hot_c * ((hot[tage_col] * w_hot_c).sum() / n_hot_c)
            else:
                delta_comp += -p_cold_c * ((cold[tage_col] * w_cold_c).sum() / n_cold_c)

        pct_comp = 100.0 * delta_comp / delta_obs if abs(delta_obs) > 1e-9 else np.nan
        pct_within = 100.0 * delta_within / delta_obs if abs(delta_obs) > 1e-9 else np.nan
        rows.append(dict(
            sample_id=sample_id, age_group=sdf[age_group_col].iloc[0],
            delta_obs=delta_obs, delta_comp=delta_comp, delta_within=delta_within,
            residual=delta_obs - (delta_comp + delta_within),
            pct_compositional=pct_comp, pct_intrinsic=pct_within,
            n_hot_mp=len(hot), n_cold_mp=len(cold),
            n_hot_spots=int(n_hot_spots), n_cold_spots=int(n_cold_spots),
        ))
    return pd.DataFrame(rows)
