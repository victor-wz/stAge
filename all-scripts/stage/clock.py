"""Relative-to-reference (differential) clock prep, elastic-net prediction, and metapixel-to-pixel propagation."""

import gc
import uuid

import numpy as np
import pandas as pd
import scanpy as sc
import sklearn
from scipy.sparse import issparse, csr_matrix


def final_clock_preparation(df, clock_model, diff_suffix=None):
    """Subtract the median of the `diff_suffix`-tagged (control/young) columns per gene, then predict."""
    if diff_suffix is not None:
        df = df.T
        control_group = df.index.str.contains(diff_suffix)
        exprs_control = df.loc[control_group].median(axis=0)
        df = df.sub(exprs_control, axis=1).T

    df.index = df.index.astype(str)

    if isinstance(clock_model, sklearn.pipeline.Pipeline):
        feature_names = clock_model.feature_names_in_
    else:
        feature_names = clock_model.feature_names
    predicted_age = clock_model.predict(df.loc[feature_names].T)

    # Fill missed features (missing-gene imputation: reindexed as NaN rows)
    missed_features = set(feature_names) - set(df.index)
    df_fixed = df.reindex(index=list(df.index) + list(missed_features))

    return df_fixed


def predict_age(df, clock_model):
    if isinstance(clock_model, sklearn.pipeline.Pipeline):
        feature_names = clock_model.feature_names_in_
    else:
        feature_names = clock_model.feature_names
    predicted_age = clock_model.predict(df.loc[feature_names].T)
    return predicted_age


def prepare_prediction_results(df, predicted_age):
    groups = ["Young" if "young" in idx else "Old" for idx in df.columns]
    data = pd.DataFrame({"Predicted Age": predicted_age, "Group": groups, "sample": df.columns})
    return data


def propagate_into_pixel_level(lowres_adata_mp, adata, age_group='', obs_to_propagate=[], propagate_expression=True):
    """
    Propagate metapixel-level expression and annotations into pixel-level adata.

    Parameters:
    - lowres_adata_mp: AnnData with metapixel-level data and annotations (obs)
    - adata: pixel-level AnnData with .obs['metapixel'] indicating group membership
    - age_group: string label to prefix pixel names (e.g. 'young', 'old')
    - obs_to_propagate: list of .obs column names to propagate from metapixel to pixels
    - propagate_expression: if False, skip building the full metapixel-broadcast expression
      matrix (obs-only fast path) -- much cheaper when the caller only needs propagated tAge
      values, since the broadcast matrix is ~90% dense after metapixel aggregation.

    Returns:
    - adata_mp: new pixel-level AnnData with propagated metapixel data and annotations
    """
    metapixel_labels = adata.obs['metapixel'].values
    unique_mp_ids = np.unique(metapixel_labels)

    # Build mp_id -> obs_name lookup once (avoids O(n^2) string scanning per obs key)
    mp_id_to_obs = {}
    for obs_name in lowres_adata_mp.obs.index:
        parts = obs_name.split('.')
        if len(parts) >= 3:
            mp_id_to_obs[parts[2]] = obs_name

    # Propagate obs annotations (always done, cheap -- just scalar lookups)
    mp_id_to_int = {mp_id: i for i, mp_id in enumerate(unique_mp_ids)}
    pixel_to_mp_int = np.array([mp_id_to_int[mp_id] for mp_id in metapixel_labels])

    obs_out = adata.obs.copy()
    for obs_key in obs_to_propagate:
        if obs_key not in lowres_adata_mp.obs:
            continue
        values = np.full(adata.n_obs, np.nan)
        for mp_id in unique_mp_ids:
            obs_name = mp_id_to_obs.get(str(mp_id))
            if obs_name is None:
                continue
            values[metapixel_labels == mp_id] = lowres_adata_mp.obs.loc[obs_name, obs_key]
        obs_out[obs_key] = pd.to_numeric(values, errors='coerce')
    obs_out['pixel_id'] = adata.obs_names

    if not propagate_expression:
        # Obs-only path: propagate tAge obs to spot-level spatial coords, no expression matrix.
        # `var` is still set (cheap -- just gene metadata, not the expression matrix itself) so
        # that callers can safely assign a real expression matrix onto `.X` afterward without a
        # shape mismatch (n_vars would otherwise default to 0).
        adata_mp = sc.AnnData(
            obs=obs_out,
            var=adata.var.copy(),
            obsm={k: v.copy() for k, v in adata.obsm.items()},
        )
        adata_mp.obs_names = [f"{age_group}.group.{pixel_id}.{uuid.uuid4()}" for pixel_id in adata.obs_names]
        return adata_mp

    # Aggregate expression by metapixel (only when propagate_expression=True)
    n_mps = len(unique_mp_ids)
    n_genes = adata.n_vars

    if issparse(adata.X):
        rows = pixel_to_mp_int
        cols = np.arange(adata.n_obs)
        A = csr_matrix((np.ones(adata.n_obs), (rows, cols)), shape=(n_mps, adata.n_obs))
        aggregated = A @ adata.X
        # NOTE: a direct `aggregated[pixel_to_mp_int]` broadcast triggers a scipy sparse
        # fancy-indexing int32 nnz overflow ("ValueError: negative dimensions are not
        # allowed") on large datasets, where the broadcast result's total nonzero count
        # exceeds 2^31-1. Use an explicit pixel->metapixel indicator matmul instead, which
        # avoids that overflow path and keeps `propagated` sparse.
        P = csr_matrix((np.ones(adata.n_obs), (np.arange(adata.n_obs), pixel_to_mp_int)),
                        shape=(adata.n_obs, n_mps))
        propagated = P @ aggregated
    else:
        aggregated = np.zeros((n_mps, n_genes), dtype=adata.X.dtype)
        np.add.at(aggregated, pixel_to_mp_int, adata.X)
        propagated = aggregated[pixel_to_mp_int]

    del aggregated
    gc.collect()

    adata_mp = sc.AnnData(
        X=propagated,
        obs=obs_out,
        var=adata.var.copy(),
        obsm={k: v.copy() for k, v in adata.obsm.items()},
    )
    adata_mp.layers['mp_count'] = propagated
    adata_mp.obs['cumulative_coverage'] = np.array(propagated.sum(axis=1)).ravel()

    del propagated
    gc.collect()

    adata_mp.obs_names = [f"{age_group}.group.{pixel_id}.{uuid.uuid4()}" for pixel_id in adata.obs_names]

    return adata_mp
