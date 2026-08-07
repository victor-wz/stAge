"""Getis-Ord Gi* spatial hotspot/coldspot classification, and metapixel-level aggregation by majority vote.

Canonical parameters (confirmed by the paper author against all four independent
in-notebook implementations found during the code audit): k=8 row-standardized
KNN spatial weights, 999 permutations, Benjamini-Hochberg FDR<0.05, |z|>1.

Source: consolidated from v_pipeline/spatial_tage_beyond_composition.ipynb's
`_knn_w_rowstd`/`_classify_gi` helpers (most recent/complete implementation),
cross-checked against the same logic in celltype_hotspot_tAge.ipynb and
spatial_tage_variance_by_age.ipynb — all three use identical parameters.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse import issparse
from sklearn.neighbors import NearestNeighbors
from statsmodels.stats.multitest import multipletests
from esda.getisord import G_Local
from libpysal.weights import WSP, WSP2W

# Canonical Gi* parameters — do not change without re-validating against the
# paper's reported hotspot/coldspot calls (Fig 6a and all figures that reuse
# this classification, e.g. Fig S6a-c, Fig S12).
K_KNN = 8
N_PERMS = 999
FDR_ALPHA = 0.05
GI_Z_THRESH = 1.0


def knn_weights_rowstd(coords: np.ndarray, k: int = K_KNN):
    """Row-standardized k-nearest-neighbor spatial weights matrix (libpysal W)."""
    n = coords.shape[0]
    nn = NearestNeighbors(n_neighbors=min(k + 1, n)).fit(coords)
    _, idx = nn.kneighbors(coords)
    rows_ = np.repeat(np.arange(n), idx.shape[1] - 1)
    cols_ = idx[:, 1:].ravel()
    A = sp.csr_matrix((np.ones_like(rows_, float), (rows_, cols_)), shape=(n, n))
    rs = np.asarray(A.sum(1)).ravel()
    rs[rs == 0] = 1.0
    A = A.multiply(1.0 / rs[:, None]).tocsr()
    w = WSP2W(WSP(A))
    w.transform = 'r'
    return w


def classify_hotspots(
    adata,
    value_col: str,
    k: int = K_KNN,
    n_perms: int = N_PERMS,
    fdr_alpha: float = FDR_ALPHA,
    z_thresh: float = GI_Z_THRESH,
    out_col: str = 'aging_type',
    seed: int = 0,
):
    """Spot-level Getis-Ord Gi* classification into hotspot/normal/coldspot.

    In-place: writes `adata.obs[out_col]` with values in {hotspot, normal, coldspot}.
    Requires `adata.obsm['spatial']` and `adata.obs[value_col]`.

    Rows with a non-finite `value_col` are excluded from the Gi* computation and
    labeled 'normal'. If fewer than 20 valid rows are present, every row is
    labeled 'normal' without running Gi* (matches source notebook behavior).
    """
    n = adata.n_obs
    aging_type = np.full(n, 'normal', dtype=object)
    values = adata.obs[value_col].astype(float).values
    valid = np.isfinite(values)
    if valid.sum() < 20:
        adata.obs[out_col] = aging_type
        return adata

    coords = adata.obsm['spatial'][valid]
    g = G_Local(
        values[valid], knn_weights_rowstd(coords, k=k),
        transform='R', permutations=n_perms, star=True, seed=seed,
    )
    z = np.asarray(g.Zs)
    p = np.where(np.isfinite(g.p_sim), g.p_sim, 1.0)
    rejected, *_ = multipletests(p, alpha=fdr_alpha, method='fdr_bh')

    sub = np.full(valid.sum(), 'normal', dtype=object)
    sub[rejected & (z > z_thresh)] = 'hotspot'
    sub[rejected & (z < -z_thresh)] = 'coldspot'
    aging_type[valid] = sub
    adata.obs[out_col] = aging_type
    return adata


def majority_vote_to_group(labels: pd.Series, group_mask: np.ndarray) -> str:
    """Majority-vote aggregation of spot-level hotspot/normal/coldspot labels up to
    a coarser spatial unit (e.g. metapixel). `labels` are the full spot-level Gi*
    labels (e.g. adata.obs['aging_type']); `group_mask` selects the member spots
    of one group (one metapixel). Ties broken by pandas value_counts().idxmax()
    (first-encountered on tie), matching source notebook behavior exactly."""
    return pd.Series(np.asarray(labels)[group_mask]).value_counts().idxmax()
