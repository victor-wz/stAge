"""SpatialGroup metaspot clustering: overlapping (KD-tree radius expansion) and non-overlapping (Leiden) metapixel construction."""

import uuid

# dask/dask.dataframe must be imported before squidpy -- squidpy pulls in spatialdata,
# which imports dask.dataframe transitively; on some dask/dask-expr version pairings
# that cold transitive import raises `NotImplementedError: The legacy implementation
# is no longer supported`, while an explicit warm import here does not. The original
# v_pipeline/st_utils.py imports dask/dask.dataframe before squidpy for this same
# reason (confirmed during Phase 3 parity checking) -- preserved here for the same fix.
import dask
import dask.dataframe as dd

import numpy as np
import pandas as pd
import scanpy as sc
import squidpy as sq
from tqdm import tqdm
from scipy.spatial import cKDTree
from scipy.sparse import issparse, csr_matrix


def select_neighbors(adata, coverage_threshold, age_group, initial_radius=3.0, max_radius=20.0):
    """
    Select neighbors for each cell by expanding a search radius until
    total coverage >= coverage_threshold or we reach max_radius.
    This avoids sorting k=len(spatial_coords) distances for each cell.
    Requires adata.obs['total_counts'], adata.obs['cell_type'], adata.obsm['spatial'],
    and adata.layers['raw_count'] to already be populated.
    """
    cells_df = adata.obs.copy()
    cells_df["cell_id"] = cells_df.index
    cell_ids = cells_df.index.to_list()
    total_counts_array = cells_df["total_counts"].values
    spatial_coords = adata.obsm["spatial"]
    counts_matrix = adata.layers["raw_count"]
    n_cells = len(cells_df)

    results = []
    summed_counts_results = []

    # Precompute cell indices for each cell type
    cell_types = cells_df["cell_type"].unique()
    for cell_type in cell_types:
        cell_type_obs = adata.obs_names[adata.obs["cell_type"] == cell_type]
        kdtree = cKDTree(adata[cell_type_obs].obsm["spatial"])

        cell_type_mask = cells_df["cell_type"] == cell_type
        cell_type_indices = np.where(cell_type_mask)[0]

        for cell_idx in tqdm(cell_type_indices, desc=f"Neighbor selection for {cell_type}"):
            radius = initial_radius
            coverage = 0
            counted = np.zeros(n_cells, dtype=bool)  # Track counted cells
            chosen_neighbors = []

            # Expand radius until coverage threshold or max_radius
            while coverage < coverage_threshold and radius <= max_radius:
                neighbor_idx = kdtree.query_ball_point(spatial_coords[cell_idx], radius)
                neighbor_idx = np.array(neighbor_idx, dtype=int)

                new_neighbors = neighbor_idx[~counted[neighbor_idx]]
                if not new_neighbors.size:
                    radius *= 2
                    continue

                coverage += total_counts_array[new_neighbors].sum()
                counted[new_neighbors] = True
                chosen_neighbors.extend(new_neighbors.tolist())

                if coverage >= coverage_threshold:
                    break
                radius *= 2

            summed_counts = counts_matrix[chosen_neighbors].sum(axis=0)
            summed_counts_results.append(summed_counts.A.flatten() if hasattr(summed_counts, "A") else summed_counts)

            cell_id = cell_ids[cell_idx]
            group_tag = f"{age_group}.group.{cell_id}.{str(uuid.uuid4())}"
            results.append(
                {
                    "cell_id": cell_id,
                    "cell_type": cell_type,
                    "group_tag": group_tag,
                    "cumulative_coverage": coverage,
                }
            )

    result_adata = sc.AnnData(
        X=np.vstack(summed_counts_results),
        var=adata.var.copy(),
        obs=pd.DataFrame(results),
    )
    result_adata.obs = result_adata.obs.set_index("group_tag")
    return result_adata


def non_overlapping_MPs(adata, age_group, lower_res=False, n_neighs=8, resolution=0.5):
    """
    Non-overlapping metapixels via spatially-constrained Leiden clustering.

    `lower_res` is accepted for call-signature compatibility with `select_neighbors`-based
    callers but has no effect here (this function always returns the metapixel-level
    AnnData) -- unlike an earlier version of this function, kept in the project's own
    version history, which also supported returning a pixel-level broadcast.
    """
    # Spatial neighbors
    sq.gr.spatial_neighbors(adata, coord_type='generic', n_neighs=n_neighs)

    # Spatially constrained clustering (Leiden)
    sc.tl.leiden(
        adata,
        adjacency=adata.obsp['spatial_connectivities'],
        resolution=resolution,
        key_added='metapixel'  # Metapixel label
    )

    # Aggregate spatial coordinates (mean per metapixel) -- small (n_metapixels x 2),
    # defines the canonical metapixel ordering used below.
    aggregated_coords = pd.DataFrame(
        adata.obsm['spatial'], index=adata.obs['metapixel']
    ).groupby(level=0).mean()
    mp_labels = aggregated_coords.index

    # Aggregate gene expression data by metapixels via a sparse indicator matmul instead
    # of densifying the full pixels x genes matrix + pandas groupby -- the dense version
    # (adata.X.toarray()) allocates an n_pixels x n_genes dense array (tens of GB on large
    # slices) and is slow; the sparse matmul below never densifies and is a single BLAS/
    # sparse-optimized op, so it's both memory-safe and much faster.
    codes = pd.Categorical(adata.obs['metapixel'], categories=mp_labels).codes
    X = adata.X if issparse(adata.X) else csr_matrix(adata.X)
    indicator = csr_matrix((np.ones(adata.n_obs), (codes, np.arange(adata.n_obs))),
                            shape=(len(mp_labels), adata.n_obs))
    aggregated_X = indicator @ X  # sparse, n_metapixels x n_genes

    # Tag samples with unique IDs for merging
    mp_obs_names = [f"{age_group}.group.{mp_id}.{uuid.uuid4()}" for mp_id in mp_labels]

    lowres_adata_mp = sc.AnnData(
        X=aggregated_X,
        obs=pd.DataFrame(index=mp_obs_names),
        obsm={"spatial": aggregated_coords.values},
        var=pd.DataFrame(index=adata.var_names)
    )

    lowres_adata_mp.obs['cumulative_coverage'] = np.asarray(lowres_adata_mp.X.sum(axis=1)).ravel()

    return lowres_adata_mp
