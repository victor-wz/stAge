# from pathlib import Path 
import dask
dask.config.set({'dataframe.query-planning': True})
import dask.dataframe as dd

import uuid
import math
import gc
import warnings
from functools import partial
from multiprocessing import Pool
from pathlib import Path
import os

import joblib
import pandas as pd
import numpy as np
import scanpy as sc
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu
from scipy import stats
import glob
import anndata
import squidpy as sq
from scipy.sparse import issparse, csr_matrix
import sklearn 
from sklearn.pipeline import Pipeline


# PIPELINE TO GENERATE tAGE PREDICTIONS FROM ST-OMICS DATA 

# Functions for the whole pipeline, from utils_mod.py:

# Creating metapixels based on KD-trees
def select_neighbors(adata, coverage_threshold, age_group, initial_radius=3.0, max_radius=20.0):
    """
    Select neighbors for each cell by expanding a search radius until
    total coverage >= coverage_threshold or we reach max_radius.
    This avoids sorting k=len(spatial_coords) distances for each cell.
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
        # Build KDTree
        # kdtree = cKDTree(adata[adata.obs["cell_type"] == cell_type, :].obsm["spatial"])
        cell_type_obs = adata.obs_names[adata.obs["cell_type"] == cell_type]
        kdtree = cKDTree(adata[cell_type_obs].obsm["spatial"])

        # Get indices for current cell type
        cell_type_mask = cells_df["cell_type"] == cell_type
        cell_type_indices = np.where(cell_type_mask)[0]

        for cell_idx in tqdm(cell_type_indices, desc=f"Neighbor selection for {cell_type}"):
            radius = initial_radius
            coverage = 0
            counted = np.zeros(n_cells, dtype=bool)  # Track counted cells
            chosen_neighbors = []

            # Expand radius until coverage threshold or max_radius
            while coverage < coverage_threshold and radius <= max_radius:
                # Query neighbors within current radius
                neighbor_idx = kdtree.query_ball_point(spatial_coords[cell_idx], radius)
                neighbor_idx = np.array(neighbor_idx, dtype=int)

                # Find new neighbors not previously counted
                new_neighbors = neighbor_idx[~counted[neighbor_idx]]
                if not new_neighbors.size:
                    # radius += 1
                    radius *= 2
                    continue

                # Update coverage and tracking
                coverage += total_counts_array[new_neighbors].sum()
                counted[new_neighbors] = True
                chosen_neighbors.extend(new_neighbors.tolist())

                # Check if coverage threshold is met
                if coverage >= coverage_threshold:
                    break
                # radius += 1
                radius *= 2

            # Calculate summed counts directly using indices
            summed_counts = counts_matrix[chosen_neighbors].sum(axis=0)
            summed_counts_results.append(summed_counts.A.flatten() if hasattr(summed_counts, "A") else summed_counts)

            # Prepare results entry
            cell_id = cell_ids[cell_idx]
            # group_tag = f"{age_group}_group_{cell_id}_{str(uuid.uuid4())}"
            group_tag = f"{age_group}.group.{cell_id}.{str(uuid.uuid4())}"
            results.append(
                {
                    "cell_id": cell_id,
                    "cell_type": cell_type,
                    "group_tag": group_tag,
                    "cumulative_coverage": coverage,
                }
            )

    # Create result AnnData
    result_adata = sc.AnnData(
        X=np.vstack(summed_counts_results),
        var=adata.var.copy(),
        obs=pd.DataFrame(results),
    )
    result_adata.obs = result_adata.obs.set_index("group_tag")
    return result_adata




# # Create non-overlapping metapixels
# def non_overlapping_MPs(adata, age_group, lower_res=False, n_neighs=8, resolution=0.5): 
#     # Spatial neighbors
#     sq.gr.spatial_neighbors(adata, coord_type='generic', n_neighs=n_neighs)

#     # Spatially constrained clustering (Leiden)
#     sc.tl.leiden(
#         adata, 
#         adjacency=adata.obsp['spatial_connectivities'], 
#         resolution=resolution, 
#         key_added='metapixel' # Metapixel label
#     )

#     # Convert sparse matrix to dense numpy array if necessary
#     if not isinstance(adata.X, np.ndarray):
#         X_dense = adata.X.toarray() if hasattr(adata.X, "toarray") else np.array(adata.X)
#     else:
#         X_dense = adata.X

#     # Aggregate gene expression data by metapixels
#     aggregated_X = pd.DataFrame(X_dense, index=adata.obs['metapixel']).groupby(level=0).sum()

#     # Aggregate spatial coordinates (mean per metapixel)
#     aggregated_coords = pd.DataFrame(
#         adata.obsm['spatial'], index=adata.obs['metapixel']
#     ).groupby(level=0).mean()

#     # Tag samples with unique IDs for merging
#     mp_obs_names = [f"{age_group}.group.{mp_id}.{uuid.uuid4()}" for mp_id in aggregated_coords.index]
#     aggregated_coords.index = mp_obs_names

#     # Create new AnnData object for metapixels
#     lowres_adata_mp = sc.AnnData(
#         X=aggregated_X.values,
#         obs=pd.DataFrame(index=aggregated_coords.index),
#         obsm={"spatial": aggregated_coords.values},
#         var=pd.DataFrame(index=adata.var_names)
#     )
    
#     lowres_adata_mp.obs['cumulative_coverage'] = lowres_adata_mp.X.sum(axis=1) # add this column to obs 

#     # Create an empty matrix for propagated metapixel expressions
#     propagated_mp_expmat = np.zeros_like(X_dense)
    
#     # Assign each pixel the expression of its metapixel
#     for mp_id in aggregated_X.index:
#         pixel_indices = adata.obs.index[adata.obs["metapixel"] == mp_id]
#         propagated_mp_expmat[adata.obs.index.isin(pixel_indices)] = aggregated_X.loc[mp_id].values
    
#     # Convert to NumPy array
#     propagated_mp_expmat = np.array(propagated_mp_expmat)

#     adata_mp = adata.copy()
    
#     # Add new layer with propagated metapixel counts
#     adata_mp.layers['mp_count'] = propagated_mp_expmat
    
#     # Prevent name repetition when merging later
#     mp_obs_names = [f"{age_group}.group.{pixel_id}.{uuid.uuid4()}" for pixel_id in adata.obs_names]
#     adata_mp.obs['pixel_id'] = adata.obs_names
#     adata_mp.obs_names = mp_obs_names
    
#     # Copy mp_counts layer into raw_counts for compatibility
#     adata_mp.layers['raw_count'] = adata_mp.layers['mp_count'].copy()
#     adata_mp.X = adata_mp.layers['mp_count'].copy()
#     adata_mp.obs['cumulative_coverage'] = adata_mp.X.sum(axis=1)
    
#     # Return either the low-resolution metapixel AnnData or the pixel-level AnnData
#     return lowres_adata_mp if lower_res else adata_mp


# Create non-overlapping metapixels
def non_overlapping_MPs(adata, age_group, lower_res=False, n_neighs=8, resolution=0.5): 
    # Spatial neighbors
    sq.gr.spatial_neighbors(adata, coord_type='generic', n_neighs=n_neighs)

    # Spatially constrained clustering (Leiden)
    sc.tl.leiden(
        adata,
        adjacency=adata.obsp['spatial_connectivities'],
        resolution=resolution,
        key_added='metapixel' # Metapixel label
    )

    # Convert sparse matrix to dense numpy array if necessary
    if not isinstance(adata.X, np.ndarray):
        X_dense = adata.X.toarray() if hasattr(adata.X, "toarray") else np.array(adata.X)
    else:
        X_dense = adata.X

    # Aggregate gene expression data by metapixels
    aggregated_X = pd.DataFrame(X_dense, index=adata.obs['metapixel']).groupby(level=0).sum()

    # Aggregate spatial coordinates (mean per metapixel)
    aggregated_coords = pd.DataFrame(
        adata.obsm['spatial'], index=adata.obs['metapixel']
    ).groupby(level=0).mean()

    # Tag samples with unique IDs for merging
    mp_obs_names = [f"{age_group}.group.{mp_id}.{uuid.uuid4()}" for mp_id in aggregated_coords.index]
    aggregated_coords.index = mp_obs_names

    # Create new AnnData object for metapixels
    lowres_adata_mp = sc.AnnData(
        X=aggregated_X.values,
        obs=pd.DataFrame(index=aggregated_coords.index),
        obsm={"spatial": aggregated_coords.values},
        var=pd.DataFrame(index=adata.var_names)
    )
    
    lowres_adata_mp.obs['cumulative_coverage'] = lowres_adata_mp.X.sum(axis=1) # add this column to obs 

    # Return either the low-resolution metapixel AnnData or the pixel-level AnnData
    return lowres_adata_mp #if lower_res else adata_mp


def filter_genes(adata, layer_name=None):
    """
    Filter genes that are expressed with at least 10 counts in at least 20% of the cells.
    """
    
    # from scipy.sparse import issparse, csr_matrix

    # Extract counts
    matrix = adata.X if layer_name is None else adata.layers[layer_name]

    # Convert to sparse matrix to sparse for memory efficiency
    if not issparse(matrix):
        matrix = csr_matrix(matrix)

    # Define threshold for gene filtering
    threshold = math.ceil(0.2 * adata.n_obs)

    # Create gene filter mask.
    # 1. (matrix >= 10) -> Boolean mask where values >= 10
    # 2. .sum(axis=0) -> Sum along columns (genes) to count cells per gene with values >= 10
    # 3. >= threshold -> Boolean mask where sum >= threshold
    gene_counts = np.array((matrix >= 10).sum(axis=0)).flatten()
    gene_filter = gene_counts >= threshold

    # Filter the AnnData object based on the gene filter
    adata_filtered = adata[:, gene_filter].copy()

    # Copy only required obsm keys explicitly if needed
    if "spatial" in adata.obsm:
        adata_filtered.obsm["spatial"] = adata.obsm["spatial"].copy()

    return adata_filtered



# Normalize before filtering for clock features 
def preprocess_counts(
    df_counts,
    gene_name_column,
    clock_model,
    original_ids,
    ncbi_reference_path=f"{os.getcwd()}/Mus_musculus.gene_info"
):
    """
    Preprocess gene count data for biological clock prediction.

    Steps include:
    1. Load NCBI gene reference data
    2. Map gene names to standardized Gene IDs
    3. Normalize and scale data
    4. Handle missing genes and duplicates
    5. Filter and align data with model's expected features
    """
    # # Get clock gene IDs
    # feature_names = [int(x) for x in clock_model.feature_names]

    # if isinstance(clock_model, sklearn.pipeline.Pipeline):
    #     feature_names = clock_model.feature_names_in_
    # else: 
    #      feature_names = clock_model.feature_names

    import sklearn

    if isinstance(clock_model, sklearn.pipeline.Pipeline):
        feature_names = [int(x) for x in clock_model.feature_names_in_]
    else: 
        feature_names = [int(x) for x in clock_model.feature_names]
        

    # 1. Load NCBI gene reference data

    # Load NCBI reference data and prepare synonym mappings
    ncbi_genes = pd.read_table(ncbi_reference_path)

    # Combine official symbols with alternative synonyms for name mapping
    ncbi_genes["Synonyms_Combined"] = ncbi_genes["Synonyms"] + "|" + ncbi_genes["Symbol"]

    # 2. Map gene names to standardized Gene IDs

    if original_ids == 'symbol':
    
        # Expand synonyms and create a direct mapping dictionary
        gene_name_to_gene_id = {}
        for _, row in ncbi_genes.iterrows():
            gene_id = row["GeneID"]
    
            synonyms = [row["Symbol"]]
    
            for synonym in synonyms:
                gene_name_to_gene_id[synonym] = gene_id  # Map each synonym to the GeneID

    elif original_ids == 'ensembl':
        # Expand synonyms and create a direct mapping dictionary
        gene_name_to_gene_id = {}
        for _, row in ncbi_genes.iterrows():
            gene_id = row["GeneID"]
            synonyms = [row["Symbol"]]  # Include gene symbol
        
            # Extract Ensembl ID from dbXrefs if present
            if isinstance(row["dbXrefs"], str) and "Ensembl:" in row["dbXrefs"]:
                ensembl_id = [x.split(":")[-1] for x in row["dbXrefs"].split("|") if x.startswith("Ensembl")][0]
                synonyms.append(ensembl_id)  # Add Ensembl ID to mapping
        
            # Map both symbols and Ensembl IDs to Entrez GeneID
            for synonym in synonyms:
                gene_name_to_gene_id[synonym] = gene_id
    

    # Map gene names to standardized Gene IDs
    df_counts["mapped_geneID"] = df_counts[gene_name_column].map(gene_name_to_gene_id)

    # Remove unmapped genes and original gene name column
    df_filtered = df_counts[df_counts["mapped_geneID"].notna()].drop(columns=[gene_name_column])
    df_filtered["mapped_geneID"] = df_filtered["mapped_geneID"].astype(int) # turn the entrez id column from float to int 
    
    # 3. Handle missing genes and duplicates

    # Drop duplicates using 'mapped_geneID' index. Select rows with the highest sum of counts.
    index_map = {}
    sums_map = {}
    for id, row in df_filtered.iterrows():
        gene_id = int(row["mapped_geneID"])
        row_selected = row[row.index != "mapped_geneID"]

        if gene_id not in index_map:
            index_map[gene_id] = id
            sums_map[gene_id] = 0

        row_sum = row_selected.sum()
        if row_sum > sums_map[gene_id]:
            index_map[gene_id] = id
            sums_map[gene_id] = row_sum
    df_filtered = df_filtered.loc[list(index_map.values())].reset_index(drop=True)

    # Set Gene ID as index and align with model's feature order
    df_filtered = df_filtered.set_index("mapped_geneID")
    df_aligned = df_filtered.reindex(feature_names)
    
    # 4. Normalize and scale data
    
    # Log-transform
    df_log = np.log1p(df_aligned)

    # Standard scale data (per sample normalization)
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df_log.values)
    df_scaled = pd.DataFrame(scaled_data, index=df_log.index, columns=df_log.columns)

    # 5. Filter and align data with model's expected features

    # Get model's required features as a set for fast lookups
    df_scaled = df_scaled.loc[df_scaled.index.isin(feature_names)]#.reset_index(drop=True)


    return df_scaled


def YuGene(data_prop, progress_bar=True):
    """
    Fast YuGene transformation for gene expression data.

    Parameters
    ----------
    data_prop : pd.DataFrame or np.ndarray
        DataFrame (or array) with genes as rows and samples as columns.
    progress_bar : bool
        Whether to show a progress bar over samples.

    Returns
    -------
    pd.DataFrame
        DataFrame of YuGene-transformed values with the same shape/index/columns.
    """
    # Shift data to non-negative range (minimum at zero)
    # Vectorized operation replaces lambda for better performance
    data_prop = data_prop - data_prop.min(axis=0)
    data_prop = data_prop.fillna(0)

    # -------------------------------------------------------------------------
    # 1. Ensure we have a NumPy array and store index/columns if DataFrame
    # -------------------------------------------------------------------------
    if isinstance(data_prop, pd.DataFrame):
        row_index = data_prop.index
        col_index = data_prop.columns
        data = data_prop.values
    else:
        # If it's already an ndarray, create default range indices
        data = np.asarray(data_prop, dtype=np.float64)
        row_index = range(data.shape[0])
        col_index = range(data.shape[1])

    # -------------------------------------------------------------------------
    # 2. Clip negative values in-place to 0 (check only if needed)
    # -------------------------------------------------------------------------
    if (data < 0).any():
        print("Warning: some negative values were set to 0")
        np.clip(data, 0, None, out=data)

    # -------------------------------------------------------------------------
    # 3. Prepare an output array
    # -------------------------------------------------------------------------
    # Same shape, will hold the YuGene-transformed values
    result = np.empty_like(data)

    # -------------------------------------------------------------------------
    # 4. Process each column (sample) in a loop (cannot avoid sorting each one)
    # -------------------------------------------------------------------------
    n_cols = data.shape[1]
    for j in tqdm(range(n_cols), disable=not progress_bar, desc="Processing samples"):
        # Extract a single column (sample)
        col_data = data[:, j]

        # Sort (descending). argsort gives ascending, so we invert with [::-1].
        sort_idx = np.argsort(col_data)[::-1]
        sorted_vals = col_data[sort_idx]

        # Compute cumulative sums
        cumsum_vals = np.cumsum(sorted_vals)
        total = cumsum_vals[-1]

        # Handle the edge case: if total == 0, everything is 0 => YuGene = 1
        if total == 0:
            result[:, j] = 1.0
            continue

        cumprop = cumsum_vals / total

        # ---------------------------------------------------------------------
        # 4a. Handle duplicates:
        #     All identical expression values should map to the same
        #     cumulative proportion. We do a single pass to propagate
        #     the "last unique" value forward.
        # ---------------------------------------------------------------------
        for i in range(1, len(cumprop)):
            if sorted_vals[i] == sorted_vals[i - 1]:
                cumprop[i] = cumprop[i - 1]

        # ---------------------------------------------------------------------
        # 4b.  Final YuGene transform is 1 - cumprop
        #      "cumprop" is sorted by descending expression; invert ordering
        # ---------------------------------------------------------------------
        final_col = 1.0 - cumprop

        # Place values back in result (unsort them to original row order)
        result[sort_idx, j] = final_col

    # -------------------------------------------------------------------------
    # 5. Convert result array back to a DataFrame
    # -------------------------------------------------------------------------
    result_df = pd.DataFrame(result, index=row_index, columns=col_index)
    return result_df


def get_YuGene_counts(df, clock_model, original_ids='symbol'):
    df_preprocessed = preprocess_counts(df.reset_index(), "geneID", clock_model=clock_model, original_ids=original_ids, ncbi_reference_path=f"{os.getcwd()}/Mus_musculus.gene_info")
    df_YuGene = YuGene(df_preprocessed)
    return df_YuGene


def get_scaled_counts(df, clock_model, original_ids='symbol'):
    df_preprocessed = preprocess_counts(df.reset_index(), "geneID", clock_model=clock_model, original_ids=original_ids)
    return df_preprocessed

# def final_clock_preparation(df, clock_model, diff_suffix=None):
#     if diff_suffix is not None:
#         # Diff
#         df = df.T
#         control_group = df.index.str.contains(diff_suffix)
#         exprs_control = df.loc[control_group].median(axis=0)
#         df = df.sub(exprs_control, axis=1).T

#     # Consistent data type for the gene ids
#     df.index = df.index.astype(str)

#     # Fill missed
#     missed_features = set(clock_model.feature_names) - set(df.index)
#     df_fixed = df.reindex(index=list(df.index) + list(missed_features))

#     return df_fixed

def final_clock_preparation(df, clock_model, diff_suffix=None):
    if diff_suffix is not None:
        # Diff
        df = df.T
        control_group = df.index.str.contains(diff_suffix)
        exprs_control = df.loc[control_group].median(axis=0)
        df = df.sub(exprs_control, axis=1).T

    # Consistent data type for the gene ids
    df.index = df.index.astype(str)

    import sklearn
    if isinstance(clock_model, sklearn.pipeline.Pipeline):
        feature_names = clock_model.feature_names_in_
    else: 
        feature_names = clock_model.feature_names
    predicted_age = clock_model.predict(df.loc[feature_names].T)

    # Fill missed
    missed_features = set(feature_names) - set(df.index)
    df_fixed = df.reindex(index=list(df.index) + list(missed_features))

    return df_fixed


# def predict_age(df, clock_model):
#     predicted_age = clock_model.predict(df.loc[clock_model.feature_names].T)
#     return predicted_age

def predict_age(df, clock_model):
    import sklearn
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



def propagate_into_pixel_level(lowres_adata_mp, adata, age_group='', obs_to_propagate=[]):
    """
    Propagate metapixel-level expression and annotations into pixel-level adata.

    Parameters:
    - lowres_adata_mp: AnnData with metapixel-level data and annotations (obs)
    - adata: pixel-level AnnData with .obs['metapixel'] indicating group membership
    - age_group: string label to prefix pixel names (e.g. 'young', 'old')
    - obs_to_propagate: list of .obs column names to propagate from metapixel to pixels

    Returns:
    - adata_mp: new pixel-level AnnData with propagated metapixel data and annotations
    """

    # --- 1. Handle sparse or dense .X ---
    if not isinstance(adata.X, np.ndarray):
        X_dense = adata.X.toarray() if hasattr(adata.X, "toarray") else np.array(adata.X)
    else:
        X_dense = adata.X

    # --- 2. Aggregate expression by metapixel ---
    aggregated_X = pd.DataFrame(X_dense, index=adata.obs['metapixel']).groupby(level=0).sum()

    # --- 3. Create propagated expression matrix ---
    propagated_mp_expmat = np.zeros_like(X_dense)
    for mp_id in aggregated_X.index:
        pixel_mask = adata.obs["metapixel"] == mp_id
        propagated_mp_expmat[pixel_mask.values] = aggregated_X.loc[mp_id].values

    # --- 4. Copy adata and assign propagated expression to layers ---
    adata_mp = adata.copy()
    adata_mp.layers['mp_count'] = propagated_mp_expmat
    adata_mp.layers['raw_count'] = propagated_mp_expmat.copy()
    adata_mp.X = propagated_mp_expmat.copy()
    adata_mp.obs['cumulative_coverage'] = adata_mp.X.sum(axis=1)

    # --- 5. Propagate selected annotations from lowres_adata_mp.obs ---
    for obs_key in obs_to_propagate:
        if obs_key in lowres_adata_mp.obs:
            propagated_values = np.empty(adata.n_obs, dtype=object)
            # for mp_id in aggregated_X.index:
            #     if mp_id in lowres_adata_mp.obs.index:
            #         pixel_mask = adata.obs["metapixel"] == mp_id
            #         value = lowres_adata_mp.obs.loc[mp_id, obs_key]
            #         propagated_values[pixel_mask.values] = value
            for mp_id in aggregated_X.index:
                # Match based on numeric ID
                matches = [idx for idx in lowres_adata_mp.obs.index if f'.{mp_id}.' in idx]
                if matches:
                    matched_idx = matches[0]
                    value = lowres_adata_mp.obs.loc[matched_idx, obs_key]
                    pixel_mask = adata.obs["metapixel"] == mp_id
                    propagated_values[pixel_mask.values] = value
            adata_mp.obs[obs_key] = propagated_values
            
            # Try coercing to float if possible
            try:
                adata_mp.obs[obs_key] = pd.to_numeric(adata_mp.obs[obs_key], errors='coerce')
            except Exception as e:
                print(f"Could not convert '{obs_key}' to numeric: {e}")

    # --- 6. Assign unique pixel names ---
    adata_mp.obs['pixel_id'] = adata.obs_names
    adata_mp.obs_names = [f"{age_group}.group.{pixel_id}.{uuid.uuid4()}" for pixel_id in adata.obs_names]

    return adata_mp


####

# Function to APPLY THE WHOLE PIPELINE on a dictionary containing a cohort of adata objects that use one (or more) of them as a control (e.g. all adatas for different slices of a tissue that use young ones as control, all slices from the same embryo experiment that use E9.5 slices as control)

def full_mp_pipeline(anndata_dict, # takes a dictionary with file names as keys and anndatas as items
                     radius_df = None,
                     control_file_pattern = '_Y_',
                     mp_coverage_threshold = 150_000,
                     save_plot = False,
                     save_result = True,
                     save_dir = '',
                     tag = '',
                     # cell_type_specific = True
                    ):
    
    tissue_adatas = []

    for file, tissue_slice in anndata_dict.items():
            
        # print(file)
    
        # Step 0: Load the file and set parameters
        # tissue_slice = sc.read_h5ad(f'{rawdata_dir}/{file}') 

        # Young files (=files containing CONTROL samples) must have an identifying pattern
        # control_file_pattern = '_Y_'
    
        # Using the maximum distance thresholds df, select the one for that tissue 
        if radius_df is not None and not radius_df.empty:
            max_dist = radius_df[radius_df['Tissue'] == tag]['Radius'].values[0]  # Extract float, not Series
        else: 
            max_dist = 400.0

        # Minumum metapixel distance, usually always 1.0  
        min_dist = 1.0
        
        print(f'Grouping cells in metapixels with radius r = ({min_dist},{max_dist}) ')

        # Specify age group to tag metapixel names later 
        age_group = "young" if control_file_pattern in file else "old"
    
        # Step 1: Group pixels into MPs with a certain coverage and distance threshold 
        grouped_adata = select_neighbors( # this function might need to be modified
                adata=tissue_slice, 
                coverage_threshold=mp_coverage_threshold, 
                age_group=age_group,  
                initial_radius=min_dist, 
                max_radius=max_dist
            )
        
        ## Pass the spatial coordinates into the new grouped_adata
        # Extract barcodes from preds_per_file and filter matching observations in adata
        obs_filter = tissue_slice.obs_names.isin(grouped_adata.obs_names.str.split(".").str[2])
        # # obs_filter = np.array([any(sub in x for sub in grouped_adata.obs_names) for x in tissue_slice.obs_names])
        
        # Assign filtered spatial coordinates
        grouped_adata.obsm["spatial"] = tissue_slice[obs_filter].obsm["spatial"].copy()

        print(f'Before filtering there are {grouped_adata.n_obs} samples left and {grouped_adata.n_vars} genes left.')
        
        ## Filter out MPs with less than 100k reads
        grouped_adata2 = grouped_adata[grouped_adata.obs['cumulative_coverage'] >= mp_coverage_threshold].copy()

        # Make names unique 
        grouped_adata2.var_names_make_unique()
        grouped_adata2.obs_names_make_unique()

        ## Add tag to keep track of the file the samples are from 
        # grouped_adata2.obs.loc[:, 'File'] = file
        grouped_adata2.obs['File'] = file

        tissue_adatas.append(grouped_adata2)

    # Concatenate tissue adatas along observations (axis=0)
    merged_adata = sc.concat(tissue_adatas, join="outer", axis=0)

    # Filter out genes with less than 10 counts in at least 20% of samples
    ad_filtered = filter_genes(merged_adata)

    print(f'After filtering there are {ad_filtered.n_obs} samples left and {ad_filtered.n_vars} genes left.')

    # Create a dataframe out of the ad_filtered
    df = pd.DataFrame(ad_filtered.X.T, index=ad_filtered.var_names, columns=ad_filtered.obs.index)
    df.index.name = 'geneID' # the get_scaled_counts function requires this 

    # # Load the clocks (ORIGINAL)
    # clock_model = joblib.load('/home/vvicente/spatial_aging/tAge_clocks/EN differential models 4.6/EN_Chronoage_Mouse_All_WT_scaleddiff.pkl')
    # clock_model_yugene = joblib.load('/home/vvicente/spatial_aging/tAge_clocks/EN differential models 4.6/EN_Chronoage_Mouse_All_WT_yugenediff.pkl')

    # Load the clocks (TMS-trained)
    clock_model = joblib.load('/home/vvicente/spatial_aging/tAge_clocks/tms_clocks/EN_Chronoage_Mouse_All_WT_scaleddiff.pkl')
    clock_model_yugene = joblib.load('/home/vvicente/spatial_aging/tAge_clocks/tms_clocks/EN_Chronoage_Mouse_All_WT_yugenediff.pkl')


    # Get the scaled counts 
    counts_scaled = get_scaled_counts(df, clock_model, 'symbol')

    # Perform YugGene transformation too 
    counts_yugene = get_YuGene_counts(df, clock_model_yugene, 'symbol')

    ## Save density plots
    if save_plot:

        import os
        os.makedirs(save_dir, exist_ok=True) # make sure the directory exists
        
        # Create a figure
        plt.figure()
    
        # Plot the scaled count values to check gene count distributions across metapixels
        for _, row in counts_scaled.T[:100].iterrows():
            sns.kdeplot(row)
    
        plt.xlabel('Scaled value')
    
        # Save the plot
        plt.savefig(f"{save_dir}/{tag}_scaled_plot.png", dpi=300, bbox_inches='tight')
    
        # Close the figure to prevent it from displaying
        plt.close()

        # Create a figure
        plt.figure()
    
        # Plot the scaled count values to check gene count distributions across metapixels
        for _, row in counts_yugene.T.iterrows():
            sns.kdeplot(row)
    
        plt.xlabel('Yugene value')
    
        # Save the plot
        plt.savefig(f"{save_dir}/{tag}_yugene_plot.png", dpi=300, bbox_inches='tight')
    
        # Close the figure to prevent it from displaying
        plt.close()

    # Subtract control samples to make expressions relative 
    preprocessed_scaled = final_clock_preparation(counts_scaled, clock_model, diff_suffix = "young")
    preprocessed_yugene = final_clock_preparation(counts_yugene, clock_model_yugene, diff_suffix = "young")

    # Apply the clock
    age_predictions_scaled = predict_age(preprocessed_scaled, clock_model)
    age_predictions_yugene = predict_age(preprocessed_yugene, clock_model_yugene)

    # Continue the pipeline 
    prediction_results_scaled = prepare_prediction_results(preprocessed_scaled, age_predictions_scaled)
    prediction_results_yugene = prepare_prediction_results(preprocessed_yugene, age_predictions_yugene)

    # Ensure 'sample' is set as the index for alignment
    prediction_results_scaled = prediction_results_scaled.set_index("sample")
    prediction_results_yugene = prediction_results_yugene.set_index("sample")

    # Assign values ensuring correct alignment with ad_filtered.obs.index
    ad_filtered.obs["tAge_SM"] = prediction_results_scaled.loc[ad_filtered.obs.index, "Predicted Age"]*48
    ad_filtered.obs["tAge_YM"] = prediction_results_yugene.loc[ad_filtered.obs.index, "Predicted Age"]*48

    # Save the predictions as an AnnData object
    ad_filtered.X = None # Delete the counts matrix to save space 

    if save_result == True:
        ad_filtered.obs.to_parquet(f'{save_dir}/{tag}_preds.parquet', index = True)
        
    # Split ad_filtered into their original dataframes and return a dictionary with the original predictions 
    # Separate the tissue sections creating a dictionary
    adatas_dict = {}

    for file in ad_filtered.obs["File"].unique():
        adatas_dict[file] = ad_filtered[ad_filtered.obs["File"] == file].copy()

    return adatas_dict
    
    # preds_per_tissue[specific_tissue] = ad_filtered # Add to dictionary 



# Function to APPLY THE WHOLE non-overlapping PIPELINE on a dictionary containing a cohort of adata objects that use one (or more) of them as a control (e.g. all adatas for different slices of a tissue that use young ones as control, all slices from the same embryo experiment that use E9.5 slices as control)
def full_nonoverlap_mp_pipeline(anndata_dict, # takes a dictionary with file names as keys and anndatas as items
                                 res=2,
                                 lower_res = False,
                                 control_file_pattern = '_Y_',
                                 mp_coverage_threshold = 150_000,
                                 save_plot = False,
                                 save_result = True,
                                 clock_folder = 'tAge_clocks/EN differential models 4.6',
                                 save_dir = '',
                                 tag = '',
                                 # cell_type_specific = True
                                ):
    tissue_adatas = []

    for file, tissue_slice in anndata_dict.items():

        print(f'Analyzing sample {file}')

        # Specify age group to tag metapixel names later 
        age_group = "young" if control_file_pattern in file else "old"

        # Step 1: Group pixels into MPs with a certain coverage and distance threshold 
        
        # NON-OVERLAPPING MP GROUPING
        grouped_adata = non_overlapping_MPs(adata=tissue_slice, 
                                            age_group=age_group, 
                                            lower_res=lower_res, 
                                            n_neighs=20, 
                                            resolution=res)
        
        ## Pass the spatial coordinates into the new grouped_adata
        # Extract barcodes from preds_per_file and filter matching observations in adata
        obs_filter = tissue_slice.obs_names.isin(grouped_adata.obs_names.str.split(".").str[2])

        # if lower_res == False:
        #     # Assign filtered spatial coordinates
        #     grouped_adata.obsm["spatial"] = tissue_slice[obs_filter].obsm["spatial"].copy()

        print(f'Before filtering there are {grouped_adata.n_obs} samples left and {grouped_adata.n_vars} genes left.')
        
        ## Filter out MPs with less than 100k reads
        grouped_adata2 = grouped_adata[grouped_adata.obs['cumulative_coverage'] >= mp_coverage_threshold].copy()

        ## Add tag to keep track of the file the samples are from 
        grouped_adata2.obs['File'] = file

        # Fix potential index issues making elements unique 
        grouped_adata2.var_names_make_unique()
        grouped_adata2.obs_names_make_unique()

        tissue_adatas.append(grouped_adata2)

    # Concatenate tissue adatas along observations (axis=0)
    merged_adata = sc.concat(tissue_adatas, join="outer", axis=0)

    # Filter out genes with less than 10 counts in at least 20% of samples
    ad_filtered = filter_genes(merged_adata)
    
    print(f'After filtering there are {ad_filtered.n_obs} samples left and {ad_filtered.n_vars} genes left.')

    # Load the clocks 
    clock_model = joblib.load(f'/home/vvicente/spatial_aging/{clock_folder}/EN_Chronoage_Mouse_All_WT_scaleddiff.pkl')
    clock_model_yugene = joblib.load(f'/home/vvicente/spatial_aging/{clock_folder}/EN_Chronoage_Mouse_All_WT_yugenediff.pkl')

    # Create a dataframe out of the ad_filtered
    df = pd.DataFrame(ad_filtered.X.T, index=ad_filtered.var_names, columns=ad_filtered.obs.index)
    df.index.name = 'geneID' # the get_scaled_counts function requires this 

    # Get the scaled counts 
    counts_scaled = get_scaled_counts(df, clock_model, 'symbol')

    # Perform YugGene transformation too 
    counts_yugene = get_YuGene_counts(df, clock_model_yugene, 'symbol')

    ## Save density plots
    if save_plot:

        import os
        os.makedirs(save_dir, exist_ok=True) # make sure the directory exists
        
        # Create a figure
        plt.figure()
    
        # Plot the scaled count values to check gene count distributions across metapixels
        for _, row in counts_scaled.T[:100].iterrows():
            sns.kdeplot(row)
    
        plt.xlabel('Scaled value')
    
        # Save the plot
        plt.savefig(f"{save_dir}/{tag}_scaled_plot.png", dpi=300, bbox_inches='tight')
    
        # Close the figure to prevent it from displaying
        plt.close()

        # Create a figure
        plt.figure()
    
        # Plot the scaled count values to check gene count distributions across metapixels
        for _, row in counts_yugene.T.iterrows():
            sns.kdeplot(row)
    
        plt.xlabel('Yugene value')
    
        # Save the plot
        plt.savefig(f"{save_dir}/{tag}_yugene_plot.png", dpi=300, bbox_inches='tight')
    
        # Close the figure to prevent it from displaying
        plt.close()

    # Subtract control samples to make expressions relative 
    preprocessed_scaled = final_clock_preparation(counts_scaled, clock_model, diff_suffix = "young")
    preprocessed_yugene = final_clock_preparation(counts_yugene, clock_model_yugene, diff_suffix = "young")

    # Apply the clock
    age_predictions_scaled = predict_age(preprocessed_scaled, clock_model)
    age_predictions_yugene = predict_age(preprocessed_yugene, clock_model_yugene)

    # Continue the pipeline 
    prediction_results_scaled = prepare_prediction_results(preprocessed_scaled, age_predictions_scaled)
    prediction_results_yugene = prepare_prediction_results(preprocessed_yugene, age_predictions_yugene)

    # Ensure 'sample' is set as the index for alignment
    prediction_results_scaled = prediction_results_scaled.set_index("sample")
    prediction_results_yugene = prediction_results_yugene.set_index("sample")

    # Assign values ensuring correct alignment with ad_filtered.obs.index
    ad_filtered.obs["tAge_SM"] = prediction_results_scaled.loc[ad_filtered.obs.index, "Predicted Age"]*48
    ad_filtered.obs["tAge_YM"] = prediction_results_yugene.loc[ad_filtered.obs.index, "Predicted Age"]*48

    # Save the predictions as an AnnData object
    # ad_filtered.X = None if not lower_res else ad_filtered.X # Delete the counts matrix to save space if it's propagated (lower_res = False)
    ad_filtered.obsm = merged_adata.obsm.copy()
    ad_filtered.obs["centroid_id"] = [f"{x}_{y}" for x, y in ad_filtered.obsm["spatial"]]

    if save_result == True:
        # ad_filtered.obs.to_parquet(f'{save_dir}/{tag}_preds.parquet', index = True)
        ad_filtered.write_h5ad(f'{save_dir}/{tag}_preds.h5ad')
        
    # Split ad_filtered into their original dataframes and return a dictionary with the original predictions 
    # Separate the tissue sections creating a dictionary
    adatas_dict = {}

    # Divide and propagate 
    if lower_res: 
        for file in ad_filtered.obs["File"].unique():
            adatas_dict[file] = ad_filtered[ad_filtered.obs["File"] == file].copy()

    elif not lower_res: 
        for file in ad_filtered.obs["File"].unique():
            adata = anndata_dict[file] # get the sample's original pixel-level adata
            lowres_adata_mp = ad_filtered[ad_filtered.obs["File"] == file].copy() # get the samle's lower resolution metapixel-level adata
            age_group = "young" if control_file_pattern in file else "old" # Specify age group to tag metapixel names later acts like a tag for the metapixel names
            # Propagate predictions and obs before saving 
            adata_mp = propagate_into_pixel_level(lowres_adata_mp, adata, 
                                                  age_group, 
                                                  obs_to_propagate=['tAge_YM', 'tAge_SM'])
            # For each pixel, save both the original pixel expression and the expression of the metapixel they belong to
            adata_mp.layers['mp_counts'] = adata_mp.X
            adata_mp.X = adata.X
            adatas_dict[file] = adata_mp # save
    else: 
        print("Error adding predictions to the dictionary...")
        
    return adatas_dict





# # Function to APPLY THE WHOLE non-overlapping PIPELINE on a dictionary containing a cohort of adata objects that use one (or more) of them as a control (e.g. all adatas for different slices of a tissue that use young ones as control, all slices from the same embryo experiment that use E9.5 slices as control)
# def full_nonoverlap_mp_pipeline(anndata_dict, # takes a dictionary with file names as keys and anndatas as items
#                                  res=2,
#                                  lower_res = False,
#                                  control_file_pattern = '_Y_',
#                                  mp_coverage_threshold = 150_000,
#                                  save_plot = False,
#                                  save_result = True,
#                                  clock_folder = 'tAge_clocks/EN differential models 4.6',
#                                  save_dir = '',
#                                  tag = '',
#                                  # cell_type_specific = True
#                                 ):
    
#     tissue_adatas = []

#     for file, tissue_slice in anndata_dict.items():

#         print(f'Analyzing sample {file}')

#         # Specify age group to tag metapixel names later 
#         age_group = "young" if control_file_pattern in file else "old"

#         # Step 1: Group pixels into MPs with a certain coverage and distance threshold 
        
#         # NON-OVERLAPPING MP GROUPING
#         grouped_adata = non_overlapping_MPs(adata=tissue_slice, 
#                                             age_group=age_group, 
#                                             lower_res=lower_res, 
#                                             n_neighs=20, 
#                                             resolution=res)
        
#         ## Pass the spatial coordinates into the new grouped_adata
#         # Extract barcodes from preds_per_file and filter matching observations in adata
#         obs_filter = tissue_slice.obs_names.isin(grouped_adata.obs_names.str.split(".").str[2])

#         if lower_res == False:
#             # Assign filtered spatial coordinates
#             grouped_adata.obsm["spatial"] = tissue_slice[obs_filter].obsm["spatial"].copy()

#         print(f'Before filtering there are {grouped_adata.n_obs} samples left and {grouped_adata.n_vars} genes left.')
        
#         ## Filter out MPs with less than 100k reads
#         grouped_adata2 = grouped_adata[grouped_adata.obs['cumulative_coverage'] >= mp_coverage_threshold].copy()

#         ## Add tag to keep track of the file the samples are from 
#         grouped_adata2.obs['File'] = file

#         # Fix potential index issues making elements unique 
#         grouped_adata2.var_names_make_unique()
#         grouped_adata2.obs_names_make_unique()

#         tissue_adatas.append(grouped_adata2)

#     # Concatenate tissue adatas along observations (axis=0)
#     merged_adata = sc.concat(tissue_adatas, join="outer", axis=0)

#     # Filter out genes with less than 10 counts in at least 20% of samples
#     ad_filtered = filter_genes(merged_adata)
    
#     print(f'After filtering there are {ad_filtered.n_obs} samples left and {ad_filtered.n_vars} genes left.')

#     # Load the clocks 
#     clock_model = joblib.load(f'/home/vvicente/spatial_aging/{clock_folder}/EN_Chronoage_Mouse_All_WT_scaleddiff.pkl')
#     clock_model_yugene = joblib.load(f'/home/vvicente/spatial_aging/{clock_folder}/EN_Chronoage_Mouse_All_WT_yugenediff.pkl')

#     # Create a dataframe out of the ad_filtered
#     df = pd.DataFrame(ad_filtered.X.T, index=ad_filtered.var_names, columns=ad_filtered.obs.index)
#     df.index.name = 'geneID' # the get_scaled_counts function requires this 

#     # Get the scaled counts 
#     counts_scaled = get_scaled_counts(df, clock_model, 'symbol')

#     # Perform YugGene transformation too 
#     counts_yugene = get_YuGene_counts(df, clock_model_yugene, 'symbol')

#     ## Save density plots
#     if save_plot:

#         import os
#         os.makedirs(save_dir, exist_ok=True) # make sure the directory exists
        
#         # Create a figure
#         plt.figure()
    
#         # Plot the scaled count values to check gene count distributions across metapixels
#         for _, row in counts_scaled.T[:100].iterrows():
#             sns.kdeplot(row)
    
#         plt.xlabel('Scaled value')
    
#         # Save the plot
#         plt.savefig(f"{save_dir}/{tag}_scaled_plot.png", dpi=300, bbox_inches='tight')
    
#         # Close the figure to prevent it from displaying
#         plt.close()

#         # Create a figure
#         plt.figure()
    
#         # Plot the scaled count values to check gene count distributions across metapixels
#         for _, row in counts_yugene.T.iterrows():
#             sns.kdeplot(row)
    
#         plt.xlabel('Yugene value')
    
#         # Save the plot
#         plt.savefig(f"{save_dir}/{tag}_yugene_plot.png", dpi=300, bbox_inches='tight')
    
#         # Close the figure to prevent it from displaying
#         plt.close()

#     # Subtract control samples to make expressions relative 
#     preprocessed_scaled = final_clock_preparation(counts_scaled, clock_model, diff_suffix = "young")
#     preprocessed_yugene = final_clock_preparation(counts_yugene, clock_model_yugene, diff_suffix = "young")

#     # Apply the clock
#     age_predictions_scaled = predict_age(preprocessed_scaled, clock_model)
#     age_predictions_yugene = predict_age(preprocessed_yugene, clock_model_yugene)

#     # Continue the pipeline 
#     prediction_results_scaled = prepare_prediction_results(preprocessed_scaled, age_predictions_scaled)
#     prediction_results_yugene = prepare_prediction_results(preprocessed_yugene, age_predictions_yugene)

#     # Ensure 'sample' is set as the index for alignment
#     prediction_results_scaled = prediction_results_scaled.set_index("sample")
#     prediction_results_yugene = prediction_results_yugene.set_index("sample")

#     # Assign values ensuring correct alignment with ad_filtered.obs.index
#     ad_filtered.obs["tAge_SM"] = prediction_results_scaled.loc[ad_filtered.obs.index, "Predicted Age"]*48
#     ad_filtered.obs["tAge_YM"] = prediction_results_yugene.loc[ad_filtered.obs.index, "Predicted Age"]*48

#     # Save the predictions as an AnnData object
#     ad_filtered.X = None if not lower_res else ad_filtered.X # Delete the counts matrix to save space if it's propagated (lower_res = False)
#     ad_filtered.obsm = merged_adata.obsm.copy()
#     ad_filtered.obs["centroid_id"] = [f"{x}_{y}" for x, y in ad_filtered.obsm["spatial"]]
#     # print(ad_filtered.obs)

#     if save_result == True:
#         # ad_filtered.obs.to_parquet(f'{save_dir}/{tag}_preds.parquet', index = True)
#         ad_filtered.write_h5ad(f'{save_dir}/{tag}_preds.h5ad')
        
#     # Split ad_filtered into their original dataframes and return a dictionary with the original predictions 
#     # Separate the tissue sections creating a dictionary
#     adatas_dict = {}

#     for file in ad_filtered.obs["File"].unique():
#         adatas_dict[file] = ad_filtered[ad_filtered.obs["File"] == file].copy()
#         # adatas_dict[file] = ad_filtered[ad_filtered.obs["File"] == file].obsm['spatial'].copy()

#     return adatas_dict




# Functions to apply HALVES OF non-overlapping PIPELINE on a dictionary containing a cohort of adata objects that use one (or more) of them as a control (e.g. all adatas for different slices of a tissue that use young ones as control, all slices from the same embryo experiment that use E9.5 slices as control)

def nonoverlap_mp_and_filter(anndata_dict, # takes a dictionary with file names as keys and anndatas as items
                                     res=4,
                                     control_file_pattern = '_Y_',
                                     mp_coverage_threshold = 150_000,
                                     lower_res=False,
                                     save_plot = False,
                                     save_result = True,
                                     save_dir = '',
                                     tag = '',
                                     # cell_type_specific = True
                                    ):
    
    tissue_adatas = []

    for file, tissue_slice in anndata_dict.items():

        print(f'Analyzing sample {file}')

        # Specify age group to tag metapixel names later 
        age_group = "young" if control_file_pattern in file else "old"

        # Step 1: Group pixels into MPs with a certain coverage and distance threshold 
        
        # # OVERLAPPING MP GROUPING
        # grouped_adata = select_neighbors( # this function might need to be modified
        #         adata=tissue_slice, 
        #         coverage_threshold=mp_coverage_threshold, 
        #         age_group=age_group,  
        #         initial_radius=min_dist, 
        #         max_radius=max_dist
        #     )
        
        # NON-OVERLAPPING MP GROUPING
        grouped_adata = non_overlapping_MPs(adata=tissue_slice, 
                                            age_group=age_group, 
                                            lower_res=lower_res, 
                                            n_neighs=20, 
                                            resolution=res)
        
        ## Pass the spatial coordinates into the new grouped_adata
        # Extract barcodes from preds_per_file and filter matching observations in adata
        obs_filter = tissue_slice.obs_names.isin(grouped_adata.obs_names.str.split(".").str[2])
        
        # Assign filtered spatial coordinates
        grouped_adata.obsm["spatial"] = tissue_slice[obs_filter].obsm["spatial"].copy()

        print(f'Before filtering there are {grouped_adata.n_obs} samples left and {grouped_adata.n_vars} genes left.')
        
        ## Filter out MPs with less than 100k reads
        grouped_adata2 = grouped_adata[grouped_adata.obs['cumulative_coverage'] >= mp_coverage_threshold].copy()

        ## Add tag to keep track of the file the samples are from 
        grouped_adata2.obs['File'] = file

        tissue_adatas.append(grouped_adata2)

        # Safely free memory without modifying dict keys
        del tissue_slice
        anndata_dict[file] = None 

    # Concatenate tissue adatas along observations (axis=0)
    merged_adata = sc.concat(tissue_adatas, join="outer", axis=0)

    # Filter out genes with less than 10 counts in at least 20% of samples
    ad_filtered = filter_genes(merged_adata)

    print(f'After filtering there are {ad_filtered.n_obs} samples left and {ad_filtered.n_vars} genes left.')

    return ad_filtered


def preprocess_and_predict(ad_filtered, # takes a dictionary with file names as keys and anndatas as items
                             # anndata_dict,
                             sample_idents, # keys of anndata_dict that identify each control or experimental sample
                             ctrl_ids=None,
                             res=4,
                             control_file_pattern = '_Y_',
                             mp_coverage_threshold = 150_000,
                             save_plot = False,
                             save_result = True,
                             clock_folder = 'tAge_clocks/EN differential models 4.6',
                             save_dir = '',
                             tag = '',
                             # cell_type_specific = True
                            ):

    # Load the clocks
    # clock_model = joblib.load(f'/home/vvicente/spatial_aging/{clock_folder}/EN_Chronoage_Mouse_All_WT_scaleddiff.pkl')
    # clock_model_yugene = joblib.load(f'/home/vvicente/spatial_aging/{clock_folder}/EN_Chronoage_Mouse_All_WT_yugenediff.pkl')
    clock_model = joblib.load(f'{clock_folder}/EN_Chronoage_Mouse_All_WT_scaleddiff.pkl')
    clock_model_yugene = joblib.load(f'{clock_folder}/EN_Chronoage_Mouse_All_WT_yugenediff.pkl')

    real_adatas_dict = {}

    # If ctrl_ids is None, define it using sample_idents
    if ctrl_ids is None:
        ctrl_ids = [x for x in sample_idents if control_file_pattern in x]
    
    # Scale tissues and run clocks by chunks (to not collapse RAM memory)
    for file in sample_idents:
        
        print(f'Scaling sample {file}')
    
        # Select control samples + current file if it is not a control sample
        selected_ids = ctrl_ids + [file] if file not in ctrl_ids else ctrl_ids  
    
        # Filter expression matrix to keep only selected samples
        s_ad_filtered = ad_filtered[ad_filtered.obs["File"].isin(selected_ids)].copy()

        # selected_ids = [x for x in sample_idents if control_file_pattern in x or file in x]
        # s_ad_filtered = ad_filtered[ad_filtered.obs["File"].isin(selected_ids)].copy()
        
        # Create a dataframe out of the ad_filtered
        df = pd.DataFrame(s_ad_filtered.X.T, index=s_ad_filtered.var_names, columns=s_ad_filtered.obs.index)
        df.index.name = 'geneID' # the get_scaled_counts function requires this 

        # # The control IDs will be used for obtaining their median later anyways, so let's just calc it now to scale matrices with less rows
        # # But only if they contain a non-contorl group
        # exp_id = [x for x in selected_ids if control_file_pattern not in x]
        
        # if exp_id != []:
        #     ctrl_ids = selected_ids.drop(exp_id)
        #     control_cols = df.columns.str.isin(ctrl_ids)
        #     exprs_control = df.loc[control_cols].median(axis=1)
        #     df[control_file_pattern]
    
        # Get the scaled counts 
        counts_scaled = get_scaled_counts(df, clock_model, 'symbol')
    
        # Perform YugGene transformation too 
        counts_yugene = get_YuGene_counts(df, clock_model_yugene, 'symbol')
    
        ## Save density plots
        if save_plot:
    
            import os
            os.makedirs(save_dir, exist_ok=True) # make sure the directory exists
            
            # Create a figure
            plt.figure()
        
            # Plot the scaled count values to check gene count distributions across metapixels
            for _, row in counts_scaled.T[:100].iterrows():
                sns.kdeplot(row)
        
            plt.xlabel('Scaled value')
        
            # Save the plot
            plt.savefig(f"{save_dir}/{tag}_scaled_plot.png", dpi=300, bbox_inches='tight')
        
            # Close the figure to prevent it from displaying
            plt.close()
    
            # Create a figure
            plt.figure()
        
            # Plot the scaled count values to check gene count distributions across metapixels
            for _, row in counts_yugene.T.iterrows():
                sns.kdeplot(row)
        
            plt.xlabel('Yugene value')
        
            # Save the plot
            plt.savefig(f"{save_dir}/{tag}_yugene_plot.png", dpi=300, bbox_inches='tight')
        
            # Close the figure to prevent it from displaying
            plt.close()
    
        # Subtract control samples to make expressions relative 
        preprocessed_scaled = final_clock_preparation(counts_scaled, clock_model, diff_suffix = "young")
        preprocessed_yugene = final_clock_preparation(counts_yugene, clock_model_yugene, diff_suffix = "young")
    
        # Apply the clock
        age_predictions_scaled = predict_age(preprocessed_scaled, clock_model)
        age_predictions_yugene = predict_age(preprocessed_yugene, clock_model_yugene)
    
        # Continue the pipeline 
        prediction_results_scaled = prepare_prediction_results(preprocessed_scaled, age_predictions_scaled)
        prediction_results_yugene = prepare_prediction_results(preprocessed_yugene, age_predictions_yugene)
    
        # Ensure 'sample' is set as the index for alignment
        prediction_results_scaled = prediction_results_scaled.set_index("sample")
        prediction_results_yugene = prediction_results_yugene.set_index("sample")
    
        # Assign values ensuring correct alignment with s_ad_filtered.obs.index
        s_ad_filtered.obs["tAge_SM"] = prediction_results_scaled.loc[s_ad_filtered.obs.index, "Predicted Age"]*48
        s_ad_filtered.obs["tAge_YM"] = prediction_results_yugene.loc[s_ad_filtered.obs.index, "Predicted Age"]*48
    
        # Save the predictions as an AnnData object
        s_ad_filtered.X = None # Delete the counts matrix to save space 

        # Split s_ad_filtered into their original dataframes and return a dictionary with the original predictions 
        # Separate the tissue sections creating a dictionary
        real_adatas_dict[file] = s_ad_filtered[s_ad_filtered.obs["File"] == file].copy()

        del s_ad_filtered

        if save_result == True:
            real_adatas_dict[file].obs.to_parquet(f'{save_dir}/{file}_preds.parquet', index = True)
            
    return real_adatas_dict













####

# SAVE THE DICTIONARY in case kernel dies
def dict_to_parquet(data_dict, output_dir):
    import os
    import pandas as pd
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Iterate through your nested dictionary and save each DataFrame
    for tissue, slices_df in data_dict.items():  # dict[tissue name][tissue slice df]
        tissue_dir = os.path.join(output_dir, tissue)
        # os.makedirs(tissue_dir, exist_ok=True)
        
        file_path = os.path.join(output_dir, f"{tissue}_preds.parquet")
        slices_df.to_parquet(file_path, index=False)
    
    print("Data saved in Parquet format successfully!")


def parquet_to_dict(directory, end_pattern=''):

    import os
    import glob 
    
    """
    Loads all Parquet files from a directory into a dictionary.
    
    - Keys: File names without the .parquet extension
    - Values: Corresponding pandas DataFrames

    Parameters:
        directory (str): Path to the directory containing Parquet files.
    
    Returns:
        dict: A dictionary with DataFrames loaded from Parquet files.
    """
    loaded_data = {}

    # Find all Parquet files in the directory
    for file_path in glob.glob(os.path.join(directory, f"*{end_pattern}.parquet")):
        file_name = os.path.basename(file_path).replace("_preds.parquet", "")
        loaded_data[file_name] = pd.read_parquet(file_path)
    
    print(f"Loaded {len(loaded_data)} files into dictionary!")
    return loaded_data



def parquet_to_adata(directory, end_pattern=''):

    import os
    import glob 
    
    """
    Loads all Parquet files from a directory into a dictionary.
    
    - Keys: File names without the .parquet extension
    - Values: Corresponding pandas DataFrames

    Parameters:
        directory (str): Path to the directory containing Parquet files.
    
    Returns:
        dict: A dictionary with DataFrames loaded from Parquet files.
    """
    loaded_data = {}

    # Find all Parquet files in the directory
    for file_path in glob.glob(os.path.join(directory, f"*{end_pattern}.parquet")):
        file_name = os.path.basename(file_path).replace("_preds.parquet", "")
        # loaded_data[file_name] = pd.read_parquet(file_path)
        loaded_data[file_name] = anndata.AnnData(X=None, obs=pd.read_parquet(file_path))
    
    print(f"Loaded {len(loaded_data)} files into dictionary!")
    return loaded_data


# SAVE THE NESTED DICTIONARY in case kernel dies
def nested_dict_to_parquet(data_dict, output_dir):
    import os
    import pandas as pd
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Iterate through your nested dictionary and save each DataFrame
    for tissue, slices in data_dict.items():  # dict[tissue name][tissue slice df]
        tissue_dir = os.path.join(output_dir, tissue)
        os.makedirs(tissue_dir, exist_ok=True)
    
        for slice_name, df in slices.items():
            file_path = os.path.join(tissue_dir, f"{slice_name}_preds.parquet")
            df.obs.to_parquet(file_path, index=True)
    
    print("Data saved in Parquet format successfully!")



# LOAD AND REASSEMBLE THE NESTED DICTIONARY in case kernel died 
def parquet_to_nested_dict(data_dict, output_dir):
    """
    Load and reassemble the nexted dictionary in case kernel died
    """
    
    import glob
    
    # Initialize an empty dictionary
    loaded_data = {}
    
    # Iterate over tissues (subdirectories)
    for tissue in os.listdir(output_dir):
        tissue_path = os.path.join(output_dir, tissue)
        if os.path.isdir(tissue_path):
            loaded_data[tissue] = {}
    
            # Iterate over parquet files in each tissue folder
            for file in glob.glob(os.path.join(tissue_path, "*.parquet")):
                slice_name = os.path.basename(file).replace(".parquet", "")
                loaded_data[tissue][slice_name] = pd.read_parquet(file)
    
    print("Data successfully loaded into a nested dictionary!")
    
    return loaded_data





##### Other downstream analyses 



# Violin or box plots
def plot_clock_distributions(preds_per_file, group_patterns, norm_cols=['tAge_SM', 'tAge_YM'], test='Mann-Whitney'):
    """
    Plot violin plots of predicted transcriptomic ages by group, with statistical comparisons.

    Parameters:
    - preds_per_file: dict of filename → AnnData
    - group_patterns: list of substrings to identify groups from filenames
    - norm_cols: list of obs columns to plot (default: ['tAge_SM', 'tAge_YM'])
    - test: statistical test for annotation (e.g. 'Mann-Whitney', 't-test_ind')
    """

    custom = ["#a1c9f4", #blu
              # "#a1c9f4",
              "lightblue",
              # "lightblue",
              "#ffb482", #orange
              # "#ffb482", 
              "salmon",
              # "salmon"
              ]

    # # For cancer 
    # custom = ["#ffb482", #orange
    #           "#a1c9f4", #blu 
    # ]
    # For WM/GM
    # custom = ["silver",
    #           "lavenderblush",
    #           ]
    
    def assign_group(filename):
        for pattern in group_patterns:
            if pattern in filename:
                return pattern
        return 'Unknown'

    # Build tidy DataFrame
    all_preds = []
    for file, adata in preds_per_file.items():
        group = assign_group(file)
        for norm in norm_cols:
            if norm in adata.obs:
                ages = adata.obs[norm]
                for age in ages:
                    all_preds.append({
                        'file': file,
                        'group': group,
                        'norm': norm,
                        'age': age
                    })

    df_preds = pd.DataFrame(all_preds)

    # Setup group order and comparisons
    group_order = group_patterns
    comparisons = list(zip(group_order[:-1], group_order[1:]))
    # remove these pairs if present
    # to_remove = {('5X_4mo', 'WT_6mo'),
    #              ('5X_6mo', 'WT_8mo'),
    #              ('5X_8mo', 'WT_12mo')}
    # comparisons = [c for c in comparisons if c not in to_remove]
    comparisons = [c for i,c in enumerate(comparisons) if i%2==0] #remove even comparisons (2and3, 4and5, etc.)

    # comparisons.append(('"33 GM", "33 WM"'))
    # comparisons.append(('"41 WM", "41 GM"'))

    # ---------- P L O T ----------------------------------------------------------
    sns.set_theme(context="talk", style="ticks", font_scale=1.15)
    
    fig, axes = plt.subplots(
        1, len(norm_cols),
        # figsize=(4.5 * len(norm_cols), 6), # 2 boxes
        # figsize=(5.5 * len(norm_cols), 6), # 3 boxes
        # figsize=(8 * len(norm_cols), 6), # 8 boxes
        figsize=(12 * len(norm_cols), 7),
        sharey=True,
        constrained_layout=True,
    )
    
    # ensure axes is iterable
    axes = axes if isinstance(axes, (list, np.ndarray)) else [axes]
    
    for ax, norm in zip(axes, norm_cols):
        df_clock = df_preds.query("norm == @norm")
    
        # line widths
        lw_box, lw_med, lw_whisk = 2.2, 2.6, 2

        # df_clock['group'] = [x.replace('.h5ad','') for x in df_clock['group']]
    
        sns.boxplot(
            data=df_clock,
            x="group", y="age",
            order=group_order,
            palette=custom,
            width=0.55,
            showcaps=True,
            showfliers=True,                    # ← show outliers again
            linewidth=lw_box,
            boxprops=dict(linewidth=lw_box),
            whiskerprops=dict(linewidth=lw_whisk),
            capprops=dict(linewidth=lw_whisk),
            medianprops=dict(linewidth=lw_med),
            flierprops=dict(                    # ← style of the outlier dots
                marker="o",
                markersize=6,
                markerfacecolor="#333333",
                markeredgecolor="#333333",
                alpha=0.9,
            ),
            ax=ax,
        )

        # --------------------------------------------------
        # 1. compute the mean for every group in this clock
        # --------------------------------------------------
        group_means = (
            df_clock
            .groupby("group", observed=True)["age"]
            .mean()
            .reindex(group_order)         # keep the visual order
        )
        
        # x-coordinates are 0,1,2,… matching the boxes
        x_pos = np.arange(len(group_order))
        
        # # --------------------------------------------------
        # # 2. overlay the markers
        # # --------------------------------------------------
        # ax.scatter(
        #     x=x_pos,
        #     y=group_means,
        #     marker="D",                   # diamond
        #     s=70,                         # size in points^2
        #     color="black",
        #     zorder=4,                     # plot above everything else
        #     linewidths=0,
        # )
        # -------------------------------------------------- 
    
        # thicker axes/ticks (looks better when scaled down)
        ax.spines["left"].set_linewidth(1.5)
        ax.spines["bottom"].set_linewidth(1.5)
        ax.tick_params(width=1.2)
    
        ax.set(title=norm, ylabel="Predicted relative age", xlabel="", 
               # ylim=(-12,35)
              )
        ax.tick_params(axis="x", rotation=30, size=5)
        ax.grid(False)
        sns.despine(ax=ax)
    
        annot = Annotator(
            ax, pairs=comparisons, data=df_clock,
            x="group", y="age", order=group_order,
        )
        annot.configure(test=test, text_format="star", loc="inside", verbose=0)
        annot.apply_and_annotate()
    
    plt.show()
