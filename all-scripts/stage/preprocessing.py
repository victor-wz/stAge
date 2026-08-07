"""Gene filtering and normalization: StandardScaler-log1p and YuGene paths, gene-symbol/Ensembl to NCBI Entrez ID mapping."""

import math

import numpy as np
import pandas as pd
import sklearn
from sklearn.preprocessing import StandardScaler
from scipy.sparse import issparse, csr_matrix
from tqdm import tqdm


def filter_genes(adata, layer_name=None):
    """Keep genes with >=10 counts in >=20% of the cells/metapixels."""
    matrix = adata.X if layer_name is None else adata.layers[layer_name]

    if not issparse(matrix):
        matrix = csr_matrix(matrix)

    threshold = math.ceil(0.2 * adata.n_obs)

    gene_counts = np.array((matrix >= 10).sum(axis=0)).flatten()
    gene_filter = gene_counts >= threshold

    adata_filtered = adata[:, gene_filter].copy()

    if "spatial" in adata.obsm:
        adata_filtered.obsm["spatial"] = adata.obsm["spatial"].copy()

    return adata_filtered


def preprocess_counts(
    df_counts,
    gene_name_column,
    clock_model,
    original_ids,
    ncbi_reference_path,
):
    """
    Preprocess gene count data for biological clock prediction.

    Steps: (1) load NCBI gene reference data, (2) map gene names to standardized Entrez
    GeneIDs, (3) log1p + per-sample StandardScaler, (4) handle missing genes/duplicates,
    (5) align to the clock model's expected feature order.

    `ncbi_reference_path` (path to a `*.gene_info` file, e.g. `Mus_musculus.gene_info`) is a
    required parameter -- the original hard-coded a project-relative default
    ("vvicente/scripts/v_pipeline/Mus_musculus.gene_info") that only resolved correctly when
    the caller's cwd was the original project root. Callers must now pass the path explicitly;
    passing the same file reproduces identical results.
    """
    if isinstance(clock_model, sklearn.pipeline.Pipeline):
        feature_names = [int(x) for x in clock_model.feature_names_in_]
    else:
        feature_names = [int(x) for x in clock_model.feature_names]

    # 1. Load NCBI gene reference data and prepare synonym mappings
    ncbi_genes = pd.read_table(ncbi_reference_path)
    ncbi_genes["Synonyms_Combined"] = ncbi_genes["Synonyms"] + "|" + ncbi_genes["Symbol"]

    # 2. Map gene names to standardized Gene IDs
    if original_ids == 'symbol':
        gene_name_to_gene_id = {}
        for _, row in ncbi_genes.iterrows():
            gene_id = row["GeneID"]
            synonyms = [row["Symbol"]]
            for synonym in synonyms:
                gene_name_to_gene_id[synonym] = gene_id

    elif original_ids == 'ensembl':
        gene_name_to_gene_id = {}
        for _, row in ncbi_genes.iterrows():
            gene_id = row["GeneID"]
            synonyms = [row["Symbol"]]

            if isinstance(row["dbXrefs"], str) and "Ensembl:" in row["dbXrefs"]:
                ensembl_id = [x.split(":")[-1] for x in row["dbXrefs"].split("|") if x.startswith("Ensembl")][0]
                synonyms.append(ensembl_id)

            for synonym in synonyms:
                gene_name_to_gene_id[synonym] = gene_id

    df_counts["mapped_geneID"] = df_counts[gene_name_column].map(gene_name_to_gene_id)

    df_filtered = df_counts[df_counts["mapped_geneID"].notna()].drop(columns=[gene_name_column])
    df_filtered["mapped_geneID"] = df_filtered["mapped_geneID"].astype(int)

    # 3. Handle missing genes and duplicates: keep the row with the highest count sum per GeneID
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

    df_filtered = df_filtered.set_index("mapped_geneID")
    df_aligned = df_filtered.reindex(feature_names)

    # 4. Log1p + per-sample StandardScaler
    df_log = np.log1p(df_aligned)
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df_log.values)
    df_scaled = pd.DataFrame(scaled_data, index=df_log.index, columns=df_log.columns)

    # 5. Align to model's expected features
    df_scaled = df_scaled.loc[df_scaled.index.isin(feature_names)]

    return df_scaled


def YuGene(data_prop, progress_bar=True):
    """
    YuGene transformation: rank-based cumulative proportion transform.

    Parameters
    ----------
    data_prop : pd.DataFrame or np.ndarray
        DataFrame (or array) with genes as rows and samples as columns.
    progress_bar : bool
        Whether to show a progress bar over samples.
    """
    data_prop = data_prop - data_prop.min(axis=0)
    data_prop = data_prop.fillna(0)

    if isinstance(data_prop, pd.DataFrame):
        row_index = data_prop.index
        col_index = data_prop.columns
        data = data_prop.values
    else:
        data = np.asarray(data_prop, dtype=np.float64)
        row_index = range(data.shape[0])
        col_index = range(data.shape[1])

    if (data < 0).any():
        print("Warning: some negative values were set to 0")
        np.clip(data, 0, None, out=data)

    result = np.empty_like(data)

    n_cols = data.shape[1]
    for j in tqdm(range(n_cols), disable=not progress_bar, desc="Processing samples"):
        col_data = data[:, j]

        sort_idx = np.argsort(col_data)[::-1]
        sorted_vals = col_data[sort_idx]

        cumsum_vals = np.cumsum(sorted_vals)
        total = cumsum_vals[-1]

        if total == 0:
            result[:, j] = 1.0
            continue

        cumprop = cumsum_vals / total

        # Duplicates: identical expression values must map to the same cumulative proportion.
        for i in range(1, len(cumprop)):
            if sorted_vals[i] == sorted_vals[i - 1]:
                cumprop[i] = cumprop[i - 1]

        final_col = 1.0 - cumprop
        result[sort_idx, j] = final_col

    result_df = pd.DataFrame(result, index=row_index, columns=col_index)
    return result_df


def get_YuGene_counts(df, clock_model, ncbi_reference_path, original_ids='symbol'):
    """`ncbi_reference_path` is required -- see `preprocess_counts` docstring."""
    df_preprocessed = preprocess_counts(
        df.reset_index(), "geneID", clock_model=clock_model,
        original_ids=original_ids, ncbi_reference_path=ncbi_reference_path,
    )
    df_YuGene = YuGene(df_preprocessed)
    return df_YuGene


def get_scaled_counts(df, clock_model, ncbi_reference_path, original_ids='symbol'):
    """`ncbi_reference_path` is required -- see `preprocess_counts` docstring."""
    df_preprocessed = preprocess_counts(
        df.reset_index(), "geneID", clock_model=clock_model,
        original_ids=original_ids, ncbi_reference_path=ncbi_reference_path,
    )
    return df_preprocessed
