"""
End-to-end tAge prediction drivers: overlapping (full_mp_pipeline), non-overlapping
(full_nonoverlap_mp_pipeline), the two-stage RAM-efficient variant (nonoverlap_mp_and_filter +
preprocess_and_predict), and the disk-based variant for very large datasets
(integrated_nonoverlap_incremental_filter + load_filter_write + preprocess_and_predict_from_disk).

Clock/reference-data paths are all explicit parameters here (no hard-coded absolute paths) --
see each function's docstring for what changed relative to the original notebooks/scripts.
"""

import gc
import math
import os
import tempfile

import joblib
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.sparse import issparse, csr_matrix

from .metapixels import select_neighbors, non_overlapping_MPs
from .preprocessing import filter_genes, get_scaled_counts, get_YuGene_counts
from .clock import final_clock_preparation, predict_age, prepare_prediction_results, propagate_into_pixel_level


def _save_density_plots(counts_scaled, counts_yugene, save_dir, tag):
    os.makedirs(save_dir, exist_ok=True)

    plt.figure()
    for _, row in counts_scaled.T[:100].iterrows():
        sns.kdeplot(row)
    plt.xlabel('Scaled value')
    plt.savefig(f"{save_dir}/{tag}_scaled_plot.png", dpi=300, bbox_inches='tight')
    plt.close()

    plt.figure()
    for _, row in counts_yugene.T.iterrows():
        sns.kdeplot(row)
    plt.xlabel('Yugene value')
    plt.savefig(f"{save_dir}/{tag}_yugene_plot.png", dpi=300, bbox_inches='tight')
    plt.close()


def full_mp_pipeline(
    anndata_dict,  # dict of {file name: AnnData}
    ncbi_reference_path,
    clock_dir,  # directory containing EN_Chronoage_Mouse_All_WT_{scaleddiff,yugenediff}.pkl (TMS-trained)
    radius_df=None,
    control_file_pattern='_Y_',
    mp_coverage_threshold=150_000,
    save_plot=False,
    save_result=True,
    save_dir='',
    tag='',
):
    """
    Overlapping SpatialGroup pipeline (KD-tree radius expansion metapixels).

    `ncbi_reference_path` and `clock_dir` are required, explicit parameters -- the original
    hard-coded `ncbi_reference_path` (via preprocess_counts' old default) and the absolute
    clock path `/home/vvicente/spatial_aging/tAge_clocks/tms_clocks/...`. Passing the
    equivalent paths reproduces identical results.
    """
    tissue_adatas = []

    for file, tissue_slice in anndata_dict.items():
        if radius_df is not None and not radius_df.empty:
            max_dist = radius_df[radius_df['Tissue'] == tag]['Radius'].values[0]
        else:
            max_dist = 400.0

        min_dist = 1.0
        print(f'Grouping cells in metapixels with radius r = ({min_dist},{max_dist}) ')

        age_group = "young" if control_file_pattern in file else "old"

        grouped_adata = select_neighbors(
            adata=tissue_slice,
            coverage_threshold=mp_coverage_threshold,
            age_group=age_group,
            initial_radius=min_dist,
            max_radius=max_dist
        )

        obs_filter = tissue_slice.obs_names.isin(grouped_adata.obs_names.str.split(".").str[2])
        grouped_adata.obsm["spatial"] = tissue_slice[obs_filter].obsm["spatial"].copy()

        print(f'Before filtering there are {grouped_adata.n_obs} samples left and {grouped_adata.n_vars} genes left.')

        grouped_adata2 = grouped_adata[grouped_adata.obs['cumulative_coverage'] >= mp_coverage_threshold].copy()

        grouped_adata2.var_names_make_unique()
        grouped_adata2.obs_names_make_unique()

        grouped_adata2.obs['File'] = file

        tissue_adatas.append(grouped_adata2)

    merged_adata = sc.concat(tissue_adatas, join="outer", axis=0)
    ad_filtered = filter_genes(merged_adata)

    print(f'After filtering there are {ad_filtered.n_obs} samples left and {ad_filtered.n_vars} genes left.')

    df = pd.DataFrame(ad_filtered.X.T, index=ad_filtered.var_names, columns=ad_filtered.obs.index)
    df.index.name = 'geneID'

    clock_model = joblib.load(os.path.join(clock_dir, 'EN_Chronoage_Mouse_All_WT_scaleddiff.pkl'))
    clock_model_yugene = joblib.load(os.path.join(clock_dir, 'EN_Chronoage_Mouse_All_WT_yugenediff.pkl'))

    counts_scaled = get_scaled_counts(df, clock_model, ncbi_reference_path, 'symbol')
    counts_yugene = get_YuGene_counts(df, clock_model_yugene, ncbi_reference_path, 'symbol')

    if save_plot:
        _save_density_plots(counts_scaled, counts_yugene, save_dir, tag)

    preprocessed_scaled = final_clock_preparation(counts_scaled, clock_model, diff_suffix="young")
    preprocessed_yugene = final_clock_preparation(counts_yugene, clock_model_yugene, diff_suffix="young")

    age_predictions_scaled = predict_age(preprocessed_scaled, clock_model)
    age_predictions_yugene = predict_age(preprocessed_yugene, clock_model_yugene)

    prediction_results_scaled = prepare_prediction_results(preprocessed_scaled, age_predictions_scaled)
    prediction_results_yugene = prepare_prediction_results(preprocessed_yugene, age_predictions_yugene)

    prediction_results_scaled = prediction_results_scaled.set_index("sample")
    prediction_results_yugene = prediction_results_yugene.set_index("sample")

    ad_filtered.obs["tAge_SM"] = prediction_results_scaled.loc[ad_filtered.obs.index, "Predicted Age"] * 48
    ad_filtered.obs["tAge_YM"] = prediction_results_yugene.loc[ad_filtered.obs.index, "Predicted Age"] * 48

    ad_filtered.X = None  # Delete the counts matrix to save space

    if save_result:
        ad_filtered.obs.to_parquet(f'{save_dir}/{tag}_preds.parquet', index=True)

    adatas_dict = {}
    for file in ad_filtered.obs["File"].unique():
        adatas_dict[file] = ad_filtered[ad_filtered.obs["File"] == file].copy()

    return adatas_dict


def full_nonoverlap_mp_pipeline(
    anndata_dict,
    ncbi_reference_path,
    clock_root,  # directory prefix under which `clock_folder` is resolved
    res=2,
    lower_res=False,
    control_file_pattern='_Y_',
    mp_coverage_threshold=150_000,
    save_plot=False,
    save_result=True,
    clock_folder='tAge_clocks/EN differential models 4.6',
    save_dir='',
    tag='',
):
    """
    Non-overlapping (Leiden-clustered) SpatialGroup pipeline.

    `clock_root` + `clock_folder` replace the original's hard-coded absolute prefix
    (`/home/vvicente/spatial_aging/{clock_folder}/...`); pass the equivalent `clock_root` to
    reproduce identical results. `ncbi_reference_path` is likewise now required -- see
    `stage.preprocessing.preprocess_counts`.
    """
    tissue_adatas = []

    for file, tissue_slice in anndata_dict.items():
        print(f'Analyzing sample {file}')

        age_group = "young" if control_file_pattern in file else "old"

        grouped_adata = non_overlapping_MPs(adata=tissue_slice,
                                             age_group=age_group,
                                             lower_res=lower_res,
                                             n_neighs=20,
                                             resolution=res)

        print(f'Before filtering there are {grouped_adata.n_obs} samples left and {grouped_adata.n_vars} genes left.')

        grouped_adata2 = grouped_adata[grouped_adata.obs['cumulative_coverage'] >= mp_coverage_threshold].copy()

        grouped_adata2.obs['File'] = file

        grouped_adata2.var_names_make_unique()
        grouped_adata2.obs_names_make_unique()

        tissue_adatas.append(grouped_adata2)

    merged_adata = sc.concat(tissue_adatas, join="outer", axis=0)
    ad_filtered = filter_genes(merged_adata)

    print(f'After filtering there are {ad_filtered.n_obs} samples left and {ad_filtered.n_vars} genes left.')

    clock_model = joblib.load(os.path.join(clock_root, clock_folder, 'EN_Chronoage_Mouse_All_WT_scaleddiff.pkl'))
    clock_model_yugene = joblib.load(os.path.join(clock_root, clock_folder, 'EN_Chronoage_Mouse_All_WT_yugenediff.pkl'))

    # ad_filtered is metapixel-level here, i.e. small, so densifying at this point is cheap --
    # unlike the pixel-level matrix upstream in non_overlapping_MPs, which is kept sparse
    # throughout to avoid OOM on large slices.
    X_ad_filtered = ad_filtered.X.toarray() if issparse(ad_filtered.X) else ad_filtered.X
    df = pd.DataFrame(X_ad_filtered.T, index=ad_filtered.var_names, columns=ad_filtered.obs.index)
    df.index.name = 'geneID'

    counts_scaled = get_scaled_counts(df, clock_model, ncbi_reference_path, 'symbol')
    counts_yugene = get_YuGene_counts(df, clock_model_yugene, ncbi_reference_path, 'symbol')

    if save_plot:
        _save_density_plots(counts_scaled, counts_yugene, save_dir, tag)

    preprocessed_scaled = final_clock_preparation(counts_scaled, clock_model, diff_suffix="young")
    preprocessed_yugene = final_clock_preparation(counts_yugene, clock_model_yugene, diff_suffix="young")

    age_predictions_scaled = predict_age(preprocessed_scaled, clock_model)
    age_predictions_yugene = predict_age(preprocessed_yugene, clock_model_yugene)

    prediction_results_scaled = prepare_prediction_results(preprocessed_scaled, age_predictions_scaled)
    prediction_results_yugene = prepare_prediction_results(preprocessed_yugene, age_predictions_yugene)

    prediction_results_scaled = prediction_results_scaled.set_index("sample")
    prediction_results_yugene = prediction_results_yugene.set_index("sample")

    ad_filtered.obs["tAge_SM"] = prediction_results_scaled.loc[ad_filtered.obs.index, "Predicted Age"] * 48
    ad_filtered.obs["tAge_YM"] = prediction_results_yugene.loc[ad_filtered.obs.index, "Predicted Age"] * 48

    ad_filtered.obsm = merged_adata.obsm.copy()
    ad_filtered.obs["centroid_id"] = [f"{x}_{y}" for x, y in ad_filtered.obsm["spatial"]]

    if save_result:
        ad_filtered.write_h5ad(f'{save_dir}/{tag}_preds.h5ad')

    adatas_dict = {}

    if lower_res:
        for file in ad_filtered.obs["File"].unique():
            adatas_dict[file] = ad_filtered[ad_filtered.obs["File"] == file].copy()
    else:
        for file in ad_filtered.obs["File"].unique():
            adata = anndata_dict[file]
            lowres_adata_mp = ad_filtered[ad_filtered.obs["File"] == file].copy()
            age_group = "young" if control_file_pattern in file else "old"
            # propagate_expression=False: see stage.clock.propagate_into_pixel_level docstring
            # -- the metapixel-broadcast expression layer this would otherwise compute is
            # immediately overwritten by adata.X on the next line and never read elsewhere.
            adata_mp = propagate_into_pixel_level(lowres_adata_mp, adata,
                                                   age_group,
                                                   obs_to_propagate=['tAge_YM', 'tAge_SM'],
                                                   propagate_expression=False)
            adata_mp.X = adata.X
            adatas_dict[file] = adata_mp

    return adatas_dict


def nonoverlap_mp_and_filter(
    anndata_dict,
    res=4,
    control_file_pattern='_Y_',
    mp_coverage_threshold=150_000,
    lower_res=False,
    save_plot=False,
    save_result=True,
    save_dir='',
    tag='',
):
    """Stage 1 of the two-stage RAM-efficient pipeline: group into metapixels + filter genes, no clock yet."""
    tissue_adatas = []
    metapixel_maps = {}

    for file, tissue_slice in anndata_dict.items():
        print(f'Analyzing sample {file}')

        age_group = "young" if control_file_pattern in file else "old"

        grouped_adata = non_overlapping_MPs(adata=tissue_slice,
                                             age_group=age_group,
                                             lower_res=lower_res,
                                             n_neighs=20,
                                             resolution=res)

        # Preserve metapixel assignments for later propagation to pixel level
        metapixel_maps[file] = tissue_slice.obs['metapixel'].copy()

        print(f'Before filtering there are {grouped_adata.n_obs} samples left and {grouped_adata.n_vars} genes left.')

        grouped_adata2 = grouped_adata[grouped_adata.obs['cumulative_coverage'] >= mp_coverage_threshold].copy()
        grouped_adata2.obs['File'] = file

        tissue_adatas.append(grouped_adata2)

        # Safely free memory without modifying dict keys
        del tissue_slice
        anndata_dict[file] = None

    merged_adata = sc.concat(tissue_adatas, join="outer", axis=0)
    ad_filtered = filter_genes(merged_adata)

    # Store metapixel mappings so callers can propagate predictions to pixel level
    ad_filtered.uns['metapixel_maps'] = {k: v.to_numpy().astype(str) for k, v in metapixel_maps.items()}

    print(f'After filtering there are {ad_filtered.n_obs} samples left and {ad_filtered.n_vars} genes left.')

    return ad_filtered


def preprocess_and_predict(
    ad_filtered,
    sample_idents,  # identifiers of each control/experimental sample present in ad_filtered.obs['File']
    ncbi_reference_path,
    clock_root,
    ctrl_ids=None,
    control_file_pattern='_Y_',
    save_plot=False,
    save_result=True,
    clock_folder='tAge_clocks/EN differential models 4.6',
    save_dir='',
    tag='',
):
    """Stage 2 of the two-stage RAM-efficient pipeline: per-sample-chunk scale + predict, from `nonoverlap_mp_and_filter`'s output."""
    clock_model = joblib.load(os.path.join(clock_root, clock_folder, 'EN_Chronoage_Mouse_All_WT_scaleddiff.pkl'))
    clock_model_yugene = joblib.load(os.path.join(clock_root, clock_folder, 'EN_Chronoage_Mouse_All_WT_yugenediff.pkl'))

    real_adatas_dict = {}

    if ctrl_ids is None:
        ctrl_ids = [x for x in sample_idents if control_file_pattern in x]

    for file in sample_idents:
        print(f'Scaling sample {file}')

        selected_ids = ctrl_ids + [file] if file not in ctrl_ids else ctrl_ids

        s_ad_filtered = ad_filtered[ad_filtered.obs["File"].isin(selected_ids)].copy()

        df = pd.DataFrame(s_ad_filtered.X.T, index=s_ad_filtered.var_names, columns=s_ad_filtered.obs.index)
        df.index.name = 'geneID'

        counts_scaled = get_scaled_counts(df, clock_model, ncbi_reference_path, 'symbol')
        counts_yugene = get_YuGene_counts(df, clock_model_yugene, ncbi_reference_path, 'symbol')

        if save_plot:
            _save_density_plots(counts_scaled, counts_yugene, save_dir, tag)

        preprocessed_scaled = final_clock_preparation(counts_scaled, clock_model, diff_suffix="young")
        preprocessed_yugene = final_clock_preparation(counts_yugene, clock_model_yugene, diff_suffix="young")

        age_predictions_scaled = predict_age(preprocessed_scaled, clock_model)
        age_predictions_yugene = predict_age(preprocessed_yugene, clock_model_yugene)

        prediction_results_scaled = prepare_prediction_results(preprocessed_scaled, age_predictions_scaled)
        prediction_results_yugene = prepare_prediction_results(preprocessed_yugene, age_predictions_yugene)

        prediction_results_scaled = prediction_results_scaled.set_index("sample")
        prediction_results_yugene = prediction_results_yugene.set_index("sample")

        s_ad_filtered.obs["tAge_SM"] = prediction_results_scaled.loc[s_ad_filtered.obs.index, "Predicted Age"] * 48
        s_ad_filtered.obs["tAge_YM"] = prediction_results_yugene.loc[s_ad_filtered.obs.index, "Predicted Age"] * 48

        s_ad_filtered.X = None

        real_adatas_dict[file] = s_ad_filtered[s_ad_filtered.obs["File"] == file].copy()

        del s_ad_filtered

        if save_result:
            real_adatas_dict[file].obs.to_parquet(f'{save_dir}/{file}_preds.parquet', index=True)

    return real_adatas_dict


# ---------------------------------------------------------------------------
# Disk-based variant (for very large datasets), from stage_big_pred.py
# ---------------------------------------------------------------------------

def integrated_nonoverlap_incremental_filter(
    anndata_dict,
    save_dir='.',
    res=4,
    control_file_pattern='_Y_',
    mp_coverage_threshold=150_000,
    threshold_percent=0.2,
    min_counts=10,
    filtered_adatas_dir=None,
):
    """
    Disk-buffered stage 1: group into metapixels per sample, write each to a temp .h5ad
    under `save_dir`, and compute the global gene-presence filter across all temp files
    without holding them all in RAM at once.

    BUG FIX (mechanical, not a behavior change): the original referenced an undefined free
    variable `ipynb_dir` when writing `genes_to_keep_names.csv` -- it would raise NameError
    if this function were ever called standalone (outside the exact notebook session that
    happened to have `ipynb_dir` set as a global). `filtered_adatas_dir` is now an explicit
    parameter (defaulting to `{save_dir}/filtered_adatas`, mirroring the apparent original
    intent of `ipynb_dir` being the notebook's own directory with a `filtered_adatas`
    subfolder), and the directory is created if missing -- the original had no `os.makedirs`
    call for this path either, so it would have raised FileNotFoundError even with
    `ipynb_dir` defined.
    """
    if filtered_adatas_dir is None:
        filtered_adatas_dir = os.path.join(save_dir, 'filtered_adatas')

    temp_files = []
    total_cells = 0
    gene_presence_counts = {}

    for file, adata in anndata_dict.items():
        print(f'Analyzing sample {file}')

        age_group = "young" if control_file_pattern in file else "old"

        grouped_adata = non_overlapping_MPs(adata=adata,
                                             age_group=age_group,
                                             lower_res=False,
                                             n_neighs=20,
                                             resolution=res)

        grouped_adata = grouped_adata[grouped_adata.obs['cumulative_coverage'] >= mp_coverage_threshold].copy()
        grouped_adata.obs['File'] = file

        if not issparse(grouped_adata.X):
            grouped_adata.X = csr_matrix(grouped_adata.X)

        gene_counts = np.array((grouped_adata.X >= min_counts).sum(axis=0)).flatten()
        gene_names = grouped_adata.var_names.tolist()

        for gene_name, count in zip(gene_names, gene_counts):
            gene_presence_counts[gene_name] = gene_presence_counts.get(gene_name, 0) + count

        total_cells += grouped_adata.n_obs

        temp_filename = tempfile.mktemp(suffix='.h5ad', dir=save_dir)
        grouped_adata.write_h5ad(temp_filename)
        temp_files.append(temp_filename)

        anndata_dict[file] = None
        del adata, grouped_adata
        gc.collect()

    global_threshold = math.ceil(total_cells * threshold_percent)
    genes_to_keep_names = [gene_name for gene_name, count in gene_presence_counts.items() if count >= global_threshold]

    os.makedirs(filtered_adatas_dir, exist_ok=True)
    pd.Series(genes_to_keep_names).to_csv(
        os.path.join(filtered_adatas_dir, 'genes_to_keep_names.csv'), index=False, header=False
    )

    print(f'Genes retained after global filtering: {len(genes_to_keep_names)}')


def load_filter_write(genes_to_keep_names, temp_dir='.'):
    """Re-read each per-sample temp .h5ad, restrict to `genes_to_keep_names`, write as `f_{file}.h5ad`."""
    temp_files = [f'{temp_dir}/{f}' for f in os.listdir(temp_dir) if 'tmp' in f]

    for f in temp_files:
        print(f'Merging file: {f}')

        temp_adata = sc.read_h5ad(f)
        temp_adata = temp_adata[:, temp_adata.var_names.isin(genes_to_keep_names)].copy()

        filename = temp_adata.obs['File'].unique()[0]
        temp_adata.write_h5ad(f'{temp_dir}/f_{filename}.h5ad')

        del temp_adata
        gc.collect()


def clear_temp_files(temp_dir='.'):
    temp_files = [f'{temp_dir}/{f}' for f in os.listdir(f'{temp_dir}') if 'tmp' in f]
    for f in temp_files:
        if os.path.exists(f):
            os.remove(f)


def preprocess_and_predict_from_disk(
    temp_dir,
    sample_idents,
    ncbi_reference_path,
    clock_dir,  # directory containing EN_Chronoage_Mouse_All_WT_{scaleddiff,yugenediff}.pkl
    ctrl_ids=None,
    control_file_pattern='_Y_',
    save_plot=False,
    save_result=True,
    save_dir='',
    tag='',
):
    """
    Predict from the `f_{file}.h5ad` files written by `load_filter_write`, per-sample-chunk, without
    ever assembling one big in-RAM AnnData.

    `ncbi_reference_path` and `clock_dir` are required, explicit parameters -- see
    `full_mp_pipeline`'s docstring for the equivalent change.

    NOTE (latent bug in the original, fixed by module-level imports here): the original
    `save_plot` branch called `plt.figure()`/`sns.kdeplot()` without importing `matplotlib.pyplot`
    or `seaborn` anywhere in the source file -- it would have raised NameError if `save_plot=True`
    was ever actually passed. Both are now imported at module level (this file already needs them
    for the other pipeline functions' density plots), which incidentally fixes this.
    """
    clock_model = joblib.load(os.path.join(clock_dir, 'EN_Chronoage_Mouse_All_WT_scaleddiff.pkl'))
    clock_model_yugene = joblib.load(os.path.join(clock_dir, 'EN_Chronoage_Mouse_All_WT_yugenediff.pkl'))

    real_adatas_dict = {}

    if ctrl_ids is None:
        ctrl_ids = [x for x in sample_idents if control_file_pattern in x]

    for file in sample_idents:
        print(f'Scaling sample {file}')

        selected_ids = ctrl_ids + [file] if file not in ctrl_ids else ctrl_ids

        filt_list = [
            sc.read_h5ad(os.path.join(temp_dir, fname))
            for fname in os.listdir(temp_dir)
            if fname.endswith(".h5ad") and any(sid in fname for sid in selected_ids)
        ]

        s_ad_filtered = sc.concat(filt_list, join='outer', axis=0)

        X = s_ad_filtered.X.toarray() if issparse(s_ad_filtered.X) else s_ad_filtered.X

        if X.ndim == 1:
            X = X.reshape(1, -1)
        elif X.shape[1] == 1:
            X = X.reshape(-1, 1)

        df = pd.DataFrame(X.T, index=s_ad_filtered.var_names, columns=s_ad_filtered.obs_names)
        df.index.name = 'geneID'

        counts_scaled = get_scaled_counts(df, clock_model, ncbi_reference_path, 'symbol')
        counts_yugene = get_YuGene_counts(df, clock_model_yugene, ncbi_reference_path, 'symbol')

        if save_plot:
            _save_density_plots(counts_scaled, counts_yugene, save_dir, tag)

        preprocessed_scaled = final_clock_preparation(counts_scaled, clock_model, diff_suffix="young")
        preprocessed_yugene = final_clock_preparation(counts_yugene, clock_model_yugene, diff_suffix="young")

        age_predictions_scaled = predict_age(preprocessed_scaled, clock_model)
        age_predictions_yugene = predict_age(preprocessed_yugene, clock_model_yugene)

        prediction_results_scaled = prepare_prediction_results(preprocessed_scaled, age_predictions_scaled)
        prediction_results_yugene = prepare_prediction_results(preprocessed_yugene, age_predictions_yugene)

        prediction_results_scaled = prediction_results_scaled.set_index("sample")
        prediction_results_yugene = prediction_results_yugene.set_index("sample")

        s_ad_filtered.obs["tAge_SM"] = prediction_results_scaled.loc[s_ad_filtered.obs.index, "Predicted Age"] * 48
        s_ad_filtered.obs["tAge_YM"] = prediction_results_yugene.loc[s_ad_filtered.obs.index, "Predicted Age"] * 48

        s_ad_filtered.X = None

        real_adatas_dict[file] = s_ad_filtered[s_ad_filtered.obs["File"] == file].copy()

        del s_ad_filtered

        if save_result:
            real_adatas_dict[file].write_h5ad(f'{save_dir}/pred_{file}.h5ad')

    return real_adatas_dict
