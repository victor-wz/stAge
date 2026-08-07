"""Module-specific clock application: apply each of a directory of module (compartment/
pathway) elastic-net clocks to a cohort of tissue AnnDatas, one clock at a time.

Maps to Figs 6e, 7d, 7h, S9, S10 (Module-specific clocks).

Source: v_pipeline/stage_modules_pred.ipynb (`full_nonoverlap_mp_pipeline_modules`,
the per-tissue x per-clock `res_dict` loop). This notebook is used in at least two
different configurations found in its own (uncommented) cell history: a mouse-tissue
config (`rawdata_dir='data/immunoglobulin'`, commented-out mouse module-clock dir
`tAge_clocks/Module clocks 4.6 bwnet3 filtered5`) and, in its last-saved execution
state, a HUMAN config (`rawdata_dir='.../hbreastcancer'`, active
`mod_dir='tAge_clocks/Module clocks 5.4 multispecies'`). This supports (but does not
confirm -- see analyses/*_module_clocks.py docstrings) treating 6e/S9/S10 as mouse-
tissue panels and 7d/7h as human-cohort panels of this same underlying analysis.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Iterable, Optional

import joblib
import pandas as pd
import scanpy as sc

from .metapixels import non_overlapping_MPs
from .preprocessing import filter_genes, get_scaled_counts
from .clock import final_clock_preparation, predict_age, prepare_prediction_results, propagate_into_pixel_level


def full_nonoverlap_mp_pipeline_modules(
    anndata_dict: Dict[str, "sc.AnnData"],
    clock_path: str,
    ncbi_reference_path: str,
    res: float = 2,
    lower_res: bool = False,
    control_file_pattern: str = '_Y_',
    mp_coverage_threshold: int = 150_000,
    save_result: bool = False,
    save_dir: str = '',
    tag: str = '',
):
    """Apply ONE module clock (`clock_path`, a path to a single `*_scaleddiff.pkl`
    file -- not a directory) to a cohort of tissue AnnDatas via the standard
    non-overlapping metapixel pipeline. Unlike `stage.pipeline.full_nonoverlap_mp_
    pipeline`, this only computes the scaled-diff prediction (no YuGene variant) --
    matches the source notebook's module-clock loop, which is run once per (tissue,
    clock) pair rather than once per (tissue, normalization).

    `ncbi_reference_path` and `clock_path` are required, explicit parameters -- the
    original hard-coded `f'/home/vvicente/spatial_aging/{clock_folder}'`. Passing the
    equivalent path reproduces identical results.
    """
    tissue_adatas = []

    for file, tissue_slice in anndata_dict.items():
        age_group = "young" if control_file_pattern in file else "old"

        grouped_adata = non_overlapping_MPs(adata=tissue_slice, age_group=age_group,
                                             lower_res=lower_res, n_neighs=8, resolution=res)

        grouped_adata2 = grouped_adata[grouped_adata.obs['cumulative_coverage'] >= mp_coverage_threshold].copy()
        grouped_adata2.obs['File'] = file
        grouped_adata2.var_names_make_unique()
        grouped_adata2.obs_names_make_unique()
        tissue_adatas.append(grouped_adata2)

    merged_adata = sc.concat(tissue_adatas, join="outer", axis=0)
    ad_filtered = filter_genes(merged_adata)

    clock_model = joblib.load(clock_path)
    df = pd.DataFrame(ad_filtered.X.T, index=ad_filtered.var_names, columns=ad_filtered.obs.index)
    df.index.name = 'geneID'

    counts_scaled = get_scaled_counts(df, clock_model, ncbi_reference_path, 'symbol')
    preprocessed_scaled = final_clock_preparation(counts_scaled, clock_model, diff_suffix="young")
    age_predictions_scaled = predict_age(preprocessed_scaled, clock_model)

    prediction_results_scaled = prepare_prediction_results(preprocessed_scaled, age_predictions_scaled).set_index("sample")
    ad_filtered.obs["tAge_SM"] = prediction_results_scaled.loc[ad_filtered.obs.index, "Predicted Age"] * 48

    ad_filtered.X = None if not lower_res else ad_filtered.X
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
            adata_mp = propagate_into_pixel_level(lowres_adata_mp, adata, age_group,
                                                   obs_to_propagate=['tAge_SM'])
            adata_mp.layers['mp_counts'] = adata_mp.X
            adata_mp.X = adata.X
            adatas_dict[file] = adata_mp

    return adatas_dict


def run_module_clock_grid(
    assembled_adatas: Dict[str, "sc.AnnData"],
    module_clock_dir: str,
    ncbi_reference_path: str,
    unique_tissues: Iterable[str],
    age_group_from_filename,
    control_file_pattern: str = '_Y_',
    resolution: float = 0.5,
    mp_coverage_threshold: int = 1_000,
    min_obs_per_slice: int = 20,
) -> Dict[str, Dict[str, "sc.AnnData"]]:
    """Run `full_nonoverlap_mp_pipeline_modules` once per (tissue, module-clock-file)
    pair found in `module_clock_dir` (every `*_Chronoage*scaleddiff*.pkl` file),
    restricted to `assembled_adatas` entries containing each tissue tag. Returns
    `res_dict`, keyed `f"{tissue}__{clock_tag}"` -> {filename: AnnData with tAge_SM}.

    Ported from v_pipeline/stage_modules_pred.ipynb's per-tissue/per-clock loop (see
    module docstring). `age_group_from_filename` is a caller-supplied callable
    (filename -> 'Young'/'Old') since the source notebook's own version of this
    helper is commented out / dataset-specific.
    """
    module_clocks = [
        f for f in os.listdir(module_clock_dir)
        if 'scaleddiff' in f and 'Chronoage' in f and f.endswith('.pkl')
    ]

    res_dict: Dict[str, Dict[str, "sc.AnnData"]] = {}
    for tissue in unique_tissues:
        subset = {k: ad for k, ad in assembled_adatas.items() if tissue in k}
        cleaned = {k: ad for k, ad in subset.items() if ad.n_obs >= min_obs_per_slice}
        if not cleaned:
            continue

        for clock_file in module_clocks:
            clock_tag = Path(clock_file).stem
            key = f"{tissue}__{clock_tag}"
            if key not in res_dict:
                res_dict[key] = full_nonoverlap_mp_pipeline_modules(
                    cleaned,
                    clock_path=os.path.join(module_clock_dir, clock_file),
                    ncbi_reference_path=ncbi_reference_path,
                    res=resolution,
                    lower_res=True,
                    control_file_pattern=control_file_pattern,
                    mp_coverage_threshold=mp_coverage_threshold,
                    save_result=False,
                )
            for fn, ad in res_dict[key].items():
                ad.obs["AgeGroup"] = age_group_from_filename(fn)

    return res_dict
