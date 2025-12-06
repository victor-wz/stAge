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


# Functions for resolution search

# --- Min-max normalize a pandas Series ---
def normalize_minmax(series):
    return (series - series.min()) / (series.max() - series.min())

# --- Select best resolution based on composite score ---
def select_best_resolution(df, score_col='score', resolution_col='Resolution', tolerance=0.01):
    """
    Selects the best resolution by:
    - Taking the highest score
    - If multiple scores are within `tolerance` of the max, picks the one with the highest resolution
    """
    max_score = df[score_col].max()
    close_scores = df[df[score_col] >= max_score - tolerance]
    best_row = close_scores.loc[close_scores[resolution_col].idxmax()]
    return best_row

# --- Main function: runs the pipeline across resolutions and selects the best one ---
def optimal_resolution_search(
    assembled_adatas,              # dict of sample AnnData objects
    ipynb_dir,                     # output directory for predictions (used in pipeline)
    pred_pipeline,  # your custom function that runs the prediction pipeline
    control_file_pattern = '_Y_', 
    res_range=[0.25, 0.5, 1, 1.5, 2, 4, 8, 16],  # resolution values to test
    coverage_thresh=10_000,       # MP coverage threshold
    cohen_weight=0.6,             # weight for Cohen's d in composite score
    tstat_weight=0.4,             # weight for t-statistic
    tolerance=0.05                # tie-breaking tolerance for score
):
    all_best_resolutions = []

    # Define the different clocks you want to test
    clock_dirs = {
        f'orig': 'tAge_clocks/EN differential models 4.6',
        f'tms': 'tAge_clocks/tms_clocks',
        f'tmsh': 'tAge_clocks/tmsh_clocks'
    }

    # Iterate over each clock variant
    for tag, clock_folder in clock_dirs.items():
        results = []

        # Try each resolution value
        for resolution in res_range:
            print(f"\n Running {tag} with resolution = {resolution}")

            # Run prediction pipeline with this resolution
            preds_per_file = pred_pipeline(
                assembled_adatas,
                res=resolution,
                control_file_pattern=control_file_pattern,  # distinguishes young vs old
                mp_coverage_threshold=coverage_thresh,
                lower_res=True,
                save_plot=False,
                save_result=False,
                clock_folder=clock_folder,
                save_dir=f'{ipynb_dir}/parquet_embryo_age_preds',
                tag=''
            )

            # Separate young vs. old predictions
            young_preds, old_preds = [], []
            for k, adata in preds_per_file.items():
                if control_file_pattern in k:
                    young_preds.extend(adata.obs['tAge_SM'].values)
                else:
                    old_preds.extend(adata.obs['tAge_SM'].values)

            young_preds = np.array(young_preds)
            old_preds = np.array(old_preds)

            # T-test: how statistically different are the two groups?
            t_stat, p_value = stats.ttest_ind(young_preds, old_preds, equal_var=False)
            mean_diff = old_preds.mean() - young_preds.mean()

            # Pooled standard deviation for Cohen's d
            def pooled_std(s1, s2, n1, n2):
                return np.sqrt(((n1 - 1)*s1**2 + (n2 - 1)*s2**2) / (n1 + n2 - 2))

            s1, s2 = young_preds.std(ddof=1), old_preds.std(ddof=1)
            n1, n2 = len(young_preds), len(old_preds)
            sp = pooled_std(s1, s2, n1, n2)
            cohen_d = mean_diff / sp  # Effect size: how big is the difference?

            # Compute mean coverage (just as an extra QC/metadata column)
            all_coverages = np.concatenate([
                adata.obs['cumulative_coverage'].values for adata in preds_per_file.values()
            ])
            mean_mp_coverage = int(np.round(np.mean(all_coverages)))

            # Save result for this resolution
            results.append({
                "Resolution": resolution,
                "Coverage": mean_mp_coverage,
                "T_stat": abs(t_stat),  # magnitude only
                "log_P_value": -np.log10(p_value),
                "Age_diff": abs(mean_diff),
                "Cohen_d": abs(cohen_d)
            })

        # --- After all resolutions have been evaluated for one clock ---

        # Convert results to DataFrame
        df = pd.DataFrame(results)

        # Normalize both metrics so we can combine them fairly
        df['norm_t'] = normalize_minmax(df['T_stat'])
        df['norm_p'] = normalize_minmax(df['log_P_value'])
        df['norm_d'] = normalize_minmax(df['Cohen_d'])

        # Composite score: weighted average of normalized stats
        df['score'] = cohen_weight * df['norm_d'] + tstat_weight * df['norm_t']
        # df['score'] = cohen_weight * df['norm_d'] + tstat_weight * df['norm_p']

        # Select best resolution based on highest score, favoring higher resolution in case of tie
        best_row = select_best_resolution(df, score_col='score', resolution_col='Resolution', tolerance=tolerance)
        print(f"\n Best resolution for {tag}: {best_row['Resolution']} (score = {best_row['score']:.3f})")

        # Add clock tag to result and collect it
        best_row['Clock'] = tag
        all_best_resolutions.append(best_row)

    # Return summary DataFrame of best resolutions per clock
    return pd.DataFrame(all_best_resolutions)
