"""Optimal resolution search (ORS): sweep Leiden resolutions, score by composite Cohen's d + t-statistic."""

import numpy as np
import pandas as pd
from scipy import stats


def normalize_minmax(series):
    return (series - series.min()) / (series.max() - series.min())


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


def optimal_resolution_search(
    assembled_adatas,              # dict of sample AnnData objects
    ipynb_dir,                     # output directory for predictions (used in pipeline)
    pred_pipeline,                 # callable that runs the prediction pipeline, e.g. stage.pipeline.full_nonoverlap_mp_pipeline
    control_file_pattern='_Y_',
    res_range=[0.25, 0.5, 1, 1.5, 2, 4, 8, 16],  # resolution values to test
    coverage_thresh=10_000,        # MP coverage threshold
    cohen_weight=0.6,              # weight for Cohen's d in composite score
    tstat_weight=0.4,              # weight for t-statistic
    tolerance=0.05,                # tie-breaking tolerance for score
    clock_dirs=None,               # dict of {clock_tag: relative clock folder}; see default below
):
    """
    Composite score S = tstat_weight * norm(|t|) + cohen_weight * norm(|Cohen's d|).

    NOTE: this function's own `tolerance` default is 0.05 (5%), but every real call site in
    the original codebase passes tolerance=0.1 (10%) explicitly -- the paper's reported ORS
    methodology uses 10%. The default is left as-is here (not silently changed to 0.1) since
    changing an existing public default would itself be an undocumented behavior change;
    callers reproducing the paper's analyses must pass tolerance=0.1 explicitly.
    """
    if clock_dirs is None:
        clock_dirs = {
            'orig': 'tAge_clocks/EN differential models 4.6',
            'tms': 'tAge_clocks/tms_clocks',
            'tmsh': 'tAge_clocks/tmsh_clocks',
        }

    all_best_resolutions = []

    for tag, clock_folder in clock_dirs.items():
        results = []

        for resolution in res_range:
            print(f"\n Running {tag} with resolution = {resolution}")

            preds_per_file = pred_pipeline(
                assembled_adatas,
                res=resolution,
                control_file_pattern=control_file_pattern,
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

            t_stat, p_value = stats.ttest_ind(young_preds, old_preds, equal_var=False)
            mean_diff = old_preds.mean() - young_preds.mean()

            def pooled_std(s1, s2, n1, n2):
                return np.sqrt(((n1 - 1) * s1 ** 2 + (n2 - 1) * s2 ** 2) / (n1 + n2 - 2))

            s1, s2 = young_preds.std(ddof=1), old_preds.std(ddof=1)
            n1, n2 = len(young_preds), len(old_preds)
            sp = pooled_std(s1, s2, n1, n2)
            cohen_d = mean_diff / sp

            all_coverages = np.concatenate([
                adata.obs['cumulative_coverage'].values for adata in preds_per_file.values()
            ])
            mean_mp_coverage = int(np.round(np.mean(all_coverages)))

            results.append({
                "Resolution": resolution,
                "Coverage": mean_mp_coverage,
                "T_stat": abs(t_stat),
                "log_P_value": -np.log10(p_value),
                "Age_diff": abs(mean_diff),
                "Cohen_d": abs(cohen_d)
            })

        df = pd.DataFrame(results)

        df['norm_t'] = normalize_minmax(df['T_stat'])
        df['norm_p'] = normalize_minmax(df['log_P_value'])
        df['norm_d'] = normalize_minmax(df['Cohen_d'])

        # Composite score: weighted average of normalized stats
        df['score'] = cohen_weight * df['norm_d'] + tstat_weight * df['norm_t']

        best_row = select_best_resolution(df, score_col='score', resolution_col='Resolution', tolerance=tolerance)
        print(f"\n Best resolution for {tag}: {best_row['Resolution']} (score = {best_row['score']:.3f})")

        best_row['Clock'] = tag
        all_best_resolutions.append(best_row)

    return pd.DataFrame(all_best_resolutions)
