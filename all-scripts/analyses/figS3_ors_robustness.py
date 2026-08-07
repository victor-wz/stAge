#!/usr/bin/env python
"""Fig S3 -- ORS (Optimal Resolution Search) robustness: leave-one-sample-out
cross-validation, bootstrapped uncertainty on S(gamma), weight-grid (alpha,
beta) sensitivity, and gamma-perturbation analysis (Cohen's d + Gi* hotspot
Jaccard), across three tissues (Hippocampus, Spinal cord, Liver).

Source: integrated_stAge/ORS_validation_stAge.ipynb.

RE-POINTED ONTO stage/ (author-approved, mechanical change): the source notebook
imported `non_overlapping_MPs`/`filter_genes`/`get_scaled_counts`/
`final_clock_preparation`/`predict_age`/`prepare_prediction_results` from its own
local `integrated_stAge/st_utils.py` fork. Per the author's Phase-1 decision,
`v_pipeline/st_utils.py` is canonical -- this script imports the equivalent
functions from `stage.metapixels`/`stage.preprocessing`/`stage.clock` instead.
The underlying algorithm is identical (Phase 1 found only a 3-line clock-path-
convention diff between the two `st_resol.py` copies); the one behavioral
consequence is that `stage.preprocessing.get_scaled_counts` now REQUIRES an
explicit `ncbi_reference_path` argument (the source notebook's own
`get_scaled_counts(df, clock_model, 'symbol')` 3-positional-arg call matched
its OWN st_utils fork's older signature, which had no `ncbi_reference_path`
parameter at all and fell back to a hard-coded default) -- ported below with
`ncbi_reference_path` threaded through explicitly via `--ncbi-reference-path`.

GAMMA_RANGE DISCREPANCY (flagged, not silently resolved): the source notebook's
live Section-0 config cell sets `GAMMA_RANGE = [1.0, 1.5, 2.0, 4.0, 8.0]` (5
values), but its own markdown documentation states the range "matches default
in optimal_resolution_search... = [0.25, 0.5, 1, 1.5, 2, 4, 8, 16]" (8 values),
and a later "kernel died, hardcoded from printed output" recovery cell at the
tail of the notebook uses the full 8-value list when reconstructing the figure
from already-computed results. This suggests the 5-value live cell is stale
debug-run leftover, not what actually produced the notebook's own results. This
script defaults `--gamma-range` to the full 8-value list (matching
`stage.resolution_search`'s own default and the notebook's stated intent),
fully overridable via CLI -- flagged here rather than silently picking either.

Gi* HOTSPOT CONSOLIDATION (flagged, deliberate, not silently equivalent): the
source notebook implements its OWN independent Gi* hotspot classifier
(`_gi_hotspot_spots`, via `libpysal.weights.KNN.from_array` for the k-NN
weights matrix) for Section 4's perturbation-Jaccard analysis -- a FIFTH
independent in-notebook Gi* implementation not found by the original Phase 1
audit (which found four). It uses the same nominal parameters (k=8, 999
permutations, BH-FDR<0.05, z>1, hotspot-only mask) as the now-consolidated
`stage.hotspots.classify_hotspots`, but builds the k-NN spatial weights matrix
via a different code path (libpysal's own `KNN.from_array` vs.
`stage.hotspots`'s sklearn-`NearestNeighbors`-based `knn_weights_rowstd`). This
script uses `stage.hotspots.classify_hotspots` (extracting the 'hotspot' label
as the boolean mask `_gi_hotspot_spots` used to return directly) for
consistency with every other figure in this release that calls Gi* -- the two
KNN-construction code paths were NOT verified bit-for-bit equivalent by this
port; if exact reproduction of this notebook's original cached Jaccard numbers
matters, that equivalence should be checked before trusting this script's
Section 4 output to match historical values precisely.

EXCLUDED FROM THIS PORT: (1) a commented-out duplicate "Panel A" figure block
in the source (iterative-editing residue, superseded by the live block
immediately above it) -- not reproduced. (2) The notebook's final ~200 lines
are a "kernel died before this cell finished" recovery block that hardcodes
numeric results transcribed from printed cell output, so the figure could
still be regenerated without re-running the (expensive) sections above --
this is backup data for one specific historical run, not reproducible
computation, and is deliberately NOT ported here; this script always
recomputes Sections 1-4 from data (with on-disk caching, see --cache-dir).
"""

from __future__ import annotations

import argparse
import os
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import joblib
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.ticker import ScalarFormatter
from scipy import stats
from scipy.sparse import issparse
from tqdm import tqdm

from stage.metapixels import non_overlapping_MPs
from stage.preprocessing import filter_genes, get_scaled_counts
from stage.clock import final_clock_preparation, predict_age, prepare_prediction_results
from stage.hotspots import classify_hotspots, K_KNN, N_PERMS, FDR_ALPHA, GI_Z_THRESH

DEFAULT_GAMMA_RANGE = [0.25, 0.5, 1.0, 1.5, 2.0, 4.0, 8.0, 16.0]  # see module docstring
ALPHA_T_DEFAULT = 0.4
BETA_D_DEFAULT = 0.6
TOLERANCE_DEFAULT = 0.1  # absolute tolerance on the min-max normalized composite score

WEIGHT_COMBOS = [(0.2, 0.8), (0.4, 0.6), (0.5, 0.5), (0.6, 0.4), (0.8, 0.2), (1.0, 0.0), (0.0, 1.0)]
PERTURBATION_STEPS_DEFAULT = [-3, -2, -1, 0, 1, 2, 3]


# --------------------------------------------------------------------------- #
# Caching (simplified from the source notebook's many per-section RECOMPUTE_S*
# flags into one --force-recompute switch -- orchestration/glue simplification,
# not a change to any statistical computation).
# --------------------------------------------------------------------------- #

def _cache_path(cache_dir: Path, tissue: str, section: str) -> Path:
    safe = tissue.replace(' ', '_')
    return cache_dir / f'ORS_validation_{safe}_{section}.pkl'


def _save_cache(obj, cache_dir: Path, tissue: str, section: str):
    with open(_cache_path(cache_dir, tissue, section), 'wb') as f:
        pickle.dump(obj, f)


def _load_cache(cache_dir: Path, tissue: str, section: str, force_recompute: bool):
    if force_recompute:
        return None
    p = _cache_path(cache_dir, tissue, section)
    if p.exists():
        with open(p, 'rb') as f:
            return pickle.load(f)
    return None


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #

def list_tissue_files(directory: str, prefix_filter, exclude_patterns):
    """List .h5ad files in directory, optionally restricted to a filename prefix.
    Mid-age samples (matching `exclude_patterns`) are dropped, mirroring the
    source notebook's ``"Mid" not in file`` filter."""
    files = sorted(f for f in os.listdir(directory) if f.endswith('.h5ad'))
    if prefix_filter is not None:
        files = [f for f in files if f.startswith(prefix_filter)]
    files = [f for f in files if not any(p in f for p in exclude_patterns)]
    return files


def load_tissue_spot_adatas(directory, prefix, young_pattern, old_pattern, exclude_patterns):
    files = list_tissue_files(directory, prefix, exclude_patterns)
    adatas = {}
    for f in files:
        ad = sc.read(os.path.join(directory, f))
        if young_pattern in f and old_pattern not in f.split('_')[0]:
            ad.uns['age_group'] = 'young'
        elif old_pattern in f:
            ad.uns['age_group'] = 'old'
        else:
            ad.uns['age_group'] = 'young' if young_pattern in f else 'old'
        ad.layers['raw_count'] = ad.X.toarray() if issparse(ad.X) else ad.X.copy()
        adatas[f] = ad
    return adatas


# --------------------------------------------------------------------------- #
# Core ORS machinery (ported from the notebook's Section 0 helpers)
# --------------------------------------------------------------------------- #

def _cohens_d_tstat(young: np.ndarray, old: np.ndarray):
    young = np.asarray(young, dtype=float)
    old = np.asarray(old, dtype=float)
    young = young[np.isfinite(young)]
    old = old[np.isfinite(old)]
    if len(young) < 2 or len(old) < 2:
        return np.nan, np.nan
    t_stat, _ = stats.ttest_ind(young, old, equal_var=False)
    s1, s2 = young.std(ddof=1), old.std(ddof=1)
    n1, n2 = len(young), len(old)
    sp = np.sqrt(((n1 - 1) * s1 ** 2 + (n2 - 1) * s2 ** 2) / (n1 + n2 - 2))
    d = (old.mean() - young.mean()) / sp if sp > 0 else np.nan
    return abs(d), abs(t_stat)


def _composite_score(t_curve: dict, d_curve: dict, alpha: float, beta: float) -> dict:
    """Min-max normalize T and d across gamma and combine into S(gamma)."""
    gammas = sorted(t_curve.keys())
    t = np.array([t_curve[g] for g in gammas], dtype=float)
    d = np.array([d_curve[g] for g in gammas], dtype=float)

    def _mm(x):
        lo, hi = np.nanmin(x), np.nanmax(x)
        if not np.isfinite(lo) or not np.isfinite(hi) or hi - lo < 1e-12:
            return np.zeros_like(x)
        return (x - lo) / (hi - lo)

    S = alpha * _mm(t) + beta * _mm(d)
    return dict(zip(gammas, S.tolist()))


def _select_optimal_gamma(S_curve: dict, tolerance: float) -> float:
    """Highest gamma whose S is within `tolerance` of Smax (absolute, on [0,1])."""
    gammas = np.array(sorted(S_curve.keys()), dtype=float)
    S = np.array([S_curve[g] for g in gammas], dtype=float)
    Smax = np.nanmax(S)
    mask = S >= (Smax - tolerance)
    return float(gammas[mask].max())


def _assemble_metaspots(adata_dict: dict, gamma: float, control_file_pattern: str, n_neighs_mp: int) -> sc.AnnData:
    """Run non_overlapping_MPs on each sample at `gamma` and concatenate.
    Mirrors the first half of `stage.pipeline.full_nonoverlap_mp_pipeline` with
    `lower_res=True`."""
    tissue_adatas = []
    for fname, ad in adata_dict.items():
        age_group = ad.uns.get('age_group', 'young' if control_file_pattern in fname else 'old')
        mp = non_overlapping_MPs(adata=ad, age_group=age_group, lower_res=True,
                                  n_neighs=n_neighs_mp, resolution=gamma)
        mp.obs['File'] = fname
        mp.obs['age_group'] = age_group
        mp.var_names_make_unique()
        mp.obs_names_make_unique()
        tissue_adatas.append(mp)
    return sc.concat(tissue_adatas, join="outer", axis=0)


def _predict_tage(ad_filtered: sc.AnnData, clock_model, ncbi_reference_path: str) -> pd.Series:
    """get_scaled_counts -> final_clock_preparation -> predict_age, aligned to
    `ad_filtered.obs_names`. Returns tAge_SM * 48 (months)."""
    X = ad_filtered.X.toarray() if issparse(ad_filtered.X) else np.asarray(ad_filtered.X)
    df = pd.DataFrame(X.T, index=ad_filtered.var_names, columns=ad_filtered.obs.index)
    df.index.name = 'geneID'
    counts_scaled = get_scaled_counts(df, clock_model, ncbi_reference_path, 'symbol')
    prepped = final_clock_preparation(counts_scaled, clock_model, diff_suffix='young')
    preds = predict_age(prepped, clock_model)
    results = prepare_prediction_results(prepped, preds).set_index('sample')
    return results.loc[ad_filtered.obs.index, 'Predicted Age'] * 48


def run_stAge_at_gamma(adata_dict, gamma, clock_model, ncbi_reference_path,
                        control_file_pattern, n_neighs_mp):
    """Run stAge at a single resolution gamma. Returns (mp_adata, cohens_d, t_stat)."""
    merged = _assemble_metaspots(adata_dict, gamma, control_file_pattern, n_neighs_mp)
    try:
        ad_filtered = filter_genes(merged)
    except Exception:
        ad_filtered = merged
    if ad_filtered.n_vars == 0:
        raise RuntimeError(f'No genes pass the filter at gamma={gamma}')

    tage = _predict_tage(ad_filtered, clock_model, ncbi_reference_path)
    ad_filtered.obs['tAge_SM'] = tage.values

    y = ad_filtered.obs.loc[ad_filtered.obs['age_group'] == 'young', 'tAge_SM'].to_numpy()
    o = ad_filtered.obs.loc[ad_filtered.obs['age_group'] == 'old', 'tAge_SM'].to_numpy()
    cohens_d, t_stat = _cohens_d_tstat(y, o)
    return ad_filtered, cohens_d, t_stat


def compute_optimal_gamma(adata_dict, gamma_range, clock_model, ncbi_reference_path,
                           control_file_pattern, n_neighs_mp, alpha, beta, tolerance,
                           precomputed_td=None):
    d_curve, t_curve = {}, {}
    for g in gamma_range:
        if precomputed_td is not None and g in precomputed_td:
            d_curve[g], t_curve[g] = precomputed_td[g]['d'], precomputed_td[g]['t']
            continue
        _, d, t_ = run_stAge_at_gamma(adata_dict, g, clock_model, ncbi_reference_path,
                                       control_file_pattern, n_neighs_mp)
        d_curve[g], t_curve[g] = d, t_
    S_curve = _composite_score(t_curve, d_curve, alpha=alpha, beta=beta)
    return _select_optimal_gamma(S_curve, tolerance), S_curve, d_curve, t_curve


# --------------------------------------------------------------------------- #
# Section 4 helper: Gi* hotspot mask via stage.hotspots (see module docstring)
# --------------------------------------------------------------------------- #

def _gi_hotspot_mask(coords: np.ndarray, values: np.ndarray, seed: int) -> np.ndarray:
    if np.isfinite(values).sum() < 20:
        return np.zeros(len(values), dtype=bool)
    tmp = sc.AnnData(obs=pd.DataFrame({'tAge_SM': values}), obsm={'spatial': coords})
    classify_hotspots(tmp, value_col='tAge_SM', k=K_KNN, n_perms=N_PERMS,
                       fdr_alpha=FDR_ALPHA, z_thresh=GI_Z_THRESH, seed=seed)
    return (tmp.obs['aging_type'] == 'hotspot').to_numpy()


def _propagate_tage_to_spots(adata_dict, gamma, mp_tage_df, control_file_pattern, n_neighs_mp):
    """Return {fname: (coords, tAge_spot_vector)} at the given gamma. Requires
    re-running `non_overlapping_MPs` to (re)populate obs['metapixel'] on the
    spot-level adata."""
    for fname, ad in adata_dict.items():
        age_group = ad.uns.get('age_group', 'young' if control_file_pattern in fname else 'old')
        non_overlapping_MPs(adata=ad, age_group=age_group, lower_res=True,
                             n_neighs=n_neighs_mp, resolution=gamma)
    by_file = {}
    for fname, sub in mp_tage_df.groupby('File'):
        mp_ids = sub.index.to_series().str.split('.').str[2]  # age.group.<id>.uuid
        by_file[fname] = dict(zip(mp_ids.values, sub['tAge_SM'].values))
    out = {}
    for fname, ad in adata_dict.items():
        mp_col = ad.obs['metapixel'].astype(str)
        lookup = by_file.get(fname, {})
        vals = np.asarray([lookup.get(m, np.nan) for m in mp_col], dtype=float)
        out[fname] = (ad.obsm['spatial'], vals)
    return out


# --------------------------------------------------------------------------- #
# Sections 1-4
# --------------------------------------------------------------------------- #

def run_section1_loso(tissue, ad_dict, gamma_range, clock_model, ncbi_reference_path,
                       control_file_pattern, n_neighs_mp, alpha, beta, tolerance,
                       baseline_gamma, baseline_S_curve, baseline_d_curve):
    young = [f for f, a in ad_dict.items() if a.uns['age_group'] == 'young']
    old = [f for f, a in ad_dict.items() if a.uns['age_group'] == 'old']
    if len(young) < 3 or len(old) < 3:
        print(f'[{tissue}] SKIPPED — insufficient samples for LOSO (young={len(young)}, old={len(old)})')
        return pd.DataFrame()

    full_S = baseline_S_curve[tissue]
    full_Smax = max(full_S.values())
    accept = {g for g, s in full_S.items() if s >= full_Smax - tolerance}
    g_full = baseline_gamma[tissue]
    d_full = baseline_d_curve[tissue][g_full]

    rows = []
    for held_out in young + old:
        age_group = ad_dict[held_out].uns['age_group']
        sub = {f: a for f, a in ad_dict.items() if f != held_out}
        print(f'  [{tissue}] holdout={held_out} ({age_group})')
        g_loso, S_l, d_l, _ = compute_optimal_gamma(
            sub, gamma_range, clock_model, ncbi_reference_path, control_file_pattern,
            n_neighs_mp, alpha, beta, tolerance)
        rows.append({
            'tissue': tissue, 'held_out_sample': held_out, 'age_group': age_group,
            'gamma_full': g_full, 'gamma_loso': g_loso, 'delta_gamma': g_loso - g_full,
            'cohens_d_loso': d_l[g_loso], 'cohens_d_full': d_full,
            'in_tolerance': g_loso in accept, 'S_loso': S_l,
        })
    return pd.DataFrame(rows)


def run_section2_bootstrap(tissue, mp_tage, gamma_range, n_bootstrap, alpha, beta, tolerance,
                            baseline_S_curve, seed):
    pools = {}
    for g, df in mp_tage.items():
        y = df.loc[df['age_group'] == 'young', 'tAge_SM'].to_numpy()
        o = df.loc[df['age_group'] == 'old', 'tAge_SM'].to_numpy()
        pools[g] = (y, o)

    rng = np.random.default_rng(seed)
    S_matrix = np.full((n_bootstrap, len(gamma_range)), np.nan)
    d_matrix = np.full_like(S_matrix, np.nan)
    t_matrix = np.full_like(S_matrix, np.nan)
    gamma_opt_boot = np.full(n_bootstrap, np.nan)

    full_S = baseline_S_curve[tissue]
    full_Smax = max(full_S.values())
    accept = {g for g, s in full_S.items() if s >= full_Smax - tolerance}

    t0 = time.time()
    for i in tqdm(range(n_bootstrap), desc=f'bootstrap {tissue}'):
        d_curve, t_curve = {}, {}
        for j, g in enumerate(gamma_range):
            y, o = pools[g]
            if len(y) < 2 or len(o) < 2:
                d_curve[g], t_curve[g] = np.nan, np.nan
                continue
            yb = y[rng.integers(0, len(y), size=len(y))]
            ob = o[rng.integers(0, len(o), size=len(o))]
            d_b, t_b = _cohens_d_tstat(yb, ob)
            d_curve[g], t_curve[g] = d_b, t_b
            d_matrix[i, j], t_matrix[i, j] = d_b, t_b
        S_curve_b = _composite_score(t_curve, d_curve, alpha=alpha, beta=beta)
        for j, g in enumerate(gamma_range):
            S_matrix[i, j] = S_curve_b[g]
        try:
            gamma_opt_boot[i] = _select_optimal_gamma(S_curve_b, tolerance)
        except Exception:
            gamma_opt_boot[i] = np.nan
        if i == 9:
            est = (time.time() - t0) / 10 * n_bootstrap
            print(f'  ~{est:.1f}s estimated for {n_bootstrap} iterations')

    return {'S_matrix': S_matrix, 'd_matrix': d_matrix, 't_matrix': t_matrix,
            'gamma_opt': gamma_opt_boot, 'gammas': list(gamma_range), 'accept_window': accept}


def run_section3_weights(tissue, baseline_d_curve, baseline_t_curve, baseline_gamma, tolerance):
    d_curve = baseline_d_curve[tissue]
    t_curve = baseline_t_curve[tissue]
    g_default = baseline_gamma[tissue]
    rows = []
    for a, b in WEIGHT_COMBOS:
        S_curve = _composite_score(t_curve, d_curve, alpha=a, beta=b)
        g_star = _select_optimal_gamma(S_curve, tolerance)
        rows.append({
            'tissue': tissue, 'alpha': a, 'beta': b, 'gamma_optimal': g_star,
            'cohens_d': d_curve[g_star], 'delta_gamma_from_default': g_star - g_default,
            'same_as_default': g_star == g_default,
        })
    return rows


def run_section4_perturbation(tissue, ad_dict, gamma_range, base_results, baseline_gamma,
                               baseline_d_curve, clock_model, ncbi_reference_path,
                               control_file_pattern, n_neighs_mp, perturbation_steps, seed):
    g_star = baseline_gamma[tissue]
    idx_star = gamma_range.index(g_star)
    d_full_star = baseline_d_curve[tissue][g_star]

    mp_tage_star = base_results[tissue]['mp_tage'][g_star]
    propagated_star = _propagate_tage_to_spots(ad_dict, g_star, mp_tage_star, control_file_pattern, n_neighs_mp)
    hot_at_star = {f: _gi_hotspot_mask(coords, vals, seed) for f, (coords, vals) in propagated_star.items()}

    rows = []
    for step in perturbation_steps:
        j = idx_star + step
        if j < 0 or j >= len(gamma_range):
            print(f'  [{tissue}] step {step:+d} out of range — skipped')
            continue
        g = gamma_range[j]
        d_here = baseline_d_curve[tissue][g]

        if step == 0:
            hot_here = hot_at_star
        else:
            if g not in base_results[tissue]['mp_tage']:
                _, d_, t_ = run_stAge_at_gamma(ad_dict, g, clock_model, ncbi_reference_path,
                                                control_file_pattern, n_neighs_mp)
            mp_tage = base_results[tissue]['mp_tage'][g]
            propagated = _propagate_tage_to_spots(ad_dict, g, mp_tage, control_file_pattern, n_neighs_mp)
            hot_here = {f: _gi_hotspot_mask(coords, vals, seed) for f, (coords, vals) in propagated.items()}

        jacc_by_group = {'young': [], 'old': []}
        for fname, ad in ad_dict.items():
            hs = hot_at_star.get(fname, np.zeros(ad.n_obs, dtype=bool))
            hp = hot_here.get(fname, np.zeros(ad.n_obs, dtype=bool))
            inter = np.logical_and(hs, hp).sum()
            union = np.logical_or(hs, hp).sum()
            if union == 0:
                jacc = 1.0 if (hs.sum() == 0 and hp.sum() == 0) else np.nan
            else:
                jacc = inter / union
            if np.isfinite(jacc):
                jacc_by_group[ad.uns['age_group']].append(jacc)
        jy = float(np.mean(jacc_by_group['young'])) if jacc_by_group['young'] else np.nan
        jo = float(np.mean(jacc_by_group['old'])) if jacc_by_group['old'] else np.nan
        jm = float(np.nanmean([jy, jo]))

        rows.append({
            'tissue': tissue, 'perturbation_step': step, 'gamma_value': g, 'cohens_d': d_here,
            'jaccard_young': jy, 'jaccard_old': jo, 'jaccard_mean': jm,
            'delta_cohens_d_from_optimal': d_here - d_full_star,
        })
        print(f'  [{tissue}] step {step:+d}  gamma={g}  d={d_here:.3f}  jaccard_y={jy:.3f}  jaccard_o={jo:.3f}')

    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Section 5: summary figure
# --------------------------------------------------------------------------- #

def _shade_tolerance(ax, gammas, S, tolerance, color='lightgrey'):
    Smax = np.nanmax(S)
    mask = S >= (Smax - tolerance)
    if mask.any():
        ax.axvspan(gammas[mask].min(), gammas[mask].max(), color=color, alpha=0.35, zorder=0)


def make_summary_figure(tissues, tissue_colors, gamma_range, baseline_S_curve, baseline_gamma,
                         loso_dfs, loso_stability_rate, bootstrap_results, bootstrap_summary_df,
                         weights_df, weight_stability_rate, perturbation_dfs, tolerance, out_dir):
    gammas = np.array(gamma_range, dtype=float)

    fig = plt.figure(figsize=(18, 14), dpi=150)
    gs = fig.add_gridspec(7, 3, hspace=0.85, wspace=0.35, height_ratios=[1, 1, 1, 1, 1.1, 1, 1])

    # ---- Panel A: LOSO ----
    for col, tissue in enumerate(tissues):
        ax = fig.add_subplot(gs[0:2, col])
        color = tissue_colors[tissue]
        full_S = baseline_S_curve[tissue]
        S_full_vec = np.array([full_S[g] for g in gamma_range])
        df = loso_dfs.get(tissue, pd.DataFrame())
        if not df.empty:
            for _, row in df.iterrows():
                sc_vec = np.array([row['S_loso'][g] for g in gamma_range])
                ax.plot(gammas, sc_vec, color=color, alpha=0.3, lw=1)
        ax.plot(gammas, S_full_vec, color=color, lw=2.5, label='full data')
        ax.axvline(baseline_gamma[tissue], ls='--', color='k', lw=1)
        _shade_tolerance(ax, gammas, S_full_vec, tolerance)
        ax.set_xscale('log')
        ax.set_xticks(gamma_range)
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.xaxis.set_minor_formatter(ScalarFormatter())
        ax.tick_params(axis='x', which='minor', labelsize=0)
        ax.set_xticklabels([str(int(g)) if g == int(g) else str(g) for g in gamma_range],
                            rotation=35, ha='right', fontsize=7)
        ax.set_xlabel(r'$\gamma$')
        ax.set_ylabel(r'$S(\gamma)$')
        ax.set_title(tissue)
        rate = loso_stability_rate.get(tissue, np.nan)
        if np.isfinite(rate):
            ax.text(0.02, 0.05, f'LOSO stability: {rate:.0%}', transform=ax.transAxes, fontsize=10,
                     bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
        ax.legend(fontsize=7, frameon=False, loc='upper right')
        if col == 0:
            ax.annotate('A', xy=(-0.18, 1.05), xycoords='axes fraction', fontsize=18, fontweight='bold')

    # ---- Panel B: Bootstrap ----
    for col, tissue in enumerate(tissues):
        ax = fig.add_subplot(gs[2:4, col])
        color = tissue_colors[tissue]
        S_mat = bootstrap_results[tissue]['S_matrix']
        g_opt = bootstrap_results[tissue]['gamma_opt']
        mean_S = np.nanmean(S_mat, axis=0)
        lo = np.nanpercentile(S_mat, 2.5, axis=0)
        hi = np.nanpercentile(S_mat, 97.5, axis=0)
        ax.fill_between(gammas, lo, hi, color=color, alpha=0.2, label='95% CI')
        ax.plot(gammas, mean_S, color=color, lw=2.5, label='bootstrap mean')
        ax.axvline(baseline_gamma[tissue], ls='--', color='k', lw=1)
        _shade_tolerance(ax, gammas, np.array([baseline_S_curve[tissue][g] for g in gamma_range]), tolerance)
        ax.set_xscale('log')
        ax.set_xlabel(r'$\gamma$')
        ax.set_ylabel(r'$S(\gamma)$ (bootstrap)')
        ax.set_title(tissue)

        row = bootstrap_summary_df.set_index('tissue').loc[tissue]
        ax.text(0.02, 0.08,
                f"Bootstrap stability: {row['stability_rate']:.0%}\n"
                f"d @γ*: {row['cohens_d_mean']:.2f} ± {row['cohens_d_std']:.2f}",
                transform=ax.transAxes, fontsize=9, bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

        ax_ins = ax.inset_axes([0.55, 0.60, 0.42, 0.35])
        bins = np.unique(gammas)
        edges = np.concatenate([[bins[0] / 1.4], (bins[:-1] + bins[1:]) / 2, [bins[-1] * 1.4]])
        ax_ins.hist(g_opt[np.isfinite(g_opt)], bins=edges, color=color, alpha=0.8)
        ax_ins.set_xscale('log')
        ax_ins.set_xlabel(r'$\gamma^*$', fontsize=8)
        ax_ins.set_ylabel('count', fontsize=8)
        ax_ins.tick_params(labelsize=7)
        ax_ins.axvline(baseline_gamma[tissue], ls='--', color='k', lw=0.8)
        if col == 0:
            ax.annotate('B', xy=(-0.18, 1.05), xycoords='axes fraction', fontsize=18, fontweight='bold')

    # ---- Panel C: Weight sensitivity heatmap ----
    ax = fig.add_subplot(gs[4, :])
    mat = np.zeros((len(tissues), len(WEIGHT_COMBOS)))
    for i, tissue in enumerate(tissues):
        tdf = weights_df[weights_df['tissue'] == tissue]
        for j, (a, b) in enumerate(WEIGHT_COMBOS):
            mat[i, j] = tdf[(tdf['alpha'] == a) & (tdf['beta'] == b)]['gamma_optimal'].iloc[0]
    default_gammas = [baseline_gamma[t] for t in tissues]
    vcenter = float(np.median(default_gammas))
    vmin, vmax = 0.0, 10.0
    norm = mpl.colors.TwoSlopeNorm(vmin=min(vmin, vcenter - 1e-3), vcenter=vcenter, vmax=max(vmax, vcenter + 1e-3))
    im = ax.imshow(mat, aspect='auto', cmap='RdBu_r', norm=norm)
    ax.set_xticks(range(len(WEIGHT_COMBOS)))
    ax.set_xticklabels([f'α={a}\nβ={b}' for a, b in WEIGHT_COMBOS])
    ax.set_yticks(range(len(tissues)))
    ax.set_yticklabels(tissues)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, f'{mat[i, j]:g}', ha='center', va='center', color='black', fontsize=9)
    try:
        def_j = WEIGHT_COMBOS.index((ALPHA_T_DEFAULT, BETA_D_DEFAULT))
        ax.add_patch(Rectangle((def_j - 0.5, -0.5), 1, len(tissues), fill=False, edgecolor='black', lw=2.5))
    except ValueError:
        pass
    plt.colorbar(im, ax=ax, label=r'$\gamma^*$')
    ax.set_title('Optimal γ across weight combinations')
    note = '  |  '.join(f'{t}: {weight_stability_rate[t]:.0%}' for t in tissues)
    ax.text(0.5, -0.45, f'Weight stability (γ* matches default): {note}', transform=ax.transAxes,
            ha='center', fontsize=10)
    ax.annotate('C', xy=(-0.07, 1.08), xycoords='axes fraction', fontsize=18, fontweight='bold')

    # ---- Panel D: Perturbation ----
    for col, tissue in enumerate(tissues):
        ax_d = fig.add_subplot(gs[5, col])
        ax_j = fig.add_subplot(gs[6, col])
        color = tissue_colors[tissue]
        pdf = perturbation_dfs[tissue].sort_values('perturbation_step')
        ax_d.plot(pdf['perturbation_step'], pdf['cohens_d'], color=color, lw=2)
        zero = pdf[pdf['perturbation_step'] == 0]
        if not zero.empty:
            ax_d.plot(0, zero['cohens_d'].iloc[0], 'o', color=color, markersize=10)
            d_opt = zero['cohens_d'].iloc[0]
            ax_d.axhline(d_opt, ls='--', color='grey', lw=1)
            ax_d.axhspan(d_opt * 0.9, d_opt * 1.1, color='grey', alpha=0.15)
        ax_d.set_xlabel('perturbation step')
        ax_d.set_ylabel("Cohen's d")
        ax_d.set_title(tissue)

        ax_j.plot(pdf['perturbation_step'], pdf['jaccard_mean'], color=color, lw=2, label='mean')
        ax_j.plot(pdf['perturbation_step'], pdf['jaccard_young'], ls='--', color=color, lw=1.2, label='young')
        ax_j.plot(pdf['perturbation_step'], pdf['jaccard_old'], ls=':', color=color, lw=1.2, label='old')
        if not zero.empty:
            ax_j.plot(0, zero['jaccard_mean'].iloc[0], 'o', color=color, markersize=10)
        ax_j.axhline(0.8, ls='--', color='grey', lw=1)
        ax_j.set_ylim(0, 1.05)
        ax_j.set_xlabel('perturbation step')
        ax_j.set_ylabel('Jaccard')
        ax_j.legend(fontsize=7, loc='lower center')
        if col == 0:
            ax_d.annotate('D', xy=(-0.18, 1.10), xycoords='axes fraction', fontsize=18, fontweight='bold')

    fig.suptitle('Optimal Resolution Search validation and robustness analysis',
                 fontsize=16, fontweight='bold', y=0.995)

    out_pdf = Path(out_dir) / 'figS3_ORS_validation.pdf'
    out_png = Path(out_dir) / 'figS3_ORS_validation.png'
    fig.savefig(out_pdf, dpi=300, bbox_inches='tight')
    fig.savefig(out_png, dpi=300, bbox_inches='tight')
    print(f'Saved: {out_pdf}\nSaved: {out_png}')
    plt.close(fig)


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--hippocampus-dir', required=True, help="Directory of Hippocampus_*.h5ad files")
    p.add_argument('--spinalcord-dir', required=True, help="Directory of Spinalcord *.h5ad files")
    p.add_argument('--liver-dir', required=True,
                    help="Directory containing Liver_*.h5ad files (prefix-filtered from a shared directory in the source)")
    p.add_argument('--clock-path', required=True, help="Path to EN_Chronoage_All_All_WT_scaleddiff.pkl")
    p.add_argument('--ncbi-reference-path', required=True)
    p.add_argument('--out-dir', required=True)
    p.add_argument('--cache-dir', default=None, help="Defaults to <out-dir>/cache")
    p.add_argument('--force-recompute', action='store_true')
    p.add_argument('--gamma-range', type=float, nargs='+', default=DEFAULT_GAMMA_RANGE)
    p.add_argument('--alpha-t', type=float, default=ALPHA_T_DEFAULT)
    p.add_argument('--beta-d', type=float, default=BETA_D_DEFAULT)
    p.add_argument('--tolerance', type=float, default=TOLERANCE_DEFAULT)
    p.add_argument('--n-neighs-mp', type=int, default=20)
    p.add_argument('--mp-coverage-threshold', type=float, default=1000)
    p.add_argument('--n-bootstrap', type=int, default=500)
    p.add_argument('--perturbation-steps', type=int, nargs='+', default=PERTURBATION_STEPS_DEFAULT)
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()

    cache_dir = Path(args.cache_dir) if args.cache_dir else Path(args.out_dir) / 'cache'
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)

    control_file_pattern = '_Y'
    young_pattern, old_pattern = '_Y', '_O'
    exclude_patterns = ('M-13M-',)

    tissue_paths = {
        'Hippocampus': (args.hippocampus_dir, None),
        'Spinal cord': (args.spinalcord_dir, None),
        'Liver': (args.liver_dir, 'Liver_'),
    }
    tissue_colors = {'Hippocampus': '#4C72B0', 'Spinal cord': '#C44E52', 'Liver': '#2d6a4f'}

    clock_model = joblib.load(args.clock_path)
    print('Clock loaded:', args.clock_path)

    spot_adata_dict = {}
    for tissue, (directory, prefix) in tissue_paths.items():
        spot_adata_dict[tissue] = load_tissue_spot_adatas(directory, prefix, young_pattern, old_pattern, exclude_patterns)
        young = [f for f, a in spot_adata_dict[tissue].items() if a.uns['age_group'] == 'young']
        old = [f for f, a in spot_adata_dict[tissue].items() if a.uns['age_group'] == 'old']
        print(f'{tissue:12s}  young={len(young):2d}  old={len(old):2d}  total={len(young) + len(old):2d}')

    # ---- Base gamma scan per tissue (cached) ----
    base_results = {}
    for tissue in tissue_paths:
        cache = _load_cache(cache_dir, tissue, 'base_gamma_scan', args.force_recompute)
        if cache is not None:
            print(f'[{tissue}] loaded base gamma scan from cache')
            base_results[tissue] = cache
            continue
        print(f'[{tissue}] running base gamma scan over {args.gamma_range} …')
        td, mp_tage = {}, {}
        for g in args.gamma_range:
            t0 = time.time()
            ad, d, t_ = run_stAge_at_gamma(spot_adata_dict[tissue], g, clock_model, args.ncbi_reference_path,
                                            control_file_pattern, args.n_neighs_mp)
            td[g] = {'d': d, 't': t_}
            mp_tage[g] = ad.obs[['File', 'age_group', 'tAge_SM']].copy()
            print(f'  γ={g:<5}  d={d:.3f}  t={t_:.2f}  n_mp={ad.n_obs:5d}  ({time.time() - t0:.1f}s)')
        base_results[tissue] = {'td': td, 'mp_tage': mp_tage}
        _save_cache(base_results[tissue], cache_dir, tissue, 'base_gamma_scan')

    baseline_gamma, baseline_S_curve, baseline_d_curve, baseline_t_curve = {}, {}, {}, {}
    for tissue in tissue_paths:
        td = base_results[tissue]['td']
        d_curve = {g: td[g]['d'] for g in args.gamma_range}
        t_curve = {g: td[g]['t'] for g in args.gamma_range}
        S_curve = _composite_score(t_curve, d_curve, args.alpha_t, args.beta_d)
        g_star = _select_optimal_gamma(S_curve, args.tolerance)
        baseline_gamma[tissue] = g_star
        baseline_S_curve[tissue] = S_curve
        baseline_d_curve[tissue] = d_curve
        baseline_t_curve[tissue] = t_curve
        print(f'[{tissue}] γ* = {g_star}  Smax = {max(S_curve.values()):.3f}')

    # ---- Section 1: LOSO ----
    loso_dfs, loso_stability_rate = {}, {}
    for tissue in tissue_paths:
        cache = _load_cache(cache_dir, tissue, 'section1_loso', args.force_recompute)
        if cache is not None:
            loso_dfs[tissue] = cache
        else:
            loso_dfs[tissue] = run_section1_loso(
                tissue, spot_adata_dict[tissue], args.gamma_range, clock_model, args.ncbi_reference_path,
                control_file_pattern, args.n_neighs_mp, args.alpha_t, args.beta_d, args.tolerance,
                baseline_gamma, baseline_S_curve, baseline_d_curve)
            _save_cache(loso_dfs[tissue], cache_dir, tissue, 'section1_loso')
        df = loso_dfs[tissue]
        if not df.empty:
            loso_stability_rate[tissue] = df['in_tolerance'].mean()
            print(f'\n=== LOSO — {tissue} — stability = {loso_stability_rate[tissue]:.0%} ===')

    # ---- Section 2: Bootstrap ----
    bootstrap_results = {}
    for tissue in tissue_paths:
        cache = _load_cache(cache_dir, tissue, 'section2_bootstrap', args.force_recompute)
        if cache is not None:
            bootstrap_results[tissue] = cache
        else:
            bootstrap_results[tissue] = run_section2_bootstrap(
                tissue, base_results[tissue]['mp_tage'], args.gamma_range, args.n_bootstrap,
                args.alpha_t, args.beta_d, args.tolerance, baseline_S_curve, args.seed)
            _save_cache(bootstrap_results[tissue], cache_dir, tissue, 'section2_bootstrap')

    rows = []
    for tissue in tissue_paths:
        b = bootstrap_results[tissue]
        g_opt = b['gamma_opt']
        g_full = baseline_gamma[tissue]
        j_full = args.gamma_range.index(g_full)
        d_at_star = b['d_matrix'][:, j_full]
        stability = np.mean([g in b['accept_window'] for g in g_opt if np.isfinite(g)])
        mode_val = pd.Series(g_opt).mode().iloc[0] if np.isfinite(g_opt).any() else np.nan
        rows.append({
            'tissue': tissue, 'gamma_optimal_full': g_full, 'gamma_bootstrap_mean': np.nanmean(g_opt),
            'gamma_bootstrap_std': np.nanstd(g_opt), 'gamma_bootstrap_mode': mode_val,
            'stability_rate': stability, 'cohens_d_mean': np.nanmean(d_at_star), 'cohens_d_std': np.nanstd(d_at_star),
        })
    bootstrap_summary_df = pd.DataFrame(rows)
    print('\n=== Bootstrap summary ===')
    print(bootstrap_summary_df.to_string(index=False))

    # ---- Section 3: Weight sensitivity ----
    rows = []
    for tissue in tissue_paths:
        cache = _load_cache(cache_dir, tissue, 'section3_weights', args.force_recompute)
        if cache is not None:
            rows.extend(cache)
        else:
            tissue_rows = run_section3_weights(tissue, baseline_d_curve, baseline_t_curve, baseline_gamma, args.tolerance)
            _save_cache(tissue_rows, cache_dir, tissue, 'section3_weights')
            rows.extend(tissue_rows)
    weights_df = pd.DataFrame(rows)
    weight_stability_rate = weights_df.groupby('tissue')['same_as_default'].mean().to_dict()
    print('\n=== Weight sensitivity ===')
    print(weights_df.to_string(index=False))

    # ---- Section 4: Perturbation ----
    perturbation_dfs = {}
    for tissue in tissue_paths:
        cache = _load_cache(cache_dir, tissue, 'section4_perturbation', args.force_recompute)
        if cache is not None:
            perturbation_dfs[tissue] = cache
        else:
            perturbation_dfs[tissue] = run_section4_perturbation(
                tissue, spot_adata_dict[tissue], args.gamma_range, base_results, baseline_gamma,
                baseline_d_curve, clock_model, args.ncbi_reference_path, control_file_pattern,
                args.n_neighs_mp, args.perturbation_steps, args.seed)
            _save_cache(perturbation_dfs[tissue], cache_dir, tissue, 'section4_perturbation')

    perturbation_df = pd.concat(perturbation_dfs.values(), ignore_index=True)
    print('\n=== Perturbation ===')
    print(perturbation_df.to_string(index=False))
    perturbation_df.to_csv(os.path.join(args.out_dir, 'figS3_perturbation.csv'), index=False)
    weights_df.to_csv(os.path.join(args.out_dir, 'figS3_weight_sensitivity.csv'), index=False)
    bootstrap_summary_df.to_csv(os.path.join(args.out_dir, 'figS3_bootstrap_summary.csv'), index=False)

    # ---- Section 5: figure ----
    make_summary_figure(list(tissue_paths.keys()), tissue_colors, args.gamma_range, baseline_S_curve,
                         baseline_gamma, loso_dfs, loso_stability_rate, bootstrap_results, bootstrap_summary_df,
                         weights_df, weight_stability_rate, perturbation_dfs, args.tolerance, args.out_dir)


if __name__ == '__main__':
    main()
