"""Self-contained GSEA: Stouffer/Liptak Z-combination from meta-analysis summary
statistics, ssGSEA-based self-contained pathway testing, and GSEA-Prerank with
cross-tissue NES correlation.

Maps to Figs 6d, S8 (Self-contained GSEA, Stouffer/Liptak).

Source: v_pipeline/stage_dstream_loop.py, functions `self_contained_from_meta`,
`self_contained_gsea_ssgsea`, `gsea_prerank_df`, `nes_correlation_pipeline`.
"""

from __future__ import annotations

import inspect
from functools import reduce
from typing import Dict, Iterable, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import gseapy as gp
from scipy.stats import norm
from statsmodels.stats.multitest import multipletests


def self_contained_from_meta(
    meta_tbl: pd.DataFrame,
    gene_sets: Dict[str, list],
    beta_col: str = 'beta',
    se_col: str = 'se',
    min_size: int = 10,
    corr_ref: Union[pd.DataFrame, None] = None,
    gene_name_upper: bool = True,
) -> pd.DataFrame:
    """Self-contained gene-set test using meta-signature summary statistics.

    For each gene set S, combine per-gene Z_i = beta_i / se_i via Stouffer/Liptak,
    with a CAMERA-style variance-inflation factor to account for inter-gene
    correlation:  Z_S = sum(Z_i) / sqrt(m * (1 + (m-1) * rho_bar))
    where rho_bar is the average pairwise correlation among genes in S, estimated
    from `corr_ref` (genes x samples expression) if provided, else 0.

    Returns a DataFrame (index=pathway term) with columns
    ['Z','P','FDR','n_genes','direction','mean_beta'].
    """
    mt = meta_tbl.copy()
    if gene_name_upper:
        mt.index = mt.index.astype(str).str.upper()
    z = (mt[beta_col] / mt[se_col]).replace([np.inf, -np.inf], np.nan).dropna()

    corr_ref_use = None
    if corr_ref is not None:
        X = corr_ref.copy()
        X.index = X.index.astype(str).str.upper()
        corr_ref_use = X.T.corr(min_periods=3)

    results = []
    for term, genes in gene_sets.items():
        g = pd.Index([str(x).upper() for x in genes])
        zs = z.reindex(g).dropna()
        m = len(zs)
        if m < min_size:
            continue

        if corr_ref_use is not None:
            g_in = [gg for gg in zs.index if gg in corr_ref_use.index]
            if len(g_in) >= 3:
                C = corr_ref_use.loc[g_in, g_in].values
                rho_bar = (C.sum() - np.trace(C)) / (len(g_in) * (len(g_in) - 1))
            else:
                rho_bar = 0.0
        else:
            rho_bar = 0.0

        vif = 1.0 + (m - 1.0) * float(rho_bar)
        Zs = zs.sum() / np.sqrt(m * max(vif, 1e-9))
        p = 2 * norm.sf(abs(Zs))
        results.append((term, Zs, p, m, np.sign(zs.mean()), mt.loc[zs.index, beta_col].mean()))

    if not results:
        return pd.DataFrame(columns=['Z', 'P', 'FDR', 'n_genes', 'direction', 'mean_beta'])

    out = pd.DataFrame(
        results, columns=['Term', 'Z', 'P', 'n_genes', 'direction', 'mean_beta']
    ).set_index('Term')
    out['FDR'] = multipletests(out['P'], method='fdr_bh')[1]
    return out.sort_values('FDR')


def _fetch_library_any(lib_name: str, organism: str = 'Mouse') -> Dict[str, list]:
    """Fetch an Enrichr/MSigDB-style library as dict{term: [genes]}, robust to
    the gseapy `get_library` signature changing across versions."""
    sig = inspect.signature(gp.get_library)
    params = set(sig.parameters.keys())
    if 'name' in params:
        return gp.get_library(name=lib_name, organism=organism)
    if 'gene_sets' in params:
        return gp.get_library(gene_sets=lib_name, organism=organism)
    return gp.get_library(lib_name, organism=organism)


def _load_gene_sets(
    gene_sets: Union[str, Dict[str, list], Iterable[Union[str, Dict[str, list]]]],
    organism: str = 'Mouse',
) -> Dict[str, list]:
    """Accept a single library name, a .gmt path, a dict {term: [genes]}, or an
    iterable mixing the above; returns one merged dict {term: [genes]}."""
    def _one(x):
        if isinstance(x, dict):
            return x
        if isinstance(x, str) and x.lower().endswith('.gmt'):
            return gp.parser.read_gmt(x)
        if isinstance(x, str):
            return _fetch_library_any(x, organism=organism)
        raise ValueError(f'Unsupported gene_sets entry: {type(x)}')

    if isinstance(gene_sets, (list, tuple, set)):
        d: Dict[str, list] = {}
        for item in gene_sets:
            d.update(_one(item))
        return d
    return _one(gene_sets)


def self_contained_gsea_ssgsea(
    expr: pd.DataFrame,
    group: pd.Series,
    gene_sets: Union[str, Dict, Iterable],
    organism: str = 'Mouse',
    min_size: int = 10,
    max_size: int = 5000,
    sample_norm_method: str = 'rank',
    use_permutation: bool = True,
    n_perms: int = 2000,
    fdr_method: str = 'fdr_bh',
    random_state: int = 42,
    covariates: Union[pd.DataFrame, None] = None,
    id_upper: bool = False,
    verbose: bool = True,
):
    """Self-contained pathway test via per-sample ssGSEA + group comparison
    (+ optional label-permutation p-values). `expr`: genes x samples (index=gene
    symbols); `group`: length n_samples, index aligned to expr.columns, exactly
    2 unique values.

    Returns (results, es_matrix): results is a DataFrame (index=pathway term,
    columns effect/t/p/p_perm/FDR/n_eff_genes/group1/group0/direction); es_matrix
    is the ssGSEA enrichment-score matrix (pathways x samples).
    """
    from scipy import stats

    rng = np.random.default_rng(random_state)

    expr = expr.copy()
    if id_upper:
        expr.index = expr.index.astype(str).str.upper()

    group = group.loc[expr.columns].astype(str)
    if covariates is not None:
        covariates = covariates.loc[expr.columns]

    groups = group.unique().tolist()
    if len(groups) != 2:
        raise ValueError(f'Expected exactly 2 groups, got: {groups}')
    g1, g0 = groups[0], groups[1]

    if verbose:
        print(f'Samples per group: {g1}={sum(group == g1)}, {g0}={sum(group == g0)}')

    gmt = _load_gene_sets(gene_sets, organism=organism)
    genes_in_expr = set(expr.index)
    eff_size = {term: len(set(genes) & genes_in_expr) for term, genes in gmt.items()}
    keep = [t for t, n in eff_size.items() if min_size <= n <= max_size]
    gmt_f = {t: list(set(gmt[t]) & genes_in_expr) for t in keep}

    if verbose:
        print(f'Gene sets: {len(gmt)} -> kept {len(gmt_f)} (min_size={min_size}, max_size={max_size})')
    if len(gmt_f) == 0:
        raise ValueError('No gene sets passed the size/coverage filter.')

    ss = gp.ssgsea(
        data=expr, gene_sets=gmt_f, outdir=None,
        sample_norm_method=sample_norm_method, no_plot=True,
        processes=1, seed=random_state, verbose=False,
    ).res2d
    es = ss.astype(float)

    idx_g1 = group[group == g1].index
    idx_g0 = group[group == g0].index

    def _welch_t(x1, x0):
        return stats.ttest_ind(x1, x0, equal_var=False)

    effects = es[idx_g1].mean(axis=1) - es[idx_g0].mean(axis=1)
    tvals, pvals = [], []

    if covariates is not None and covariates.shape[1] > 0:
        import statsmodels.api as sm
        X = pd.get_dummies(group, drop_first=True)
        X = pd.concat([X, covariates], axis=1).assign(const=1.0)
        for term in es.index:
            y = es.loc[term, X.index].values
            m = sm.OLS(y, X.values).fit()
            tvals.append(m.tvalues[0])
            pvals.append(m.pvalues[0])
    else:
        for term in es.index:
            t, p = _welch_t(es.loc[term, idx_g1], es.loc[term, idx_g0])
            tvals.append(t)
            pvals.append(p)

    res = pd.DataFrame({
        'effect': effects.values,
        't': np.array(tvals, dtype=float),
        'p': np.array(pvals, dtype=float),
        'n_eff_genes': [len(gmt_f[t]) for t in es.index],
        'group1': g1, 'group0': g0,
    }, index=es.index)

    if use_permutation:
        obs = res['effect'].values
        perm_counts = np.zeros_like(obs, dtype=int)
        grp_arr = group.values
        idx_all = group.index.values
        for _ in range(n_perms):
            grp_perm = pd.Series(grp_arr[rng.permutation(len(grp_arr))], index=idx_all)
            g1_idx = grp_perm[grp_perm == g1].index
            g0_idx = grp_perm[grp_perm == g0].index
            if len(g1_idx) == 0 or len(g0_idx) == 0:
                continue
            eff_perm = es[g1_idx].mean(axis=1) - es[g0_idx].mean(axis=1)
            perm_counts += (np.abs(eff_perm.values) >= np.abs(obs)).astype(int)
        p_perm = (perm_counts + 1) / (n_perms + 1)
        res['p_perm'] = p_perm
        res['FDR'] = multipletests(p_perm, method=fdr_method)[1]
    else:
        res['p_perm'] = np.nan
        res['FDR'] = multipletests(res['p'].values, method=fdr_method)[1]

    res['direction'] = np.where(res['effect'] > 0, f'up_in_{g1}', f'up_in_{g0}')
    res = res.sort_values(['FDR', 'effect'], ascending=[True, False])
    return res, es


def _fetch_library(lib_name: str, organism: str = 'Mouse') -> Dict[str, list]:
    sig = inspect.signature(gp.get_library)
    if len(sig.parameters) == 1:
        return gp.get_library(lib_name)
    if 'gene_sets' in sig.parameters:
        return gp.get_library(gene_sets=lib_name, organism=organism)
    return gp.get_library(lib_name, organism=organism)


def gsea_prerank_df(
    ranking: pd.Series,
    gene_sets=('KEGG_2019_Mouse',),
    organism: str = 'Mouse',
    n_perm: int = 999,
    min_size: int = 15,
    max_size: int = 500,
    threads: int = 4,
    seed: int = 42,
) -> pd.DataFrame:
    """Run GSEA-Prerank, return a DataFrame with columns ['NES','FDR'] (index=pathway).

    `ranking`: pd.Series, index=gene, values=signed statistic. `gene_sets`: iterable
    of library names / .gmt paths / dicts, merged into one gmt dict.
    """
    if isinstance(gene_sets, (list, tuple)):
        lib_dicts = []
        for lib in gene_sets:
            if isinstance(lib, str) and lib.lower().endswith('.gmt'):
                lib_dicts.append(gp.parser.gsea_gmt_parser(lib))
            elif isinstance(lib, str):
                lib_dicts.append(_fetch_library(lib, organism))
            elif isinstance(lib, dict):
                lib_dicts.append(lib)
            else:
                raise ValueError(f'Cannot handle gene_sets entry {lib}')
        gmt = reduce(lambda a, b: {**a, **b}, lib_dicts)
    elif isinstance(gene_sets, dict):
        gmt = gene_sets
    else:
        gmt = _fetch_library(gene_sets, organism)

    res = gp.prerank(
        rnk=ranking, gene_sets=gmt, permutation_num=n_perm,
        min_size=min_size, max_size=max_size, threads=threads,
        seed=seed, outdir=None, no_plot=True, verbose=False,
    ).res2d

    cols_lower = {c.lower(): c for c in res.columns}
    term_col = cols_lower.get('term')
    if term_col in res.columns:
        res = res.set_index(term_col)
    res = res.rename_axis('Term')
    nes_col = cols_lower.get('nes')
    fdr_col = None
    for cand in ('fdr', 'fdr q-val', 'fdr q value', 'fdr qvalue', 'fdr q-val.'):
        if cand in cols_lower:
            fdr_col = cols_lower[cand]
            break
    if fdr_col is None:
        raise ValueError('Cannot find FDR column in gseapy result')

    return res[[nes_col, fdr_col]].rename(columns={nes_col: 'NES', fdr_col: 'FDR'}).astype(float)


def nes_correlation_pipeline(
    stat_tables: Dict[str, pd.DataFrame],
    score_col: str = 'Coef_interaction',
    gene_sets=('MSigDB_Hallmark_2020', 'Reactome_2022'),
    threads: int = 4,
    corr_method: str = 'spearman',
    cmap: str = 'vlag',
) -> Dict[str, pd.DataFrame]:
    """Build a pathway NES matrix (one column per tissue) via `gsea_prerank_df`,
    compute the inter-tissue NES correlation, and draw a clustered heatmap.

    NOTE(bug): the source (`stage_dstream_loop.py`) calls an undefined function
    `_gsea_prerank(rank_vec, gene_sets=gene_sets, processes=processes)` here —
    there is no `_gsea_prerank` anywhere in that file, only `gsea_prerank_df`
    (different signature: `threads`, not `processes`). Calling this function in
    the original codebase would raise `NameError` — it is confirmed dead/unreachable
    as originally written, not verified to have ever produced a real figure this
    way. This reimplementation calls `gsea_prerank_df` directly (the evident intent)
    so the function is actually runnable; flagged here rather than silently passed
    through as broken. Confirm with the author whether this was ever actually used
    to produce a paper figure before trusting its output.
    """
    nes_columns = []
    for tissue, df in stat_tables.items():
        if {'Gene', score_col}.difference(df.columns):
            raise ValueError(f"{tissue}: missing 'Gene' or '{score_col}'")
        rank_vec = df.dropna(subset=[score_col]).set_index('Gene')[score_col].sort_values(ascending=False)
        nes = gsea_prerank_df(rank_vec, gene_sets=gene_sets, threads=threads)['NES']
        nes.name = tissue
        nes_columns.append(nes)

    nes_mat = pd.concat(nes_columns, axis=1)
    corr = nes_mat.T.corr(method=corr_method)

    sns.clustermap(corr, cmap=cmap, center=0, linewidths=.3, figsize=(6, 6),
                    cbar_kws=dict(label=f'{corr_method.capitalize()}'))
    plt.suptitle('Inter-tissue similarity based on pathway NES', y=1.03)
    plt.show()

    return dict(NES_matrix=nes_mat, Corr_matrix=corr)
