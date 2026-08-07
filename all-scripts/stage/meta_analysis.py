"""DerSimonian-Laird inverse-variance random-effects meta-analysis across tissues.

Maps to Figs 6c, S7 (Inverse-variance meta-signatures).

Source: v_pipeline/stage_dstream_loop.py. That file defines this exact trio of
functions (`_prep_deg_table`/`random_effects_meta`/`mixed_meta_from_deg_tables`)
TWICE, verbatim (byte-identical bodies, confirmed by diff) — the second definition
silently shadows the first at import time, so it's the one that actually runs today.
This module keeps a single copy; the duplicate in the source is genuinely dead
redundant code, not a behavior change.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm
from statsmodels.stats.multitest import multipletests


def _prep_deg_table(
    df,
    gene_col: str = 'gene',
    beta_col: str = 'Coef_interaction',
    se_col: str = 'SE_interaction',
    p_col: str = 'P_interaction',
) -> pd.DataFrame:
    """Return a DataFrame with index=Gene and columns ['beta','se']. If `se_col` is
    missing (or all-NA), back-calculate SE from beta & p (two-sided normal approx)."""
    req = [gene_col, beta_col]
    missing = [c for c in req if c not in df.columns]
    if missing:
        raise KeyError(f"input DEG table missing columns: {missing}")
    d = df[[gene_col, beta_col]].copy()
    d = d.rename(columns={beta_col: 'beta'})
    if se_col in df.columns and df[se_col].notna().any():
        d['se'] = df[se_col].values
    else:
        if p_col not in df.columns:
            raise KeyError(f"Cannot derive SE: `{se_col}` absent and `{p_col}` not found.")
        p = df[p_col].astype(float).clip(lower=np.finfo(float).tiny)
        z = norm.isf(p / 2.0)
        beta = df[beta_col].astype(float)
        d['se'] = (beta.abs() / z).replace([np.inf, -np.inf], np.nan)
    d = d.set_index(gene_col).dropna(subset=['beta', 'se'])
    d = d.loc[d['se'] > 0]
    return d


def random_effects_meta(long_df: pd.DataFrame) -> pd.DataFrame:
    """DerSimonian-Laird random-effects meta-analysis, one row per gene.

    `long_df` columns: gene, tissue, beta, se. Returns a per-gene table with
    beta, se, z, p, FDR (BH), tau2, I2, k (number of tissues contributing).
    """
    def _meta_one(g):
        yi = g['beta'].to_numpy(float)
        vi = g['se'].to_numpy(float) ** 2
        wi = 1.0 / vi

        fixed_beta = np.sum(wi * yi) / np.sum(wi)
        Q = np.sum(wi * (yi - fixed_beta) ** 2)
        k = yi.size

        if k > 1:
            c = np.sum(wi) - (np.sum(wi ** 2) / np.sum(wi))
            tau2 = max(0.0, (Q - (k - 1.0)) / c) if c > 0 else 0.0
            I2 = max(0.0, (Q - (k - 1.0)) / Q) * 100.0 if Q > (k - 1.0) else 0.0
        else:
            tau2 = 0.0
            I2 = 0.0

        w_star = 1.0 / (vi + tau2)
        beta = np.sum(w_star * yi) / np.sum(w_star)
        se = np.sqrt(1.0 / np.sum(w_star))
        z = beta / se if se > 0 else np.nan
        p = 2 * norm.sf(abs(z)) if np.isfinite(z) else np.nan

        return pd.Series(dict(beta=beta, se=se, z=z, p=p, tau2=tau2, I2=I2, k=k))

    meta = long_df.groupby('gene', sort=False).apply(_meta_one).sort_index()
    meta['FDR'] = multipletests(meta['p'], method='fdr_bh')[1]
    return meta


def mixed_meta_from_deg_tables(
    per_tissue_deg_tables: dict,
    gene_col: str = 'gene',
    beta_col: str = 'logfoldchange',
    se_col: str = 'SE_interaction',
    p_col: str = 'P_interaction',
) -> pd.DataFrame:
    """Convenience wrapper: build the long (gene, tissue, beta, se) table from a
    dict {tissue -> per-tissue DEG DataFrame} and run `random_effects_meta` on it."""
    long_parts = []
    for tissue, df in per_tissue_deg_tables.items():
        sub = _prep_deg_table(df, gene_col, beta_col, se_col, p_col)
        sub = sub.assign(tissue=tissue)
        long_parts.append(sub.reset_index())
    if not long_parts:
        raise ValueError('no valid DEG tables.')
    long_df = pd.concat(long_parts, ignore_index=True)
    return random_effects_meta(long_df)
