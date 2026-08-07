"""Marker-gene brain-region annotation (mouse), and human WM/GM annotation (GMM +
spatial label smoothing).

Maps to: Marker-gene brain region annotation (mouse: Brain 2g/3g/25); human WM/GM
annotation (feeds module-clock figures, see analyses/*_module_clocks.py).

Source: v_pipeline/celltype_hotspot_tAge.ipynb's Region Analysis section (marker
panel `REGION_MARKERS`, Leiden clustering parameters) cross-checked against its
reuse in spatial_tage_beyond_composition.ipynb Panel C. celltype_hotspot_tAge.ipynb
contains 4 near-duplicate full re-executions of this section (iterative-editing
residue) — parameters were confirmed identical across the final iteration and the
verbatim reuse in spatial_tage_beyond_composition.ipynb (n_top_genes=2000, PCA
n_comps=30, neighbors n_neighbors=15, Leiden resolution=0.8), so there is no
drift to reconcile.

Per-cluster mean expression uses a sparse cluster-indicator matmul rather than
`.toarray()` on the full spots x genes matrix — this is the memory-safety fix
documented in FigureS11.md (the dense `.toarray()` version OOM'd on Brain 25's
~85k spots under this environment's 200GB cgroup limit; both approaches compute
the same algorithm and were confirmed to produce the same result, only one
doesn't crash). No dense fallback is provided since there's no reason to carry
the broken variant forward.
"""

from __future__ import annotations

import gc

import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
from scipy.sparse import issparse

# 7 canonical mouse-brain regions (Wang et al. mouse brain atlas marker conventions).
# Reused verbatim across celltype_hotspot_tAge.ipynb and spatial_tage_beyond_composition.ipynb.
REGION_MARKERS = {
    "Isocortex": ["Lamp5", "Trbc2", "Gm11549", "Ovol2", "Ntn5", "Tcap", "Wnt10a", "Myl4", "Satb2", "Arc", "Stx1a"],
    "Hypothalamus": ["Hcrt", "Pmch", "Th", "Lhx1os", "Pitx2", "Slc18a2", "Dlk1", "BC039966", "Gal", "Gm5741"],
    "Hippocampus": ["Cabp7", "Lct", "Dsp", "C1ql2", "Crf1", "Spink8", "Fibcd1", "Lefty1", "Tnfrsf25", "Wipf3"],
    "OLF_CTX": ["Ccn3", "Lypd1", "Col23a1", "Apaf1", "Kcng1", "Bmp3", "Moxd1", "Syt17", "Nptxr", "Atp2b4"],
    "Fiber_tracts": ["Opalin", "Fa2h", "Mal", "Plp1", "Mog", "Anln", "Mobp", "Ermn", "Ppp1r14a", "Enpp6"],
    "Thalamus": ["Prkcd", "Abhd12b", "Tnnt1", "Plekhg1", "Shox2", "Gbx2", "Rgs16", "Ramp3", "Slitrk6", "Synpo2"],
    "Striatum_CNu": ["Gpr88", "Adora2a", "Rgs9", "Penk", "Ppp1r1b", "Drd2", "Drd1", "Tac1", "Pde10a", "Syndig1l"],
}

UNKNOWN_Z_THRESHOLD = 0.5


def annotate_brain_regions(
    raw_adata,
    batch_key: str = 'sample',
    n_top_genes: int = 2000,
    n_pcs: int = 30,
    n_neighbors: int = 15,
    leiden_resolution: float = 0.8,
    region_markers: dict = REGION_MARKERS,
    unknown_z_threshold: float = UNKNOWN_Z_THRESHOLD,
) -> pd.Series:
    """Marker-gene-scored Leiden-clustering anatomical region annotation.

    normalize_total -> log1p -> batch-aware HVG -> scale -> PCA -> neighbors ->
    Leiden -> each cluster labeled by its highest mean z-scored marker-gene-panel
    score across `region_markers`; 'Unknown' if the best z-score is below
    `unknown_z_threshold`.

    `raw_adata` should be raw (unnormalized) counts, spots/cells x genes, with
    `batch_key` in `.obs` if it spans multiple samples. Returns a per-obs Series
    of region labels aligned to `raw_adata.obs_names` (does not modify `raw_adata`).

    Memory note: per-cluster mean expression is computed via a sparse
    cluster-indicator matmul, never densifying the full expression matrix — this
    matters for large concatenated cohorts (see module docstring).
    """
    norm_adata = raw_adata.copy()
    sc.pp.normalize_total(norm_adata, target_sum=1e4)
    sc.pp.log1p(norm_adata)
    sc.pp.highly_variable_genes(norm_adata, n_top_genes=n_top_genes, batch_key=batch_key)
    hvg_mask = norm_adata.var['highly_variable'].values
    full_var_names = pd.Index(norm_adata.var_names)
    lognorm_X = norm_adata.X

    hvg_adata = norm_adata[:, hvg_mask].copy()
    sc.pp.scale(hvg_adata, max_value=10)
    sc.tl.pca(hvg_adata, n_comps=n_pcs)
    sc.pp.neighbors(hvg_adata, n_neighbors=n_neighbors, n_pcs=n_pcs, key_added='graph')
    sc.tl.leiden(hvg_adata, resolution=leiden_resolution, key_added='cluster',
                 adjacency=hvg_adata.obsp['graph_connectivities'])
    cluster_labels = hvg_adata.obs['cluster'].values.copy()
    del hvg_adata
    gc.collect()

    unique_clusters = sorted(pd.unique(cluster_labels), key=lambda x: int(x))
    n_cells = len(cluster_labels)
    cl_idx = {cl: i for i, cl in enumerate(unique_clusters)}
    rows_ = np.array([cl_idx[c] for c in cluster_labels])
    indicator = sp.csr_matrix(
        (np.ones(n_cells), (rows_, np.arange(n_cells))),
        shape=(len(unique_clusters), n_cells),
    )
    cluster_sizes = np.asarray(indicator.sum(axis=1)).ravel()
    cluster_sizes[cluster_sizes == 0] = 1.0
    if not issparse(lognorm_X):
        lognorm_X = sp.csr_matrix(lognorm_X)
    cluster_sums = indicator @ lognorm_X
    mean_expr_df = pd.DataFrame(
        np.asarray(cluster_sums.todense()) / cluster_sizes[:, None],
        index=unique_clusters, columns=full_var_names,
    )
    del norm_adata, lognorm_X, cluster_sums, indicator
    gc.collect()

    col_mean = mean_expr_df.mean(axis=0)
    col_std = mean_expr_df.std(axis=0).replace(0, np.nan)
    mean_expr_z = ((mean_expr_df - col_mean) / col_std).fillna(0.0)

    region_scores = {}
    for region, markers in region_markers.items():
        present = [g for g in markers if g in mean_expr_z.columns]
        region_scores[region] = (
            mean_expr_z[present].mean(axis=1) if present else pd.Series(0.0, index=unique_clusters)
        )
    score_df = pd.DataFrame(region_scores, index=unique_clusters)
    best_region = score_df.idxmax(axis=1)
    best_score = score_df.max(axis=1)
    best_region[best_score < unknown_z_threshold] = 'Unknown'
    cluster_to_region = best_region.to_dict()

    return pd.Series(cluster_labels, index=raw_adata.obs_names).map(cluster_to_region)



# ---------------------------------------------------------------------------
# Human white-matter / grey-matter annotation (GMM + spatial label smoothing).
#
# CORRECTION TO PHASE 1/EARLY PHASE 2: this was originally stubbed as a confirmed
# gap ("no GMM code located anywhere in the audited tree"). On a closer read of
# v_pipeline/stage_modules_pred.ipynb (~lines 500-690 of the nbconvert'd script,
# cell "Functions to Annotate HUMAN brain into Grey and White matter regions
# 'WM_label_smoothed'"), the real implementation was found — it was missed by the
# earlier pass because it lives in the module-clocks notebook, not in
# celltype_hotspot_tAge.ipynb where it was expected. Ported verbatim below. This
# is the code that produces the `WM_label_smoothed` column consumed downstream by
# stage_res_pred.ipynb (~lines 886-1129) and stage_modules_pred.ipynb's own
# WM/GM module-clock heatmap section.
# ---------------------------------------------------------------------------

import re as _re
from sklearn.mixture import GaussianMixture

try:
    # dask/dask.dataframe must be imported before squidpy on some dask/dask-expr
    # version pairings -- see stage/metapixels.py's module docstring for the full
    # explanation (same fix applied here since this file also imports squidpy).
    import dask
    import dask.dataframe as dd
    import squidpy as sq
except ImportError:  # pragma: no cover - squidpy is an optional dep for this function only
    sq = None

# Canonical human WM/GM marker panels (v_pipeline/stage_modules_pred.ipynb).
WM_GENES_CANON = ["MBP", "PLP1", "MOG", "MAG", "MAL", "OPALIN", "CLDN11", "CNP", "OLIG1", "OLIG2", "SOX10", "TF"]
GM_GENES_CANON = ["SLC17A7", "SLC17A6", "SLC6A1", "GAD1", "GAD2", "RBFOX3", "SNAP25", "SYT1", "CAMK2A", "MAP2"]

# Common aliases to improve matching.
WM_GM_ALIASES = {
    "RBFOX3": ["NEUN"],       # Neuronal marker
    "SLC17A7": ["VGLUT1"],    # same protein
    "BCL11B": ["CTIP2"],      # layer 5 TF
}

_CANDIDATE_SYMBOL_COLS = [
    "gene_symbols", "gene_symbol", "gene", "symbol", "Gene", "Symbol",
    "gene_name", "feature_name", "features", "name", "SYMBOL",
]


def _strip_version(x):
    if x is None:
        return ""
    return str(x).split(".")[0]


def _norm_token(x):
    return _strip_version(x).strip().upper()


def _pick_symbol_series(adata):
    for col in _CANDIDATE_SYMBOL_COLS:
        if col in adata.var and adata.var[col].notna().any():
            return adata.var[col].astype(str)
    return pd.Series(adata.var_names, index=adata.var_names)


def _make_symbol_index(adata):
    sym = _pick_symbol_series(adata).fillna("").astype(str)
    upper = sym.map(_norm_token)
    title = sym.str.title()
    vn = pd.Series(adata.var_names, index=adata.var_names)

    idx_upper = {}
    for orig, u in zip(sym.index, upper):
        idx_upper.setdefault(u, set()).add(orig)
    for orig in vn.index:
        u = _norm_token(orig)
        idx_upper.setdefault(u, set()).add(orig)

    return {
        "sym_series": sym,
        "upper_index": idx_upper,
        "present_upper": set(idx_upper.keys()),
        "present_title": set(title.unique()),
    }


def _expand_wm_gm_aliases(genes):
    out = set()
    for g in genes:
        out.add(g)
        for k, alist in WM_GM_ALIASES.items():
            if g == k:
                out.update(alist)
        for canon, alist in WM_GM_ALIASES.items():
            if g in alist:
                out.add(canon)
    return list(out)


def match_markers(adata, genes, assume_species="auto"):
    """Return var_names keys matching `genes`, robust to symbol col, case, aliases,
    and Ensembl versions. `assume_species`: 'auto' | 'mouse' | 'human'."""
    genes = _expand_wm_gm_aliases([_norm_token(g) for g in genes])
    idx = _make_symbol_index(adata)
    hits = set()

    var_names_upper_frac = np.mean([v.isupper() for v in map(_norm_token, adata.var_names)])
    mouse_like = var_names_upper_frac < 0.3 if assume_species == "auto" else (assume_species == "mouse")

    for g in genes:
        if g in idx["present_upper"]:
            hits.update(idx["upper_index"][g])
            continue
        if mouse_like:
            g_title = g.title()
            if g_title in idx["present_title"]:
                sym = _pick_symbol_series(adata)
                hits.update(sym.index[sym == g_title].tolist())

    return list(hits)


def score_wm_gm(adata, wm_genes=WM_GENES_CANON, gm_genes=GM_GENES_CANON, layer=None, min_hits=3, verbose=True):
    """`sc.tl.score_genes` on matched WM/GM marker panels; writes `WM_score`,
    `GM_score`, `WM_GM_delta` = WM_score - GM_score into `adata.obs`."""
    import scanpy as sc

    wm_hits = match_markers(adata, wm_genes)
    gm_hits = match_markers(adata, gm_genes)

    if verbose:
        print(f"[{adata.uns.get('sample_id', 'section')}] WM hits: {len(wm_hits)}, GM hits: {len(gm_hits)}")
        if len(wm_hits) < min_hits:
            print(f"  Warning: low WM marker matches ({len(wm_hits)}).")
        if len(gm_hits) < min_hits:
            print(f"  Warning: low GM marker matches ({len(gm_hits)}).")

    if len(wm_hits) < min_hits or len(gm_hits) < min_hits:
        raise ValueError("Too few matched markers for WM/GM scoring in this section.")

    sc.tl.score_genes(adata, gene_list=wm_hits, score_name="WM_score", use_raw=False, layer=layer)
    sc.tl.score_genes(adata, gene_list=gm_hits, score_name="GM_score", use_raw=False, layer=layer)
    adata.obs["WM_GM_delta"] = adata.obs["WM_score"] - adata.obs["GM_score"]
    return adata


def otsu_like_threshold(x):
    """Two-component GMM fit to a 1-D score; threshold = mean of the two component means."""
    x = np.asarray(x).reshape(-1, 1)
    gmm = GaussianMixture(n_components=2, covariance_type="full", random_state=0).fit(x)
    mu = np.sort(gmm.means_.flatten())
    return float(np.mean(mu))


def spatial_smooth_labels(adata, label_key="WM_label", n_rings=1, beta=0.8):
    """Diffuse a binary WM/GM label across the spatial neighbor graph (`n_rings` steps
    of weight `beta`), writing `{label_key}_smoothed`. Requires `adata.obsm['spatial']`."""
    if sq is None:
        raise ImportError("squidpy is required for spatial_smooth_labels (sq.gr.spatial_neighbors).")
    if "spatial" not in adata.obsm:
        raise ValueError("No 'spatial' coordinates found in adata.obsm; add XY coords to .obsm['spatial'].")
    sq.gr.spatial_neighbors(adata, coord_type="grid")

    G = adata.obsp["spatial_connectivities"]
    y = adata.obs[label_key].map({"WM": 1, "GM": 0}).astype(int).to_numpy()
    y_soft = y.astype(float).copy()

    for _ in range(n_rings):
        y_nb = G.dot(y_soft) / (G.sum(1).A1 + 1e-9)
        y_soft = (1 - beta) * y_soft + beta * y_nb

    adata.obs[label_key + "_smoothed"] = pd.Series((y_soft >= 0.5)).map({True: "WM", False: "GM"}).values
    return adata


def annotate_wm_gm(adata, layer=None, smooth=True, min_hits=3, verbose=True):
    """Human WM/GM annotation: marker-gene score -> GMM threshold -> optional spatial
    label smoothing. Writes `WM_label` (and `WM_label_smoothed` if `smooth=True`) into
    `adata.obs`. Source: v_pipeline/stage_modules_pred.ipynb (ported verbatim, see
    module docstring)."""
    adata = score_wm_gm(adata, layer=layer, min_hits=min_hits, verbose=verbose)
    thr = otsu_like_threshold(adata.obs["WM_GM_delta"].values)
    adata.obs["WM_label"] = pd.Series((adata.obs["WM_GM_delta"].values >= thr)).map({True: "WM", False: "GM"}).values
    if smooth:
        adata = spatial_smooth_labels(adata, "WM_label")
    return adata
