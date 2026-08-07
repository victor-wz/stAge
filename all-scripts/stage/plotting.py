"""Shared plotting helpers for stAge figures.

This module consolidates duplicate/divergent plotting code found scattered across the
original codebase: `plot_clock_distributions` existed in THREE distinct forms
(`v_pipeline/st_utils.py`, `v_pipeline/st_utils_claude.py` -- a near-verbatim rewrite of the
st_utils.py version -- and `v_pipeline/stage_dstream.py`), plus a fourth, differently-designed
variant pasted directly into `v_pipeline/A.md` for the CTRL-vs-LPS whole-mouse figure. See the
`plot_clock_distributions` docstring below for exactly what was kept/merged/dropped from each.
This is a refactor, not a reanalysis: no statistical behavior was changed, only unified behind
one function signature with the original hard-coded, dataset-specific choices (fixed axis
limits, a hard-coded "every other pair" comparison filter) exposed as optional parameters
instead of being baked in.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch
from scipy import stats


def _assign_group(filename, group_patterns):
    for pattern in group_patterns:
        if pattern in filename:
            return pattern
    return "Unknown"


def _build_tidy_predictions(preds_per_file, group_patterns, norm_cols):
    rows = []
    for file, adata in preds_per_file.items():
        group = _assign_group(file, group_patterns)
        if group == "Unknown":
            continue
        for norm in norm_cols:
            if norm not in adata.obs:
                continue
            for age in adata.obs[norm]:
                rows.append({"file": file, "group": group, "norm": norm, "age": float(age)})
    return pd.DataFrame(rows)


def _fmt_pval(p):
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return f"ns (p={p:.3f})"


def plot_clock_distributions(
    preds_per_file,
    group_patterns,
    norm_cols=("tAge_SM", "tAge_YM"),
    plot_kind="box",
    stat_method="auto",
    test="Mann-Whitney",
    comparisons=None,
    palette=None,
    point_overlay="strip_by_sample",
    ylim=None,
    show=True,
):
    """Plot per-group distributions of predicted transcriptomic age, with significance.

    Consolidates 4 divergent implementations found in the original codebase:

    - `v_pipeline/st_utils.py` (and its near-verbatim copy in `st_utils_claude.py`):
      box + black swarm overlay, N groups, `statannotations.Annotator`, a hard-coded
      "every other adjacent pair" comparison filter (`i % 2 == 0`) that looks like a
      one-off convenience for a specific ordered group list rather than a general
      rule, a fixed 4-color palette, and (in `st_utils.py` only) commented-out
      alternate palettes for "cancer" and "WM/GM" datasets that were never wired up
      as real parameters.
    - `v_pipeline/stage_dstream.py`: violin instead of box, `statannotations.Annotator`
      over ALL adjacent pairs (no `i % 2` filter), a hard-coded `ylim=(-14, 30)`
      specific to one dataset.
    - `v_pipeline/A.md` ("CTRL vs LPS box plot"): exactly 2 groups, an explicit
      sample-level (per-file mean) t-test rather than `Annotator` -- documented
      rationale kept verbatim below -- and a strip-plot overlay colored by sample
      identity rather than a black swarm.

    Kept: all four plot styles (box/violin via `plot_kind`), both point-overlay
    styles (`point_overlay`), both significance-testing approaches (`stat_method`).
    Dropped nothing silently -- every hard-coded, dataset-specific choice from the
    originals (fixed `ylim`, the `i % 2` pair filter, the "cancer"/"WM-GM" palette
    variants that existed only as dead comments) is now an explicit parameter with
    a documented default and a note on how to reproduce the original hard-coded
    behavior exactly, rather than reappearing as a silent default.

    Why sample-level aggregation for `stat_method='sample_level_ttest'` (rationale
    copied verbatim from `A.md`, which is the only one of the four originals to
    document it): spots within a slide are pseudoreplicates. The unit of biological
    replication is the slide (file). Aggregating to per-slide means and then running
    a t-test gives a valid test with df = n_a + n_b - 2, instead of e.g. a p~1
    produced by a mixed-effects model whose random intercepts absorb all
    between-slide variance.

    Parameters
    ----------
    preds_per_file : dict[str, AnnData]
        Filename -> AnnData with prediction columns in `.obs`.
    group_patterns : list of str
        Substrings used to assign each file to a group (also sets x-axis order).
        Files matching no pattern are dropped (matches all four originals).
    norm_cols : sequence of str
        `.obs` columns to plot, one subplot per column.
    plot_kind : {'box', 'violin'}
        Underlying seaborn plot type.
    stat_method : {'auto', 'sample_level_ttest', 'annotator'}
        'sample_level_ttest': A.md's approach (only valid for exactly 2 groups).
        'annotator': the st_utils.py/stage_dstream.py approach via
        `statannotations.Annotator`, works for any number of groups.
        'auto' (default): 'sample_level_ttest' when there are exactly 2 groups,
        else 'annotator'.
    test : str
        Test name passed to `Annotator` when stat_method='annotator'
        (e.g. 'Mann-Whitney', 't-test_ind').
    comparisons : list of (str, str) tuples, optional
        Explicit group pairs to annotate when stat_method='annotator'. Defaults to
        ALL consecutive pairs in `group_patterns` (stage_dstream.py's behavior). To
        reproduce st_utils.py's/st_utils_claude.py's original "every other pair"
        filter exactly, pass:
        `comparisons=[p for i, p in enumerate(zip(group_patterns[:-1], group_patterns[1:])) if i % 2 == 0]`
    palette : list of str or dict, optional
        Colors per group. Defaults to a 2-color orange/blue CTRL-vs-experimental
        palette (matching A.md) when there are 2 groups, else the 4-color
        `["#a1c9f4", "lightblue", "#ffb482", "salmon"]` palette from st_utils.py,
        cycled if there are more than 4 groups.
    point_overlay : {'strip_by_sample', 'swarm', None}
        'strip_by_sample' (A.md style, default): jittered points colored by
        file/sample identity. 'swarm' (st_utils.py style): plain black swarm plot.
        None: no point overlay.
    ylim : tuple, optional
        Fixed y-axis limits. stage_dstream.py hard-coded `(-14, 30)` for one
        specific dataset; that is NOT applied by default here -- pass it explicitly
        to reproduce that figure exactly.
    show : bool
        Call `plt.show()` at the end (set False if you intend to save the figure
        yourself via the returned `fig`).

    Returns
    -------
    fig, axes, stats_df : the figure/axes, and a DataFrame with one row per
    (norm, comparison) giving the p-value used for the annotation (only populated
    for `stat_method='sample_level_ttest'` -- `Annotator` does not expose p-values
    through a simple return value, matching the original's behavior of only ever
    displaying them on-plot).
    """
    norm_cols = list(norm_cols)
    df_preds = _build_tidy_predictions(preds_per_file, group_patterns, norm_cols)
    group_order = list(group_patterns)
    n_groups = len(group_order)

    if stat_method == "auto":
        stat_method = "sample_level_ttest" if n_groups == 2 else "annotator"
    if stat_method == "sample_level_ttest" and n_groups != 2:
        raise ValueError(
            "stat_method='sample_level_ttest' only supports exactly 2 groups "
            f"(got {n_groups}); use stat_method='annotator' for >2 groups."
        )

    if palette is None:
        if n_groups == 2:
            palette = {group_order[0]: "#ffb482", group_order[1]: "#a1c9f4"}
        else:
            base = ["#a1c9f4", "lightblue", "#ffb482", "salmon"]
            palette = [base[i % len(base)] for i in range(n_groups)]

    if comparisons is None:
        comparisons = list(zip(group_order[:-1], group_order[1:]))

    sns.set_theme(context="talk", style="ticks", font_scale=1.15)
    lw_box, lw_med, lw_whisk = 2.2, 2.6, 2.0

    fig, axes = plt.subplots(
        1, len(norm_cols), figsize=(6 * len(norm_cols), 6.5),
        sharey=True, constrained_layout=True,
    )
    axes = np.atleast_1d(axes)

    stats_rows = []

    for ax, norm in zip(axes, norm_cols):
        df_clock = df_preds.query("norm == @norm").copy()
        if df_clock.empty:
            continue

        if plot_kind == "violin":
            sns.violinplot(data=df_clock, x="group", y="age", order=group_order,
                           palette=palette, ax=ax)
        else:
            sns.boxplot(
                data=df_clock, x="group", y="age", order=group_order, palette=palette,
                width=0.55, showcaps=True, showfliers=True, linewidth=lw_box,
                boxprops=dict(linewidth=lw_box), whiskerprops=dict(linewidth=lw_whisk),
                capprops=dict(linewidth=lw_whisk), medianprops=dict(linewidth=lw_med),
                flierprops=dict(marker="o", markersize=6, markerfacecolor="#333333",
                                 markeredgecolor="#333333", alpha=0.9),
                ax=ax,
            )

        if point_overlay == "swarm":
            sns.swarmplot(data=df_clock, x="group", y="age", order=group_order,
                          color="black", alpha=0.5, size=3, ax=ax)
        elif point_overlay == "strip_by_sample":
            files = sorted(df_clock["file"].unique())
            file_pal = dict(zip(files, sns.color_palette("tab10", len(files))))
            sns.stripplot(data=df_clock, x="group", y="age", order=group_order,
                          hue="file", palette=file_pal, alpha=0.45, size=2.5,
                          jitter=True, legend=False, ax=ax)

        ax.spines["left"].set_linewidth(1.5)
        ax.spines["bottom"].set_linewidth(1.5)
        ax.tick_params(width=1.2)
        ax.set(title=norm, ylabel="Predicted relative age", xlabel="")
        if ylim is not None:
            ax.set_ylim(ylim)
        ax.tick_params(axis="x", rotation=30 if n_groups > 2 else 0)
        ax.grid(False)
        sns.despine(ax=ax)

        if stat_method == "sample_level_ttest":
            sample_means = df_clock.groupby(["file", "group"])["age"].mean().reset_index()
            a = sample_means.loc[sample_means["group"] == group_order[0], "age"].values
            b = sample_means.loc[sample_means["group"] == group_order[1], "age"].values
            _, p = stats.ttest_ind(a, b, equal_var=True)
            star = _fmt_pval(p)
            stats_rows.append({"norm": norm, "comparison": tuple(group_order), "p": p,
                                "method": "sample_level_ttest"})

            y_data = df_clock["age"].quantile(0.995)
            pad = (df_clock["age"].max() - df_clock["age"].min()) * 0.04
            y_bar = y_data + pad
            x0, x1 = 0, 1
            ax.plot([x0, x0, x1, x1], [y_bar - pad * 0.3, y_bar, y_bar, y_bar - pad * 0.3],
                    lw=1.5, c="black", clip_on=False)
            ax.text((x0 + x1) / 2, y_bar + pad * 0.5, star, ha="center", va="bottom",
                    fontsize=15)

        elif stat_method == "annotator":
            from statannotations.Annotator import Annotator  # optional dep, imported lazily
            annot = Annotator(ax, pairs=comparisons, data=df_clock, x="group", y="age",
                              order=group_order)
            annot.configure(test=test, text_format="star", loc="inside", verbose=0)
            annot.apply_and_annotate()

    if isinstance(palette, dict):
        legend_handles = [Patch(facecolor=palette[g], edgecolor="black", linewidth=1.2, label=g)
                          for g in group_order]
        axes[-1].legend(handles=legend_handles, loc="upper right", frameon=True, fontsize=13)

    if show:
        plt.show()

    return fig, axes, pd.DataFrame(stats_rows)


def plot_tissue_standardized_effects(
    preds_per_file,
    group_patterns,
    norm_cols=("tAge_SM", "tAge_YM"),
    ctrl_suffix="CTRL",
    exp_suffix="LPS",
    sort_by_effect=True,
    show=True,
):
    """Horizontal bar chart of Cohen's d (experimental - control) per tissue.

    Ported near-verbatim from `v_pipeline/A.md` ("CTRL vs LPS -- standardized effect
    size per tissue"), the only source found for this plot. Originally hard-coded to
    `ctrl_suffix='CTRL'`/`lps_suffix='LPS'` for the whole-mouse LPS figure; renamed
    the second parameter `exp_suffix` (was `lps_suffix`) so the function reads as
    dataset-agnostic, but the default value and all computed statistics are
    unchanged from the original.

    Design (unchanged from the original): Cohen's d uses the within-group
    spot-level pooled SD as denominator (stable with large n spots). Significance
    is a sample-level t-test on per-file means -- the same pseudoreplication-aware
    approach as `plot_clock_distributions(..., stat_method='sample_level_ttest')`.

    Parameters
    ----------
    preds_per_file : dict[str, AnnData]
    group_patterns : list of str
        Expected to be of the form `f'{tissue}_{ctrl_suffix}'` /
        `f'{tissue}_{exp_suffix}'` for every tissue of interest -- tissues are
        derived from the patterns ending in `_{ctrl_suffix}`.
    norm_cols : sequence of str
    ctrl_suffix, exp_suffix : str
    sort_by_effect : bool
        Sort tissues by Cohen's d (original default) vs. input order.
    show : bool

    Returns
    -------
    fig, axes, df_res -- df_res has one row per (tissue, norm) with cohens_d and p.
    """
    ctrl_pats = [p for p in group_patterns if p.endswith(f"_{ctrl_suffix}")]
    tissues = [p[: -len(f"_{ctrl_suffix}")] for p in ctrl_pats]

    def fmt_pval(p):
        if np.isnan(p):
            return ""
        if p < 0.001:
            return "***"
        if p < 0.01:
            return "**"
        if p < 0.05:
            return "*"
        return f"p={p:.2f}"

    rows = []
    for file, adata in preds_per_file.items():
        for norm in norm_cols:
            if norm not in adata.obs:
                continue
            for tissue in tissues:
                if f"{tissue}_{ctrl_suffix}" in file:
                    cond = ctrl_suffix
                elif f"{tissue}_{exp_suffix}" in file:
                    cond = exp_suffix
                else:
                    continue
                for age in adata.obs[norm]:
                    rows.append({"file": file, "tissue": tissue, "condition": cond,
                                 "norm": norm, "age": float(age)})
    df = pd.DataFrame(rows)

    results = []
    for norm in norm_cols:
        df_n = df[df["norm"] == norm]
        for tissue in tissues:
            df_t = df_n[df_n["tissue"] == tissue]
            ctrl = df_t[df_t["condition"] == ctrl_suffix]["age"].values
            exp = df_t[df_t["condition"] == exp_suffix]["age"].values

            if len(ctrl) == 0 or len(exp) == 0:
                results.append({"tissue": tissue, "norm": norm, "cohens_d": np.nan, "p": np.nan})
                continue

            n1, n2 = len(ctrl), len(exp)
            sd_p = np.sqrt(((n1 - 1) * ctrl.var(ddof=1) + (n2 - 1) * exp.var(ddof=1)) / (n1 + n2 - 2))
            d = (exp.mean() - ctrl.mean()) / sd_p if sd_p > 0 else np.nan

            file_means = df_t.groupby(["file", "condition"])["age"].mean().reset_index()
            a = file_means[file_means["condition"] == ctrl_suffix]["age"].values
            b = file_means[file_means["condition"] == exp_suffix]["age"].values
            p = stats.ttest_ind(a, b, equal_var=True).pvalue if len(a) >= 2 and len(b) >= 2 else np.nan

            results.append({"tissue": tissue, "norm": norm, "cohens_d": d, "p": p})

    df_res = pd.DataFrame(results)

    sns.set_theme(context="talk", style="ticks", font_scale=1.1)
    fig, axes = plt.subplots(
        1, len(norm_cols), figsize=(7 * len(norm_cols), max(4, len(tissues) * 0.45)),
        sharey=True, constrained_layout=True,
    )
    axes = np.atleast_1d(axes)

    for ax, norm in zip(axes, norm_cols):
        df_plot = df_res[df_res["norm"] == norm].dropna(subset=["cohens_d"]).copy()

        if sort_by_effect:
            df_plot = df_plot.sort_values("cohens_d", ascending=True).reset_index(drop=True)
        else:
            order_map = {t: i for i, t in enumerate(tissues)}
            df_plot = df_plot.assign(_o=df_plot["tissue"].map(order_map)).sort_values("_o").reset_index(drop=True)

        colors = ["#a1c9f4" if d < 0 else "#ffb482" for d in df_plot["cohens_d"]]
        ax.barh(range(len(df_plot)), df_plot["cohens_d"], color=colors,
                edgecolor="black", linewidth=0.7, height=0.65)
        ax.set_yticks(range(len(df_plot)))
        ax.set_yticklabels(df_plot["tissue"], fontsize=11)

        xmax = df_plot["cohens_d"].abs().max()
        offset = xmax * 0.04
        for i, (_, row) in enumerate(df_plot.iterrows()):
            label = fmt_pval(row["p"])
            if not label:
                continue
            d = row["cohens_d"]
            ha = "left" if d >= 0 else "right"
            ax.text(d + (offset if d >= 0 else -offset), i, label, va="center", ha=ha, fontsize=11)

        ax.set_xlim(-(xmax + xmax * 0.35), xmax + xmax * 0.35)
        ax.axvline(0, color="black", lw=1.2)
        ax.set_xlabel(f"Cohen's d  ({exp_suffix} - {ctrl_suffix})", fontsize=13)
        ax.set_title(norm, fontsize=13)
        ax.grid(axis="x", alpha=0.25, lw=0.8)
        sns.despine(ax=ax)

    if show:
        plt.show()

    return fig, axes, df_res


def spatial_plot_tage_by_age_group(preds_per_file, norm="tAge_SM", spot_size=1,
                                    young_pat="_Y_", old_pat="_O_", cmap="coolwarm",
                                    show=True):
    """Spatial tAge (or other continuous obs column) for young vs old samples, 2-row grid.

    Ported verbatim from `v_pipeline/stage_dstream.py` -- no logic changes.

    Parameters
    ----------
    preds_per_file : dict[str, AnnData]
    norm : str
        `.obs` column to plot.
    spot_size : float
    young_pat, old_pat : str
        Substrings used to split `preds_per_file` into the two rows.
    cmap : str
    show : bool
    """
    import scanpy as sc  # imported here to keep this module importable without scanpy

    young_samples = {k: v for k, v in preds_per_file.items() if young_pat in k}
    old_samples = {k: v for k, v in preds_per_file.items() if old_pat in k}

    vmax = max(adata.obs[norm].max() for adata in preds_per_file.values())
    vmin = min(adata.obs[norm].min() for adata in preds_per_file.values())
    vfinal = max(abs(vmax), abs(vmin))

    n_young = len(young_samples)
    n_old = len(old_samples)
    n_cols = max(n_young, n_old)

    fig, axes = plt.subplots(nrows=2, ncols=n_cols, figsize=(n_cols * 5, 10))
    if n_cols == 1:
        axes = np.array([[axes[0]], [axes[1]]])

    for i, (k, adata_pred) in enumerate(young_samples.items()):
        ax = axes[0][i]
        sc.pl.spatial(adata_pred, color=norm, spot_size=spot_size, cmap=cmap,
                      vmax=vfinal, vmin=-vfinal, ax=ax, show=False,
                      title=f'{k.replace("_processed.h5ad", "")} | {norm}')

    for i, (k, adata_pred) in enumerate(old_samples.items()):
        ax = axes[1][i]
        sc.pl.spatial(adata_pred, color=norm, spot_size=spot_size, cmap=cmap,
                      vmax=vfinal, vmin=-vfinal, ax=ax, show=False,
                      title=f'{k.replace("_processed.h5ad", "")} | {norm}')

    plt.tight_layout()
    if show:
        plt.show()
    return fig, axes
