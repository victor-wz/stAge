"""Fig 4e-g: Whole-mouse LPS -- CTRL vs LPS tAge distributions and per-tissue effect sizes.

REAL, WIRED UP: the plotting/statistics for both panels are the consolidated
`stage.plotting.plot_clock_distributions` (sample-level t-test, CTRL-vs-LPS box plot) and
`stage.plotting.plot_tissue_standardized_effects` (per-tissue Cohen's d bar chart), both
originally sourced from `v_pipeline/A.md`'s "CTRL vs LPS box plot" / "CTRL vs LPS --
standardized effect size per tissue" paste-in snippets (see `stage/plotting.py`'s docstring
for exactly what was preserved). No statistical logic was changed here.

TODO(needs confirmation): the data-loading step that builds `preds_per_file` for the actual
whole-mouse LPS dataset is NOT wired up -- it was not independently located/confirmed in the
code audit. `v_pipeline/big_stage_pred.ipynb` is the most likely source (its default
`rawdata_dir` was `vvicente/stomics_datasets/notion2/as_h5ad/wholemouse`, and it already runs
the RAM-efficient prediction pipeline end-to-end via `stage.pipeline`), but this was not
confirmed to be CTRL/LPS-labeled data specifically -- confirm with the author which raw
dataset and file-naming convention (e.g. `{Tissue}_CTRL_*.h5ad` / `{Tissue}_LPS_*.h5ad`) this
figure actually uses before relying on `load_wholemouse_lps_predictions` below.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

from stage.plotting import plot_clock_distributions, plot_tissue_standardized_effects

# Tissues used in the standardized-effect-size panel (Fig 4g), per the group-pattern
# convention documented in stage.plotting.plot_tissue_standardized_effects
# ("{tissue}_{ctrl_suffix}" / "{tissue}_{exp_suffix}").
WHOLEMOUSE_TISSUES = [
    'Bone Marrow', 'Brain', 'Brown Fat', 'Colon', 'Heart', 'Kidney', 'Liver', 'Lung',
    'Lymph Node', 'Muscle', 'Other', 'Pancreas', 'Skin', 'Small Intestine',
    'Spleen', 'Stomach', 'Thymus',
]


def load_wholemouse_lps_predictions(pred_dir: str | Path) -> Dict[str, "AnnData"]:
    """TODO(needs confirmation): build {filename: AnnData} for the whole-mouse CTRL/LPS cohort.

    Not implemented -- see module docstring. Once the correct raw dataset/prediction directory
    is confirmed, this should call `stage.pipeline.preprocess_and_predict_from_disk` (or
    whichever pipeline entry point was actually used for this dataset) and return one AnnData
    per file with `tAge_SM`/`tAge_YM` populated in `.obs`, keyed by a filename that encodes
    tissue + CTRL/LPS condition (e.g. `Liver_CTRL_1.h5ad`), matching the `group_patterns`
    convention both plotting functions below expect.
    """
    raise NotImplementedError(
        "Whole-mouse LPS prediction loading is unconfirmed -- see this function's docstring "
        "and stAge-release/INVENTORY.md gap list item 4."
    )


def run_fig4e_g(preds_per_file: Dict[str, "AnnData"], save_dir: str | Path = "."):
    """Fig 4e: CTRL vs LPS box plot (sample-level t-test). Fig 4f/4g: per-tissue Cohen's d bars.

    `preds_per_file`: dict of filename -> AnnData with tAge_SM/tAge_YM in `.obs`, as returned
    by `load_wholemouse_lps_predictions` (or an equivalent loader once confirmed).
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    fig_e, _, stats_e = plot_clock_distributions(
        preds_per_file, group_patterns=['CTRL', 'LPS'], norm_cols=['tAge_SM'], show=False,
    )
    fig_e.savefig(save_dir / 'fig4e_ctrl_vs_lps_boxplot.pdf', bbox_inches='tight', dpi=300)

    tissue_patterns = [f'{tissue}_{cond}' for tissue in WHOLEMOUSE_TISSUES for cond in ('CTRL', 'LPS')]
    fig_fg, _, stats_fg = plot_tissue_standardized_effects(
        preds_per_file, group_patterns=tissue_patterns, norm_cols=['tAge_SM'],
        ctrl_suffix='CTRL', exp_suffix='LPS', show=False,
    )
    fig_fg.savefig(save_dir / 'fig4f-g_tissue_effect_sizes.pdf', bbox_inches='tight', dpi=300)

    return dict(fig_e=fig_e, fig_fg=fig_fg, stats_e=stats_e, stats_fg=stats_fg)
