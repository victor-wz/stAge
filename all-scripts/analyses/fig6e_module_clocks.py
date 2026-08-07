"""Fig 6e -- Module-specific clocks (mouse tissue panel).

# INFERRED (unconfirmed): the source bullet "Module-specific clocks [Figs 6e, 7d, 7h,
S9, S10]" covers one underlying analysis (stage.module_clocks / v_pipeline/
stage_modules_pred.ipynb) applied across several figures. No source document
confirms which figure gets which dataset. This file's mouse-tissue assignment is
inferred from stage_modules_pred.ipynb's own cell history: an earlier (commented-out)
configuration used `rawdata_dir='data/immunoglobulin'` with a MOUSE module-clock
directory (`tAge_clocks/Module clocks 4.6 bwnet3 filtered5`, itself commented out in
favor of the human config in the notebook's last-saved state) -- Fig 6 is the main
mouse-hotspot figure elsewhere in this analysis list, making a mouse-tissue panel a
reasonable (not confirmed) fit for 6e. figS9_module_clocks.py and
figS10_module_clocks.py share this same inference; fig7d/fig7h are the human-cohort
counterparts. Confirm the real dataset/clock-set assignment with the author before
treating this file's defaults as the literal Fig 6e content.

Runs stage.module_clocks.run_module_clock_grid across a mouse tissue cohort and every
module clock in a mouse module-clock directory, then builds a per-tissue x per-clock
tAge delta summary. Uses `stage.plotting` for the heatmap.

CLI:
    python fig6e_module_clocks.py --rawdata-dir DIR --ncbi-reference-path PATH \\
        --module-clock-dir DIR --save-dir DIR
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
import scanpy as sc

from stage.module_clocks import run_module_clock_grid


def age_group_from_filename(fn: str, control_file_pattern: str = '_Y_') -> str:
    return "Young" if control_file_pattern in fn else "Old"


def summarize_module_clock_grid(res_dict: dict) -> pd.DataFrame:
    """Per (tissue, clock) mean tAge_SM by AgeGroup -- the table a Δ-heatmap is built from."""
    rows = []
    for key, adatas in res_dict.items():
        tissue, clock_tag = key.split('__', 1)
        for fn, ad in adatas.items():
            rows.append({
                'tissue': tissue, 'clock': clock_tag, 'file': fn,
                'age_group': ad.obs['AgeGroup'].iloc[0] if 'AgeGroup' in ad.obs else np.nan,
                'mean_tAge_SM': ad.obs['tAge_SM'].mean(),
            })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--rawdata-dir', required=True, help='Directory of per-sample raw mouse .h5ad files.')
    ap.add_argument('--ncbi-reference-path', required=True)
    ap.add_argument('--module-clock-dir', required=True,
                     help='Directory of mouse module-clock *_scaleddiff.pkl files, '
                          'e.g. "tAge_clocks/Module clocks 4.6 bwnet3 filtered5".')
    ap.add_argument('--save-dir', required=True)
    ap.add_argument('--control-file-pattern', default='_Y_')
    ap.add_argument('--resolution', type=float, default=0.5)
    ap.add_argument('--mp-coverage-threshold', type=int, default=1_000)
    args = ap.parse_args()

    assembled_adatas = {
        f: sc.read_h5ad(os.path.join(args.rawdata_dir, f))
        for f in os.listdir(args.rawdata_dir) if f.endswith('.h5ad')
    }
    unique_tissues = sorted({k.split('_')[2] for k in assembled_adatas if len(k.split('_')) > 2})

    res_dict = run_module_clock_grid(
        assembled_adatas, args.module_clock_dir, args.ncbi_reference_path, unique_tissues,
        age_group_from_filename=lambda fn: age_group_from_filename(fn, args.control_file_pattern),
        control_file_pattern=args.control_file_pattern,
        resolution=args.resolution, mp_coverage_threshold=args.mp_coverage_threshold,
    )

    summary = summarize_module_clock_grid(res_dict)
    os.makedirs(args.save_dir, exist_ok=True)
    summary.to_csv(os.path.join(args.save_dir, 'fig6e_module_clocks_summary.csv'), index=False)
    print(summary)


if __name__ == '__main__':
    main()
