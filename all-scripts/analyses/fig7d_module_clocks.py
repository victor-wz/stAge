"""Fig 7d -- Module-specific clocks (human cohort panel).

# INFERRED (unconfirmed): see fig6e_module_clocks.py's module docstring for the
mouse/human split rationale. This file's human-cohort assignment is better supported
than the mouse-tissue files': v_pipeline/stage_modules_pred.ipynb's LAST-SAVED
execution state (not a commented-out alternate) is actively configured with
`rawdata_dir='vvicente/stomics_datasets/notion2/as_h5ad/hbreastcancer'` and
`mod_dir='tAge_clocks/Module clocks 5.4 multispecies'` (comment in source: "for
HUMANS!!!!!!!"). Still, no source document confirms this maps to Fig 7d specifically
rather than 7h (or both) -- Fig 7 spans multiple human validation cohorts (AD,
breast cancer) elsewhere in this analysis list, and it isn't confirmed which panel
this dataset feeds. Confirm with the author; fig7h_module_clocks.py is an identical
placeholder pending that confirmation.

Runs stage.module_clocks.run_module_clock_grid across a human cohort and every
module clock in the human multispecies module-clock directory.

CLI:
    python fig7d_module_clocks.py --rawdata-dir DIR --ncbi-reference-path PATH \\
        --module-clock-dir "tAge_clocks/Module clocks 5.4 multispecies" --save-dir DIR
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
    ap.add_argument('--rawdata-dir', required=True,
                     help='Directory of per-sample raw human .h5ad files, e.g. an "hbreastcancer"-style cohort.')
    ap.add_argument('--ncbi-reference-path', required=True,
                     help='NOTE: source notebook uses the mouse gene_info reference even for human data '
                          '(no separate human reference path was found in this cell) -- pass '
                          'Homo_sapiens.gene_info if that turns out to be wrong; flagged, not resolved.')
    ap.add_argument('--module-clock-dir', default='tAge_clocks/Module clocks 5.4 multispecies')
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
    summary.to_csv(os.path.join(args.save_dir, 'fig7d_module_clocks_summary.csv'), index=False)
    print(summary)


if __name__ == '__main__':
    main()
