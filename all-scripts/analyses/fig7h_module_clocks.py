"""Fig 7h -- Module-specific clocks (human cohort panel).

# INFERRED (unconfirmed): see fig7d_module_clocks.py's module docstring for the full
explanation -- this file is a copy of that same human-cohort placeholder (output
filename changed) pending confirmation of whether this dataset feeds 7d, 7h, or both,
with the author.

CLI:
    python fig7h_module_clocks.py --rawdata-dir DIR --ncbi-reference-path PATH \\
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
                     help='NOTE: source notebook uses the mouse gene_info reference even for human data -- '
                          'flagged, not resolved. See fig7d_module_clocks.py.')
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
    summary.to_csv(os.path.join(args.save_dir, 'fig7h_module_clocks_summary.csv'), index=False)
    print(summary)


if __name__ == '__main__':
    main()
