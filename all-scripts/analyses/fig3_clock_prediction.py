"""Fig 3 -- Clock prediction pipeline (filtering, log1p, StandardScaler/YuGene,
relative-to-reference subtraction, missing-gene imputation, elastic net).

TODO(needs author input): see fig2_clock_prediction.py's module docstring for the full
explanation -- this file is an identical structurally-correct placeholder copy (same
underlying pipeline call, `--tag` changed) pending confirmation of what actually
distinguishes the Fig 3 panel from Figs 2, S4, S5.

CLI:
    python fig3_clock_prediction.py --rawdata-dir DIR --ncbi-reference-path PATH \\
        --clock-root DIR --save-dir DIR [--clock-folder "tAge_clocks/EN differential models 4.6"]
"""

from __future__ import annotations

import argparse
import os

import scanpy as sc

from stage.pipeline import full_nonoverlap_mp_pipeline


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--rawdata-dir', required=True, help='Directory of per-sample raw .h5ad files.')
    ap.add_argument('--ncbi-reference-path', required=True, help='Path to a *.gene_info file (e.g. Mus_musculus.gene_info).')
    ap.add_argument('--clock-root', required=True, help='Root directory containing --clock-folder.')
    ap.add_argument('--clock-folder', default='tAge_clocks/EN differential models 4.6')
    ap.add_argument('--save-dir', required=True)
    ap.add_argument('--control-file-pattern', default='_Y_')
    ap.add_argument('--resolution', type=float, default=2.0)
    ap.add_argument('--mp-coverage-threshold', type=int, default=150_000)
    ap.add_argument('--tag', default='fig3')
    args = ap.parse_args()

    anndata_dict = {
        f: sc.read_h5ad(os.path.join(args.rawdata_dir, f))
        for f in os.listdir(args.rawdata_dir) if f.endswith('.h5ad')
    }

    full_nonoverlap_mp_pipeline(
        anndata_dict,
        ncbi_reference_path=args.ncbi_reference_path,
        clock_root=args.clock_root,
        clock_folder=args.clock_folder,
        res=args.resolution,
        control_file_pattern=args.control_file_pattern,
        mp_coverage_threshold=args.mp_coverage_threshold,
        save_plot=True,
        save_result=True,
        save_dir=args.save_dir,
        tag=args.tag,
    )


if __name__ == '__main__':
    main()
