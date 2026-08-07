#!/usr/bin/env python
"""Fig S8 — Self-contained GSEA (Stouffer/Liptak) on the cross-tissue meta-signature.

Logic-identical to fig6d_self_contained_gsea.py (see that file for the full method
description and source citation) -- consumes figS7_meta_signatures.py's output CSV
(or recomputes it via --per-tissue-degs). Same open questions apply: which cached
DE-table combination and which of the raw-beta/z-score variants correspond to the
published Fig S8 was not established by this audit.
"""

from __future__ import annotations

import argparse
import gzip
import os
import pickle

import numpy as np
import pandas as pd

from stage.meta_analysis import mixed_meta_from_deg_tables
from stage.gsea import self_contained_from_meta
from fig6c_meta_signatures import load_gene_sets

FIGURE_TAG = 'figS8'


def _load_meta_tbl(args) -> pd.DataFrame:
    if args.meta_table_csv:
        return pd.read_csv(args.meta_table_csv, index_col=0)
    if not args.per_tissue_degs:
        raise SystemExit("Must pass either --meta-table-csv (figS7's output) or --per-tissue-degs (to recompute it).")
    with gzip.open(args.per_tissue_degs, 'rb') as f:
        per_tissue_tables_de = pickle.load(f)
    return mixed_meta_from_deg_tables(
        per_tissue_tables_de,
        gene_col='gene', beta_col='Coef_interaction', se_col='SE_interaction', p_col='P_interaction',
    )


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--meta-table-csv', default=None)
    p.add_argument('--per-tissue-degs', default=None)
    p.add_argument('--hallmark-gmt', required=True)
    p.add_argument('--reactome-gmt', required=True)
    p.add_argument('--use-zscore', action='store_true')
    p.add_argument('--min-size', type=int, default=10)
    p.add_argument('--out-dir', required=True)
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    meta_tbl = _load_meta_tbl(args)
    gene_sets_combined = load_gene_sets(args.hallmark_gmt, args.reactome_gmt)

    if args.use_zscore:
        meta_tbl = meta_tbl.copy()
        meta_tbl['z_score'] = meta_tbl['beta'] / meta_tbl['se'].replace(0, np.nan)
        meta_tbl = meta_tbl.dropna(subset=['z_score'])
        # see fig6d_self_contained_gsea.py for why a unit-se column is needed here
        # (self_contained_from_meta indexes mt[se_col] unconditionally; se_col=None crashes).
        meta_tbl['_unit_se'] = 1.0
        sc_gsea = self_contained_from_meta(meta_tbl, gene_sets=gene_sets_combined, beta_col='z_score',
                                            se_col='_unit_se', min_size=args.min_size, corr_ref=None)
    else:
        sc_gsea = self_contained_from_meta(meta_tbl, gene_sets=gene_sets_combined, beta_col='beta',
                                            se_col='se', min_size=args.min_size, corr_ref=None)

    out_csv = os.path.join(args.out_dir, f'{FIGURE_TAG}_self_contained_gsea.csv')
    sc_gsea.to_csv(out_csv)
    print(f"Wrote {out_csv}  ({(sc_gsea['FDR'] < 0.05).sum()} pathways significant at FDR<0.05)")


if __name__ == '__main__':
    main()
