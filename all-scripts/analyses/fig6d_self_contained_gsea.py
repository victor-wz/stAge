#!/usr/bin/env python
"""Fig 6d — Self-contained GSEA (Stouffer/Liptak) on the cross-tissue meta-signature.

Pathway-level second stage of the two-stage analysis whose gene-level first stage is
Fig 6c (fig6c_meta_signatures.py). Takes that script's meta-signature CSV (or
recomputes it in one step via --per-tissue-degs) and runs a self-contained Stouffer/
Liptak gene-set test (stage.gsea.self_contained_from_meta) against Hallmark + Reactome
mouse gene sets, with a CAMERA-style variance-inflation correction for inter-gene
correlation.

The source notebook has two variants of this step: one using the meta-analysis's raw
beta/se, and one using a standardized z-score (beta/se) instead -- both present as
sequential, near-duplicate cells (iterative-editing residue, not two different
figures). `--use-zscore` selects between them; which variant matches the published
Fig 6d was not established by this audit -- flagged, not guessed.

# TODO(needs author input): same open question as fig6c_meta_signatures.py regarding
# which cached --per-tissue-degs file (if recomputing from scratch rather than
# consuming a fig6c CSV) corresponds to the published Fig 6d.

Source: v_pipeline/dstream_loop.ipynb, "Calculate META-SIGNATURES" cells (~lines
2903-2920 and 2960-2975 of the nbconvert'ed script, self-contained-test portions).
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

FIGURE_TAG = 'fig6d'


def _load_meta_tbl(args) -> pd.DataFrame:
    if args.meta_table_csv:
        return pd.read_csv(args.meta_table_csv, index_col=0)
    if not args.per_tissue_degs:
        raise SystemExit("Must pass either --meta-table-csv (fig6c's output) or --per-tissue-degs (to recompute it).")
    with gzip.open(args.per_tissue_degs, 'rb') as f:
        per_tissue_tables_de = pickle.load(f)
    return mixed_meta_from_deg_tables(
        per_tissue_tables_de,
        gene_col='gene', beta_col='Coef_interaction', se_col='SE_interaction', p_col='P_interaction',
    )


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--meta-table-csv', default=None, help="fig6c_meta_signatures.py's output CSV (preferred, avoids recompute)")
    p.add_argument('--per-tissue-degs', default=None, help="Alternative: recompute the meta-table from a per_tissue_tables_de_*.pkl.gz cache")
    p.add_argument('--hallmark-gmt', required=True, help="Path to mh.all.v2025.1.Mm.symbols.gmt")
    p.add_argument('--reactome-gmt', required=True, help="Path to m2.cp.reactome.v2025.1.Mm.symbols.gmt")
    p.add_argument('--use-zscore', action='store_true',
                    help="Use standardized z-score (beta/se) instead of raw beta -- see module docstring")
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
        # self_contained_from_meta always computes Z_i = beta_col / se_col per gene before
        # combining across the gene set; since z_score is already a standardized statistic,
        # pass a unit "se" column so Z_i reduces to z_score itself (se_col=None would raise
        # a KeyError inside self_contained_from_meta, which indexes mt[se_col] unconditionally).
        meta_tbl['_unit_se'] = 1.0
        sc_gsea = self_contained_from_meta(meta_tbl, gene_sets=gene_sets_combined, beta_col='z_score',
                                            se_col='_unit_se', min_size=args.min_size, corr_ref=None)
    else:
        sc_gsea = self_contained_from_meta(meta_tbl, gene_sets=gene_sets_combined, beta_col='beta',
                                            se_col='se', min_size=args.min_size, corr_ref=None)

    out_csv = os.path.join(args.out_dir, f'{FIGURE_TAG}_self_contained_gsea.csv')
    sc_gsea.to_csv(out_csv)
    print(f"Wrote {out_csv}  ({(sc_gsea['FDR'] < 0.05).sum()} pathways significant at FDR<0.05)")
    print("\nTop pathways (self-contained on meta):")
    print(sc_gsea.sort_values('FDR').head(10))


if __name__ == '__main__':
    main()
