#!/usr/bin/env python
"""Fig 6c — Inverse-variance meta-signature (DerSimonian-Laird) across tissues.

Gene-level stage of a two-stage analysis: build a long (gene, tissue, beta, se) table
from per-tissue DESeq2 results and run DerSimonian-Laird random-effects meta-analysis
(stage.meta_analysis) to get one meta-analytic effect size per gene across tissues.
The pathway-level second stage (self-contained Stouffer/Liptak test run ON this
meta-signature) is Fig 6d (fig6d_self_contained_gsea.py), which consumes this script's
output CSV -- confirmed by reading the source notebook cell, where the DerSimonian-Laird
call and the self-contained-test call are two sequential steps against the same
`meta_tbl`, matching the "meta-signature" (6c) vs. "self-contained GSEA on the
meta-signature" (6d) framing in the original analysis-list bullets.

# TODO(needs author input): "Figs 6c, S7" as a single original bullet still leaves open
# WHICH cached per-tissue DESeq2 table (main text vs. supplementary) each of Fig 6c and
# Fig S7 uses. The source notebook (v_pipeline/dstream_loop.ipynb) caches DE tables
# under a `per_tissue_tables_de_{YOUNG,OLD}_{brains,other}_Gi>{0,1}_DEseq2.pkl.gz`
# naming convention -- at least "OLD_brains_Gi>1" and "YOUNG_other_Gi>1" variants exist
# on disk. This script and figS7_meta_signatures.py are logic-identical; which
# --per-tissue-degs file each should point at needs author confirmation.

Source: v_pipeline/dstream_loop.ipynb, "Calculate META-SIGNATURES" cells (~lines
2860-2920 of the nbconvert'ed script, DerSimonian-Laird portion only).
"""

from __future__ import annotations

import argparse
import gzip
import os
import pickle

from stage.meta_analysis import mixed_meta_from_deg_tables

FIGURE_TAG = 'fig6c'


def load_gene_sets(hallmark_gmt: str, reactome_gmt: str) -> dict:
    """Shared by fig6d/figS8 too -- Hallmark + Reactome mouse gene sets, Reactome wins
    on key clashes (matches the source's `dict(ChainMap(*gene_sets_list))` order,
    where reactome was appended after hallmark)."""
    import gseapy as gp
    gene_sets = {}
    gene_sets.update(gp.parser.read_gmt(hallmark_gmt))
    gene_sets.update(gp.parser.read_gmt(reactome_gmt))
    return gene_sets


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--per-tissue-degs', required=True,
                    help="Path to a per_tissue_tables_de_*.pkl.gz cache (dict {tissue: DESeq2 results DataFrame}), "
                         "produced upstream by the Fig 6b DESeq2 pipeline. See module docstring for the "
                         "{YOUNG,OLD}x{brains,other}xGi-threshold naming convention and the open question "
                         "about which one is Fig 6c.")
    p.add_argument('--out-dir', required=True)
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    with gzip.open(args.per_tissue_degs, 'rb') as f:
        per_tissue_tables_de = pickle.load(f)

    meta_tbl = mixed_meta_from_deg_tables(
        per_tissue_tables_de,
        gene_col='gene', beta_col='Coef_interaction', se_col='SE_interaction', p_col='P_interaction',
    )

    out_csv = os.path.join(args.out_dir, f'{FIGURE_TAG}_meta_signature_genes.csv')
    meta_tbl.to_csv(out_csv)
    print(f"Wrote {out_csv}  ({(meta_tbl['FDR'] < 0.05).sum()} genes significant at FDR<0.05)")
    print("\nTop meta hits (random-effects):")
    print(meta_tbl.sort_values('FDR').head(10))


if __name__ == '__main__':
    main()
