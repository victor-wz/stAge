#!/usr/bin/env python
"""Fig S7 — Inverse-variance meta-signature (DerSimonian-Laird) across tissues.

Logic-identical to fig6c_meta_signatures.py (see that file for the full method
description and source citation) -- see its TODO for the open question of which
cached --per-tissue-degs file distinguishes this supplementary panel from Fig 6c.

NOTE: this Fig S7 (Inverse-variance meta-signatures) is UNRELATED to the "Figure S7"
title used internally in this repo's v_pipeline/FigS7.md draft methods document --
that document's content (Brown-Forsythe spatial-variance test + Gi* + DESeq2 +
cross-tissue Spearman convergence) was confirmed during the Phase 1/2 audit to
actually correspond to the user's Fig S6a-c, not this Fig S7. Do not conflate the two
when writing the release README/methods.
"""

from __future__ import annotations

import argparse
import gzip
import os
import pickle

from stage.meta_analysis import mixed_meta_from_deg_tables

FIGURE_TAG = 'figS7'


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--per-tissue-degs', required=True)
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


if __name__ == '__main__':
    main()
