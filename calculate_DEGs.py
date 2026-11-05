"""
Differentially expressed genes (DEGs) of every perturbation, used by L_non.

For each perturbation p, a Wilcoxon rank-sum test (scanpy) compares the cells of p with the non-targeting cells.
The DEGs D(p) are the genes with BH-adjusted p < 0.05, sorted by |log2 fold change|; all other genes are the
non-DEGs penalized by L_non. One table per perturbation is written to AAA_DEGs/<dataset>/true_degs/ (finished
perturbations are skipped when the script is re-run) and the gene indices to AAA_DEGs/<dataset>/true_degs_dict.pkl.

Run inside research/:
    python ../calculate_DEGs.py --dataset_name K562_single_cell_line
"""
import argparse
import os
import pickle

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from tqdm import tqdm

PERT_KEY = "gene_name"
CONTROL_LABEL = "non-targeting"
PADJ_THRESHOLD = 0.05
SAVE_DIR = "AAA_DEGs"


def store_degs_per_perturbation(adata, perts, save_dir):
    """Test every perturbation against the controls and write one CSV of DEGs per perturbation.

    The genes with BH-adjusted p < PADJ_THRESHOLD are kept, sorted by |log2FC|.
    """
    os.makedirs(save_dir, exist_ok=True)
    print(f"{len(perts)} perturbations")

    for pert in tqdm(perts):
        file_path = os.path.join(save_dir, f"{pert}.csv")

        # resume: perturbations that already have a file are skipped
        if os.path.exists(file_path):
            continue

        # test on the cells of this perturbation and the controls only (much faster than the full object)
        idx_mask = (adata.obs[PERT_KEY] == pert) | (adata.obs[PERT_KEY] == CONTROL_LABEL)
        subset = adata[idx_mask].copy()

        sc.tl.rank_genes_groups(
            subset,
            groupby=PERT_KEY,
            groups=[pert],
            reference=CONTROL_LABEL,
            method='wilcoxon',
            key_added='temp_res'
        )

        df = sc.get.rank_genes_groups_df(subset, group=pert, key='temp_res')
        df['abs_lfc'] = df['logfoldchanges'].abs()
        df = df[df['pvals_adj'] < PADJ_THRESHOLD]

        # an empty table also marks the perturbation as done
        if df.empty:
            df.to_csv(file_path, index=False)
            continue

        top_df = df.nlargest(len(df), 'abs_lfc')[['names', 'logfoldchanges', 'pvals_adj', 'scores']]
        top_df['pert_name'] = pert
        top_df.to_csv(file_path, index=False)

    print(f"Done. Results in {save_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default='K562_single_cell_line')
    args = parser.parse_args()

    dataset_name = args.dataset_name
    test_adata = ad.read_h5ad(f"cache/{dataset_name}/de_adata_test.h5ad")
    save_dir = os.path.join(SAVE_DIR, dataset_name, "true_degs")
    perts = [p for p in test_adata.obs[PERT_KEY].unique() if p != CONTROL_LABEL]

    print(f"Calculating DEGs for {dataset_name}, saving to {save_dir} ...")
    store_degs_per_perturbation(test_adata, perts, save_dir)

    deg_idx_list = {}
    for pert in perts:
        df = pd.read_csv(os.path.join(save_dir, f"{pert}.csv"))
        deg_idx_list[pert] = np.array([test_adata.var_names.get_loc(gene) for gene in df['names']])
    out_path = os.path.join(SAVE_DIR, dataset_name, 'true_degs_dict.pkl')
    with open(out_path, 'wb') as f:
        pickle.dump(deg_idx_list, f)
    print(f"DEG indices of {len(deg_idx_list)} perturbations saved to {out_path}")
