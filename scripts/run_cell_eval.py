#!/usr/bin/env python
"""
cell-eval metrics of the predictions written by scripts/get_predictions.py.

Usage:
    python scripts/run_cell_eval.py \
        --pred predictions/adapert_k562.h5ad \
        --real research/cache/K562_single_cell_line/de_adata_test.h5ad \
        --output eval_results/adapert_k562 \
        --name adapert_k562

Writes to the output directory: <name>_results.csv (per perturbation), <name>_agg_results.csv, <name>_summary.csv
(means), and the DE tables <name>_pred_de.csv and <name>_real_de.csv.
"""
import argparse
import logging
import os

import anndata as ad
import polars as pl
from cell_eval import MetricsEvaluator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

CONTROL_PERT = "non-targeting"


def parse_args():
    parser = argparse.ArgumentParser(description="Run cell-eval metrics")
    parser.add_argument("--pred", type=str, required=True, help="Path to predicted h5ad file")
    parser.add_argument("--real", type=str, required=True, help="Path to real/ground truth h5ad file")
    parser.add_argument("--output", type=str, required=True, help="Output directory for results")
    parser.add_argument("--name", type=str, required=True, help="Model name (prefix for output files)")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)

    adata_pred = ad.read_h5ad(args.pred)
    adata_real = ad.read_h5ad(args.real)
    adata_pred.obs_names_make_unique()
    adata_real.obs_names_make_unique()

    # perturbation column 'target'
    adata_pred.obs['target'] = adata_pred.obs['pert_name']
    adata_real.obs['target'] = adata_real.obs['gene_name']

    # observed cells of the predicted perturbations and the control cells
    keep_perts = set(adata_pred.obs['target'].unique()) | {CONTROL_PERT}
    adata_real = adata_real[adata_real.obs['target'].isin(keep_perts)].copy()
    logger.info(f"Pred shape: {adata_pred.shape}, perts: {adata_pred.obs['target'].nunique()}")
    logger.info(f"Real shape: {adata_real.shape}, perts: {adata_real.obs['target'].nunique()}")

    evaluator = MetricsEvaluator(
        adata_pred=adata_pred,
        adata_real=adata_real,
        control_pert=CONTROL_PERT,
        pert_col='target',
        de_method='wilcoxon',
        num_threads=8,
        outdir=args.output,
        prefix=args.name,
    )
    results, agg_results = evaluator.compute(
        profile="full",
        skip_metrics=["pearson_edistance", "clustering_agreement"],
        write_csv=True,
    )

    # means over the perturbations
    agg_df = agg_results.to_pandas()
    mean_row = agg_df[agg_df['statistic'] == 'mean'].iloc[0]
    summary = {col: [mean_row[col]] for col in agg_df.columns if col != 'statistic'}
    print("\nAggregated Results (mean):")
    print("-" * 40)
    for col, (value,) in summary.items():
        print(f"  {col}: {value:.4f}")

    summary_path = os.path.join(args.output, f"{args.name}_summary.csv")
    pl.DataFrame(summary).write_csv(summary_path)
    logger.info(f"Results saved to: {args.output}")


if __name__ == "__main__":
    main()
