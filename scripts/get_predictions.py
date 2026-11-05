"""
Predict the test perturbations with a trained checkpoint. The predictions and the control cells are written to an
h5ad file for scripts/run_cell_eval.py. The data and model settings are read from the checkpoint.

Usage:
    python scripts/get_predictions.py \
        --ckpt_path checkpoints/adapert-k562-seed2.ckpt \
        --output predictions/adapert_k562.h5ad
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import anndata
import numpy as np
import pandas as pd
import torch
from scipy import sparse

import src.constants as cs
from src.data.datamodule import PertDataModule
from src.data.graphmodule import GSPGraph
from src.evaluation import evaluate
from src.predictor import PertPredictor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--output", type=str, default="predictions.h5ad", help="Output file path")
    args = parser.parse_args()

    hparams = torch.load(args.ckpt_path, map_location="cpu", weights_only=False)["hyper_parameters"]

    # Data and split as in training (the DEG masks enter only the training loss)
    datamodule = PertDataModule(**{**hparams["data_args"], "mask_degs": None})
    datamodule.prepare_data()
    datamodule.setup()
    for key in ("input_dim", "p_emb_dim", "adata_output_dim"):
        assert getattr(datamodule, key) == hparams[key], (
            f"{key}: data {getattr(datamodule, key)} != checkpoint {hparams[key]}")

    # Model with the settings stored in the checkpoint; every weight must match
    graph = GSPGraph(gene2id=datamodule.gene2id)
    model = PertPredictor.load_from_checkpoint(args.ckpt_path, map_location="cpu", graph=graph, strict=True)
    model = model.to("cuda:0" if torch.cuda.is_available() else "cpu")

    results = evaluate(datamodule.test_dataloader(), model)

    # Predicted cells, then the control cells (perturbation "non-targeting", needed by cell-eval)
    control = datamodule.control_data
    control_X = control.X.toarray() if sparse.issparse(control.X) else control.X
    n_controls = control_X.shape[0]
    obs = pd.DataFrame({
        "pert_name": [p.replace("+ctrl", "") for p in results[cs.RES_PERT_CAT]] + ["non-targeting"] * n_controls,
        "celltype_name": list(results[cs.RES_CELL_TYPE]) + list(control.obs[cs.CELL_TYPE].values),
        "gem_group": list(results[cs.RES_EXP_BATCH]) + list(control.obs[cs.BATCH].values),
    })
    adata_pred = anndata.AnnData(
        X=sparse.csr_matrix(np.vstack([results[cs.RES_PRED], control_X])),
        obs=obs,
        var=pd.DataFrame(index=datamodule.adata.var_names),
    )
    adata_pred.write_h5ad(args.output)

    print(f"Predictions saved to {args.output}")
    print(f"  Shape: {adata_pred.shape}")
    print(f"  Perturbations: {len(np.unique(results[cs.RES_PERT_CAT]))}")


if __name__ == "__main__":
    main()
