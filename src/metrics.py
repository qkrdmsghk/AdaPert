"""
Validation and test metrics logged during training.

For each perturbation, the mean predicted and the mean observed expression of its cells are compared, either directly
(pearson) or as changes Δ from the mean control expression of the cells' batches (pearson_delta, spearman_delta).
Each metric is the mean over the perturbations; val_pearson_delta selects the checkpoint.
"""
import logging
from typing import Dict

import anndata
import numpy as np
import pandas as pd
from lightning import LightningModule
from scipy.sparse import issparse
from scipy.stats import pearsonr, spearmanr

import src.constants as cs

logger = logging.getLogger(__name__)


def batch_control_means(adata: anndata.AnnData) -> pd.DataFrame:
    """Mean expression of the control cells of each batch (index: batch as str, columns: genes)."""
    ctrl_mask = adata.obs[cs.CONDITION] == cs.CONTROL_LABEL
    x = adata[ctrl_mask].X
    df = pd.DataFrame(x.toarray() if issparse(x) else np.asarray(x))
    batches = adata.obs.loc[ctrl_mask, cs.BATCH].values.astype(str)
    return df.groupby(batches).apply(lambda t: t.mean(0))


def nan_to_zero(v: float) -> float:
    return 0.0 if np.isnan(v) else float(v)


def compute_metrics(results: Dict[str, np.ndarray], adata: anndata.AnnData) -> Dict[str, float]:
    """Mean over the perturbations of pearson, pearson_delta and spearman_delta.

    Args:
        results: Output of src.evaluation.evaluate (pred and truth [n_cells, G], pert_cat and experimental_batch
            [n_cells]).
        adata: Data with the control cells.
    """
    batch_ctrl = batch_control_means(adata)
    values: Dict[str, list] = {"pearson": [], "pearson_delta": [], "spearman_delta": []}
    for pert in np.unique(results[cs.RES_PERT_CAT]):
        mask = results[cs.RES_PERT_CAT] == pert
        ctrl_mean = batch_ctrl.loc[results[cs.RES_EXP_BATCH][mask].astype(str), :].to_numpy().mean(0)   # [G]
        pred_mean = results[cs.RES_PRED][mask].mean(0)                                                    # [G]
        truth_mean = results[cs.RES_TRUTH][mask].mean(0)                                                  # [G]
        values["pearson"].append(nan_to_zero(pearsonr(pred_mean, truth_mean)[0]))
        values["pearson_delta"].append(nan_to_zero(pearsonr(pred_mean - ctrl_mean, truth_mean - ctrl_mean)[0]))
        values["spearman_delta"].append(nan_to_zero(spearmanr(pred_mean - ctrl_mean, truth_mean - ctrl_mean)[0]))
    return {name: float(np.mean(v)) for name, v in values.items()}


def log_metrics(metrics: Dict[str, float], stage: str, lm: LightningModule) -> None:
    """Print the metrics and log them as <stage>_<name>."""
    for name, value in metrics.items():
        logger.info(f"{stage}_{name}: {value}")
        lm.log(f"{stage}_{name}", value)
