from typing import Dict

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import src.constants as cs


def evaluate(loader: DataLoader, model) -> Dict[str, np.ndarray]:
    """Run the model in inference mode on a data loader.

    Args:
        loader: Data loader (validation or test cells).
        model: PertPredictor.

    Returns:
        pred and truth [n_cells, G]; pert_cat (e.g. "UBL5+ctrl"), cell_type and experimental_batch [n_cells].
    """
    print('Getting the predictions...')
    model.eval()
    pert_cat, pred, truth, cell_types, experimental_batches = [], [], [], [], []
    for batch in tqdm(loader):
        with torch.no_grad():
            pred.append(model(batch.control, batch.pert_idxs, batch.p).cpu())
            truth.append(batch.x.cpu())
            for itr, cond_name in enumerate(batch.pert_cond_names):
                pert_cat.append(cond_name.split("_")[1])      # condition name, e.g. K562_UBL5+ctrl_1+1
                cell_types.append(batch.cell_types[itr])
                experimental_batches.append(batch.experimental_batches[itr])

    return {
        cs.RES_PERT_CAT: np.array(pert_cat),
        cs.RES_CELL_TYPE: np.array(cell_types),
        cs.RES_PRED: torch.cat(pred).numpy(),
        cs.RES_TRUTH: torch.cat(truth).numpy(),
        cs.RES_EXP_BATCH: np.array(experimental_batches),
    }
