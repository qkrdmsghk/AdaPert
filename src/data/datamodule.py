"""
Data module: the cells of one Perturb-seq screen (research/cache/<task_type>/de_adata_test.h5ad) with its predefined
train / validation / test split of the perturbations.

Each item is a perturbed cell (in training also a control cell) with the mean expression of the control cells of its
batch, the semantic embedding s_p of its perturbation and its DEG mask.
"""
import os
import pickle
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import anndata
import joblib
import lightning as L
import numpy as np
import pandas as pd
import torch
from scipy.sparse import issparse
from torch.utils.data import DataLoader, Dataset

import src.constants as cs

# GenePT embeddings of the NCBI / UniProt gene descriptions (3,072-d)
EMBEDDING_PATH = cs.CACHE_DIR / "GenePT_emebdding_v2/GenePT_gene_protein_embedding_model_3_text.pickle"


def pert_name(condition_name: str) -> str:
    """Perturbed gene of a condition name: K562_UBL5+ctrl_1+1 -> UBL5 (control cells: K562_ctrl_1 -> ctrl)."""
    return condition_name.split("_")[1].split("+")[0]


@dataclass
class Payload:
    """A batch of cells."""

    x: torch.Tensor                 # [B, G] expression
    control: torch.Tensor           # [B, G] mean expression of the control cells of the cell's batch
    p: torch.Tensor                 # [B, p_emb_dim] semantic embedding s_p of the perturbation
    mask_degs: torch.Tensor         # [B, G] bool, True for the DEGs of the perturbation
    pert_idxs: Tuple                # graph node ids of the perturbation of each cell (["ctrl"] for control cells)
    pert_cond_names: Tuple          # condition names, e.g. K562_UBL5+ctrl_1+1
    cell_types: Tuple
    experimental_batches: Tuple


def collate_fn(batch) -> Payload:
    x, control, p, mask_degs, pert_idxs, pert_cond_names, cell_types, experimental_batches = zip(*batch)
    return Payload(
        x=torch.stack(x),
        control=torch.stack(control),
        p=torch.stack(p),
        mask_degs=torch.stack(mask_degs),
        pert_idxs=pert_idxs,
        pert_cond_names=pert_cond_names,
        cell_types=cell_types,
        experimental_batches=experimental_batches,
    )


def to_tensor(x) -> torch.Tensor:
    """Expression matrix as a dense float32 tensor (kept on the CPU; batches are moved to the device)."""
    if issparse(x):
        x = x.toarray()
    return torch.tensor(x, dtype=torch.float32)


class PertDataModule(L.LightningDataModule):
    """
    Args:
        task_type: Dataset folder in research/cache, e.g. K562_single_cell_line.
        batch_size: Batch size.
        mask_degs: DEG file written by calculate_DEGs.py. None: no DEG masks (inference).
    """

    def __init__(self, task_type: str = "K562_single_cell_line", batch_size: int = 64, mask_degs: Optional[str] = None):
        super().__init__()
        self.task_type = task_type
        self.batch_size = batch_size
        self.cache_path = cs.CACHE_DIR / task_type
        self.data_prepared = False

        # graph node id of every gene; the perturbations are the genes of the GEARS gene set
        gene_df = pd.read_csv(cs.DATA_DIR / "master_gene_set_sorted.csv")
        self.gene2id = dict(zip(gene_df["gene_name"], gene_df["gene_id"]))
        id2pert = pd.read_csv(cs.DATA_DIR / "gears_gene_set.csv", index_col=0)["0"].to_dict()
        self.pert2id = {v: k for k, v in id2pert.items()}

        with open(EMBEDDING_PATH, "rb") as f:
            self.p_emb_dict = pickle.load(f)
        self.p_emb_dim = len(next(iter(self.p_emb_dict.values())))

        self.mask_degs_dict = None
        if mask_degs is not None:
            if not os.path.exists(mask_degs):
                raise FileNotFoundError(f"DEG file not found: {mask_degs} (run calculate_DEGs.py first)")
            with open(mask_degs, "rb") as f:
                self.mask_degs_dict = pickle.load(f)

    def prepare_data(self):
        """Load the cells and the split."""
        if self.data_prepared:
            return
        self.adata = anndata.read_h5ad(self.cache_path / "de_adata_test.h5ad")
        self.control_data = self.adata[self.adata.obs[cs.CONTROL] == 1]
        self.treatment_data = self.adata[self.adata.obs[cs.CONTROL] == 0]

        # keep the cells whose perturbed genes are all in the GEARS gene set
        genes = [
            [p for p in conds if p != cs.CONTROL_LABEL]
            for conds in self.treatment_data.obs[cs.CONDITION].str.split("+").tolist()
        ]
        valid = np.array([all(p in self.pert2id for p in g) for g in genes])
        self.treatment_data = self.treatment_data[valid]
        self.conditions = pd.Series([[self.pert2id[p] for p in g] for g, v in zip(genes, valid) if v])
        self.conditions.index = self.treatment_data.obs.index
        self.condition_names = self.treatment_data.obs[cs.CONDITION_NAME]

        self.input_dim = self.adata_output_dim = self.adata.X.shape[1]
        self.condition_group = joblib.load(self.cache_path / "splits/predefined/train_test_split.pkl")
        self.data_prepared = True

    def setup(self, stage: Optional[str] = None):
        """Datasets of the training (with the control cells), validation and test perturbations."""
        self.train_data = self._dataset("train", include_controls=True)
        self.val_data = self._dataset("val")
        self.test_data = self._dataset("test")

    def _dataset(self, split: str, include_controls: bool = False) -> "PerturbDataset":
        mask = self.treatment_data.obs[cs.CONDITION].isin(self.condition_group[split])
        return PerturbDataset(
            adata=self.treatment_data[mask],
            control_data=self.control_data,
            pert_conditions=self.conditions[mask],
            condition_names=self.condition_names[mask],
            p_emb_dict=self.p_emb_dict,
            mask_degs_dict=self.mask_degs_dict,
            include_controls=include_controls,
        )

    def _loader(self, dataset: "PerturbDataset", shuffle: bool) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            collate_fn=collate_fn,
            drop_last=shuffle,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True,
        )

    def train_dataloader(self):
        return self._loader(self.train_data, shuffle=True)

    def val_dataloader(self):
        return self._loader(self.val_data, shuffle=False)

    def test_dataloader(self):
        return self._loader(self.test_data, shuffle=False)


class PerturbDataset(Dataset):
    """Cells of a set of perturbations.

    Args:
        adata: Perturbed cells.
        control_data: Control cells.
        pert_conditions: Graph node ids of the perturbation of each cell.
        condition_names: Condition name of each cell.
        p_emb_dict: Semantic embedding of each gene.
        mask_degs_dict: DEG indices of each perturbation, or None.
        include_controls: Also include the control cells (perturbation "ctrl"), as in training.

    Items: (x, control, p_emb, mask_degs, pert_idxs, cond_name, cell_type, batch).
    """

    def __init__(
        self,
        adata: anndata.AnnData,
        control_data: anndata.AnnData,
        pert_conditions: pd.Series,
        condition_names: pd.Series,
        p_emb_dict: Dict,
        mask_degs_dict: Optional[Dict],
        include_controls: bool = False,
    ):
        self.p_emb_dict = p_emb_dict
        self.p_emb_fallback = next(iter(p_emb_dict.values()))   # control cells and genes without an embedding
        self.mask_degs_dict = mask_degs_dict

        self.pert_data = to_tensor(adata.X)                         # [n_cells, G]
        self.control_data = to_tensor(control_data.X)               # [n_controls, G]
        self.output_dim = self.pert_data.shape[1]
        self.pert_conditions = pert_conditions
        self.condition_names = condition_names
        self.treatment_cell_types = adata.obs[cs.CELL_TYPE]
        self.treatment_cell_batches = adata.obs[cs.BATCH]

        # control cells of each batch
        control_batches = control_data.obs[cs.BATCH]
        self.control_batch_indices = {b: np.where(control_batches == b)[0] for b in control_batches.unique()}

        if include_controls:
            n_controls = self.control_data.shape[0]
            self.pert_data = torch.cat([self.pert_data, self.control_data])
            self.pert_conditions = pd.concat(
                [self.pert_conditions, pd.Series([[cs.CONTROL_LABEL]] * n_controls)], ignore_index=True
            )
            self.condition_names = pd.concat(
                [self.condition_names, control_data.obs[cs.CONDITION_NAME]], ignore_index=True
            )
            self.treatment_cell_types = pd.concat(
                [self.treatment_cell_types, control_data.obs[cs.CELL_TYPE]], ignore_index=True
            )
            self.treatment_cell_batches = pd.concat(
                [self.treatment_cell_batches, control_batches], ignore_index=True
            )

    def __len__(self):
        return len(self.pert_data)

    def __getitem__(self, idx: int):
        cond_name = self.condition_names.iloc[idx]
        return (
            self.pert_data[idx],
            self._get_control(idx),
            self._get_pert_embedding(cond_name),
            self._get_mask_degs(cond_name),
            self.pert_conditions.iloc[idx],
            cond_name,
            self.treatment_cell_types.iloc[idx],
            self.treatment_cell_batches.iloc[idx],
        )

    def _get_control(self, idx: int) -> torch.Tensor:
        """Mean expression [G] of the control cells of the cell's batch."""
        indices = self.control_batch_indices[self.treatment_cell_batches.iloc[idx]]
        return self.control_data[indices].mean(0)

    def _get_pert_embedding(self, cond_name: str) -> torch.Tensor:
        return torch.tensor(self.p_emb_dict.get(pert_name(cond_name), self.p_emb_fallback), dtype=torch.float32)

    def _get_mask_degs(self, cond_name: str) -> torch.Tensor:
        """[G] bool, True for the DEGs of the perturbation (all True for control cells or without DEG file)."""
        deg_indices = None if self.mask_degs_dict is None else self.mask_degs_dict.get(pert_name(cond_name))
        if deg_indices is None:
            return torch.ones(self.output_dim, dtype=torch.bool)
        mask = torch.zeros(self.output_dim, dtype=torch.bool)
        mask[deg_indices] = True
        return mask
