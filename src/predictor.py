import copy
import logging
from typing import Any, Dict

import torch
from lightning import LightningModule
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

import src.constants as cs
from src.data.graphmodule import GSPGraph
from src.evaluation import evaluate
from src.metrics import compute_metrics, log_metrics
from src.models.adapert import AdaPert

logger = logging.getLogger(__name__)


class PertPredictor(LightningModule):
    """
    Lightning module: training step, validation and test metrics, optimizer.

    Args:
        input_dim (int): Number of genes of the control expression.
        p_emb_dim (int): Dimension of the semantic perturbation embedding s_p.
        adata_output_dim (int): Number of predicted genes.
        model_args (Dict[str, Any]): Model options (`model` section of the config).
        graph (GSPGraph): Knowledge graph.
        min_lr (float): Smallest learning rate of the scheduler.
        lr_scheduler_args (Dict[str, Any]): ReduceLROnPlateau options (factor, patience).
        weight_decay (float): AdamW weight decay.
        data_args (Dict[str, Any]): Data module arguments. Only stored in the checkpoint: scripts/get_predictions.py
            rebuilds the data from it.
    """

    def __init__(
        self,
        input_dim: int,
        p_emb_dim: int,
        adata_output_dim: int,
        model_args: Dict[str, Any],
        graph: GSPGraph = None,
        min_lr: float = 0.0,
        lr_scheduler_args: Dict[str, Any] = None,
        weight_decay: float = 0.0,
        data_args: Dict[str, Any] = None,
    ):
        super(PertPredictor, self).__init__()

        self.save_hyperparameters(ignore=["graph"])
        model_args = copy.deepcopy(model_args)  # the pop below must not change the stored hyperparameters

        self.lr = model_args.pop("lr", 1e-3)
        self.min_lr = min_lr
        self.lr_scheduler_args = lr_scheduler_args or {}
        self.weight_decay = weight_decay

        self.model = AdaPert(
            input_dim=input_dim,
            p_emb_dim=p_emb_dim,
            adata_output_dim=adata_output_dim,
            graph=graph,
            **model_args,
        )

    def forward(self, cntr, pert_idxs, p_emb):
        return self.model(cntr.to(self.device), pert_idxs, p_emb.to(self.device))

    def training_step(self, batch, batch_idx):
        prediction = self(batch.control, batch.pert_idxs, batch.p)
        loss, logs = self.model.loss(
            prediction,
            target=batch.x.to(self.device),
            control=batch.control.to(self.device),
            mask_degs=batch.mask_degs.to(self.device),
        )
        self.log("train/loss", loss, prog_bar=True)
        self.log_dict({f"train/{k}": v for k, v in logs.items()})
        return loss

    def evaluate_stage(self, stage: str, loader: DataLoader) -> None:
        """Predict the cells of a loader and log the metrics of the stage (val or test)."""
        results = evaluate(loader, self)
        log_metrics(compute_metrics(results, self.trainer.datamodule.adata), stage, self)

    def validation_step(self, batch, batch_idx):
        pass  # needed: without validation_step Lightning skips the validation loop and on_validation_epoch_end

    def on_validation_epoch_end(self):
        self.evaluate_stage(cs.VAL, self.trainer.datamodule.val_dataloader())
        self.evaluate_stage(cs.TEST, self.trainer.datamodule.test_dataloader())

    def configure_optimizers(self):
        """AdamW (model.param_groups) and ReduceLROnPlateau on the validation Pearson delta."""
        optimizer = torch.optim.AdamW(self.model.param_groups(self.lr), lr=self.lr, weight_decay=self.weight_decay)
        logger.info("Optimizer learning rates: %s", [g["lr"] for g in optimizer.param_groups])
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=self.lr_scheduler_args.get("factor", 0.5),
            patience=self.lr_scheduler_args.get("patience", 10),
            min_lr=self.min_lr,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_pearson_delta",
                "interval": "epoch",
                "frequency": 1,
                "strict": False,  # validation runs every 2 epochs
            },
        }
