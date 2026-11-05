"""
Train AdaPert (Hydra): python main.py --config-name adapert-k562
"""
import os

import hydra
import torch
from lightning import Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from omegaconf import DictConfig, OmegaConf

import src.constants as cs
from src.data.datamodule import PertDataModule
from src.data.graphmodule import GSPGraph
from src.models.adapert import compute_effects
from src.predictor import PertPredictor
from src.utils import set_seed


@hydra.main(config_name="adapert-k562", config_path="configs", version_base="1.3")
def main(cfg: DictConfig):
    set_seed(cfg.seed)

    data_args = OmegaConf.to_container(cfg.datamodule, resolve=True)
    datamodule = PertDataModule(**data_args)
    datamodule.prepare_data()

    os.makedirs(cfg.wandb.save_dir, exist_ok=True)  # otherwise W&B writes the run to the system temp directory
    wandb_logger = WandbLogger(
        project=cfg.wandb.project,
        name=cfg.wandb.name,
        save_dir=cfg.wandb.save_dir,
        log_model=True,
        config=OmegaConf.to_container(cfg, resolve=True),
    )
    checkpoint_callback = ModelCheckpoint(
        dirpath=cs.CHECKPOINT_DIR / cfg.datamodule.task_type / f"{cfg.wandb.name}_{wandb_logger.experiment.id}",
        filename="best-{epoch}-{step}",
        save_top_k=1,
        monitor="val_pearson_delta",
        mode="max",
    )
    # patience counts validation checks (one every `check_val_every_n_epoch` epochs)
    early_stop_callback = EarlyStopping(monitor="val_pearson_delta", mode="max", patience=cfg.early_stop_patience)

    trainer = Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        max_epochs=cfg.max_epochs,
        check_val_every_n_epoch=2,
        logger=wandb_logger,
        num_sanity_val_steps=0,
        callbacks=[checkpoint_callback, early_stop_callback],
        gradient_clip_val=1.0,
        enable_progress_bar=False,
    )

    datamodule.setup("fit")
    graph = GSPGraph(gene2id=datamodule.gene2id)  # knowledge graph used by the GNN

    # Reset seed right before model init to ensure reproducible
    # weight initialization regardless of prior random state consumption
    # (e.g., from imports, data loading, graph construction).
    set_seed(cfg.seed)

    model = PertPredictor(
        input_dim=datamodule.input_dim,
        p_emb_dim=datamodule.p_emb_dim,
        adata_output_dim=datamodule.adata_output_dim,
        graph=graph,
        model_args=OmegaConf.to_container(cfg.model, resolve=True),
        min_lr=cfg.min_lr,
        lr_scheduler_args=OmegaConf.to_container(cfg.lr_scheduler, resolve=True),
        weight_decay=cfg.weight_decay,
        data_args=data_args,  # stored in the checkpoint: scripts/get_predictions.py rebuilds the data from it
    )
    # AdaPert context: observed pseudo-bulk ΔX of the training perturbations (training cells only)
    node_ids, effects = compute_effects(datamodule)
    model.model.set_effect_table(node_ids, effects)

    trainer.fit(model, datamodule)


if __name__ == "__main__":
    main()
