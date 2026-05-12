#!/usr/bin/env python3
"""Training script for MLP VQ-VAE on L1 Trigger data."""

import hydra
from omegaconf import DictConfig

import lightning as pl
import torch

from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

from src.models.registry import MODEL_REGISTRY

from src.data.data_loading import L1TriggerDataModule
from src.models.mlp_vqvae import MLPVQVAE


@hydra.main(
    version_base=None,
    config_path="configs",
    config_name="config"
)
def main(cfg: DictConfig):

    # Set seed for reproducibility
    pl.seed_everything(cfg.trainer.seed, workers=True)

    # -----------------------
    # DataModule
    # -----------------------

    data_module = L1TriggerDataModule(
        parquet_dirs_train=cfg.data.train_path,
        parquet_dirs_val=cfg.data.val_path,
        parquet_dirs_test=cfg.data.test_path,
        max_particles=cfg.data.max_particles,
        batch_size=cfg.trainer.batch_size,
        features=list(cfg.data.features),
        preprocessing=cfg.data.preprocessing
    )

    # -----------------------
    # Model
    # -----------------------

    model_name = cfg.model.name
    ModelClass = MODEL_REGISTRY[model_name]

    model = ModelClass(cfg.model, lr=cfg.trainer.lr)

    # -----------------------
    # Logger
    # -----------------------

    logger = TensorBoardLogger(
        save_dir=cfg.paths.logs_dir,
        name=cfg.experiment.name
    )

    # -----------------------
    # Checkpoints
    # -----------------------

    checkpoint_callback = ModelCheckpoint(
        dirpath=f"{cfg.paths.checkpoint_dir}/{cfg.experiment.name}",
        filename=f"v{logger.version}" + "-{epoch:02d}-{val_loss:.4f}",
        monitor="val_loss",
        mode="min",
        save_top_k=3,
        save_last=True
    )

    # -----------------------
    # Trainer
    # -----------------------

    trainer = pl.Trainer(
        max_epochs=cfg.trainer.max_epochs,
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        log_every_n_steps=cfg.trainer.log_every_n_steps,
        logger=logger,
        callbacks=[checkpoint_callback]
    )

    trainer.fit(model, datamodule=data_module)


if __name__ == "__main__":
    main()