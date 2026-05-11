#!/usr/bin/env python3
"""Training script for MLP VQ-VAE on L1 Trigger data."""

from pathlib import Path
import yaml

import lightning as pl
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

from src.data.data_loading import L1TriggerDataset, L1TriggerDataModule
from src.models.mlp_vqvae import MLPVQVAE

seed = 42

pl.seed_everything(seed, workers=True)


def load_config(config_path: str) -> dict:
    """Load YAML config file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def main():
    # Paths
    data_config_path = "configs/data.yaml"
    model_config_path = "configs/model.yaml"

    # Load configs
    data_config = load_config(data_config_path)
    model_config = load_config(model_config_path)

    # Extract data config
    dirs_train = data_config["data"]["train_path"]
    dirs_val = data_config["data"]["val_path"]
    dirs_test = data_config["data"]["test_path"]
    features_cols = data_config["data"]["features"]
    max_part = data_config["data"]["max_particles"]
    prep = data_config["data"]["preprocessing"]

    # Extract model config
    input_dim = model_config["MLP_VQVAE"]["input_dim"]
    hidden_dim = model_config["MLP_VQVAE"]["hidden_dim"]
    latent_dim = model_config["MLP_VQVAE"]["latent_dim"]
    codebook_size = model_config["MLP_VQVAE"]["codebook_size"]
    rot_trick = model_config["MLP_VQVAE"]["rotation_trick"]
    decay = model_config["MLP_VQVAE"]["decay"]
    beta = model_config["MLP_VQVAE"]["beta"]

    # Extract training config
    lr = model_config["training"]["lr"]
    max_epochs = model_config["training"]["max_epochs"]
    batch_size = model_config["training"]["batch_size"]

    # Initialize data module
    data_module = L1TriggerDataModule(
        parquet_dirs_train=dirs_train,
        parquet_dirs_val=dirs_val,
        parquet_dirs_test=dirs_test,
        max_particles=max_part,
        batch_size=batch_size,
        #num_workers=7,  
        features=features_cols,
        preprocessing=prep
    )

    # Initialize model
    model = MLPVQVAE(
        input_dim=input_dim,
        hidden_dims=hidden_dim,
        decay=decay,
        embedding_dim=latent_dim,
        num_embeddings=codebook_size,
        rot_trick=rot_trick,
        commitment_cost=beta,
        lr=lr
    )
    
    # TensorBoard logger
    logger = TensorBoardLogger(
        save_dir="logs",
        name="mlp_vqvae_rot"
    )
    
    # Save only the top-3 checkpoint
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints/",
        filename=f"{logger.name}-v{logger.version}" + "-{epoch:02d}-{val_loss:.4f}",
        monitor="val_loss",
        mode="min",
        save_top_k=3,        
        save_last=True       
    )

    

    # Initialize trainer
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator="auto",
        devices="auto",
        log_every_n_steps=10,
        callbacks=[checkpoint_callback],
        logger=logger
    )

    # Train
    trainer.fit(model, datamodule=data_module)


if __name__ == "__main__":
    main()