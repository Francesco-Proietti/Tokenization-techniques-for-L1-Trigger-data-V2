#!/usr/bin/env python3
"""Script for producing the histogram of the tokens' distribution for the MLP-VQ-VAE model."""


import torch
import numpy as np
import yaml
import matplotlib.pyplot as plt

from src.data.data_loading import L1TriggerDataModule
from src.models.mlp_vqvae import MLPVQVAE

# Open config file
with open("configs/data/default.yaml") as f:
    config = yaml.safe_load(f)

# Extract config info
dirs_train = config["train_path"]
dirs_val = config["val_path"]
dirs_test = config["test_path"]

max_part = config["max_particles"]

feat = config["features"]

# Initialize DataModule (with test data)
dm_prep = L1TriggerDataModule(
    parquet_dirs_train=dirs_train,
    parquet_dirs_val=dirs_val,
    parquet_dirs_test=dirs_test,
    max_particles=max_part,
    batch_size=32,
    features=feat,
    preprocessing=True,
    #num_workers=0,
)

loader_prep = dm_prep.test_dataloader()

# Ask user for the checkpoint string
checkpoint = input("Insert the checkpoint string: ")

# Ask user for the plot name
plot_name = input("Insert the plot name (without extension): ")

# Load the trained MLP-VQ-VAE model
ckpt = torch.load(
    f"checkpoints/mlp/{checkpoint}",
    map_location="cpu",
    weights_only=False,  
)

model = MLPVQVAE(**ckpt["hyper_parameters"])

model.load_state_dict(ckpt["state_dict"])

model.eval()
model.freeze()

device = "cuda"

model.to(device)

# Initialize indices list
indices = []

with torch.inference_mode():

    for batch in loader_prep:

        features, mask = batch

        features = features.to(device)
        mask = mask.to(device)

        outputs = model(features, mask)

        indices.extend(np.array(outputs[2].squeeze(0).cpu()))

# Codebook usage 
cb_usage = len(set(indices))/512

# Plot
plt.figure(figsize=(12, 6))
plt.hist(indices, bins=512, density=True)
plt.title("Histogram of tokens' distribution using test dataset")
plt.xlabel("Token ID")
plt.ylabel("Density")
plt.text(0.8, 0.85, f"MLP-VQ-VAE\nCB_size=512\nCB_usage={cb_usage:.3f}",
         transform=plt.gca().transAxes)

plt.savefig(f"CB_MLPVQVAE_plots/{plot_name}.png")

print("DONE!")

