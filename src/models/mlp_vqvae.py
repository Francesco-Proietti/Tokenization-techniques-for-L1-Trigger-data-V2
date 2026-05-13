"""
MLP VQ-VAE Implementation

A simple Vector Quantized Variational Autoencoder with MLP encoder/decoder.
"""

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import lightning as pl

from vector_quantize_pytorch import VectorQuantize


class MLPEncoder(nn.Module):
    """MLP Encoder for VQ-VAE."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        latent_dim: int,
    ):
        """
        Initialize the Encoder.

        Args:
            input_dim: Dimension of input features.
            hidden_dims: List of hidden layer dimensions.
            latent_dim: Dimension of the latent space (output).
        """
        super().__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU()
            ])
            prev_dim = hidden_dim

        self.encoder = nn.Sequential(*layers)
        self.projector = nn.Linear(prev_dim, latent_dim)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through encoder."""
        
        h = self.encoder(x)

        return self.projector(h)


class MLPDecoder(nn.Module):
    """MLP Decoder for VQ-VAE."""

    def __init__(
        self,
        latent_dim: int,
        hidden_dims: List[int],
        output_dim: int,
    ):
        """
        Initialize the Decoder.

        Args:
            latent_dim: Dimension of the latent space.
            hidden_dims: List of hidden layer dimensions.
            output_dim: Dimension of reconstructed output.
        """
        super().__init__()

        layers = []
        prev_dim = latent_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU()
            ])
            prev_dim = hidden_dim

        self.decoder = nn.Sequential(*layers)
        self.reconstructor = nn.Linear(prev_dim, output_dim)

    def forward(self, z: Tensor) -> Tensor:
        """Forward pass through decoder."""

        h = self.decoder(z)
        return self.reconstructor(h)


class MLPVQVAE(pl.LightningModule):
    """MLP Vector Quantized Variational Autoencoder."""

    def __init__(
        self,
        cfg,
        lr: float = 1e-3
    ):
        """
        Initialize the MLP VQ-VAE.

        Args:
            input_dim: Dimension of input features.
            hidden_dims: Shared hidden dimensions for encoder/decoder.
            latent_dim: Dimension of the latent space.
            codebook_size: Number of codebook vectors.
            commitment_cost: Weight for commitment loss.
            reconstruction_weight: Weight for reconstruction loss.
            encoder_hidden_dims: Optional custom hidden dims for encoder (overrides hidden_dims).
            decoder_hidden_dims: Optional custom hidden dims for decoder (overrides hidden_dims).
        """
        super().__init__()

        self.save_hyperparameters()

        self.input_dim = cfg.input_dim
        self.hidden_dims = cfg.hidden_dims
        self.latent_dim = cfg.latent_dim
        self.codebook_size = cfg.codebook_size
        self.rot_trick = cfg.rotation_trick
        self.decay = cfg.decay
        self.beta = cfg.beta
        self.lr = lr

        self.encoder_hidden_dims = cfg.encoder_hidden_dims or self.hidden_dims
        self.decoder_hidden_dims = cfg.decoder_hidden_dims or list(reversed(self.hidden_dims))

        self.encoder = MLPEncoder(
            input_dim=self.input_dim,
            hidden_dims=self.encoder_hidden_dims,
            latent_dim=self.latent_dim
        )

        # Vector Quantizer
        self.quantizer = VectorQuantize(
            dim=self.latent_dim,
            codebook_size=self.codebook_size,
            rotation_trick=self.rot_trick,
            commitment_weight=self.beta,
            decay=self.decay
        )

        self.decoder = MLPDecoder(
            latent_dim=self.latent_dim,
            hidden_dims=self.decoder_hidden_dims,
            output_dim=self.input_dim
        )

    def forward(self, x: Tensor, mask: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Forward pass through the VQ-VAE.

        Args:
            x: Input tensor of shape [batch_size, num_particles, num_features]
            mask: Boolean tensor indicating valid particles [batch_size, num_particles]
        Returns:
            reconstruction: Reconstructed input
            commitment_loss: Commitment loss
            indices: Indices of the quantized vectors
        """
        
        B, N, F = x.size()

        # Encode
        z_e = self.encoder(x)

        # Flatten mask
        flat_mask = mask.view(-1) # [B*N]

        flat_z_e = z_e.view(-1, self.latent_dim) # [B*N, latent_dim]
        
        valid_z_e = flat_z_e[flat_mask]

        valid_z_e_3d = valid_z_e.unsqueeze(0) 

        # Quantize
        z_q, indices, commit_loss = self.quantizer(valid_z_e_3d)

        z_q_valid = z_q.squeeze(0) 

        z_q_padded = torch.zeros_like(flat_z_e)

        z_q_padded[flat_mask] = z_q_valid

        z_q = z_q_padded.view(B, N, -1)

        # Decode
        x_recon = self.decoder(z_q) 

        x_recon = x_recon * mask.unsqueeze(-1)

        return x_recon, commit_loss, indices
        
    # Training Step
    def training_step(self, batch, batch_idx):

        x, mask = batch 

        x_recon, commit_loss, _ = self(x, mask)

        # Reconstruction loss 
        recon_loss = (x - x_recon) ** 2

        # Apply mask
        mask = mask.unsqueeze(-1)
        recon_loss = recon_loss * mask

        # Average only with valid values
        recon_loss = recon_loss.sum() / mask.sum()

        # Total loss
        loss = recon_loss + 10 * commit_loss
        
        # Log
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_recon_loss", recon_loss, prog_bar=True)
        self.log("train_commit_loss", commit_loss, prog_bar=True)

        return loss
    
    # Validation Step
    def validation_step(self, batch, batch_idx):

        x, mask = batch

        x_recon, commit_loss, _ = self(x, mask)
        
        # Reconstruction loss
        recon_loss = (x - x_recon) ** 2

        # Apply mask
        mask = mask.unsqueeze(-1)
        recon_loss = recon_loss * mask

        # Average only valid values
        recon_loss = recon_loss.sum() / mask.sum()
        
        # Total loss
        loss = recon_loss + 10 * commit_loss
        
        # Log
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_recon_loss", recon_loss, prog_bar=True)
        self.log("val_commit_loss", commit_loss, prog_bar=True)
    

    # Test step
    def test_step(self, batch, batch_idx):
        
        x, mask = batch

        x_recon, commit_loss, _ = self(x, mask)

        # Reconstruction loss
        recon_loss = (x - x_recon) ** 2

        # Apply mask
        mask = mask.unsqueeze(-1)
        recon_loss = recon_loss * mask

        # Average only valid values
        recon_loss = recon_loss.sum() / mask.sum()

        # Total loss
        loss = recon_loss + 10 * commit_loss

        # Log
        self.log("test_loss", loss, prog_bar=True)
        self.log("test_recon_loss", recon_loss, prog_bar=True)
        self.log("test_commit_loss", commit_loss, prog_bar=True)

    # Optimizer
    def configure_optimizers(self):
        
        # Adam
        return torch.optim.Adam(self.parameters(), lr=self.lr)
