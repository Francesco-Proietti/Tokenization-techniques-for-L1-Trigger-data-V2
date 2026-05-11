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
        embedding_dim: int,
    ):
        """
        Initialize the Encoder.

        Args:
            input_dim: Dimension of input features.
            hidden_dims: List of hidden layer dimensions.
            embedding_dim: Dimension of the embedding space (output).
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
        self.projector = nn.Linear(prev_dim, embedding_dim)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through encoder."""
        # If input has more than 2 dimensions, flatten except batch dim
        if x.dim() > 2:
            batch_size = x.size(0)
            num_part = x.size(1)
            num_features = x.size(2)
            x = x.view(batch_size*num_part, num_features)

        h = self.encoder(x)
        return self.projector(h)


class MLPDecoder(nn.Module):
    """MLP Decoder for VQ-VAE."""

    def __init__(
        self,
        embedding_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        dropout: float = 0.1,
    ):
        """
        Initialize the Decoder.

        Args:
            embedding_dim: Dimension of the latent embedding.
            hidden_dims: List of hidden layer dimensions.
            output_dim: Dimension of reconstructed output.
        """
        super().__init__()

        layers = []
        prev_dim = embedding_dim

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
        # If input has more than 2 dimensions, flatten except batch dim
        if z.dim() > 2:
            batch_size = z.size(0)
            num_part = z.size(1)
            num_features = z.size(2)
            z = z.view(batch_size*num_part, num_features)

        h = self.decoder(z)
        return self.reconstructor(h)


class MLPVQVAE(pl.LightningModule):
    """MLP Vector Quantized Variational Autoencoder."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        decay: float,
        embedding_dim: int = 64,
        num_embeddings: int = 512,
        rot_trick: bool = True,
        commitment_cost: float = 0.25,
        reconstruction_weight: float = 1.0,
        encoder_hidden_dims: Optional[List[int]] = None,
        decoder_hidden_dims: Optional[List[int]] = None,
        lr: float = 1e-3
    ):
        """
        Initialize the MLP VQ-VAE.

        Args:
            input_dim: Dimension of input features.
            hidden_dims: Shared hidden dimensions for encoder/decoder.
            embedding_dim: Dimension of the embedding space.
            num_embeddings: Number of codebook vectors.
            commitment_cost: Weight for commitment loss.
            reconstruction_weight: Weight for reconstruction loss.
            encoder_hidden_dims: Optional custom hidden dims for encoder (overrides hidden_dims).
            decoder_hidden_dims: Optional custom hidden dims for decoder (overrides hidden_dims).
        """
        super().__init__()

        self.save_hyperparameters()

        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.rot_trick = rot_trick
        self.decay = decay
        self.beta = commitment_cost
        self.reconstruction_weight = reconstruction_weight
        self.lr = lr

        self.encoder_hidden_dims = encoder_hidden_dims or hidden_dims
        self.decoder_hidden_dims = decoder_hidden_dims or list(reversed(hidden_dims))
        
        self.encoder = MLPEncoder(
            input_dim=self.input_dim,
            hidden_dims=self.encoder_hidden_dims,
            embedding_dim=self.embedding_dim
        )

        # Vector Quantizer
        self.quantizer = VectorQuantize(
            dim=self.embedding_dim,
            codebook_size=self.num_embeddings,
            rotation_trick=self.rot_trick,
            commitment_weight=self.beta,
            decay=self.decay
        )

        self.decoder = MLPDecoder(
            embedding_dim=self.embedding_dim,
            hidden_dims=self.decoder_hidden_dims,
            output_dim=self.input_dim
        )

    def forward(self, x: Tensor, mask: Tensor) -> Tuple[Tensor, dict]:
        """
        Forward pass through the VQ-VAE.

        Args:
            x: Input tensor of shape [batch_size, num_particles, num_features]
            mask: Boolean tensor indicating valid particles [batch_size, num_particles]
        Returns:
            reconstruction: Reconstructed input
            losses: Dictionary of loss components
        """
        
        B, N, F = x.size()

        # Mask valid particles
        x_valid = x[mask]    

        # Encode
        z_e = self.encoder(x_valid)

        # Quantize
        z_q, indices, vq_loss = self.quantizer(z_e)

        # Decode
        x_recon_valid = self.decoder(z_q)

        x_recon = torch.zeros(B, N, F, device=x.device)

        x_recon[mask] = x_recon_valid    

        return x_recon, vq_loss, indices
        
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
        loss = recon_loss + commit_loss

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
        loss = recon_loss + commit_loss
        
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
        loss = recon_loss + commit_loss

        # Log
        self.log("test_loss", loss, prog_bar=True)
        self.log("test_recon_loss", recon_loss, prog_bar=True)
        self.log("test_commit_loss", commit_loss, prog_bar=True)

    # Optimizer
    def configure_optimizers(self):
        
        # Adam
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    #def encode(self, x: Tensor) -> Tensor:
    #    """Encode input to latent representation."""
    #    z_e = self.encoder(x)
    #    _, _, _ = self.quantizer(z_e)  # Forward to get quantized
    #    return z_e

    #def decode(self, z_q: Tensor) -> Tensor:
    #    """Decode latent representation to reconstruction."""
    #    return self.decoder(z_q)

    #def generate(self, x: Tensor) -> Tensor:
    #    """Generate reconstruction of input."""
    #    return self.forward(x)[0]
