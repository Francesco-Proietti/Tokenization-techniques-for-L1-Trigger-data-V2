"""
Transformer VQ-VAE Implementation

A Vector Quantized Variational Autoencoder with Transformer encoder/decoder.
"""

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as pl
from torch import Tensor

from vector_quantize_pytorch import VectorQuantize


class TransformerEncoder(nn.Module):
    """Transformer Encoder for VQ-VAE."""

    def __init__(
        self,
        input_dim: int, 
        latent_dim: int, 
        n_heads: int = 4, 
        n_layers: int = 3
    ):
        """"
        Initialize the Encoder.

        Args:
            input_dim: Dimension of input features.
            latent_dim: Dimension of the latent space.
            n_heads: Number of attention heads.
            n_layers: Number of transformer layers.
        """
        super().__init__()

        self.input_proj = nn.Linear(input_dim, latent_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=latent_dim,
            nhead=n_heads,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers
        )

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        """Forward pass through encoder."""
        
        x = self.input_proj(x)  

        #mask (invert True-False)
        attn_mask = ~mask 

        z = self.transformer(x, src_key_padding_mask=attn_mask)

        return z
    

class TransformerDecoder(nn.Module):
    """Transformer Decoder for VQ-VAE."""

    def __init__(
        self, 
        latent_dim: int, 
        output_dim: int, 
        n_heads: int = 4, 
        n_layers: int = 3
    ):
        """
        Initialize the Decoder.

        Args:
            latent_dim: Dimension of the latent embedding.
            n_heads: Number of attention heads.
            n_layers: Number of transformer layers.
            output_dim: Dimension of reconstructed output.
        """
        super().__init__()

        decoder_layer = nn.TransformerEncoderLayer(
            d_model=latent_dim,
            nhead=n_heads,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            decoder_layer,
            num_layers=n_layers
        )

        self.output_proj = nn.Linear(latent_dim, output_dim)

    def forward(self, z: Tensor, mask: Tensor) -> Tensor:
        """Forward pass through decoder."""
        attn_mask = ~mask

        z = self.transformer(z, src_key_padding_mask=attn_mask)
        x_recon = self.output_proj(z)

        return x_recon
    

class TransformerVQVAE(pl.LightningModule):
    """Transformer Vector Quantized Variational Autoencoder."""

    def __init__(
        self,
        input_dim: int = 3, 
        latent_dim: int = 128, 
        codebook_size: int = 256, 
        n_heads: int = 4, 
        n_layers: int = 3, 
        dec: float = 0.8, 
        beta: float = 0.25, 
        rot_trick: bool = True, 
        lr: float = 1e-3
    ):
        """
        Initialize the transformer VQ-VAE.

        Args:
            input_dim: Dimension of input features.
            latent_dim: Dimension of the latent space.
            codebook_size: Number of codebook vectors.
            n_heads: Number of attention heads.
            n_layers: Number of transformer layers.
            dec: Decay rate for exponential moving average.
            beta: Weight for commitment loss.
            rot_trick: Whether to use rotation trick.
            lr: Learning rate.
        """
        
        super().__init__()

        self.save_hyperparameters()

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.codebook_size = codebook_size
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.decay = dec
        self.beta = beta
        self.rot_trick = rot_trick
        self.lr = lr
        
        #Encoder
        self.encoder = TransformerEncoder(self.input_dim, self.latent_dim, self.n_heads, self.n_layers)
        
        #Decoder (not a "real" transformer decoder)
        self.decoder = TransformerDecoder(self.latent_dim, self.input_dim, self.n_heads, self.n_layers)

        self.quantizer = VectorQuantize(
            dim=self.latent_dim,
            codebook_size=self.codebook_size,
            decay=self.decay,
            commitment_weight=self.beta,
            rotation_trick=self.rot_trick
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
        
        # Encode
        z_e = self.encoder(x, mask)  
        
        # Quantize
        z_q, indices, vq_loss = self.quantizer(z_e)
        
        # Decode
        x_recon = self.decoder(z_q, mask)

        return x_recon, vq_loss, indices
    

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
        
        #Log
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_recon_loss", recon_loss, prog_bar=True)
        self.log("val_commit_loss", commit_loss, prog_bar=True)
    

    #Test step
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