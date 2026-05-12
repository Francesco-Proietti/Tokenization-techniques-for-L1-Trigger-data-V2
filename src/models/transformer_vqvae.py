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


class NormFormerBlock(nn.Module):

    def __init__(
        self,
        input_dim: int = 128,
        num_heads: int = 8,
        mlp_dim: int = 4,
        dropout: float = 0.1
    ):
        
        super().__init__()

        self.input_dim = input_dim
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.dropout = dropout

        # MHA and LayerNorm
        self.norm1 = nn.LayerNorm(self.input_dim)

        self.attn = nn.MultiheadAttention(
            embed_dim=self.input_dim,
            num_heads=self.num_heads,
            dropout=self.dropout,
            batch_first=True
        )

        self.norm2 = nn.LayerNorm(self.input_dim)
        
        # MLP with layer normalization
        self.mlp = nn.Sequential(
            nn.LayerNorm(self.input_dim),  
            nn.Linear(self.input_dim, self.mlp_dim),
            nn.SiLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.mlp_dim, self.input_dim),
        )

        # Initialize to 0 for stable training at the beginning
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)
        nn.init.zeros_(self.norm1.weight)

    def forward(self, x, mask):
      
        # Self-attention
        x_norm = self.norm1(x)

        key_padding_mask = ~mask

        attn_output, _ = self.attn(
            x_norm, 
            x_norm, 
            x_norm, 
            key_padding_mask=key_padding_mask
        )

        attn_res = self.norm2(attn_output) + x

        output = self.mlp(attn_res) + attn_res

        return output
    

class Transformer(torch.nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        hidden_dim,
        num_heads=1,
        num_blocks=2,
    ):
        super().__init__()
         
        self.input_dim = input_dim
        self.num_blocks = num_blocks
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.output_dim = output_dim

        self.project_in = nn.Linear(input_dim, hidden_dim)

        self.blocks = nn.ModuleList(
            [
                NormFormerBlock(input_dim=self.hidden_dim, mlp_dim=self.hidden_dim, num_heads=self.num_heads)
                for _ in range(self.num_blocks)
            ]
        )
        self.project_out = nn.Linear(self.hidden_dim, self.output_dim)

    def forward(self, x, mask):

        x = self.project_in(x)

        for i, block in enumerate(self.blocks):
            x = block(x, mask)

        x = self.project_out(x) * mask.unsqueeze(-1)
        
        return x


class TransformerVQVAE(pl.LightningModule):
    """Transformer Vector Quantized Variational Autoencoder."""

    def __init__(
        self,
        cfg,
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

        self.input_dim = cfg.input_dim
        self.hidden_dim = cfg.hidden_dim
        self.n_heads = cfg.n_heads
        self.dropout = cfg.dropout
        self.depth = cfg.depth
        self.latent_dim = cfg.latent_dim
        self.codebook_size = cfg.codebook_size
        self.decay = cfg.decay
        self.beta = cfg.beta
        self.rot_trick = cfg.rotation_trick
        self.lr = lr
        
        self.encoder = Transformer(
            input_dim=self.input_dim,
            output_dim=self.latent_dim,
            hidden_dim=self.hidden_dim,
            num_heads=self.n_heads,
            num_blocks=self.depth,
        )

        self.quantizer = VectorQuantize(
            dim=self.latent_dim,
            codebook_size=self.codebook_size,
            decay=self.decay,
            commitment_weight=self.beta,
            rotation_trick=self.rot_trick,
        )
        
        self.decoder = Transformer(
            input_dim=self.latent_dim,
            output_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            num_heads=self.n_heads,
            num_blocks=self.depth,
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
        
        B, N, F = x.shape
        
        x = self.encoder(x, mask)

        z_embed = x * mask.unsqueeze(-1)
        
        z, indices, vq_loss = self.quantizer(z_embed)
        
        x_reco = self.decoder(z, mask)

        return x_reco, vq_loss, indices

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
        loss = recon_loss + 10 *commit_loss

        # Log
        self.log("test_loss", loss, prog_bar=True)
        self.log("test_recon_loss", recon_loss, prog_bar=True)
        self.log("test_commit_loss", commit_loss, prog_bar=True)
    
    # Optimizer
    def configure_optimizers(self):
        
        # Adam
        return torch.optim.Adam(self.parameters(), lr=self.lr)