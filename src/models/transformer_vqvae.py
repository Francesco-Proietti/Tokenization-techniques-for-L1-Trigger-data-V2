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


# ============================================================
# NormFormer Block
# ============================================================

class NormFormerBlock(nn.Module):

    def __init__(
        self,
        dim: int = 128,
        num_heads: int = 8,
        mlp_ratio: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()

        # --------------------------------------------------
        # Attention
        # --------------------------------------------------

        self.norm1 = nn.LayerNorm(dim)

        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        self.post_attn_norm = nn.LayerNorm(dim)

        # --------------------------------------------------
        # FeedForward
        # --------------------------------------------------

        self.norm2 = nn.LayerNorm(dim)

        hidden_dim = dim * mlp_ratio

        self.ff = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim)
        )

        self.post_ff_norm = nn.LayerNorm(dim)

    def forward(self, x, mask):

        # --------------------------------------------------
        # Attention block
        # --------------------------------------------------

        residual = x

        x_norm = self.norm1(x)

        key_padding_mask = ~mask

        attn_out, _ = self.attn(
            x_norm,
            x_norm,
            x_norm,
            key_padding_mask=key_padding_mask,
            need_weights=False
        )

        attn_out = self.post_attn_norm(attn_out)

        x = residual + attn_out

        # --------------------------------------------------
        # FeedForward block
        # --------------------------------------------------

        residual = x

        x_norm = self.norm2(x)

        ff_out = self.ff(x_norm)

        ff_out = self.post_ff_norm(ff_out)

        x = residual + ff_out

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
        self.embedding_dim = cfg.embedding_dim
        self.n_heads = cfg.n_heads
        self.mlp_ratio = cfg.mlp_ratio
        self.dropout = cfg.dropout
        self.depth = cfg.depth
        self.latent_dim = cfg.latent_dim
        self.codebook_size = cfg.codebook_size
        self.n_layers = cfg.n_layers
        self.decay = cfg.decay
        self.beta = cfg.beta
        self.rot_trick = cfg.rotation_trick
        self.lr = lr
        
        # Input projection
        self.input_proj = nn.Linear(self.input_dim, self.embedding_dim)

        # Encoder
        self.encoder = nn.ModuleList([
            NormFormerBlock(
                dim=self.embedding_dim, 
                num_heads=self.n_heads, 
                mlp_ratio=self.mlp_ratio,
                dropout=self.dropout
            )
            for _ in range(self.depth)              
        ])

        # Pre-VQ projection
        self.to_quant = nn.Linear(self.embedding_dim, self.latent_dim)

        self.quantizer = VectorQuantize(
            dim=self.latent_dim,
            codebook_size=self.codebook_size,
            decay=self.decay,
            commitment_weight=self.beta,
            rotation_trick=self.rot_trick
        )
        
        # Post-VQ projection
        self.from_quant = nn.Linear(self.latent_dim, self.embedding_dim)
        
        # Decoder 
        self.decoder = nn.ModuleList([
            NormFormerBlock(
                dim=self.embedding_dim,
                num_heads=self.n_heads,
                mlp_ratio=self.mlp_ratio,
                dropout=self.dropout
            )
            for _ in range(self.depth)
        ])
        
        # Output projection
        self.output_proj = nn.Linear(self.embedding_dim, self.input_dim)

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
        
        # Input projection
        x = self.input_proj(x)

        # Encode
        for block in self.encoder:
            x = block(x, mask)
        
        # Pre-VQ projection
        z_e = self.to_quant(x) 
        
        B, N, D = z_e.size()

        z_e_flat = z_e.view(B*N, D)
        mask_flat = mask.view(B*N)
        
        z_e_valid = z_e_flat[mask_flat]
        
        # Quantize
        z_q, indices, vq_loss = self.quantizer(z_e_valid)

        z_q_flat = torch.zeros(B*N, D, device=x.device)
        z_q_flat[mask_flat] = z_q

        z_q = z_q_flat.view(B, N, D)
        
        # Post-VQ projection
        x = self.from_quant(z_q)

        # Decode
        for block in self.decoder:
            x = block(x, mask)

        # Final output projection
        x_recon = self.output_proj(x)

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