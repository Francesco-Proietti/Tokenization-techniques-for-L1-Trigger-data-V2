from src.models.mlp_vqvae import MLPVQVAE
from src.models.transformer_vqvae import TransformerVQVAE

MODEL_REGISTRY = {
    "mlp": MLPVQVAE,
    "transformer": TransformerVQVAE,
}