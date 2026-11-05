import torch
from torch import nn


class BasalMLP(nn.Module):
    """Encoder of the basal cell state: control expression x_c [B, G] -> z_c [B, d].

    Args:
        input_dim: Number of input genes G.
        latent_dim: Output dimension d.
        hidden_dim: Hidden size.
        dropout: Dropout rate.
    """

    def __init__(self, input_dim: int, latent_dim: int, hidden_dim: int = 512, dropout: float = 0.2):
        super().__init__()
        self.encoder_z = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
        )
        self.out_nn = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out_nn(self.encoder_z(x))                          # [B, d]
