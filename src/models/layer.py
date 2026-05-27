"""Reusable layers used by the recommender model."""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding (Vaswani et al., 2017).

    Adds a fixed position-dependent bias to each timestep so a Transformer
    can distinguish order. Buffered (not a parameter), so it travels with
    `state_dict` but isn't trained.
    """

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))  # [1, max_len, d_model]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, d_model]
        return x + self.pe[:, : x.size(1)]


def build_mlp(input_dim: int,
              hidden_dims: list[int],
              dropout: float = 0.0,
              output_dim: int | None = None,
              activation: type[nn.Module] = nn.ReLU) -> nn.Sequential:
    """Build a vanilla MLP. Final layer (output_dim) is a plain Linear
    without activation/norm, so it can output logits."""
    layers: list[nn.Module] = []
    prev = input_dim
    for h in hidden_dims:
        layers.extend([nn.Linear(prev, h), activation(), nn.BatchNorm1d(h), nn.Dropout(dropout)])
        prev = h
    if output_dim is not None:
        layers.append(nn.Linear(prev, output_dim))
    return nn.Sequential(*layers)
