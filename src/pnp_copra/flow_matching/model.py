"""Conditional vector-field networks for pulse flow matching."""

from __future__ import annotations

import math

import torch
from torch import nn

from .config import ModelConfig


def _group_count(channels: int) -> int:
    for groups in (8, 4, 2):
        if channels % groups == 0:
            return groups
    return 1


class FourierTimeEmbedding(nn.Module):
    """Sinusoidal embedding for scalar flow time values in ``[0, 1]``."""

    def __init__(self, dim: int, max_frequency: float = 128.0):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError("FourierTimeEmbedding dim must be even.")
        frequencies = torch.exp(torch.linspace(0.0, math.log(max_frequency), dim // 2))
        self.register_buffer("frequencies", frequencies, persistent=False)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        if t.ndim == 0:
            t = t[None]
        t = t.reshape(-1, 1)
        angles = 2.0 * math.pi * t * self.frequencies.reshape(1, -1)
        return torch.cat((torch.sin(angles), torch.cos(angles)), dim=-1)


class TraceEncoder(nn.Module):
    """Compact convolutional encoder for measured FROG traces."""

    def __init__(self, condition_dim: int, base_channels: int, layers: int):
        super().__init__()
        blocks: list[nn.Module] = []
        in_channels = 1
        channels = int(base_channels)
        for _ in range(int(layers)):
            blocks.extend(
                [
                    nn.Conv2d(in_channels, channels, kernel_size=3, stride=2, padding=1),
                    nn.GroupNorm(_group_count(channels), channels),
                    nn.GELU(),
                ]
            )
            in_channels = channels
            channels = min(channels * 2, 256)
        self.encoder = nn.Sequential(*blocks)
        self.projection = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(in_channels, condition_dim),
            nn.GELU(),
            nn.Linear(condition_dim, condition_dim),
        )

    def forward(self, trace: torch.Tensor) -> torch.Tensor:
        if trace.ndim == 3:
            trace = trace.unsqueeze(1)
        if trace.ndim != 4 or trace.shape[1] != 1:
            raise ValueError(f"Expected trace shape (B, 1, N, N), got {tuple(trace.shape)}.")
        return self.projection(self.encoder(trace))


class ResidualFieldBlock(nn.Module):
    """FiLM-conditioned residual block over the time-domain field axis."""

    def __init__(self, channels: int, condition_dim: int, dropout: float):
        super().__init__()
        groups = _group_count(channels)
        self.norm1 = nn.GroupNorm(groups, channels)
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(groups, channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.condition = nn.Linear(condition_dim, 2 * channels)
        self.dropout = nn.Dropout(float(dropout)) if dropout > 0 else nn.Identity()
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        scale, shift = self.condition(condition).chunk(2, dim=-1)
        scale = scale.unsqueeze(-1)
        shift = shift.unsqueeze(-1)

        residual = x
        h = self.conv1(self.activation(self.norm1(x)))
        h = self.norm2(h)
        h = h * (1.0 + scale) + shift
        h = self.conv2(self.dropout(self.activation(h)))
        return residual + h


class FlowMatchingModel(nn.Module):
    """Trace-conditioned vector field ``v_theta(x_t, t, trace)``."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.condition_on_source = bool(config.condition_on_source)
        field_input_channels = 3 + (2 if self.condition_on_source else 0)

        self.trace_encoder = TraceEncoder(
            condition_dim=config.condition_dim,
            base_channels=config.trace_base_channels,
            layers=config.trace_layers,
        )
        self.time_encoder = nn.Sequential(
            FourierTimeEmbedding(config.time_embedding_dim),
            nn.Linear(config.time_embedding_dim, config.condition_dim),
            nn.GELU(),
            nn.Linear(config.condition_dim, config.condition_dim),
        )
        self.condition_norm = nn.LayerNorm(config.condition_dim)
        self.input_projection = nn.Conv1d(
            field_input_channels,
            config.hidden_channels,
            kernel_size=3,
            padding=1,
        )
        self.blocks = nn.ModuleList(
            [
                ResidualFieldBlock(
                    channels=config.hidden_channels,
                    condition_dim=config.condition_dim,
                    dropout=config.dropout,
                )
                for _ in range(config.residual_blocks)
            ]
        )
        self.output = nn.Sequential(
            nn.GroupNorm(_group_count(config.hidden_channels), config.hidden_channels),
            nn.GELU(),
            nn.Conv1d(config.hidden_channels, 2, kernel_size=3, padding=1),
        )

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        trace: torch.Tensor,
        source: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if x_t.ndim != 3 or x_t.shape[1] != 2:
            raise ValueError(f"Expected x_t shape (B, 2, N), got {tuple(x_t.shape)}.")
        batch, _, field_length = x_t.shape
        if field_length != int(self.config.field_length):
            raise ValueError(
                f"Model was configured for field_length={self.config.field_length}, "
                f"got {field_length}."
            )
        if t.ndim == 0:
            t = t.expand(batch)
        t = t.reshape(batch).to(device=x_t.device, dtype=x_t.dtype)
        trace = trace.to(device=x_t.device, dtype=x_t.dtype)

        position = torch.linspace(
            -1.0,
            1.0,
            field_length,
            device=x_t.device,
            dtype=x_t.dtype,
        ).reshape(1, 1, field_length)
        position = position.expand(batch, -1, -1)
        pieces = [x_t, position]
        if self.condition_on_source:
            if source is None:
                raise ValueError("source must be provided when condition_on_source=True.")
            pieces.append(source.to(device=x_t.device, dtype=x_t.dtype))
        h = self.input_projection(torch.cat(pieces, dim=1))

        condition = self.condition_norm(self.trace_encoder(trace) + self.time_encoder(t))
        for block in self.blocks:
            h = block(h, condition)
        return self.output(h)
