"""Flow-matching objectives."""

from __future__ import annotations

import torch

from .config import FlowLossConfig


def sample_flow_times(batch_size: int, *, device: torch.device, config: FlowLossConfig) -> torch.Tensor:
    """Sample flow times while avoiding exactly 0 and 1."""

    eps = float(config.time_eps)
    return eps + (1.0 - 2.0 * eps) * torch.rand(batch_size, device=device)


def interpolate_straight_path(
    source: torch.Tensor,
    target: torch.Tensor,
    t: torch.Tensor,
    *,
    noise_std: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Interpolate between source and target and return ``(x_t, velocity)``."""

    shape = (source.shape[0],) + (1,) * (source.ndim - 1)
    t_view = t.reshape(shape)
    velocity = target - source
    x_t = (1.0 - t_view) * source + t_view * target
    if noise_std > 0.0:
        bridge_scale = torch.sqrt(torch.clamp(t_view * (1.0 - t_view), min=0.0))
        x_t = x_t + float(noise_std) * bridge_scale * torch.randn_like(x_t)
    return x_t, velocity


def weighted_channel_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor | None,
) -> torch.Tensor:
    """Per-sample channel MSE with optional source weights."""

    per_sample = (prediction - target).square().flatten(1).mean(dim=1)
    if weight is None:
        return per_sample.mean()
    weight = weight.to(device=per_sample.device, dtype=per_sample.dtype).reshape(-1)
    weight = weight / torch.clamp(weight.mean(), min=1e-6)
    return (per_sample * weight).mean()


def flow_matching_loss(
    model,
    batch: dict[str, torch.Tensor],
    config: FlowLossConfig,
) -> dict[str, torch.Tensor]:
    """Compute the straight-line conditional flow-matching loss."""

    source = batch["source"]
    target = batch["target"]
    trace = batch["trace"]
    weight = batch.get("weight")

    t = sample_flow_times(source.shape[0], device=source.device, config=config)
    x_t, velocity = interpolate_straight_path(
        source,
        target,
        t,
        noise_std=config.path_noise_std,
    )
    prediction = model(x_t, t, trace, source=source)
    loss = weighted_channel_mse(prediction, velocity, weight)
    unweighted = weighted_channel_mse(prediction, velocity, None)
    return {
        "loss": loss,
        "velocity_mse": unweighted.detach(),
    }
