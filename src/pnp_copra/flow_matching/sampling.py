"""ODE samplers for trained flow-matching pulse models."""

from __future__ import annotations

import torch

from .config import SamplingConfig


@torch.no_grad()
def euler_sample(
    model,
    source: torch.Tensor,
    trace: torch.Tensor,
    config: SamplingConfig,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Integrate the learned vector field with explicit Euler steps."""

    model.eval()
    steps = int(config.steps)
    if steps < 1:
        raise ValueError("SamplingConfig.steps must be at least 1.")
    x = source.clone()
    trajectory = [x.detach().clone()] if config.return_trajectory else None
    dt = 1.0 / float(steps)
    batch = x.shape[0]
    for index in range(steps):
        t = torch.full((batch,), index / float(steps), device=x.device, dtype=x.dtype)
        velocity = model(x, t, trace, source=source)
        x = x + dt * velocity
        if trajectory is not None:
            trajectory.append(x.detach().clone())
    if trajectory is not None:
        return x, torch.stack(trajectory, dim=1)
    return x


@torch.no_grad()
def rk4_sample(
    model,
    source: torch.Tensor,
    trace: torch.Tensor,
    config: SamplingConfig,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Integrate the learned vector field with fourth-order Runge-Kutta."""

    model.eval()
    steps = int(config.steps)
    if steps < 1:
        raise ValueError("SamplingConfig.steps must be at least 1.")
    x = source.clone()
    trajectory = [x.detach().clone()] if config.return_trajectory else None
    dt = 1.0 / float(steps)
    batch = x.shape[0]
    for index in range(steps):
        t0 = index / float(steps)
        t = torch.full((batch,), t0, device=x.device, dtype=x.dtype)
        half = torch.full((batch,), min(t0 + 0.5 * dt, 1.0), device=x.device, dtype=x.dtype)
        next_t = torch.full((batch,), min(t0 + dt, 1.0), device=x.device, dtype=x.dtype)
        k1 = model(x, t, trace, source=source)
        k2 = model(x + 0.5 * dt * k1, half, trace, source=source)
        k3 = model(x + 0.5 * dt * k2, half, trace, source=source)
        k4 = model(x + dt * k3, next_t, trace, source=source)
        x = x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        if trajectory is not None:
            trajectory.append(x.detach().clone())
    if trajectory is not None:
        return x, torch.stack(trajectory, dim=1)
    return x


def sample(
    model,
    source: torch.Tensor,
    trace: torch.Tensor,
    config: SamplingConfig,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Sample using the configured ODE integration method."""

    method = config.method.lower()
    if method == "euler":
        return euler_sample(model, source, trace, config)
    if method == "rk4":
        return rk4_sample(model, source, trace, config)
    raise ValueError("Sampling method must be 'euler' or 'rk4'.")
