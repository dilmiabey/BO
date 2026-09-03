"""
noise/current.py
================
Noise variance functions used in the original experiment.
All functions return rho(x) — the noise VARIANCE.
"""

import torch


def homo_noise_current(x: torch.Tensor, **kwargs) -> torch.Tensor:
    """
    Homoscedastic noise:  rho(x) = 0.01  (sigma_noise = 0.1).
    """
    return torch.full((x.shape[0],), 0.01, dtype=x.dtype)


def hetero_noise_sine_current(x: torch.Tensor, **kwargs) -> torch.Tensor:
    """
    Heteroscedastic noise for Sine:
        rho(x) = (0.05 + 0.3 |sin(x)|)^2
    High near peaks, low near zero-crossings.
    """
    std = 0.05 + 0.3 * torch.abs(torch.sin(x.squeeze(-1)))
    return std ** 2


def hetero_noise_branin_current(x: torch.Tensor, **kwargs) -> torch.Tensor:
    """
    Heteroscedastic noise for Branin (input z in [0,1]^2):
        rho(z) = (0.05 + 0.3 z1 z2)^2
    Increases toward upper-right corner.
    """
    std = 0.05 + 0.3 * x[:, 0] * x[:, 1]
    return std ** 2