"""
Data/noise/zero.py
==================
Noise-free observations: rho(x) = 0 for every input.
"""

import torch


def zero_noise(x: torch.Tensor, **kwargs) -> torch.Tensor:
    """rho(x) = 0, one value per row of x."""
    return torch.zeros(x.shape[0], dtype=x.dtype, device=x.device)
