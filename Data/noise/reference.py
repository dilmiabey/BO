"""
Data/noise/reference.py
=======================
Noise variance functions from:
    Makarova, Usmanova, Bogunovic, Krause.
    "Risk-averse Heteroscedastic Bayesian Optimization." NeurIPS 2021.

Only heteroscedastic noise is defined (the reference studies
input-dependent noise).  All functions return rho(x) — noise VARIANCE.

Sine:   rho(x)       = sigmoid(30(x - 1)) + 0.1
Branin: rho(x0, x1)  = [sigmoid(2(x0 - 3.2)) + 0.2] * [sigmoid(2 x1) + 0.2]
        where x0, x1 are in the original Branin domain.
"""

import torch


def hetero_noise_sine_reference(x: torch.Tensor, **kwargs) -> torch.Tensor:
    """rho(x) = sigmoid(30(x - 1)) + 0.1,   x in [0, 2pi]."""
    return torch.sigmoid(30.0 * (x.squeeze(-1) - 1.0)) + 0.1


def hetero_noise_branin_reference(x: torch.Tensor, **kwargs) -> torch.Tensor:
    """
    rho(z) = [sigmoid(2(x0 - 3.2)) + 0.2] * [sigmoid(2 x1) + 0.2]
    Input z in [0,1]^2, un-normalised to original Branin domain:
        x0 = 15 z1 - 5   (in [-5, 10])
        x1 = 15 z2        (in [ 0, 15])
    """
    x0 = x[:, 0] * 15.0 - 5.0
    x1 = x[:, 1] * 15.0
    var0 = torch.sigmoid(2.0 * (x0 - 3.2)) + 0.2
    var1 = torch.sigmoid(2.0 * x1) + 0.2
    return var0 * var1