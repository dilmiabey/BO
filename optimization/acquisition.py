"""
optimization/acquisition.py
============================
Acquisition function values and exploration parameter schedules.

acq_value() takes a generic predict_fn(x) -> (mu, sigma2) so it works
with either a fitted GP or a cold-start prior.  Returns NEGATED value
(for minimisation by L-BFGS).
"""

import math
import torch
from typing import Callable


def beta_srinivas(iteration: int, dim: int, delta: float = 0.1) -> float:
    """
    Srinivas (2010) exploration parameter schedule:
        beta = 2 log(d * iteration^2 * pi^2 / (6 * delta))

    Parameters
    ----------
    iteration : 1-indexed BO iteration count (i.e., n - n0 + 1)
    dim       : input dimension d
    delta     : confidence parameter (default 0.1)
    """
    return max(2.0 * math.log(dim * iteration**2 * math.pi**2
                              / (6.0 * delta)), 0.1)


def acq_value(x: torch.Tensor,
              predict_fn: Callable,
              noise_fn: Callable,
              algorithm: str,
              beta: float) -> torch.Tensor:
    """
    Compute the NEGATED acquisition at x of shape (1, dim).
    Negated because L-BFGS minimises.

    predict_fn(x) -> (mu, sigma2)   # from GP posterior or prior
    noise_fn(x)   -> rho            # aleatoric noise variance

    GP-UCB:     alpha = mu_n + sqrt(beta_n) * sigma_n
    GP-UCB-PV:  alpha = mu_n + sqrt(beta_n) * sqrt(sigma_n^2 + rho)
    """
    mu, sigma2 = predict_fn(x)
    sqrt_beta = math.sqrt(beta)

    if algorithm == "gpucb":
        acq = mu + sqrt_beta * torch.sqrt(sigma2.clamp(min=1e-12))
    else:  # gpucb_pv
        rho = noise_fn(x)
        total_var = (sigma2 + rho).clamp(min=1e-12)
        acq = mu + sqrt_beta * torch.sqrt(total_var)

    return -acq