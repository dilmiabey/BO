"""
optimization/candidate.py
=========================
Acquisition maximisation via exhaustive evaluation on a fixed candidate set.
Fast but coarse; resolution limited by n_candidates, especially in d >= 2.
"""

import math
import torch
import numpy as np
from typing import Callable

from .acquisition import acq_value


def candidate_grid(dim: int, lo: float, hi: float,
                   n_candidates: int = 2000) -> torch.Tensor:
    """
    Deterministic candidate set over [lo, hi]^dim.
    1-D: uniform grid.  d >= 2: scrambled Sobol.
    """
    if dim == 1:
        return torch.linspace(lo, hi, n_candidates).unsqueeze(-1)
    else:
        sobol = torch.quasirandom.SobolEngine(dimension=dim, scramble=True,
                                              seed=0)
        return sobol.draw(n_candidates).float() * (hi - lo) + lo


def optimize_candidate(predict_fn: Callable,
                       noise_fn: Callable,
                       algorithm: str,
                       beta: float,
                       dim: int,
                       lo: float,
                       hi: float,
                       n_candidates: int = 2000,
                       **kwargs) -> torch.Tensor:
    """
    Maximise the acquisition by evaluating all candidates and returning
    the best.

    Returns
    -------
    x_best : Tensor of shape (1, dim)
    """
    cands = candidate_grid(dim, lo, hi, n_candidates)

    with torch.no_grad():
        neg_acq = torch.stack([
            acq_value(cands[i:i+1], predict_fn, noise_fn, algorithm, beta)
            for i in range(cands.shape[0])
        ]).squeeze()

    best_idx = neg_acq.argmin()
    return cands[best_idx].unsqueeze(0).clone()