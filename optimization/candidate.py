"""
optimization/candidate.py
=========================
Acquisition maximisation via exhaustive evaluation on a fixed candidate set.
All candidates are evaluated in a SINGLE batched call to the predict function
and a SINGLE batched call to the noise function — critical for TabPFN speed
(1 transformer forward pass instead of n_candidates).
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
    Maximise the acquisition by evaluating all candidates in one batch
    and returning the best.
    """
    cands = candidate_grid(dim, lo, hi, n_candidates)   # (n, dim)

    with torch.no_grad():
        # Single batched call — returns negated acquisition for all candidates
        neg_acq = acq_value(cands, predict_fn, noise_fn, algorithm, beta)

    best_idx = neg_acq.argmin()
    return cands[best_idx].unsqueeze(0).clone()