"""
optimization/candidate.py
=========================
Acquisition maximisation via exhaustive evaluation on a fixed candidate set.
All candidates are evaluated in a SINGLE batched call to the predict function
and a SINGLE batched call to the noise function — critical for TabPFN speed
(1 transformer forward pass instead of n_candidates).

Options (both off by default, which reproduces the original behaviour):
  grid_seed    : seed for the candidate set, so different trial seeds see
                 different candidates (fixed within a trial).
  random_ties  : break exact ties in the acquisition uniformly at random
                 instead of always taking the lowest index.
"""

import math
import torch
import numpy as np
from typing import Callable, Optional

from .acquisition import acq_value


def candidate_grid(dim: int, lo: float, hi: float,
                   n_candidates: int = 2000,
                   seed: Optional[int] = None) -> torch.Tensor:
    """
    Candidate set over [lo, hi]^dim.
    1-D: uniform grid; with a seed, the grid is shifted by a random offset
         within one grid spacing.
    d >= 2: scrambled Sobol (Sobol seed = `seed`, or 0 if None).
    """
    if dim == 1:
        if seed is None:
            return torch.linspace(lo, hi, n_candidates).unsqueeze(-1)
        step   = (hi - lo) / n_candidates
        offset = np.random.default_rng(seed).uniform(0.0, step)
        grid   = lo + offset + step * torch.arange(n_candidates,
                                                   dtype=torch.float32)
        return grid.unsqueeze(-1)
    else:
        sobol = torch.quasirandom.SobolEngine(dimension=dim, scramble=True,
                                              seed=0 if seed is None else seed)
        return sobol.draw(n_candidates).float() * (hi - lo) + lo


def optimize_candidate(predict_fn: Callable,
                       noise_fn: Callable,
                       algorithm: str,
                       beta: float,
                       dim: int,
                       lo: float,
                       hi: float,
                       n_candidates: int = 2000,
                       grid_seed: Optional[int] = None,
                       random_ties: bool = False,
                       rng: np.random.Generator = None,
                       incumbent_x: Optional[torch.Tensor] = None,
                       **kwargs) -> torch.Tensor:
    """
    Maximise the acquisition by evaluating all candidates in one batch
    and returning the best.
    """
    cands = candidate_grid(dim, lo, hi, n_candidates, seed=grid_seed)

    with torch.no_grad():
        # Single batched call — returns negated acquisition for all candidates
        neg_acq = acq_value(cands, predict_fn, noise_fn, algorithm, beta,
                            incumbent_x=incumbent_x)

    if random_ties:
        # One rng draw every iteration (even with a single best candidate),
        # so all algorithms consume the trial rng identically
        ties = torch.nonzero(neg_acq == neg_acq.min()).squeeze(-1)
        best_idx = ties[int(rng.integers(len(ties)))]
    else:
        best_idx = neg_acq.argmin()
    return cands[best_idx].unsqueeze(0).clone()