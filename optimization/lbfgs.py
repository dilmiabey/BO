"""
optimization/lbfgs.py
=====================
Acquisition maximisation via L-BFGS with multiple random restarts
and sigmoid reparametrisation for box constraints.
"""

import math
import torch
import numpy as np
from typing import Callable

from .acquisition import acq_value


def optimize_lbfgs(predict_fn: Callable,
                   noise_fn: Callable,
                   algorithm: str,
                   beta: float,
                   dim: int,
                   lo: float,
                   hi: float,
                   n_restarts: int = 10,
                   lbfgs_steps: int = 100,
                   rng: np.random.Generator = None,
                   **kwargs) -> torch.Tensor:
    """
    Maximise the acquisition via L-BFGS with multi-start.

    Reparametrisation: x = lo + (hi - lo) * sigmoid(u), u in R^d.
    L-BFGS operates on the unconstrained u; sigmoid maps to [lo, hi].

    Returns
    -------
    x_best : Tensor of shape (1, dim)
    """
    if rng is None:
        rng = np.random.default_rng(0)

    x_starts = rng.uniform(lo, hi, size=(n_restarts, dim)).astype(np.float32)

    best_val = math.inf
    best_x = None

    for i in range(n_restarts):
        x0_norm = (x_starts[i] - lo) / (hi - lo)
        x0_norm = np.clip(x0_norm, 1e-4, 1 - 1e-4)
        u0 = np.log(x0_norm / (1.0 - x0_norm)).astype(np.float32)

        u = torch.tensor(u0, requires_grad=True)

        optimiser = torch.optim.LBFGS(
            [u],
            max_iter=lbfgs_steps,
            line_search_fn="strong_wolfe",
        )

        def closure():
            optimiser.zero_grad()
            x_cand = lo + (hi - lo) * torch.sigmoid(u)
            x_cand = x_cand.unsqueeze(0)
            loss = acq_value(x_cand, predict_fn, noise_fn, algorithm, beta)
            loss.backward()
            return loss

        try:
            optimiser.step(closure)
        except Exception:
            continue

        with torch.no_grad():
            x_opt = lo + (hi - lo) * torch.sigmoid(u)
            x_opt = x_opt.unsqueeze(0)
            val = acq_value(x_opt, predict_fn, noise_fn, algorithm, beta)
            val = val.item()

        if val < best_val:
            best_val = val
            best_x = x_opt.detach().clone()

    if best_x is None:
        x_fb = rng.uniform(lo, hi, size=(1, dim)).astype(np.float32)
        best_x = torch.from_numpy(x_fb)

    return best_x