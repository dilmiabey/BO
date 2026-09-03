"""
benchmarks.py
=============
Benchmark objective functions and domain definitions.
"""

import math
import torch
from typing import Tuple


def sine_f(x: torch.Tensor) -> torch.Tensor:
    """f(x) = sin(x),  x in [0, 2pi].  Input: (N, 1)."""
    return torch.sin(x.squeeze(-1))


def branin_f(x: torch.Tensor) -> torch.Tensor:
    """
    Negated Branin (higher = better).
    Input x in [0,1]^2  ->  mapped to standard domain
    x1 in [-5, 10], x2 in [0, 15].
    Input: (N, 2).
    """
    x1 = x[:, 0] * 15.0 - 5.0
    x2 = x[:, 1] * 15.0

    a = 1.0
    b = 5.1 / (4.0 * math.pi ** 2)
    c = 5.0 / math.pi
    r = 6.0
    s = 10.0
    t = 1.0 / (8.0 * math.pi)

    result = (a * (x2 - b * x1**2 + c * x1 - r)**2
              + s * (1 - t) * torch.cos(x1) + s)
    return -result


def get_domain(benchmark: str) -> Tuple[int, float, float]:
    """Returns (dim, lo, hi)."""
    if benchmark == "sine":
        return 1, 0.0, 2.0 * math.pi
    elif benchmark == "branin":
        return 2, 0.0, 1.0
    raise ValueError(f"Unknown benchmark: {benchmark}")


def get_benchmark_fn(benchmark: str):
    """Factory: returns the objective function."""
    if benchmark == "sine":
        return sine_f
    elif benchmark == "branin":
        return branin_f
    raise ValueError(f"Unknown benchmark: {benchmark}")