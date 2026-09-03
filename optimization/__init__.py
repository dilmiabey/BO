"""
optimization/
=============
Acquisition function computation and optimisation methods.
"""

from .candidate import optimize_candidate
from .lbfgs import optimize_lbfgs


_METHODS = {
    "candidate": optimize_candidate,
    "lbfgs":     optimize_lbfgs,
}


def get_optimizer(method: str):
    """Returns the optimiser function for the given method name."""
    if method not in _METHODS:
        raise ValueError(f"Unknown optimisation method: {method}. "
                         f"Choose from {list(_METHODS.keys())}")
    return _METHODS[method]