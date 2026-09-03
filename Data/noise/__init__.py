"""
Data/noise/
===========
Noise variance functions.  Use get_noise_fn() to obtain the correct
function for a given (benchmark, noise_type, source) combination.

Reference noise (Makarova et al.) supports heteroscedastic only.
"""

from .current import (
    homo_noise_current,
    hetero_noise_sine_current,
    hetero_noise_branin_current,
)
from .reference import (
    hetero_noise_sine_reference,
    hetero_noise_branin_reference,
)

_REGISTRY = {
    # ── current noise models ────────────────────────────────────────────
    ("current", "sine",   "homo"):   homo_noise_current,
    ("current", "sine",   "hetero"): hetero_noise_sine_current,
    ("current", "branin", "homo"):   homo_noise_current,
    ("current", "branin", "hetero"): hetero_noise_branin_current,
    # ── reference noise models (heteroscedastic only) ───────────────────
    ("reference", "sine",   "hetero"): hetero_noise_sine_reference,
    ("reference", "branin", "hetero"): hetero_noise_branin_reference,
}


def get_noise_fn(benchmark: str, noise_type: str,
                 source: str = "current", **kwargs):
    """
    Factory: returns a noise variance function  rho(x) -> Tensor.

    Raises KeyError if the combination is not supported
    (e.g. reference + homoscedastic).
    """
    key = (source, benchmark, noise_type)
    if key not in _REGISTRY:
        raise KeyError(
            f"Unsupported noise config: source={source}, "
            f"benchmark={benchmark}, noise_type={noise_type}. "
            f"Reference noise only supports heteroscedastic."
        )
    return _REGISTRY[key]