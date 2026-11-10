"""
optimization/acquisition.py
============================
Acquisition function values and exploration parameter schedules.

acq_value() takes a generic predict_fn(x) -> (mu, sigma2) so it works
with either a fitted GP or a cold-start prior.  Supports GP-UCB(-PV) and
expected improvement EI(-PV).  Returns NEGATED value
(for minimisation by L-BFGS).
"""

import math
import torch
from typing import Callable, Optional


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
              beta: float,
              incumbent_x: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Compute the NEGATED acquisition at x of shape (m, dim).
    Negated because L-BFGS minimises.

    predict_fn(x) -> (mu, sigma2)   # from GP posterior or prior
    noise_fn(x)   -> rho            # aleatoric noise variance

    GP-UCB:        alpha = mu_n + sqrt(beta_n) * sigma_n
    GP-UCB-PV:     alpha = mu_n + sqrt(beta_n) * sqrt(sigma_n^2 + rho)
    GP-UCB-PV-PPD: alpha = mu_n + sqrt(beta_n) * sqrt(sigma2)
                   where predict_fn already returns the total predictive
                   variance sigma2 ~ sigma_n^2 + rho (e.g. TabPFN's PPD),
                   so rho is not added again.

    Expected improvement (Wang & de Freitas, 2014), beta is ignored:
        alpha = s(x) phi(z) + (mu_n(x) - mu_n(x^+)) Phi(z),
        z     = (mu_n(x) - mu_n(x^+)) / s(x),
        x^+   = argmax_{x_i in D_n} mu_n(x_i)   (incumbent = best predictive
                                                 mean over observed inputs)
    EI:        s(x) = sigma_n(x)                  (epistemic only)
    EI-PV:     s(x) = sqrt(sigma_n^2(x) + rho(x)) (epistemic + aleatoric)
    EI-PV-PPD: s(x) = sqrt(sigma2(x)), sigma2 already the total predictive
               variance (TabPFN PPD), so rho is not added again.

    incumbent_x : (n, dim) observed inputs used to compute mu_n(x^+).  They
                  are appended to x so the surrogate is queried in a SINGLE
                  call (important for the expensive TabPFN-CLT surrogate).
                  If None/empty (cold start), mu_n(x^+) = max_x mu_n(x).
    """
    if algorithm in _EI_ALGORITHMS:
        return -_expected_improvement(x, predict_fn, noise_fn, algorithm,
                                      incumbent_x)

    mu, sigma2 = predict_fn(x)
    sqrt_beta = math.sqrt(beta)

    if algorithm in ("gpucb", "gpucb_pv_ppd"):
        acq = mu + sqrt_beta * torch.sqrt(sigma2.clamp(min=1e-12))
    elif algorithm == "gpucb_pv":
        rho = noise_fn(x)
        total_var = (sigma2 + rho).clamp(min=1e-12)
        acq = mu + sqrt_beta * torch.sqrt(total_var)
    else:
        raise ValueError(f"Unknown acquisition: {algorithm}")

    return -acq


_EI_ALGORITHMS = {"ei", "ei_pv", "ei_pv_ppd"}


def _expected_improvement(x: torch.Tensor,
                          predict_fn: Callable,
                          noise_fn: Callable,
                          algorithm: str,
                          incumbent_x: Optional[torch.Tensor]) -> torch.Tensor:
    m = x.shape[0]
    has_inc = incumbent_x is not None and incumbent_x.shape[0] > 0
    x_all = torch.cat([x, incumbent_x.to(x.dtype)], dim=0) if has_inc else x

    mu_all, sigma2_all = predict_fn(x_all)
    mu, sigma2 = mu_all[:m], sigma2_all[:m]

    # Incumbent mu_n(x^+): a constant w.r.t. x (no gradient through it)
    if has_inc:
        mu_best = mu_all[m:].max().detach()
    else:
        mu_best = mu.max().detach()

    if algorithm == "ei_pv":
        var = sigma2 + noise_fn(x)
    else:  # "ei" (epistemic) or "ei_pv_ppd" (PPD variance is already total)
        var = sigma2
    s = torch.sqrt(var.clamp(min=1e-12))

    diff = mu - mu_best
    z    = diff / s
    std_normal = torch.distributions.Normal(
        torch.zeros((), dtype=x.dtype), torch.ones((), dtype=x.dtype))
    pdf = torch.exp(std_normal.log_prob(z))
    cdf = std_normal.cdf(z)
    return s * pdf + diff * cdf
