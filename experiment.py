"""
experiment.py
=============
Core module implementing:
  - Benchmark functions (Sine, Branin)
  - Noise models (homoscedastic, heteroscedastic)
  - GP surrogate (GPyTorch, exact GP)
  - GP-UCB acquisition (standard predictive-mean variance)
  - GP-UCB-PV acquisition (predictive variance = model + noise)
  - run_experiment() : runs one seed × benchmark × noise × algorithm trial
"""

from __future__ import annotations

import math
import torch
import gpytorch
import numpy as np
from dataclasses import dataclass
from typing import Literal, Callable

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------
BenchmarkName = Literal["sine", "branin"]
NoiseType = Literal["homo", "hetero"]
AlgorithmName = Literal["gpucb", "gpucb_pv"]


# ---------------------------------------------------------------------------
# Benchmark functions
# ---------------------------------------------------------------------------

def sine_f(x: torch.Tensor) -> torch.Tensor:
    """f(x) = sin(x),  x in [0, 2π]"""
    return torch.sin(x.squeeze(-1))


def branin_f(x: torch.Tensor) -> torch.Tensor:
    """
    Branin function (negated for maximisation convention, shifted to ~[0,1]).
    Inputs: x in [0,1]^2  mapped to standard Branin domain.
    Standard domain: x1 in [-5,10], x2 in [0,15].
    """
    x1 = x[:, 0] * 15.0 - 5.0
    x2 = x[:, 1] * 15.0

    a = 1.0
    b = 5.1 / (4.0 * math.pi ** 2)
    c = 5.0 / math.pi
    r = 6.0
    s = 10.0
    t = 1.0 / (8.0 * math.pi)

    result = a * (x2 - b * x1 ** 2 + c * x1 - r) ** 2 + s * (1 - t) * torch.cos(x1) + s
    return -result   # negate: higher = better


# ---------------------------------------------------------------------------
# Noise variance functions  ρ(x)
# ---------------------------------------------------------------------------

def homo_noise(x: torch.Tensor, sigma: float = 0.1) -> torch.Tensor:
    """Constant noise variance: ρ(x) = σ²"""
    return torch.full((x.shape[0],), sigma ** 2)


def hetero_noise_sine(x: torch.Tensor) -> torch.Tensor:
    """
    Heteroscedastic noise for Sine benchmark.
    ρ(x) = (0.05 + 0.3 * |sin(x)|)²
    Variance is low near zero-crossings and high near peaks.
    """
    std = 0.05 + 0.3 * torch.abs(torch.sin(x.squeeze(-1)))
    return std ** 2


def hetero_noise_branin(x: torch.Tensor) -> torch.Tensor:
    """
    Heteroscedastic noise for Branin benchmark.
    ρ(x) = (0.05 + 0.3 * x1 * x2)²   where x1, x2 in [0,1]
    Variance increases toward the upper-right corner.
    """
    std = 0.05 + 0.3 * x[:, 0] * x[:, 1]
    return std ** 2


# ---------------------------------------------------------------------------
# Domain helpers
# ---------------------------------------------------------------------------

def get_domain(benchmark: BenchmarkName):
    """Returns (dim, bounds_low, bounds_high) as floats."""
    if benchmark == "sine":
        return 1, 0.0, 2 * math.pi
    elif benchmark == "branin":
        return 2, 0.0, 1.0
    raise ValueError(benchmark)


def candidate_grid(benchmark: BenchmarkName, n_cand: int = 2000, generator=None) -> torch.Tensor:
    """
    Return a fixed candidate set for the discrete argmax of the acquisition.
    Uses a reproducible Sobol/grid draw over the domain.
    """
    dim, lo, hi = get_domain(benchmark)
    if dim == 1:
        cands = torch.linspace(lo, hi, n_cand).unsqueeze(-1)
    else:
        # Sobol sequence for 2-D (deterministic, no generator needed)
        sobol = torch.quasirandom.SobolEngine(dimension=dim, scramble=True, seed=0)
        cands = sobol.draw(n_cand).float() * (hi - lo) + lo
    return cands


# ---------------------------------------------------------------------------
# GP model (exact, RBF + noise)
# ---------------------------------------------------------------------------

class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel()
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def fit_gp(train_x: torch.Tensor, train_y: torch.Tensor,
           n_iter: int = 50, noise_var: float = 0.01) -> tuple:
    """Fit an exact GP by marginal likelihood optimisation."""
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    likelihood.noise = noise_var
    model = ExactGPModel(train_x, train_y, likelihood)

    model.train()
    likelihood.train()

    optimiser = torch.optim.Adam(model.parameters(), lr=0.1)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    for _ in range(n_iter):
        optimiser.zero_grad()
        output = model(train_x)
        loss = -mll(output, train_y)
        loss.backward()
        optimiser.step()

    model.eval()
    likelihood.eval()
    return model, likelihood


def gp_predict(model, likelihood, x_test: torch.Tensor):
    """Return (mean, variance) at x_test."""
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        pred = likelihood(model(x_test))
    return pred.mean, pred.variance


# ---------------------------------------------------------------------------
# Beta schedule  (GP-UCB, Srinivas 2010 heuristic)
# ---------------------------------------------------------------------------

def beta_t(t: int, dim: int, delta: float = 0.1) -> float:
    """β_t = 2 * log(dim * t² * π² / (6*δ))"""
    return max(2.0 * math.log(dim * (t ** 2) * (math.pi ** 2) / (6.0 * delta)), 0.1)


# ---------------------------------------------------------------------------
# Acquisition functions
# ---------------------------------------------------------------------------

def acq_gpucb(mu: torch.Tensor, sigma2: torch.Tensor, beta: float) -> torch.Tensor:
    """
    Standard GP-UCB:
        α(x) = μ(x) + √β · σ(x)
    σ is the *predictive* std (model uncertainty only, no noise).
    """
    return mu + math.sqrt(beta) * torch.sqrt(sigma2.clamp(min=1e-12))


def acq_gpucb_pv(mu: torch.Tensor, sigma2: torch.Tensor,
                 rho: torch.Tensor, beta: float) -> torch.Tensor:
    """
    GP-UCB under Predictive Variance:
        α(x) = μ(x) + √β · √(σ²(x) + ρ(x))
    Combines model uncertainty and noise variance.
    """
    total_var = (sigma2 + rho).clamp(min=1e-12)
    return mu + math.sqrt(beta) * torch.sqrt(total_var)


# ---------------------------------------------------------------------------
# Single trial runner
# ---------------------------------------------------------------------------

@dataclass
class TrialConfig:
    benchmark: BenchmarkName
    noise_type: NoiseType
    algorithm: AlgorithmName
    seed: int
    n_init: int = 5
    n_iter: int = 50
    n_cand: int = 2000
    gp_train_iter: int = 50


def run_trial(cfg: TrialConfig) -> np.ndarray:
    """
    Run one BO trial.

    Returns
    -------
    f_vals : np.ndarray of shape (n_iter,)
        The true f(x_t) value at each selected point x_t.
    """
    # ── reproducibility ────────────────────────────────────────────────────
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)

    # ── setup ──────────────────────────────────────────────────────────────
    dim, lo, hi = get_domain(cfg.benchmark)

    # Function and noise
    if cfg.benchmark == "sine":
        f_true: Callable = sine_f
        noise_fn: Callable = hetero_noise_sine if cfg.noise_type == "hetero" else homo_noise
    else:
        f_true = branin_f
        noise_fn = hetero_noise_branin if cfg.noise_type == "hetero" else homo_noise

    # Candidate set (fixed, deterministic)
    cands = candidate_grid(cfg.benchmark, n_cand=cfg.n_cand)  # (n_cand, dim)

    # ── initial design (LHS-like random) ───────────────────────────────────
    init_x_np = rng.uniform(lo, hi, size=(cfg.n_init, dim)).astype(np.float32)
    train_x = torch.from_numpy(init_x_np)  # (n_init, dim)

    # Observe noisy y = f(x) + ε
    f_init = f_true(train_x)
    rho_init = noise_fn(train_x)
    eps_init = torch.from_numpy(
        rng.standard_normal(cfg.n_init).astype(np.float32)
    ) * torch.sqrt(rho_init)
    train_y = f_init + eps_init

    f_vals = []

    # ── BO loop ────────────────────────────────────────────────────────────
    for t in range(1, cfg.n_iter + 1):
        # Fit GP
        noise_var_init = float(train_y.var().clamp(min=1e-4)) * 0.1
        model, likelihood = fit_gp(
            train_x, train_y,
            n_iter=cfg.gp_train_iter,
            noise_var=noise_var_init
        )

        # Predict on candidates
        mu, sigma2 = gp_predict(model, likelihood, cands)

        # Compute beta
        beta = beta_t(t, dim=dim)

        # Acquisition
        if cfg.algorithm == "gpucb":
            acq = acq_gpucb(mu, sigma2, beta)
        else:  # gpucb_pv
            rho_cands = noise_fn(cands)
            acq = acq_gpucb_pv(mu, sigma2, rho_cands, beta)

        # Select next point
        best_idx = acq.argmax()
        x_next = cands[best_idx].unsqueeze(0)  # (1, dim)

        # Observe
        f_next = f_true(x_next)
        rho_next = noise_fn(x_next)
        eps_next = torch.from_numpy(
            rng.standard_normal(1).astype(np.float32)
        ) * torch.sqrt(rho_next)
        y_next = f_next + eps_next

        # Record the *true* f value (no noise)
        f_vals.append(float(f_next.squeeze()))

        # Update dataset
        train_x = torch.cat([train_x, x_next], dim=0)
        train_y = torch.cat([train_y, y_next.reshape(1)], dim=0)

    return np.array(f_vals, dtype=np.float32)