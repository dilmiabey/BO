"""
experiment.py
=============
Core module implementing:
  - Benchmark functions (Sine, Branin)
  - Noise models (homoscedastic, heteroscedastic)
  - GP surrogate (GPyTorch, exact GP)
  - GP-UCB acquisition (standard predictive variance)
  - GP-UCB-PV acquisition (predictive variance = model + noise)
  - optimize_acq_lbfgs(): gradient-based acquisition optimisation (L-BFGS-B,
    multi-start, sigmoid reparametrisation for box constraints)
  - run_trial(): runs one seed x benchmark x noise x algorithm trial
"""

from __future__ import annotations

import math
import torch
import gpytorch
import numpy as np
from dataclasses import dataclass, field
from typing import Literal, Callable, Tuple

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------
BenchmarkName = Literal["sine", "branin"]
NoiseType     = Literal["homo", "hetero"]
AlgorithmName = Literal["gpucb", "gpucb_pv"]


# ---------------------------------------------------------------------------
# Benchmark functions
# ---------------------------------------------------------------------------

def sine_f(x: torch.Tensor) -> torch.Tensor:
    """f(x) = sin(x),  x in [0, 2π].  Input shape: (N, 1)."""
    return torch.sin(x.squeeze(-1))


def branin_f(x: torch.Tensor) -> torch.Tensor:
    """
    Negated Branin (higher = better).
    Input x in [0,1]^2, mapped to standard Branin domain
    x1 in [-5, 10], x2 in [0, 15].
    Input shape: (N, 2).
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
    return -result   # negate: higher = better


# ---------------------------------------------------------------------------
# Noise variance functions  ρ(x)
# ---------------------------------------------------------------------------

def homo_noise(x: torch.Tensor, sigma: float = 0.1) -> torch.Tensor:
    """Constant noise variance: ρ(x) = σ²."""
    return torch.full((x.shape[0],), sigma ** 2, dtype=x.dtype)


def hetero_noise_sine(x: torch.Tensor) -> torch.Tensor:
    """
    Heteroscedastic noise for Sine.
    ρ(x) = (0.05 + 0.3|sin(x)|)²  — low near zero-crossings, high at peaks.
    """
    std = 0.05 + 0.3 * torch.abs(torch.sin(x.squeeze(-1)))
    return std ** 2


def hetero_noise_branin(x: torch.Tensor) -> torch.Tensor:
    """
    Heteroscedastic noise for Branin.
    ρ(x) = (0.05 + 0.3 x₁ x₂)²  — increases toward upper-right corner.
    """
    std = 0.05 + 0.3 * x[:, 0] * x[:, 1]
    return std ** 2


# ---------------------------------------------------------------------------
# Domain helpers
# ---------------------------------------------------------------------------

def get_domain(benchmark: BenchmarkName) -> Tuple[int, float, float]:
    """Returns (dim, bounds_low, bounds_high)."""
    if benchmark == "sine":
        return 1, 0.0, 2.0 * math.pi
    elif benchmark == "branin":
        return 2, 0.0, 1.0
    raise ValueError(f"Unknown benchmark: {benchmark}")


# ---------------------------------------------------------------------------
# GP model  (exact, RBF kernel)
# ---------------------------------------------------------------------------

class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module  = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel()
        )

    def forward(self, x):
        mean_x  = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def fit_gp(train_x: torch.Tensor, train_y: torch.Tensor,
           n_iter: int = 50, noise_var: float = 0.01):
    """Fit an exact GP via marginal-likelihood optimisation (Adam)."""
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
        loss   = -mll(output, train_y)
        loss.backward()
        optimiser.step()

    model.eval()
    likelihood.eval()
    return model, likelihood


# ---------------------------------------------------------------------------
# Beta schedule  (Srinivas 2010 heuristic)
# ---------------------------------------------------------------------------

def beta_t(t: int, dim: int, delta: float = 0.1) -> float:
    """β_t = 2 log(d · t² · π² / (6δ))."""
    return max(2.0 * math.log(dim * t**2 * math.pi**2 / (6.0 * delta)), 0.1)


# ---------------------------------------------------------------------------
# Acquisition value  (differentiable, operates on a single point)
# ---------------------------------------------------------------------------

def acq_value(x: torch.Tensor,
              model: ExactGPModel,
              likelihood,
              noise_fn: Callable,
              algorithm: AlgorithmName,
              beta: float) -> torch.Tensor:
    """
    Compute the (negated) acquisition at a single point x of shape (1, dim).

    We negate because L-BFGS-B minimises, so we minimise −α(x).

    Gradients flow through the GP posterior: we use gpytorch.settings.fast_pred_var()
    but do NOT wrap in torch.no_grad(), so autograd can differentiate through
    the predictive mean and variance.
    """
    with gpytorch.settings.fast_pred_var():
        pred   = likelihood(model(x))
        mu     = pred.mean          # scalar
        sigma2 = pred.variance      # scalar, model uncertainty only

    sqrt_beta = math.sqrt(beta)

    if algorithm == "gpucb":
        # α(x) = μ(x) + √β · σ(x)
        acq = mu + sqrt_beta * torch.sqrt(sigma2.clamp(min=1e-12))
    else:
        # α(x) = μ(x) + √β · √(σ²(x) + ρ(x))
        rho        = noise_fn(x)    # shape (1,)
        total_var  = (sigma2 + rho).clamp(min=1e-12)
        acq        = mu + sqrt_beta * torch.sqrt(total_var)

    return -acq   # negate for minimisation


# ---------------------------------------------------------------------------
# L-BFGS-B acquisition optimisation  (multi-start, sigmoid reparametrisation)
# ---------------------------------------------------------------------------

def optimize_acq_lbfgs(model: ExactGPModel,
                       likelihood,
                       noise_fn: Callable,
                       algorithm: AlgorithmName,
                       beta: float,
                       dim: int,
                       lo: float,
                       hi: float,
                       n_restarts: int = 10,
                       lbfgs_steps: int = 100,
                       rng: np.random.Generator = None) -> torch.Tensor:
    """
    Maximise the acquisition function via L-BFGS with multiple random restarts.

    Strategy
    --------
    We reparametrise x = lo + (hi - lo) · sigmoid(u) so that u is unconstrained
    and we let L-BFGS operate freely in ℝᵈ.  This avoids the need to project
    back onto [lo, hi] at every step and gives smooth gradients at the boundary.

    Parameters
    ----------
    n_restarts  : number of random starting points
    lbfgs_steps : max L-BFGS iterations per restart

    Returns
    -------
    x_best : torch.Tensor of shape (1, dim) — best point found
    """
    if rng is None:
        rng = np.random.default_rng(0)

    # Draw random starting points in [lo, hi]
    x_starts = rng.uniform(lo, hi, size=(n_restarts, dim)).astype(np.float32)

    best_val  = math.inf
    best_x    = None

    for i in range(n_restarts):
        # Initialise unconstrained parameter u such that sigmoid(u) maps to
        # the normalised starting point in [0,1], then scaled to [lo,hi].
        x0_norm = (x_starts[i] - lo) / (hi - lo)               # in (0,1)
        x0_norm = np.clip(x0_norm, 1e-4, 1 - 1e-4)             # keep away from boundary
        u0      = np.log(x0_norm / (1.0 - x0_norm)).astype(np.float32)  # logit

        u = torch.tensor(u0, requires_grad=True)

        optimiser = torch.optim.LBFGS(
            [u],
            max_iter=lbfgs_steps,
            line_search_fn="strong_wolfe",
        )

        def closure():
            optimiser.zero_grad()
            # Reparametrise: x in [lo, hi]
            x_cand = lo + (hi - lo) * torch.sigmoid(u)   # (dim,)
            x_cand = x_cand.unsqueeze(0)                  # (1, dim)
            loss   = acq_value(x_cand, model, likelihood,
                               noise_fn, algorithm, beta)
            loss.backward()
            return loss

        try:
            optimiser.step(closure)
        except Exception:
            # Occasionally L-BFGS diverges; skip this restart
            continue

        with torch.no_grad():
            x_opt  = lo + (hi - lo) * torch.sigmoid(u)
            x_opt  = x_opt.unsqueeze(0)                   # (1, dim)
            val    = acq_value(x_opt, model, likelihood,
                               noise_fn, algorithm, beta)
            val    = val.item()

        if val < best_val:
            best_val = val
            best_x   = x_opt.detach().clone()

    if best_x is None:
        # Fallback: return a random point (should be extremely rare)
        x_fb   = rng.uniform(lo, hi, size=(1, dim)).astype(np.float32)
        best_x = torch.from_numpy(x_fb)

    return best_x   # shape (1, dim)


# ---------------------------------------------------------------------------
# Trial configuration
# ---------------------------------------------------------------------------

@dataclass
class TrialConfig:
    benchmark:      BenchmarkName
    noise_type:     NoiseType
    algorithm:      AlgorithmName
    seed:           int
    n_init:         int = 5
    n_iter:         int = 50
    n_restarts:     int = 10    # L-BFGS random restarts per BO iteration
    lbfgs_steps:    int = 100   # max L-BFGS steps per restart
    gp_train_iter:  int = 50


# ---------------------------------------------------------------------------
# Single trial runner
# ---------------------------------------------------------------------------

def run_trial(cfg: TrialConfig) -> np.ndarray:
    """
    Run one BO trial.

    Returns
    -------
    f_vals : np.ndarray of shape (n_iter,)
        The true (noiseless) f(x_t) at each selected point x_t.
    """
    # ── reproducibility ────────────────────────────────────────────────────
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)

    # ── setup ──────────────────────────────────────────────────────────────
    dim, lo, hi = get_domain(cfg.benchmark)

    if cfg.benchmark == "sine":
        f_true:   Callable = sine_f
        noise_fn: Callable = hetero_noise_sine if cfg.noise_type == "hetero" else homo_noise
    else:
        f_true   = branin_f
        noise_fn = hetero_noise_branin if cfg.noise_type == "hetero" else homo_noise

    # ── initial random design ───────────────────────────────────────────────
    init_x_np = rng.uniform(lo, hi, size=(cfg.n_init, dim)).astype(np.float32)
    train_x   = torch.from_numpy(init_x_np)

    f_init    = f_true(train_x)
    rho_init  = noise_fn(train_x)
    eps_init  = torch.from_numpy(
        rng.standard_normal(cfg.n_init).astype(np.float32)
    ) * torch.sqrt(rho_init)
    train_y   = f_init + eps_init

    f_vals = []

    # ── Bayesian optimisation loop ──────────────────────────────────────────
    for t in range(1, cfg.n_iter + 1):

        # 1. Fit GP
        noise_var_init = float(train_y.var().clamp(min=1e-4)) * 0.1
        model, likelihood = fit_gp(
            train_x, train_y,
            n_iter=cfg.gp_train_iter,
            noise_var=noise_var_init,
        )

        # 2. Maximise acquisition via L-BFGS (multi-start)
        beta = beta_t(t, dim=dim)

        x_next = optimize_acq_lbfgs(
            model       = model,
            likelihood  = likelihood,
            noise_fn    = noise_fn,
            algorithm   = cfg.algorithm,
            beta        = beta,
            dim         = dim,
            lo          = lo,
            hi          = hi,
            n_restarts  = cfg.n_restarts,
            lbfgs_steps = cfg.lbfgs_steps,
            rng         = rng,
        )   # shape (1, dim)

        # 3. Observe
        f_next   = f_true(x_next)
        rho_next = noise_fn(x_next)
        eps_next = torch.from_numpy(
            rng.standard_normal(1).astype(np.float32)
        ) * torch.sqrt(rho_next)
        y_next   = f_next + eps_next

        # 4. Record noiseless value and update dataset
        f_vals.append(float(f_next.squeeze()))

        train_x = torch.cat([train_x, x_next],        dim=0)
        train_y = torch.cat([train_y, y_next.reshape(1)], dim=0)

    return np.array(f_vals, dtype=np.float32)