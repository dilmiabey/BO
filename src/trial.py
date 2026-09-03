"""
src/trial.py
============
Orchestrates a single BO trial (one seed x benchmark x noise x algorithm).

Notation (matching the LaTeX writeup)
-------------------------------------
  n0    : number of initial observations (warm start) or 0 (cold start)
  N     : total evaluation budget  (N = n0 + n_iter)
  n     : current dataset size     (loop variable, n0 <= n < N)
  D_n   : dataset of n observations {(x_i, y_i)}_{i=1}^n
  x_{n+1}: next point selected by the acquisition function

The BO loop runs for n = n0, n0+1, ..., N-1.  At each step the
algorithm has D_n, selects x_{n+1}, and observes y_{n+1}.
"""

import torch
import numpy as np

from Data.benchmarks import get_benchmark_fn, get_domain
from Data.noise import get_noise_fn
from src.surrogate import fit_gp, make_gp_predict_fn, make_prior_predict_fn
from optimization.acquisition import beta_srinivas
from optimization import get_optimizer


def run_trial(cfg: dict) -> np.ndarray:
    """
    Run one BO trial.

    Expected keys in cfg
    --------------------
    benchmark, noise_type, algorithm, seed,
    start_type, n_init, n_iter,
    noise_source, reference_sigma,
    opt_method, n_restarts, lbfgs_steps, n_candidates,
    gp_train_iter, gp_lr, prior_mean, prior_var,
    beta_type ("srinivas" | "constant"),
    beta_value (float, used when beta_type == "constant"),
    beta_delta (float, used when beta_type == "srinivas")

    Returns
    -------
    f_vals : np.ndarray of shape (n_iter,)
        Noiseless f(x_{n+1}) recorded at each BO step.
    """
    # ── reproducibility ────────────────────────────────────────────────────
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    rng = np.random.default_rng(cfg["seed"])

    # ── resolve components ─────────────────────────────────────────────────
    f_true   = get_benchmark_fn(cfg["benchmark"])
    noise_fn = get_noise_fn(
        benchmark=cfg["benchmark"],
        noise_type=cfg["noise_type"],
        source=cfg["noise_source"],
        reference_sigma=cfg.get("reference_sigma", 1.0),
    )
    optimize_acq = get_optimizer(cfg["opt_method"])
    dim, lo, hi  = get_domain(cfg["benchmark"])

    # ── notation ───────────────────────────────────────────────────────────
    n0     = cfg["n_init"] if cfg["start_type"] == "warm" else 0
    n_iter = cfg["n_iter"]
    N      = n0 + n_iter          # total budget

    beta_type  = cfg.get("beta_type", "srinivas")
    beta_value = cfg.get("beta_value", 2.0)
    beta_delta = cfg.get("beta_delta", 0.1)

    # ── initial design  D_{n0} ─────────────────────────────────────────────
    if n0 > 0:
        init_x_np = rng.uniform(lo, hi,
                                size=(n0, dim)).astype(np.float32)
        train_x = torch.from_numpy(init_x_np)
        f_init  = f_true(train_x)
        rho_init = noise_fn(train_x)
        eps_init = torch.from_numpy(
            rng.standard_normal(n0).astype(np.float32)
        ) * torch.sqrt(rho_init)
        train_y = f_init + eps_init
    else:
        # Cold start: D_0 = empty
        train_x = torch.empty(0, dim)
        train_y = torch.empty(0)

    f_vals = []

    # ── BO loop:  n = n0, n0+1, ..., N-1 ──────────────────────────────────
    for n in range(n0, N):

        # 1. Build predict function from D_n
        if n == 0:
            # Cold start first step: no data, use GP prior
            predict_fn = make_prior_predict_fn(
                prior_mean=cfg.get("prior_mean", 0.0),
                prior_var=cfg.get("prior_var", 1.0),
            )
        else:
            # Fit GP on the n observations in D_n
            if n <= 1:
                noise_var_init = 0.01
            else:
                noise_var_init = float(train_y.var().clamp(min=1e-4)) * 0.1
            model, likelihood = fit_gp(
                train_x, train_y,
                n_iter=cfg["gp_train_iter"],
                lr=cfg.get("gp_lr", 0.1),
                noise_var=noise_var_init,
            )
            predict_fn = make_gp_predict_fn(model, likelihood)

        # 2. Compute exploration parameter beta_n
        bo_iteration = n - n0 + 1     # 1-indexed BO iteration count
        if beta_type == "constant":
            beta_n = beta_value
        else:
            beta_n = beta_srinivas(bo_iteration, dim=dim, delta=beta_delta)

        # 3. Select x_{n+1} by maximising the acquisition function
        x_next = optimize_acq(
            predict_fn   = predict_fn,
            noise_fn     = noise_fn,
            algorithm    = cfg["algorithm"],
            beta         = beta_n,
            dim          = dim,
            lo           = lo,
            hi           = hi,
            n_restarts   = cfg.get("n_restarts", 10),
            lbfgs_steps  = cfg.get("lbfgs_steps", 100),
            n_candidates = cfg.get("n_candidates", 2000),
            rng          = rng,
        )  # shape (1, dim)

        # 4. Observe y_{n+1} = f(x_{n+1}) + epsilon_{n+1}
        f_next   = f_true(x_next)
        rho_next = noise_fn(x_next)
        eps_next = torch.from_numpy(
            rng.standard_normal(1).astype(np.float32)
        ) * torch.sqrt(rho_next)
        y_next = f_next + eps_next

        # 5. Record noiseless value and update D_{n+1} = D_n ∪ {(x_{n+1}, y_{n+1})}
        f_vals.append(float(f_next.squeeze()))
        train_x = torch.cat([train_x, x_next], dim=0)
        train_y = torch.cat([train_y, y_next.reshape(1)], dim=0)

    return np.array(f_vals, dtype=np.float32)