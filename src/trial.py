"""
src/trial.py
============
Orchestrates a single BO trial (one seed x benchmark x noise x algorithm).

Notation
--------
  n0 : initial dataset size (warm start) or 0 (cold start)
  N  : total evaluation budget  (N = n0 + n_iter)
  n  : current dataset size     (loop variable, n0 <= n < N)
  D_n: dataset {(x_i, y_i)}_{i=1}^n

The GP uses the exact heteroscedastic noise:
  Lambda_n = diag(rho(x_1), ..., rho(x_n))
Hyperparameters (output_scale, lengthscale) are fixed throughout.
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
    benchmark, noise_type, algorithm, seed
    start_type ("warm"|"cold"), n_init
    n_iter
    noise_source
    opt_method, n_candidates
    output_scale, lengthscale    (fixed GP hyperparameters)
    prior_mean, prior_var        (cold-start first iteration)
    beta_type ("srinivas"|"constant"), beta_value, beta_delta

    Returns
    -------
    f_vals : np.ndarray of shape (n_iter,)
    """
    # ── reproducibility ────────────────────────────────────────────────────
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    rng = np.random.default_rng(cfg["seed"])

    # ── resolve components ─────────────────────────────────────────────────
    f_true       = get_benchmark_fn(cfg["benchmark"])
    noise_fn     = get_noise_fn(cfg["benchmark"], cfg["noise_type"],
                                cfg["noise_source"])
    optimize_acq = get_optimizer(cfg["opt_method"])
    dim, lo, hi  = get_domain(cfg["benchmark"])

    output_scale = cfg.get("output_scale", 1.0)
    lengthscale  = cfg.get("lengthscale",  1.0)

    beta_type    = cfg.get("beta_type",  "srinivas")
    beta_value   = cfg.get("beta_value", 2.0)
    beta_delta   = cfg.get("beta_delta", 0.1)

    n0 = cfg["n_init"] if cfg["start_type"] == "warm" else 0
    N  = n0 + cfg["n_iter"]

    # ── initial design D_{n0} ─────────────────────────────────────────────
    if n0 > 0:
        init_x_np = rng.uniform(lo, hi, size=(n0, dim)).astype(np.float32)
        train_x   = torch.from_numpy(init_x_np)
        rho_train = noise_fn(train_x)
        f_init    = f_true(train_x)
        eps_init  = torch.from_numpy(
            rng.standard_normal(n0).astype(np.float32)
        ) * torch.sqrt(rho_train)
        train_y = f_init + eps_init
    else:
        train_x   = torch.empty(0, dim)
        train_y   = torch.empty(0)
        rho_train = torch.empty(0)

    f_vals = []

    # ── BO loop:  n = n0, n0+1, ..., N-1 ──────────────────────────────────
    for n in range(n0, N):

        # 1. Build predict function from D_n
        if n == 0:
            # Cold start, first step: no data yet — use GP prior
            predict_fn = make_prior_predict_fn(
                prior_mean=cfg.get("prior_mean", 0.0),
                prior_var=cfg.get("prior_var",  output_scale),
            )
        else:
            # Fit GP with exact noise Lambda_n = diag(rho(x_1),...,rho(x_n))
            model, likelihood = fit_gp(
                train_x, train_y, rho_train,
                output_scale=output_scale,
                lengthscale=lengthscale,
            )
            predict_fn = make_gp_predict_fn(model, likelihood)

        # 2. Compute beta_n
        bo_iteration = n - n0 + 1
        if beta_type == "constant":
            beta_n = beta_value
        else:
            beta_n = beta_srinivas(bo_iteration, dim=dim, delta=beta_delta)

        # 3. Select x_{n+1} = argmax acquisition
        x_next = optimize_acq(
            predict_fn   = predict_fn,
            noise_fn     = noise_fn,
            algorithm    = cfg["algorithm"],
            beta         = beta_n,
            dim          = dim,
            lo           = lo,
            hi           = hi,
            n_candidates = cfg.get("n_candidates", 2000),
            rng          = rng,
        )  # (1, dim)

        # 4. Observe y_{n+1} = f(x_{n+1}) + eps,  eps ~ N(0, rho(x_{n+1}))
        f_next    = f_true(x_next)
        rho_next  = noise_fn(x_next)           # shape (1,)
        eps_next  = torch.from_numpy(
            rng.standard_normal(1).astype(np.float32)
        ) * torch.sqrt(rho_next)
        y_next = f_next + eps_next

        # 5. Record f(x_{n+1}) and update D_{n+1}
        f_vals.append(float(f_next.squeeze()))
        train_x   = torch.cat([train_x,   x_next],           dim=0)
        train_y   = torch.cat([train_y,   y_next.reshape(1)], dim=0)
        rho_train = torch.cat([rho_train, rho_next],          dim=0)

    return np.array(f_vals, dtype=np.float32)