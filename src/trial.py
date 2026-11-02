"""
src/trial.py
============
Orchestrates a single BO trial (one seed x benchmark x noise x algorithm).

Supported algorithms:
  - "gpucb"            : GP surrogate,              GP-UCB    acquisition
  - "gpucb_pv"         : GP surrogate,              GP-UCB-PV acquisition
  - "gpucb_pv_tabpfn"  : TabPFN surrogate (PPD),    GP-UCB-PV acquisition
  - "gpucb_tabpfn_clt" : TabPFN surrogate (CLT),    GP-UCB    acquisition
"""

import time
import torch
import numpy as np

from Data.benchmarks import get_benchmark_fn, get_domain
from Data.noise import get_noise_fn
from src.surrogate            import fit_gp, make_gp_predict_fn, make_prior_predict_fn
from src.tabpfn_surrogate     import make_tabpfn_predict_fn
from src.tabpfn_clt_surrogate import make_tabpfn_clt_predict_fn
from optimization.acquisition import beta_srinivas
from optimization             import get_optimizer


_PV_ALGORITHMS         = {"gpucb_pv", "gpucb_pv_tabpfn"}
_TABPFN_PPD_ALGORITHMS = {"gpucb_pv_tabpfn"}
_TABPFN_CLT_ALGORITHMS = {"gpucb_tabpfn_clt"}
_TABPFN_MIN_POINTS     = 2


def _acq_name(algorithm: str) -> str:
    """gpucb_pv variants use PV acquisition; others use plain GP-UCB."""
    return "gpucb_pv" if algorithm in _PV_ALGORITHMS else "gpucb"


def run_trial(cfg: dict, log_every: int = 5) -> np.ndarray:
    """
    Run one BO trial.  Prints progress every `log_every` iterations.
    """
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    rng = np.random.default_rng(cfg["seed"])

    f_true       = get_benchmark_fn(cfg["benchmark"])
    noise_fn     = get_noise_fn(cfg["benchmark"], cfg["noise_type"],
                                cfg["noise_source"])
    optimize_acq = get_optimizer(cfg["opt_method"])
    dim, lo, hi  = get_domain(cfg["benchmark"])

    output_scale = cfg.get("output_scale", 1.0)
    lengthscale  = cfg.get("lengthscale",  1.0)

    standardize_y = cfg.get("standardize_y", False)
    grid_seed     = cfg["seed"] if cfg.get("seed_grid", False) else None
    random_ties   = cfg.get("random_ties", False)

    clt_n_estimators = cfg.get("clt_n_estimators", 1)
    clt_n_perms      = cfg.get("clt_n_perms",      1)
    clt_start_k      = cfg.get("clt_start_k",      _TABPFN_MIN_POINTS + 1)

    beta_type    = cfg.get("beta_type",  "srinivas")
    beta_value   = cfg.get("beta_value", 2.0)
    beta_delta   = cfg.get("beta_delta", 0.1)

    algorithm        = cfg["algorithm"]
    use_tabpfn_ppd   = algorithm in _TABPFN_PPD_ALGORITHMS
    use_tabpfn_clt   = algorithm in _TABPFN_CLT_ALGORITHMS
    needs_min_points = use_tabpfn_ppd or use_tabpfn_clt
    acq_name         = _acq_name(algorithm)

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

    f_vals  = []
    t_start = time.time()

    for n in range(n0, N):
        iter_t0 = time.time()
        bo_iteration = n - n0 + 1

        # 1. Build predict function
        if (n == 0 or (needs_min_points and n < _TABPFN_MIN_POINTS)
                or (use_tabpfn_clt and n < clt_start_k)):
            predict_fn = make_prior_predict_fn(
                prior_mean=cfg.get("prior_mean", 0.0),
                prior_var=cfg.get("prior_var",  output_scale),
            )
        elif use_tabpfn_clt:
            predict_fn = make_tabpfn_clt_predict_fn(
                train_x, train_y,
                n_estimators=clt_n_estimators,
                n_perms=clt_n_perms,
                start_k=clt_start_k,
                perm_seed=cfg["seed"] * 100003 + n,
            )
        elif use_tabpfn_ppd:
            predict_fn = make_tabpfn_predict_fn(train_x, train_y)
        else:
            model, likelihood = fit_gp(
                train_x, train_y, rho_train,
                output_scale=output_scale, lengthscale=lengthscale,
                standardize_y=standardize_y,
            )
            predict_fn = make_gp_predict_fn(model, likelihood)

        # 2. beta_n
        if beta_type == "constant":
            beta_n = beta_value
        else:
            beta_n = beta_srinivas(bo_iteration, dim=dim, delta=beta_delta)

        # 3. x_{n+1} = argmax acquisition
        x_next = optimize_acq(
            predict_fn=predict_fn, noise_fn=noise_fn,
            algorithm=acq_name, beta=beta_n,
            dim=dim, lo=lo, hi=hi,
            n_candidates=cfg.get("n_candidates", 2000),
            grid_seed=grid_seed, random_ties=random_ties,
            rng=rng,
        )

        # 4. Observe
        f_next   = f_true(x_next)
        rho_next = noise_fn(x_next)
        eps_next = torch.from_numpy(
            rng.standard_normal(1).astype(np.float32)
        ) * torch.sqrt(rho_next)
        y_next = f_next + eps_next

        # 5. Record and update
        f_vals.append(float(f_next.squeeze()))
        train_x   = torch.cat([train_x,   x_next],            dim=0)
        train_y   = torch.cat([train_y,   y_next.reshape(1)], dim=0)
        rho_train = torch.cat([rho_train, rho_next],          dim=0)

        iter_elapsed  = time.time() - iter_t0
        total_elapsed = time.time() - t_start

        if bo_iteration == 1 or bo_iteration % log_every == 0 or n == N - 1:
            print(f"    iter {bo_iteration:3d}/{N-n0}  "
                  f"f(x)={float(f_next.squeeze()):+.4f}  "
                  f"best={max(f_vals):+.4f}  "
                  f"iter_time={iter_elapsed:.1f}s  "
                  f"total={total_elapsed:.1f}s",
                  flush=True)

    return np.array(f_vals, dtype=np.float32)