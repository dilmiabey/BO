"""
src/tabpfn_clt_surrogate.py
===========================
TabPFN surrogate using the Predictive CLT method.

For one ordering pi of D_n:
    g_k(x)      = TabPFN mean prediction at x after conditioning on the
                  first k points of D_n under ordering pi
    Delta_k(x)  = g_k(x) - g_{k-1}(x)
    V_n(x)      = (1/n) * sum_{k=K1}^n k^2 * Delta_k(x)^2

Averaged over P orderings pi_1..pi_P:
    V_bar_n(x)  = (1/P) * sum_p V_n^{pi_p}(x)
    Posterior:  f(x) | D_n  ~  N( g_n(x), V_bar_n(x) / n )

Choices
-------
1. Start index K1 (`start_k`)
   TabPFN needs >= 2 context points, so g_0, g_1 have no TabPFN value.
   Rather than inventing g_0 (e.g. g_0 = 0, which is far from Branin's
   ~ -50 mean), the sum starts at K1 >= MIN_CONTEXT + 1, so every term is a
   genuine TabPFN-to-TabPFN update.  Dropping a fixed, finite number of
   leading terms does not change the asymptotic CLT limit, since their
   contribution (1/n) * sum_{k<K1} k^2 Delta_k^2 -> 0 as n -> inf.

2. Averaging over orderings (`n_perms`)
   A single ordering puts the BO points last, where the k^2 weight is
   largest, so V_n is dominated by the most recent acquisitions.  Averaging
   V_n over random permutations of D_n removes this order dependence and
   reduces the variance of the estimator.  g_n(x) uses all n points, so it
   is the same for every ordering and is computed once.
   n_perms = 1 keeps the original data order (the previous behaviour).
   If an ordering starts with repeated x's (BO often re-selects the same
   candidate), that ordering's sum starts at the first prefix TabPFN can fit.
   The permutations are drawn once when the predict function is built, so
   predict_fn is a deterministic function of x (needed by any acquisition
   optimiser, candidate-based or continuous).

3. `n_estimators`
   Number of TabPFN ensemble members per forward pass.  More members average
   out TabPFN's preprocessing jitter, which would otherwise leak into
   Delta_k as spurious "uncertainty".  Cost scales roughly linearly.

4. TabPFN wrappers are reused across fit() calls (one per n_estimators
   value).  fit() only stores the context; the transformer weights are fixed.
"""

import torch
import numpy as np
from typing import Callable, Optional, Tuple

PredictFn = Callable[[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]

MIN_CONTEXT = 2        # TabPFN requires >= 2 training points

# ── Module-level TabPFN wrappers, keyed by n_estimators ───────────────────
_MODELS = {}


def _get_model(n_estimators: int):
    if n_estimators not in _MODELS:
        from tabpfn import TabPFNRegressor
        _MODELS[n_estimators] = TabPFNRegressor(
            ignore_pretraining_limits=True,
            n_estimators=n_estimators,
            random_state=0,
        )
    return _MODELS[n_estimators]


def _clt_V_n(model, x_np: np.ndarray, X: np.ndarray, Y: np.ndarray,
             g_n: np.ndarray, start_k: int) -> np.ndarray:
    """
    V_n(x) = (1/n) sum_{k=start_k}^n k^2 (g_k - g_{k-1})^2 for one ordering
    of (X, Y).  g_n is passed in (it does not depend on the ordering).
    """
    n = X.shape[0]
    V_n = np.zeros(x_np.shape[0])

    # TabPFN rejects a context whose features are all constant, which happens
    # when an ordering puts repeated BO points (same x) first.  Start this
    # ordering's sum at the first k whose prefix X[:k-1] has a varying
    # feature; dropping finitely many leading terms leaves the CLT limit
    # unchanged (same argument as for K1).
    while start_k <= n and not np.any(np.ptp(X[:start_k - 1], axis=0) > 0):
        start_k += 1
    if start_k > n:
        return V_n

    model.fit(X[:start_k - 1], Y[:start_k - 1])
    g_prev = np.asarray(model.predict(x_np, output_type="mean"))

    for k in range(start_k, n + 1):
        if k == n:
            g_k = g_n
        else:
            model.fit(X[:k], Y[:k])
            g_k = np.asarray(model.predict(x_np, output_type="mean"))
        V_n += (k ** 2) * (g_k - g_prev) ** 2
        g_prev = g_k

    return V_n / n


def make_tabpfn_clt_predict_fn(train_x: torch.Tensor,
                                train_y: torch.Tensor,
                                n_estimators: int = 1,
                                n_perms: int = 1,
                                start_k: int = MIN_CONTEXT + 1,
                                perm_seed: Optional[int] = None) -> PredictFn:
    """
    Build a predict function (x -> (mu, sigma2)) using the predictive-CLT
    method on top of TabPFN, with V_n averaged over `n_perms` orderings.
    """
    n = train_x.shape[0]
    if start_k < MIN_CONTEXT + 1:
        raise ValueError(f"start_k must be >= {MIN_CONTEXT + 1} "
                         f"(g_{{K1-1}} needs >= {MIN_CONTEXT} points)")
    if n < start_k:
        raise ValueError(f"Need n >= start_k ({start_k}), got n = {n}")

    train_x_np = train_x.detach().cpu().numpy()
    train_y_np = train_y.detach().cpu().numpy()

    # Orderings are fixed here so predict_fn is deterministic in x
    if n_perms == 1:
        perms = [np.arange(n)]
    else:
        perm_rng = np.random.default_rng(perm_seed)
        perms = [perm_rng.permutation(n) for _ in range(n_perms)]

    def predict_fn(x: torch.Tensor):
        with torch.no_grad():
            x_np = x.detach().cpu().numpy()
            model = _get_model(n_estimators)

            # ── 1. Posterior mean g_n(x): same for every ordering ─────────
            model.fit(train_x_np, train_y_np)
            g_n = np.asarray(model.predict(x_np, output_type="mean"))

            # ── 2. V_bar_n(x): average V_n over orderings ────────────────
            V_bar = np.zeros(x_np.shape[0])
            for perm in perms:
                V_bar += _clt_V_n(model, x_np,
                                  train_x_np[perm], train_y_np[perm],
                                  g_n, start_k)
            V_bar /= len(perms)

            # ── 3. Posterior:  mu = g_n,  sigma^2 = V_bar_n / n ──────────
            mu     = torch.as_tensor(g_n,       dtype=x.dtype)
            sigma2 = torch.as_tensor(V_bar / n, dtype=x.dtype)

        return mu, sigma2

    return predict_fn
