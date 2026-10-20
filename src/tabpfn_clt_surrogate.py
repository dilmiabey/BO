"""
src/tabpfn_clt_surrogate.py
===========================
TabPFN surrogate using the Predictive CLT method.

    g_k(x)      = TabPFN mean prediction at x after conditioning on D_k
    Delta_k(x)  = g_k(x) - g_{k-1}(x)
    V_n(x)      = (1/n) * sum_{k=1}^n k^2 * Delta_k(x)^2
    Posterior:  f(x) | D_n  ~  N( g_n(x), V_n(x) / n )

Performance notes
-----------------
1. `n_estimators=1`
   TabPFN's default is to ensemble over several data-permutation forward
   passes and average.  We set n_estimators=1 to run one forward pass per
   predict call: ~4x faster, slightly noisier predictions.

2. Single TabPFN wrapper reused across all fit() calls
   TabPFN's pre-trained transformer weights never change — they're loaded
   once from the cache file and used as a fixed feature extractor.  fit()
   just stores the context (X, y); it does not train anything.  So we
   instantiate the regressor ONCE at module level, and reuse it across
   every fit()/predict() call in every BO iteration of every trial.
   This avoids repeatedly re-allocating the Python wrapper and (on some
   TabPFN versions) re-attaching weights to GPU.
"""

import torch
import numpy as np
from typing import Callable, Tuple

PredictFn = Callable[[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]

MIN_CONTEXT = 2        # TabPFN requires >= 2 training points
G0_VALUE    = 0.0      # g_0(x), matches GP ConstantMean = 0

# ── Module-level TabPFN singleton ─────────────────────────────────────────
# Created lazily on first use; reused for the lifetime of the Python process.
# TabPFN has no persistent state between fit() calls — each fit replaces the
# context data outright — so reuse is semantically identical to a fresh
# instantiation, just much faster.
_MODEL = None


def _get_model():
    global _MODEL
    if _MODEL is None:
        from tabpfn import TabPFNRegressor
        _MODEL = TabPFNRegressor(
            ignore_pretraining_limits=True,
            n_estimators=1,
        )
    return _MODEL


def make_tabpfn_clt_predict_fn(train_x: torch.Tensor,
                                train_y: torch.Tensor) -> PredictFn:
    """
    Build a predict function (x -> (mu, sigma2)) using the predictive-CLT
    method on top of TabPFN.
    """
    n = train_x.shape[0]
    train_x_np = train_x.detach().cpu().numpy()
    train_y_np = train_y.detach().cpu().numpy()

    def predict_fn(x: torch.Tensor):
        with torch.no_grad():
            x_np = x.detach().cpu().numpy()
            m = x_np.shape[0]

            # ── 1. Collect g_k(x) for k = 0, 1, ..., n ─────────────────────
            g_by_k = {0: np.full(m, G0_VALUE)}
            for k in range(1, MIN_CONTEXT):
                g_by_k[k] = g_by_k[0]

            # Reuse the single TabPFN wrapper across all fit() calls
            model = _get_model()
            for k in range(MIN_CONTEXT, n + 1):
                model.fit(train_x_np[:k], train_y_np[:k])
                g_by_k[k] = np.asarray(
                    model.predict(x_np, output_type="mean")
                )

            # ── 2. V_n(x) = (1/n) sum_{k=1}^n k^2 (g_k - g_{k-1})^2 ─────
            V_n = np.zeros(m)
            for k in range(1, n + 1):
                delta_k = g_by_k[k] - g_by_k[k - 1]
                V_n += (k ** 2) * (delta_k ** 2)
            V_n /= n

            # ── 3. Posterior:  mu = g_n,  sigma^2 = V_n / n ─────────────
            mu     = torch.as_tensor(g_by_k[n], dtype=x.dtype)
            sigma2 = torch.as_tensor(V_n / n,   dtype=x.dtype)

        return mu, sigma2

    return predict_fn