"""
src/tabpfn_clt_surrogate.py
===========================
TabPFN surrogate using the Predictive CLT method to extract the posterior
mean and (epistemic) variance from the sequence of predictive updates.

Method (adapted from the Beta-Bernoulli Transformer paper):

For a query point x and training sequence D_n = {(x_i, y_i)}_{i=1}^n,
define the sequence of predictive means at x:

    g_k(x) = E_phi[Y | x, (x_{1:k}, y_{1:k})]
           = TabPFN mean prediction at x conditioned on first k observations

The predictive updates and their weighted sum are:

    Delta_k(x) = g_k(x) - g_{k-1}(x)
    V_n(x)     = (1/n) * sum_{k=S1}^n k^2 * Delta_k(x)^2

The predictive-CLT posterior at x is:

    f(x) | D_n  ~  N( g_n(x) , V_n(x) / n )

Here g_n(x) is the posterior mean and V_n(x)/n is the epistemic variance.
Both are returned by predict_fn(x) → (mu, sigma2).

Cost: n forward passes per predict_fn call (one per context prefix),
each batched across all query points.
"""

import torch
import numpy as np
from typing import Callable, Tuple

PredictFn = Callable[[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]

# TabPFN requires at least 2 training points
MIN_CONTEXT = 2
# Prior guess for g_0(x); 0 matches our GP ConstantMean default
G0_VALUE = 0.0


def make_tabpfn_clt_predict_fn(train_x: torch.Tensor,
                                train_y: torch.Tensor) -> PredictFn:
    """
    Build a predict function (x -> (mu, sigma2)) using the predictive-CLT
    method on top of TabPFN.
    """
    from tabpfn import TabPFNRegressor

    n = train_x.shape[0]
    train_x_np = train_x.detach().cpu().numpy()
    train_y_np = train_y.detach().cpu().numpy()

    def predict_fn(x: torch.Tensor):
        with torch.no_grad():
            x_np = x.detach().cpu().numpy()
            m = x_np.shape[0]

            # ── 1. Collect g_k(x) for k = 0, 1, ..., n ─────────────────────
            # g_by_k[k] has shape (m,)
            g_by_k = {0: np.full(m, G0_VALUE)}

            # For k < MIN_CONTEXT, TabPFN can't be queried; fall back to g_0
            for k in range(1, MIN_CONTEXT):
                g_by_k[k] = g_by_k[0]

            # For k >= MIN_CONTEXT, run one TabPFN forward pass per prefix
            for k in range(MIN_CONTEXT, n + 1):
                model = TabPFNRegressor(ignore_pretraining_limits=True)
                model.fit(train_x_np[:k], train_y_np[:k])
                g_by_k[k] = np.asarray(
                    model.predict(x_np, output_type="mean")
                )

            # ── 2. Compute V_n(x) = (1/n) sum_{k=1}^n k^2 (g_k - g_{k-1})^2 ─
            V_n = np.zeros(m)
            for k in range(1, n + 1):
                delta_k = g_by_k[k] - g_by_k[k - 1]
                V_n += (k ** 2) * (delta_k ** 2)
            V_n /= n

            # ── 3. Posterior:  mu = g_n,  sigma^2 = V_n / n ───────────────
            mu     = torch.as_tensor(g_by_k[n], dtype=x.dtype)
            sigma2 = torch.as_tensor(V_n / n,   dtype=x.dtype)

        return mu, sigma2

    return predict_fn