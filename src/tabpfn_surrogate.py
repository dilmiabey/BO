"""
src/tabpfn_surrogate.py
=======================
TabPFN surrogate using the posterior predictive distribution (PPD) directly.

Exposes the same interface as surrogate.py:
    predict_fn(x) -> (mu, sigma2)

TabPFN outputs P(y | x, D_n) as a full-support bar distribution over K bins
with borders b_0 < ... < b_K and probabilities p_i(x): uniform density inside
the inner bins and half-normal tails on the two outer bins.  The moments are
taken exactly from this distribution (TabPFN's criterion.mean / .variance):

    mu(x)      = E[y | x, D_n]
    sigma^2(x) = Var[y | x, D_n]

This is a predictive distribution of the noisy observation y, so sigma^2(x)
is the TOTAL predictive variance, epistemic + aleatoric:

    sigma^2(x)  ~=  sigma_n^2(x) + rho(x)

It is therefore used in GP-UCB-PV directly as the total variance; rho(x) is
NOT added again in the acquisition (see "gpucb_pv_ppd" in acquisition.py).

The TabPFN wrapper is reused across fit() calls (one per n_estimators value);
fit() only stores the context, the transformer weights are fixed.
"""

import torch
import numpy as np
from typing import Callable, Tuple

PredictFn = Callable[[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]

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


def make_tabpfn_predict_fn(train_x: torch.Tensor,
                           train_y: torch.Tensor,
                           n_estimators: int = 1) -> PredictFn:
    """
    Fit TabPFN on (train_x, train_y) and return a predict function
    x -> (mu, sigma2), where sigma2 is the total PPD variance.
    """
    model = _get_model(n_estimators)
    model.fit(train_x.detach().cpu().numpy(),
              train_y.detach().cpu().numpy())

    def predict_fn(x: torch.Tensor):
        with torch.no_grad():
            full = model.predict(x.detach().cpu().numpy(), output_type="full")
            # criterion: bar distribution in raw y units; logits: (n, K).
            # Logits go to the criterion's device (its borders are used
            # directly in variance()).
            criterion = full["criterion"]
            logits    = full["logits"].detach().float().to(
                criterion.borders.device)
            mu_t      = criterion.mean(logits)
            sigma2_t  = criterion.variance(logits).clamp(min=0.0)

            mu     = mu_t.cpu().to(dtype=x.dtype)
            sigma2 = sigma2_t.cpu().to(dtype=x.dtype)
        return mu, sigma2

    return predict_fn
