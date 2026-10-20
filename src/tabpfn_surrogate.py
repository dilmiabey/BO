"""
src/tabpfn_surrogate.py
=======================
TabPFN surrogate model — a pre-trained transformer for tabular regression.

Exposes the same interface as surrogate.py:
    predict_fn(x) -> (mu, sigma2)

TabPFN outputs a discrete probability distribution P(y|x,D) over K bins
with midpoints m_i and softmax probabilities p_i.  The posterior mean and
variance are computed directly from this discrete distribution:

    mu(x)      = sum_i p_i(x) * m_i
    sigma^2(x) = sum_i p_i(x) * (m_i - mu(x))^2

This is exact for any shape of distribution (no Gaussian assumption).
If the TabPFN version does not expose the raw bin probabilities, we fall
back to a quantile-based approximation.
"""

import torch
import numpy as np
from typing import Callable, Tuple

PredictFn = Callable[[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]


def _mu_sigma2_from_full(full_output) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute mu and sigma^2 directly from TabPFN's discrete posterior.

    Expects `full_output` to be a dict or object exposing:
      - probs or logits over K bins (shape: (n, K))
      - bin midpoints or borders (shape: (K,) or (K+1,))
    """
    # Extract probabilities (softmax of logits if needed)
    if isinstance(full_output, dict):
        if "probs" in full_output:
            probs = np.asarray(full_output["probs"])              # (n, K)
        elif "logits" in full_output:
            logits = np.asarray(full_output["logits"])            # (n, K)
            # softmax
            logits = logits - logits.max(axis=-1, keepdims=True)
            exp_l = np.exp(logits)
            probs = exp_l / exp_l.sum(axis=-1, keepdims=True)
        else:
            raise KeyError("No 'probs' or 'logits' in full output")

        # Extract bin midpoints
        if "midpoints" in full_output:
            midpoints = np.asarray(full_output["midpoints"])      # (K,)
        elif "borders" in full_output:
            borders = np.asarray(full_output["borders"])          # (K+1,)
            midpoints = 0.5 * (borders[:-1] + borders[1:])
        else:
            raise KeyError("No 'midpoints' or 'borders' in full output")
    else:
        raise TypeError(f"Unsupported full output type: {type(full_output)}")

    # Direct discrete-distribution moments
    mu     = (probs * midpoints[None, :]).sum(axis=-1)                     # (n,)
    sigma2 = (probs * (midpoints[None, :] - mu[:, None])**2).sum(axis=-1)  # (n,)
    return mu, sigma2


def _mu_sigma2_from_quantiles(model, x_np) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fallback: estimate mu and sigma^2 from many quantiles of the posterior.
    Non-parametric — approximates the moments by averaging over the
    inverse CDF evaluated at a dense quantile grid.
    """
    q_grid = np.linspace(0.01, 0.99, 99)
    q = model.predict(x_np, output_type="quantiles", quantiles=q_grid.tolist())
    q = np.asarray(q)
    if q.ndim == 2 and q.shape[0] == len(q_grid):
        q = q.T                                   # (n, K)
    mu     = q.mean(axis=1)                       # (n,)
    sigma2 = ((q - mu[:, None]) ** 2).mean(axis=1)
    return mu, sigma2


def make_tabpfn_predict_fn(train_x: torch.Tensor,
                           train_y: torch.Tensor) -> PredictFn:
    """
    Fit a TabPFN regressor on (train_x, train_y) and return a predict function
    with the same signature as make_gp_predict_fn: x -> (mu, sigma2).
    """
    from tabpfn import TabPFNRegressor

    model = TabPFNRegressor(ignore_pretraining_limits=True)
    model.fit(train_x.detach().cpu().numpy(),
              train_y.detach().cpu().numpy())

    def predict_fn(x: torch.Tensor):
        with torch.no_grad():
            x_np = x.detach().cpu().numpy()

            # Try the exact bin-based computation first
            try:
                full = model.predict(x_np, output_type="full")
                mu_np, sigma2_np = _mu_sigma2_from_full(full)
            except Exception:
                # Fallback: non-parametric estimate from quantiles
                mu_np, sigma2_np = _mu_sigma2_from_quantiles(model, x_np)

            mu     = torch.as_tensor(mu_np,     dtype=x.dtype)
            sigma2 = torch.as_tensor(sigma2_np, dtype=x.dtype)
        return mu, sigma2

    return predict_fn