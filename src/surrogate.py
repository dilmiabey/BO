"""
src/surrogate.py
================
Gaussian process model (exact GP, RBF kernel) via GPyTorch.

Hyperparameters (output_scale, lengthscale) are fixed — not learned.
The observation noise covariance Lambda_n = diag(rho(x_1),...,rho(x_n))
is passed in directly, so the GP uses the exact heteroscedastic noise.

Optional y-standardisation (standardize_y=True): the GP is fit on
(y - m) / s with noise rho / s^2, and predictions are mapped back to the
original units (mu * s + m, sigma2 * s^2).  The objective and noise are
unchanged; only the fixed prior (mean 0, output_scale) is applied on a
unit scale instead of the raw one (Branin values reach ~ -300).
"""

import torch
import gpytorch
from typing import Callable, Tuple

PredictFn = Callable[[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]


class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel()
        )

    def forward(self, x):
        return gpytorch.distributions.MultivariateNormal(
            self.mean_module(x), self.covar_module(x)
        )


def fit_gp(train_x: torch.Tensor,
           train_y: torch.Tensor,
           rho_train: torch.Tensor,
           output_scale: float = 1.0,
           lengthscale: float = 1.0,
           standardize_y: bool = False):
    """
    Build an exact GP with fixed hyperparameters and exact noise covariance.

    Parameters
    ----------
    train_x      : (n, d) input points
    train_y      : (n,)   noisy observations
    rho_train    : (n,)   exact noise variances rho(x_i) at training points
                          used to set Lambda_n = diag(rho(x_1),...,rho(x_n))
    output_scale : sigma_f^2  (fixed)
    lengthscale  : ell        (fixed)
    standardize_y: fit on standardised y (see module docstring)
    """
    y_mean, y_std = 0.0, 1.0
    if standardize_y and train_y.shape[0] >= 2:
        y_mean = float(train_y.mean())
        y_std  = float(train_y.std()) if float(train_y.std()) > 1e-8 else 1.0
    train_y   = (train_y - y_mean) / y_std
    rho_train = rho_train / y_std ** 2

    # GPyTorch's FixedNoiseGaussianLikelihood accepts per-point noise variances
    likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(
        noise=rho_train,
        learn_additional_noise=False,
    )

    model = ExactGPModel(train_x, train_y, likelihood)

    # Set fixed hyperparameters
    model.covar_module.outputscale = output_scale
    model.covar_module.base_kernel.lengthscale = lengthscale

    # Stored so the predict function can map back to original units
    model.y_mean, model.y_std = y_mean, y_std

    model.eval()
    likelihood.eval()
    return model, likelihood


def make_gp_predict_fn(model, likelihood) -> PredictFn:
    """
    Returns a differentiable predict function (x -> (mu, sigma2)).

    sigma2 is the LATENT (epistemic) posterior variance of f(x), without
    observation noise; GP-UCB-PV adds rho(x) itself in acq_value().
    model(x) is used rather than likelihood(model(x)): the fixed-noise
    likelihood would add the TRAINING noise vector whenever x happens to
    have the same number of rows as the training set.
    """
    def predict_fn(x: torch.Tensor):
        with gpytorch.settings.fast_pred_var():
            pred = model(x)
        return (pred.mean * model.y_std + model.y_mean,
                pred.variance * model.y_std ** 2)
    return predict_fn


def make_prior_predict_fn(prior_mean: float = 0.0,
                          prior_var: float = 1.0) -> PredictFn:
    """Constant prior for cold-start first iteration (no data yet)."""
    def predict_fn(x: torch.Tensor):
        n = x.shape[0]
        mu     = torch.full((n,), prior_mean, dtype=x.dtype)
        sigma2 = torch.full((n,), prior_var,  dtype=x.dtype)
        return mu, sigma2
    return predict_fn