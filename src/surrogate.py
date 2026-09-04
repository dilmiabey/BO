"""
src/surrogate.py
================
Gaussian process model (exact GP, RBF kernel) via GPyTorch.

Supports two modes:
  - learn_hyperparams=True:  optimise sigma_f^2, ell, noise via MLL (Adam)
  - learn_hyperparams=False: use fixed hyperparameters (default)
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


def fit_gp(train_x: torch.Tensor, train_y: torch.Tensor,
           learn_hyperparams: bool = False,
           n_iter: int = 50, lr: float = 0.1,
           output_scale: float = 1.0,
           lengthscale: float = 1.0,
           noise_var: float = 0.1):
    """
    Build and (optionally) train an exact GP.

    Parameters
    ----------
    learn_hyperparams : if False (default), set fixed hyperparameters
                        and skip optimisation.
                        if True, optimise via marginal likelihood.
    output_scale      : sigma_f^2  (used as fixed value, or as initialisation)
    lengthscale       : ell        (used as fixed value, or as initialisation)
    noise_var         : likelihood noise  (used as fixed value, or as initialisation)
    """
    likelihood = gpytorch.likelihoods.GaussianLikelihood()

    # Set hyperparameters
    likelihood.noise = noise_var
    model = ExactGPModel(train_x, train_y, likelihood)
    model.covar_module.outputscale = output_scale
    model.covar_module.base_kernel.lengthscale = lengthscale

    if learn_hyperparams:
        model.train()
        likelihood.train()
        optimiser = torch.optim.Adam(model.parameters(), lr=lr)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
        for _ in range(n_iter):
            optimiser.zero_grad()
            output = model(train_x)
            loss = -mll(output, train_y)
            loss.backward()
            optimiser.step()

    model.eval()
    likelihood.eval()
    return model, likelihood


def make_gp_predict_fn(model, likelihood) -> PredictFn:
    """
    Returns a differentiable predict function (x -> (mu, sigma2)).
    """
    def predict_fn(x: torch.Tensor):
        with gpytorch.settings.fast_pred_var():
            pred = likelihood(model(x))
        return pred.mean, pred.variance
    return predict_fn


def make_prior_predict_fn(prior_mean: float = 0.0,
                          prior_var: float = 1.0) -> PredictFn:
    """
    Constant prior predict function for cold-start first iteration.
    """
    def predict_fn(x: torch.Tensor):
        n = x.shape[0]
        mu = torch.full((n,), prior_mean, dtype=x.dtype)
        sigma2 = torch.full((n,), prior_var, dtype=x.dtype)
        return mu, sigma2
    return predict_fn