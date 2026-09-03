"""
surrogate.py
============
Gaussian process model (exact GP, RBF kernel) via GPyTorch.
Provides:
  - ExactGPModel         : GPyTorch model class
  - fit_gp()             : train hypers by MLL
  - make_gp_predict_fn() : returns a callable (x -> (mu, sigma2))
  - make_prior_predict_fn(): prior callable for cold-start first iteration
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
           n_iter: int = 50, lr: float = 0.1,
           noise_var: float = 0.01):
    """Fit exact GP via marginal-likelihood optimisation (Adam)."""
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    likelihood.noise = noise_var
    model = ExactGPModel(train_x, train_y, likelihood)

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
    Does NOT wrap in torch.no_grad() so gradients flow for L-BFGS.
    """
    def predict_fn(x: torch.Tensor):
        with gpytorch.settings.fast_pred_var():
            pred = likelihood(model(x))
        return pred.mean, pred.variance
    return predict_fn


def make_prior_predict_fn(prior_mean: float = 0.0,
                          prior_var: float = 1.0) -> PredictFn:
    """
    Returns a predict function that outputs constant prior mean/variance.
    Used for the first iteration of cold-start BO.

    Under GP-UCB:     acquisition is flat -> first point is random.
    Under GP-UCB-PV:  acquisition depends on rho(x) -> explores high-noise
                      regions even before any data.
    """
    def predict_fn(x: torch.Tensor):
        n = x.shape[0]
        mu = torch.full((n,), prior_mean, dtype=x.dtype)
        sigma2 = torch.full((n,), prior_var, dtype=x.dtype)
        return mu, sigma2
    return predict_fn