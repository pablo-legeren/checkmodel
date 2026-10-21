"""R² and leave-one-out criteria for Bayesian models."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import Any

import numpy as np

from .._bayes import BayesianModel, as_bayesian
from .._mcmc import LooResult, hdi, mad, psis_loo, relative_eff
from .._types import CheckNotApplicableError, FloatArray, UnsupportedModelError

__all__ = ["BayesR2Result", "looic", "r2_bayes"]


def _require(model: Any) -> BayesianModel:
    bayes = as_bayesian(model)
    if bayes is None:
        raise UnsupportedModelError(
            "Expected a Bayesian model: the result of `checkmodel.bayesian()` or an ArviZ "
            "DataTree/InferenceData."
        )
    return bayes


class BayesR2Result(Mapping[str, float]):
    """Posterior summary of Bayesian R²: a mapping of point estimates.

    Attributes
    ----------
    se
        Spread of each R² posterior (MAD when ``robust``, else SD).
    ci
        Highest density interval of each R² posterior.
    draws
        The R² posterior draws themselves.
    """

    def __init__(
        self,
        values: dict[str, float],
        se: dict[str, float],
        ci: dict[str, tuple[float, float]],
        draws: dict[str, FloatArray],
        ci_level: float,
    ) -> None:
        self._values = values
        self.se = se
        self.ci = ci
        self.draws = draws
        self.ci_level = ci_level

    def __getitem__(self, key: str) -> float:
        return self._values[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._values)

    def __len__(self) -> int:
        return len(self._values)

    def __repr__(self) -> str:
        mixed = "R2_marginal" in self._values
        labels = {"R2": "Conditional R2" if mixed else "R2", "R2_marginal": "Marginal R2"}
        width = max(len(labels[k]) for k in self._values)
        pct = f"{self.ci_level:.0%}"
        lines = ["# Bayesian R2 with Compatibility Interval", ""]
        for key, value in self._values.items():
            lo, hi = self.ci[key]
            lines.append(f"  {labels[key]:>{width}}: {value:.3f} ({pct} CI [{lo:.3f}, {hi:.3f}])")
        return "\n".join(lines)

    __str__ = __repr__


def _r2_draws(bayes: BayesianModel) -> dict[str, FloatArray]:
    mu = BayesianModel.flat(bayes.mu)
    var_mu = mu.var(axis=1, ddof=1)
    if bayes.is_mixed:
        # performance::r2_posterior() for mixed models: residual-based, conditional
        # and marginal (rstantools::bayes_R2() on posterior_epred()).
        def residual_based(pred: FloatArray) -> FloatArray:
            var_pred = pred.var(axis=1, ddof=1)
            var_err = (pred - bayes.y).var(axis=1, ddof=1)
            return np.asarray(var_pred / (var_pred + var_err))

        marginal = BayesianModel.flat(bayes.mu_marginal)  # type: ignore[arg-type]
        return {"R2": residual_based(mu), "R2_marginal": residual_based(marginal)}
    # rstanarm::bayes_R2(): model-based residual variance (Gelman et al., 2019).
    if bayes.family == "gaussian":
        if bayes.sigma is None:
            raise CheckNotApplicableError("Bayesian R2 of a Gaussian model needs `sigma` draws.")
        sigma2 = bayes.sigma.reshape(-1) ** 2
    elif bayes.family == "binomial":
        sigma2 = (mu * (1.0 - mu)).mean(axis=1)
    else:
        raise CheckNotApplicableError(
            "Bayesian R2 is available for Gaussian and binomial models (and any multilevel model)."
        )
    return {"R2": np.asarray(var_mu / (var_mu + sigma2))}


def r2_bayes(model: Any, *, ci: float = 0.95, robust: bool = True) -> BayesR2Result:
    """Bayesian R² (Gelman, Goodrich, Gabry & Vehtari, 2019), as ``performance::r2_bayes()``.

    One R² per posterior draw: ``var(mu) / (var(mu) + residual variance)``,
    with the model's residual variance (sigma² for Gaussian models,
    ``mean(mu (1 - mu))`` for binomial ones). For multilevel models the
    residual variance is that of ``y - mu`` and a marginal R² (without the
    group-level effects) is reported too, as R does for ``stanreg`` fits.

    Parameters
    ----------
    model
        Output of :func:`checkmodel.bayesian`, or a DataTree from PyMC.
    ci
        Probability mass of the highest density interval.
    robust
        Summarise with median and MAD (default, as R) instead of mean and SD.

    Returns
    -------
    BayesR2Result
    """
    bayes = _require(model)
    draws = _r2_draws(bayes)
    values = {k: float(np.median(v) if robust else np.mean(v)) for k, v in draws.items()}
    se = {k: mad(v) if robust else float(np.std(v, ddof=1)) for k, v in draws.items()}
    intervals = {k: hdi(v, ci) for k, v in draws.items()}
    return BayesR2Result(values, se, intervals, draws, ci)


def looic(model: Any, *, use_relative_eff: bool = False) -> LooResult:
    """PSIS leave-one-out cross-validation, as ``performance::looic()`` / ``loo::loo()``.

    Parameters
    ----------
    model
        Output of :func:`checkmodel.bayesian` with pointwise log-likelihood.
    use_relative_eff
        Account for autocorrelation of the draws when choosing the Pareto tail
        length, as ``loo::loo(r_eff = relative_eff(...))``. Off by default
        because ``loo()`` on an ``rstanarm`` fit, and therefore
        ``model_performance()`` in R, uses ``r_eff = 1``.

    Returns
    -------
    LooResult
        ELPD, LOOIC and effective number of parameters with their standard
        errors, plus the Pareto k diagnostic of each observation.
    """
    bayes = _require(model)
    if bayes.log_lik is None:
        raise CheckNotApplicableError(
            "LOO needs the pointwise log-likelihood: sample with "
            "`idata_kwargs={'log_likelihood': True}` or pass the model to `bayesian()`."
        )
    r_eff = relative_eff(bayes.log_lik) if use_relative_eff else np.ones(bayes.n_obs)
    return psis_loo(bayes.log_lik, r_eff=r_eff)
