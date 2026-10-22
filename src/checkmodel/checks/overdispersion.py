"""Overdispersion of generalized linear (mixed) models."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

from .._introspection import BaseAdapter, as_adapter
from .._types import ArrayLike, CheckNotApplicableError, Model, ModelType
from ._base import CheckResult, format_p
from .residuals import SimulatedResiduals, _alternative, simulate_residuals, simulation_p_value

__all__ = ["OverdispersionResult", "check_overdispersion"]


@dataclass(repr=False)
class OverdispersionResult(CheckResult):
    """Result of :func:`check_overdispersion`.

    With Pearson residuals, ``statistic`` is Pearson's chi-squared and
    ``dispersion_ratio`` that statistic divided by the residual degrees of
    freedom ``df``. With simulated residuals there is no chi-squared statistic:
    ``dispersion_ratio`` compares the variance of the observed residuals with
    that of the simulated ones, and ``simulation`` holds the simulations.
    """

    key = "overdispersion"
    title = "Overdispersion"

    dispersion_ratio: float = float("nan")
    df: int | None = None
    simulation: SimulatedResiduals | None = None


def _is_glm(info: ModelType) -> bool:
    # R's `inherits(model, "glm")`: least-squares linear models are not GLMs.
    return info.model_class == "GLM" or not info.is_linear


def _use_simulated(info: ModelType, residual_type: str | None) -> bool:
    """Decide between Pearson and simulated residuals as R's ``performance`` does."""
    if residual_type is not None:
        kind = {"normal": "pearson"}.get(residual_type, residual_type)
        if kind not in ("pearson", "simulated"):
            raise ValueError(
                f"`residual_type` must be 'pearson' or 'simulated', got {residual_type!r}."
            )
        return kind == "simulated"
    if info.is_mixed or info.family == "negbin":
        return True
    return _is_glm(info) and info.family != "poisson"


def _verdict(passed: bool, ratio: float, detail: str) -> str:
    if passed:
        return f"OK: No overdispersion detected ({detail})."
    if ratio > 1:
        return f"Warning: Overdispersion detected ({detail})."
    return f"Warning: Underdispersion detected ({detail})."


def _simulated_test(
    simulation: SimulatedResiduals, alpha: float, alternative: str
) -> OverdispersionResult:
    # check_overdispersion.performance_simres(): DHARMa's dispersion test.
    variance = float(np.var(simulation.simulated, ddof=1))
    observed = float(np.var(simulation.observed - simulation.fitted, ddof=1)) / variance
    simulated = np.var(simulation.simulated - simulation.fitted, axis=1, ddof=1) / variance
    p = simulation_p_value(observed, simulated, alternative)
    ratio = observed / float(np.mean(simulated))
    passed = p > alpha
    return OverdispersionResult(
        passed=passed,
        method="Simulated residuals (DHARMa dispersion test)",
        statistic=None,
        p_value=p,
        message=_verdict(passed, ratio, f"dispersion ratio = {ratio:.3f}, {format_p(p)}"),
        adapter=simulation.adapter,
        dispersion_ratio=ratio,
        simulation=simulation,
    )


def _pearson_test(adapter: BaseAdapter, alpha: float) -> OverdispersionResult:
    info = adapter.info
    if info.is_mixed:
        raise CheckNotApplicableError(
            "Pearson-based overdispersion tests are not available for mixed models; "
            "use `residual_type='simulated'`."
        )
    if not info.is_count and not info.is_binomial:
        raise CheckNotApplicableError(
            "Overdispersion checks can only be used for models from Poisson families or "
            "binomial families with trials > 1."
        )
    if info.is_bernoulli:
        raise CheckNotApplicableError("Overdispersion checks cannot be used for Bernoulli models.")
    chisq = float(np.sum(adapter.pearson_residuals**2))
    df = adapter.n_obs - adapter.n_params
    ratio = chisq / df
    p = float(stats.chi2.sf(chisq, df))
    passed = p > alpha
    return OverdispersionResult(
        passed=passed,
        method="Pearson chi-squared",
        statistic=chisq,
        p_value=p,
        message=_verdict(passed, ratio, f"dispersion ratio = {ratio:.3f}, {format_p(p)}"),
        adapter=adapter,
        dispersion_ratio=ratio,
        df=df,
    )


def check_overdispersion(
    model: Model | SimulatedResiduals,
    X: ArrayLike | None = None,
    y: ArrayLike | None = None,
    *,
    alpha: float = 0.05,
    residual_type: str | None = None,
    alternative: str = "two-sided",
    iterations: int = 250,
    random_state: int | np.random.Generator | None = None,
) -> OverdispersionResult:
    """Test a generalized linear (mixed) model for over- or underdispersion.

    Port of ``performance::check_overdispersion()``. Which residuals are used
    follows R:

    * Poisson GLMs: the Pearson chi-squared statistic ``sum((y - mu)^2 / mu)``
      against a chi-squared distribution on ``n - k`` degrees of freedom
      (Gelman & Hill, 2007).
    * Bernoulli, binomial, negative binomial and other GLMs, and mixed models:
      DHARMa's test on simulated residuals (see :func:`simulate_residuals`),
      which compares the variance of the observed residuals with the variance
      of residuals simulated from the model.

    A dispersion ratio well above 1 means the variance exceeds what the model
    allows, so standard errors are too small; consider a negative binomial or
    quasi-Poisson model.

    Parameters
    ----------
    model
        A fitted GLM or mixed model, or the output of :func:`simulate_residuals`.
    X, y
        Training data; required for scikit-learn models.
    alpha
        Significance level. The check fails when ``p <= alpha``.
    residual_type
        ``"pearson"`` or ``"simulated"`` to override the choice above (R's
        ``"normal"`` is accepted for ``"pearson"``).
    alternative
        Alternative hypothesis of the simulated test: ``"two-sided"``
        (default), ``"greater"`` (overdispersion only) or ``"less"``.
    iterations
        Number of simulations for the simulated test.
    random_state
        Seed for the simulations.

    Returns
    -------
    OverdispersionResult

    Raises
    ------
    CheckNotApplicableError
        For linear models fitted by least squares, and for Pearson residuals
        of Bernoulli, non-count or mixed models.
    """
    alternative = _alternative(alternative)
    if isinstance(model, SimulatedResiduals):
        return _simulated_test(model, alpha, alternative)
    adapter = as_adapter(model, X, y)
    info = adapter.info
    if not info.is_mixed and not _is_glm(info):
        raise CheckNotApplicableError(
            "Overdispersion checks apply to generalized linear (mixed) models; for a linear "
            "model, check the residual variance with `check_heteroscedasticity()`."
        )
    if _use_simulated(info, residual_type):
        simulation = simulate_residuals(adapter, iterations=iterations, random_state=random_state)
        return _simulated_test(simulation, alpha, alternative)
    return _pearson_test(adapter, alpha)
