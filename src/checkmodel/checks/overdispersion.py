"""Overdispersion of count models."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

from .._introspection import BaseAdapter, as_adapter
from .._types import ArrayLike, CheckNotApplicableError, Model
from ._base import CheckResult, format_p

__all__ = ["OverdispersionResult", "check_overdispersion"]


@dataclass(repr=False)
class OverdispersionResult(CheckResult):
    """Result of :func:`check_overdispersion`.

    ``statistic`` is Pearson's chi-squared; ``dispersion_ratio`` is that
    statistic divided by the residual degrees of freedom ``df``.
    """

    key = "overdispersion"
    title = "Overdispersion"

    dispersion_ratio: float = float("nan")
    df: int = 0


def _require_poisson(adapter: BaseAdapter, what: str) -> None:
    info = adapter.info
    if info.family == "poisson" and not info.is_mixed:
        return
    if info.is_bernoulli:
        raise CheckNotApplicableError(f"{what} checks cannot be used for Bernoulli models.")
    if info.family == "negbin" or info.is_mixed:
        raise CheckNotApplicableError(
            f"{what} checks for negative binomial and mixed models need simulated residuals "
            "(as R does with DHARMa), which are not supported yet."
        )
    raise CheckNotApplicableError(f"{what} checks can only be used for Poisson models.")


def check_overdispersion(
    model: Model,
    X: ArrayLike | None = None,
    y: ArrayLike | None = None,
    *,
    alpha: float = 0.05,
) -> OverdispersionResult:
    """Test a Poisson model for over- or underdispersion.

    Port of ``performance::check_overdispersion()`` for Poisson GLMs: the
    Pearson chi-squared statistic ``sum((y - mu)^2 / mu)`` is compared with a
    chi-squared distribution on ``n - k`` degrees of freedom (Gelman & Hill,
    2007). A dispersion ratio well above 1 means the variance exceeds the
    mean, so standard errors are too small; consider a negative binomial or
    quasi-Poisson model.

    Parameters
    ----------
    model
        A fitted Poisson GLM.
    X, y
        Unused; accepted for a uniform signature.
    alpha
        Significance level. The check fails when ``p <= alpha``.

    Returns
    -------
    OverdispersionResult

    Raises
    ------
    CheckNotApplicableError
        For non-Poisson models.
    """
    adapter = as_adapter(model, X, y)
    _require_poisson(adapter, "Overdispersion")
    y0, mu = adapter.y, adapter.mu
    chisq = float(np.sum((y0 - mu) ** 2 / mu))
    df = adapter.n_obs - adapter.n_params
    ratio = chisq / df
    p = float(stats.chi2.sf(chisq, df))
    passed = p > alpha
    detail = f"dispersion ratio = {ratio:.3f}, {format_p(p)}"
    if passed:
        message = f"OK: No overdispersion detected ({detail})."
    elif ratio > 1:
        message = f"Warning: Overdispersion detected ({detail})."
    else:
        message = f"Warning: Underdispersion detected ({detail})."
    return OverdispersionResult(
        passed=passed,
        method="Pearson chi-squared",
        statistic=chisq,
        p_value=p,
        message=message,
        adapter=adapter,
        dispersion_ratio=ratio,
        df=df,
    )
