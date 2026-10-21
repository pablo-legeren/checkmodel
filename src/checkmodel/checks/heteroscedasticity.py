"""Constant error variance (homoscedasticity)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

from .._introspection import as_adapter
from .._types import ArrayLike, CheckNotApplicableError, Model
from ._base import CheckResult, format_p

__all__ = ["HeteroscedasticityResult", "check_heteroscedasticity"]


@dataclass(repr=False)
class HeteroscedasticityResult(CheckResult):
    """Result of :func:`check_heteroscedasticity`.

    ``statistic`` is the score chi² with ``df`` degrees of freedom.
    """

    key = "heteroscedasticity"
    title = "Homogeneity of variance"

    df: int = 1


def check_heteroscedasticity(
    model: Model,
    X: ArrayLike | None = None,
    y: ArrayLike | None = None,
    *,
    alpha: float = 0.05,
) -> HeteroscedasticityResult:
    """Test a linear model for non-constant error variance.

    Port of ``performance::check_heteroscedasticity()``: the original
    (non-studentized) Breusch-Pagan / Cook-Weisberg score test of the
    squared Pearson residuals against the fitted values, as in
    ``car::ncvTest()``. It is equivalent to
    ``statsmodels.stats.diagnostic.het_breuschpagan(resid, [1, fitted],
    robust=False)``.

    Parameters
    ----------
    model
        A fitted linear model.
    X, y
        Training data; required for scikit-learn models.
    alpha
        Significance level. The check fails when ``p < alpha``.

    Returns
    -------
    HeteroscedasticityResult

    Raises
    ------
    CheckNotApplicableError
        For non-Gaussian models.
    """
    adapter = as_adapter(model, X, y)
    if not adapter.info.is_linear:
        message = "The Breusch-Pagan test currently only works for Gaussian models."
        if adapter.info.is_count:
            message += " Check count models for overdispersion or zero-inflation instead."
        raise CheckNotApplicableError(message)
    r = adapter.pearson_residuals
    # S² = df_resid * sigma² / n: RSS / n for OLS, the REML sigma for mixed models.
    u = r**2 / (adapter.df_resid * adapter.scale / r.shape[0])
    Z = np.column_stack([np.ones_like(adapter.mu), adapter.mu])
    coef, *_ = np.linalg.lstsq(Z, u, rcond=None)
    explained = Z @ coef - u.mean()
    statistic = float(np.sum(explained**2) / 2.0)
    p = float(stats.chi2.sf(statistic, df=1))
    passed = p >= alpha
    if passed:
        message = f"OK: Error variance appears to be homoscedastic ({format_p(p)})."
    else:
        message = (
            f"Warning: Heteroscedasticity (non-constant error variance) detected ({format_p(p)})."
        )
    return HeteroscedasticityResult(
        passed=passed,
        method="Breusch-Pagan",
        statistic=statistic,
        p_value=p,
        message=message,
        adapter=adapter,
        df=1,
    )
