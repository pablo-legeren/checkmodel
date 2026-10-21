"""Variance decomposition of mixed models: ICC and Nakagawa's R²."""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np

from .._introspection import BaseAdapter, StatsmodelsMixedAdapter, as_adapter
from .._types import ArrayLike, CheckNotApplicableError, Model
from .r2 import R2Result

__all__ = ["ICCResult", "VarianceComponents", "icc", "r2_nakagawa", "variance_components"]


@dataclass(frozen=True)
class VarianceComponents:
    """Variance decomposition of a linear mixed model, as ``insight::get_variance()``.

    Attributes
    ----------
    fixed
        Variance of the fixed-effects predictions, ``var(X @ beta)``.
    random
        Mean variance explained by the random effects, ``tr(Z G Z') / n``
        (Johnson, 2014); ``nan`` for a singular fit.
    residual
        Residual variance, ``sigma²``.
    """

    fixed: float
    random: float
    residual: float


def _mixed(model: Model, X: ArrayLike | None, y: ArrayLike | None) -> StatsmodelsMixedAdapter:
    adapter: BaseAdapter = as_adapter(model, X, y)
    if not isinstance(adapter, StatsmodelsMixedAdapter):
        raise CheckNotApplicableError("`model` has no random effects.")
    return adapter


def _is_singular(adapter: StatsmodelsMixedAdapter, tolerance: float) -> bool:
    return bool(np.linalg.det(adapter.cov_re) < tolerance)


def variance_components(
    model: Model,
    X: ArrayLike | None = None,
    y: ArrayLike | None = None,
    *,
    tolerance: float = 1e-8,
) -> VarianceComponents:
    """Split the variance of a linear mixed model into fixed, random and residual parts.

    Parameters
    ----------
    model
        A fitted statsmodels ``MixedLM`` result.
    X, y
        Unused; accepted for a uniform signature.
    tolerance
        Below this determinant of the random-effects covariance the fit is
        singular and the random variance is reported as ``nan``, as in R.

    Returns
    -------
    VarianceComponents
    """
    adapter = _mixed(model, X, y)
    beta = np.asarray(adapter.model.fe_params, dtype=float)
    fixed = float(np.var(adapter.X @ beta, ddof=1))
    if _is_singular(adapter, tolerance):
        random = float("nan")
    else:
        Z = adapter.Z
        random = float(np.mean(np.sum((Z @ adapter.cov_re) * Z, axis=1)))
    return VarianceComponents(fixed=fixed, random=random, residual=adapter.scale)


class ICCResult(R2Result):
    """Read-only mapping with ``ICC_adjusted``, ``ICC_conditional`` and ``ICC_unadjusted``."""

    def __repr__(self) -> str:
        return "\n".join(
            [
                "# Intraclass Correlation Coefficient",
                f"    Adjusted ICC: {self['ICC_adjusted']:.3f}",
                f"  Unadjusted ICC: {self['ICC_unadjusted']:.3f}",
            ]
        )

    __str__ = __repr__


def icc(
    model: Model,
    X: ArrayLike | None = None,
    y: ArrayLike | None = None,
    *,
    tolerance: float = 1e-5,
) -> ICCResult:
    """Intraclass correlation coefficient of a linear mixed model.

    Port of ``performance::icc()``. The adjusted ICC is the share of the
    random-effects variance in the random plus residual variance; the
    unadjusted (conditional) ICC also counts the fixed-effects variance in
    the denominator. With random slopes the random variance is averaged over
    the observed covariate values (Johnson, 2014).

    Parameters
    ----------
    model
        A fitted statsmodels ``MixedLM`` result.
    X, y
        Unused; accepted for a uniform signature.
    tolerance
        Singularity tolerance. For a singular fit R returns ``NA``; here the
        values are ``nan`` and a warning is raised.

    Returns
    -------
    ICCResult

    Raises
    ------
    CheckNotApplicableError
        If the model has no random effects.
    """
    adapter = _mixed(model, X, y)
    v = variance_components(adapter, tolerance=tolerance)
    if np.isnan(v.random):
        warnings.warn(
            "Can't compute ICC. Some variance components equal zero; the model may be "
            "singular (see `check_singularity()`).",
            UserWarning,
            stacklevel=2,
        )
    adjusted = v.random / (v.random + v.residual)
    unadjusted = v.random / (v.fixed + v.random + v.residual)
    return ICCResult(
        {"ICC_adjusted": adjusted, "ICC_conditional": unadjusted, "ICC_unadjusted": unadjusted},
        "Mixed Models",
    )


def r2_nakagawa(
    model: Model,
    X: ArrayLike | None = None,
    y: ArrayLike | None = None,
    *,
    tolerance: float = 1e-8,
) -> R2Result:
    """Nakagawa's marginal and conditional R² for linear mixed models.

    Port of ``performance::r2_nakagawa()``. The marginal R² is the variance
    explained by the fixed effects; the conditional R² by fixed and random
    effects together (Nakagawa, Johnson & Schielzeth, 2017).

    Parameters
    ----------
    model
        A fitted statsmodels ``MixedLM`` result.
    X, y
        Unused; accepted for a uniform signature.
    tolerance
        Singularity tolerance. For a singular fit the conditional R² is
        ``nan`` and the marginal R² ignores the random effects, as in R.

    Returns
    -------
    R2Result
        ``R2_conditional`` and ``R2_marginal``.
    """
    adapter = _mixed(model, X, y)
    v = variance_components(adapter, tolerance=tolerance)
    if np.isnan(v.random):
        warnings.warn(
            "Random effect variances not available. Returned R2 does not account for "
            "random effects.",
            UserWarning,
            stacklevel=2,
        )
        marginal = v.fixed / (v.fixed + v.residual)
        conditional = float("nan")
    else:
        total = v.fixed + v.random + v.residual
        marginal = v.fixed / total
        conditional = (v.fixed + v.random) / total
    return R2Result({"R2_conditional": conditional, "R2_marginal": marginal}, "Mixed Models")
