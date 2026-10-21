"""Convergence of the model fit."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .._introspection import (
    BaseAdapter,
    SklearnAdapter,
    StatsmodelsGLMAdapter,
    StatsmodelsLinearAdapter,
    StatsmodelsMixedAdapter,
    as_adapter,
)
from .._types import ArrayLike, CheckNotApplicableError, Model
from ._base import CheckResult

__all__ = ["ConvergenceResult", "check_convergence"]


@dataclass(repr=False)
class ConvergenceResult(CheckResult):
    """Result of :func:`check_convergence`.

    For mixed models ``statistic`` is the largest absolute relative gradient
    at the optimum; it is ``None`` when only the optimizer's flag is known.
    """

    key = "convergence"
    title = "Convergence"


def _mixed_relative_gradient(adapter: StatsmodelsMixedAdapter) -> float | None:
    # lme4 / performance: max |H^-1 g| at the optimum, on statsmodels' own scale.
    result = adapter.model
    try:
        gradient = result.model.score(result.params_object, profile_fe=False)
        hessian = result.model.hessian(result.params_object)
        hessian = hessian[0] if isinstance(hessian, tuple) else hessian
        return float(np.max(np.abs(np.linalg.solve(hessian, gradient))))
    except Exception:
        return None


def _sklearn_converged(adapter: SklearnAdapter) -> tuple[bool, str]:
    model = adapter.model
    n_iter = getattr(model, "n_iter_", None)
    max_iter = getattr(model, "max_iter", None)
    if n_iter is None or max_iter is None:
        return True, "closed form"
    return bool(np.max(np.asarray(n_iter)) < max_iter), "iteration limit"


def check_convergence(
    model: Model,
    X: ArrayLike | None = None,
    y: ArrayLike | None = None,
    *,
    tolerance: float = 1e-3,
) -> ConvergenceResult:
    """Check whether the model fit converged.

    Port of ``performance::check_convergence()``. For mixed models the fit
    counts as converged when the optimizer reports success and the largest
    absolute relative gradient (``H^-1 g``) at the optimum is below
    ``tolerance``, the criterion R uses for ``lme4`` fits. Other models use
    their optimizer's own report: IRLS for GLMs, the iteration limit for
    iterative scikit-learn estimators; OLS and WLS are solved exactly.

    Parameters
    ----------
    model
        Any supported model.
    X, y
        Training data; required for scikit-learn models.
    tolerance
        Largest acceptable relative gradient (mixed models only).

    Returns
    -------
    ConvergenceResult

    Notes
    -----
    statsmodels and lme4 parameterize the random effects differently, so the
    gradient value itself is not comparable with R's; the verdict is.
    """
    adapter: BaseAdapter = as_adapter(model, X, y)
    statistic: float | None = None
    if isinstance(adapter, StatsmodelsMixedAdapter):
        flag = bool(getattr(adapter.model, "converged", True))
        statistic = _mixed_relative_gradient(adapter)
        converged = flag and (statistic is None or statistic < tolerance)
        method = "Relative gradient"
    elif isinstance(adapter, StatsmodelsGLMAdapter):
        converged = bool(getattr(adapter.model, "converged", True))
        method = "IRLS"
    elif isinstance(adapter, StatsmodelsLinearAdapter):
        converged, method = True, "closed form"
    elif isinstance(adapter, SklearnAdapter):
        converged, method = _sklearn_converged(adapter)
    else:
        reported = getattr(adapter.model, "converged", None)
        if reported is None:
            raise CheckNotApplicableError("The model does not report convergence information.")
        converged, method = bool(reported), "reported by the model"

    if converged:
        message = "OK: Model converged."
    elif statistic is not None and statistic >= tolerance:
        message = (
            f"Warning: Model failed to converge (max |relative gradient| = {statistic:.3g} "
            f"> {tolerance:g}). Try refitting with another optimizer, e.g. "
            "`.fit(method='powell')`, especially if a variance is close to zero."
        )
    else:
        message = "Warning: Model failed to converge."
    return ConvergenceResult(
        passed=converged,
        method=method,
        statistic=statistic,
        p_value=None,
        message=message,
        adapter=adapter,
    )
