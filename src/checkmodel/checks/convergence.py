"""Convergence of the model fit."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .._bayes import BayesianModel, as_bayesian
from .._introspection import (
    BaseAdapter,
    SklearnAdapter,
    StatsmodelsGLMAdapter,
    StatsmodelsLinearAdapter,
    StatsmodelsMixedAdapter,
    as_adapter,
)
from .._mcmc import ess_bulk, ess_tail, rhat
from .._types import ArrayLike, CheckNotApplicableError, Model
from ._base import CheckResult

__all__ = ["ConvergenceResult", "check_convergence"]


@dataclass(repr=False)
class ConvergenceResult(CheckResult):
    """Result of :func:`check_convergence`.

    For mixed models ``statistic`` is the largest absolute relative gradient
    at the optimum; it is ``None`` when only the optimizer's flag is known.
    For Bayesian models it is the largest R-hat, and ``table`` lists R-hat
    and bulk/tail effective sample sizes per parameter.
    """

    key = "convergence"
    title = "Convergence"

    table: pd.DataFrame = field(default_factory=pd.DataFrame)

    def to_dataframe(self) -> pd.DataFrame:
        """Return the per-parameter diagnostics (Bayesian models) or the summary row."""
        return self.table.copy() if not self.table.empty else super().to_dataframe()


RHAT_MAX = 1.01
ESS_PER_CHAIN = 100


def _bayesian_convergence(bayes: BayesianModel) -> ConvergenceResult:
    rows = [
        {
            "Parameter": name,
            "Rhat": rhat(draws),
            "ESS_bulk": ess_bulk(draws),
            "ESS_tail": ess_tail(draws),
        }
        for name, draws in bayes.parameters.items()
    ]
    if not rows:
        raise CheckNotApplicableError("The posterior holds no scalar parameters to diagnose.")
    table = pd.DataFrame(rows)
    min_ess = ESS_PER_CHAIN * bayes.chains
    bad_rhat = table[table["Rhat"] >= RHAT_MAX]
    low_ess = table[(table["ESS_bulk"] < min_ess) | (table["ESS_tail"] < min_ess)]
    worst = float(table["Rhat"].max())
    problems = []
    if not bad_rhat.empty:
        problems.append(f"R-hat >= {RHAT_MAX} for {', '.join(bad_rhat['Parameter'])}")
    if not low_ess.empty:
        problems.append(f"ESS < {min_ess} for {', '.join(low_ess['Parameter'])}")
    if problems:
        message = (
            f"Warning: Chains have not converged: {'; '.join(problems)}. "
            "Run longer chains or reparameterize the model."
        )
    else:
        message = (
            f"OK: Chains converged (max R-hat = {worst:.3f}, "
            f"min bulk ESS = {table['ESS_bulk'].min():.0f})."
        )
    return ConvergenceResult(
        passed=not problems,
        method="R-hat and effective sample size",
        statistic=worst,
        p_value=None,
        message=message,
        adapter=bayes,  # type: ignore[arg-type]
        table=table,
    )


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

    Bayesian models converge when every parameter has rank-normalized
    R-hat below 1.01 and bulk and tail effective sample sizes of at least 100
    per chain (Vehtari et al., 2021), computed as R's ``posterior`` package.

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
    bayes = as_bayesian(model)
    if bayes is not None:
        return _bayesian_convergence(bayes)
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
