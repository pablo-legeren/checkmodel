"""Zero-inflation of count models."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .._introspection import as_adapter
from .._types import ArrayLike, CheckNotApplicableError, Model
from ._base import CheckResult, format_p
from .residuals import SimulatedResiduals, _alternative, simulate_residuals, simulation_p_value

__all__ = ["ZeroInflationResult", "check_zeroinflation"]


@dataclass(repr=False)
class ZeroInflationResult(CheckResult):
    """Result of :func:`check_zeroinflation`. ``statistic`` is predicted / observed zeros.

    ``p_value`` is set, and ``simulation`` holds the simulations, when the
    check uses simulated residuals.
    """

    key = "zeroinflation"
    title = "Zero-inflation"

    predicted_zeros: int = 0
    observed_zeros: int = 0
    tolerance: float = 0.05
    simulation: SimulatedResiduals | None = None


def check_zeroinflation(
    model: Model | SimulatedResiduals,
    X: ArrayLike | None = None,
    y: ArrayLike | None = None,
    *,
    tolerance: float | None = None,
    alternative: str = "two-sided",
    iterations: int = 250,
    random_state: int | np.random.Generator | None = None,
) -> ZeroInflationResult:
    """Compare the zeros a count model predicts with the zeros observed.

    Port of ``performance::check_zeroinflation()``. For Poisson models the
    predicted number of zeros is ``round(sum(exp(-mu)))``. For negative
    binomial models, or when given the output of :func:`simulate_residuals`,
    it is the mean number of zeros over the simulated datasets, and a p-value
    compares the observed count with the simulated ones. When the ratio of
    predicted to observed zeros falls below ``1 - tolerance`` the model
    underfits zeros, a sign of zero-inflation; consider a zero-inflated or
    hurdle model.

    Parameters
    ----------
    model
        A fitted Poisson or negative binomial model, or the output of
        :func:`simulate_residuals`.
    X, y
        Training data; required for scikit-learn models.
    tolerance
        Accepted relative deviation of the ratio from 1. Defaults to 0.05,
        or 0.1 with simulated residuals, as in R.
    alternative
        Alternative hypothesis of the simulated test.
    iterations
        Number of simulations for negative binomial models.
    random_state
        Seed for the simulations.

    Returns
    -------
    ZeroInflationResult

    Raises
    ------
    CheckNotApplicableError
        For models that are not count models, or when the response has no zeros.
    """
    alternative = _alternative(alternative)
    simulation: SimulatedResiduals | None = None
    if isinstance(model, SimulatedResiduals):
        simulation = model
        adapter = model.adapter
    else:
        adapter = as_adapter(model, X, y)
        if not adapter.info.is_count:
            raise CheckNotApplicableError(
                "Zero-inflation checks can only be used for count models (Poisson or negative "
                "binomial)."
            )
    observed = int(np.sum(adapter.y == 0))
    if observed == 0:
        raise CheckNotApplicableError("Model has no observed zeros in the response variable.")
    if simulation is None and adapter.info.family == "negbin":
        simulation = simulate_residuals(adapter, iterations=iterations, random_state=random_state)

    p: float | None = None
    if simulation is None:
        tolerance = 0.05 if tolerance is None else tolerance
        predicted = round(float(np.sum(np.exp(-adapter.mu))))
        ratio = predicted / observed
        method = "Predicted vs observed zeros"
    else:
        tolerance = 0.1 if tolerance is None else tolerance
        zeros = np.sum(simulation.simulated == 0, axis=1).astype(float)
        p = simulation_p_value(float(observed), zeros, alternative)
        predicted = round(float(np.mean(zeros)))
        ratio = float(np.mean(zeros)) / observed
        method = "Simulated vs observed zeros"

    counts = f"{observed} observed, {predicted} predicted, ratio {ratio:.2f}"
    if p is not None:
        counts += f", {format_p(p)}"
    if ratio < 1 - tolerance:
        passed = False
        message = f"Warning: Model is underfitting zeros (probable zero-inflation): {counts}."
    elif ratio > 1 + tolerance:
        passed = False
        message = f"Warning: Model is overfitting zeros: {counts}."
    else:
        passed = True
        message = f"OK: Observed and predicted zeros are within the tolerance range ({counts})."
    return ZeroInflationResult(
        passed=passed,
        method=method,
        statistic=ratio,
        p_value=p,
        message=message,
        adapter=adapter,
        predicted_zeros=predicted,
        observed_zeros=observed,
        tolerance=tolerance,
        simulation=simulation,
    )
