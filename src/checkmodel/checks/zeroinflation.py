"""Zero-inflation of count models."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .._introspection import as_adapter
from .._types import ArrayLike, CheckNotApplicableError, Model
from ._base import CheckResult
from .overdispersion import _require_poisson

__all__ = ["ZeroInflationResult", "check_zeroinflation"]


@dataclass(repr=False)
class ZeroInflationResult(CheckResult):
    """Result of :func:`check_zeroinflation`. ``statistic`` is predicted / observed zeros."""

    key = "zeroinflation"
    title = "Zero-inflation"

    predicted_zeros: int = 0
    observed_zeros: int = 0
    tolerance: float = 0.05


def check_zeroinflation(
    model: Model,
    X: ArrayLike | None = None,
    y: ArrayLike | None = None,
    *,
    tolerance: float = 0.05,
) -> ZeroInflationResult:
    """Compare the zeros a Poisson model predicts with the zeros observed.

    Port of ``performance::check_zeroinflation()`` for Poisson GLMs. The
    predicted number of zeros is ``round(sum(exp(-mu)))``. When the ratio of
    predicted to observed zeros falls below ``1 - tolerance`` the model
    underfits zeros, a sign of zero-inflation; consider a zero-inflated or
    hurdle model.

    Parameters
    ----------
    model
        A fitted Poisson GLM.
    X, y
        Unused; accepted for a uniform signature.
    tolerance
        Accepted relative deviation of the ratio from 1.

    Returns
    -------
    ZeroInflationResult

    Raises
    ------
    CheckNotApplicableError
        For non-Poisson models, or when the response has no zeros.
    """
    adapter = as_adapter(model, X, y)
    _require_poisson(adapter, "Zero-inflation")
    observed = int(np.sum(adapter.y == 0))
    if observed == 0:
        raise CheckNotApplicableError("Model has no observed zeros in the response variable.")
    predicted = round(float(np.sum(np.exp(-adapter.mu))))
    ratio = predicted / observed
    counts = f"{observed} observed, {predicted} predicted, ratio {ratio:.2f}"
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
        method="Predicted vs observed zeros",
        statistic=ratio,
        p_value=None,
        message=message,
        adapter=adapter,
        predicted_zeros=predicted,
        observed_zeros=observed,
        tolerance=tolerance,
    )
