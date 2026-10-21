"""Singular fits of mixed models."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .._introspection import StatsmodelsMixedAdapter, as_adapter
from .._types import ArrayLike, CheckNotApplicableError, Model
from ._base import CheckResult

__all__ = ["SingularityResult", "check_singularity"]


@dataclass(repr=False)
class SingularityResult(CheckResult):
    """Result of :func:`check_singularity`.

    ``statistic`` is the determinant of the random-effects covariance matrix;
    ``variances`` lists the variance of each random-effect term.
    """

    key = "singularity"
    title = "Singularity"

    variances: pd.Series = field(default_factory=pd.Series)

    def to_dataframe(self) -> pd.DataFrame:
        """Return one row per random-effect term with its variance."""
        return self.variances.rename("Variance").rename_axis("Term").reset_index()


def check_singularity(
    model: Model,
    X: ArrayLike | None = None,
    y: ArrayLike | None = None,
    *,
    tolerance: float = 1e-5,
) -> SingularityResult:
    """Check whether a mixed model's random-effects structure is singular.

    Port of ``performance::check_singularity()``: the fit is singular when
    the determinant of the random-effects variance-covariance matrix is below
    ``tolerance``, that is when some variance is (close to) zero or some
    correlation is (close to) ±1. A singular fit usually means the
    random-effects structure is too complex for the data.

    Parameters
    ----------
    model
        A fitted statsmodels ``MixedLM`` result.
    X, y
        Unused; accepted for a uniform signature.
    tolerance
        Determinant below which the fit counts as singular.

    Returns
    -------
    SingularityResult

    Raises
    ------
    CheckNotApplicableError
        If the model has no random effects.
    """
    adapter = as_adapter(model, X, y)
    if not isinstance(adapter, StatsmodelsMixedAdapter):
        raise CheckNotApplicableError("Singularity can only be checked for mixed models.")
    det = float(np.linalg.det(adapter.cov_re))
    singular = det < tolerance
    variances = pd.Series(
        np.diag(adapter.cov_re),
        index=[f"{adapter.group_name}: {t}" for t in adapter.random_terms()],
    )
    if singular:
        message = (
            f"Warning: Singular fit: random-effect variances are (close to) zero or their "
            f"correlations (close to) ±1 (determinant {det:.3g} < {tolerance:g}). Consider "
            "simplifying the random-effects structure."
        )
    else:
        message = "OK: The random-effects structure is not singular."
    return SingularityResult(
        passed=not singular,
        method="Determinant of random-effects covariance",
        statistic=det,
        p_value=None,
        message=message,
        adapter=adapter,
        variances=variances,
    )
