"""Multicollinearity via (generalized) variance inflation factors."""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import special, stats

from .._introspection import as_adapter
from .._types import ArrayLike, CheckNotApplicableError, Model
from ._base import CheckResult

__all__ = ["CollinearityResult", "check_collinearity"]

LOW_VIF = 5.0
HIGH_VIF = 10.0


@dataclass(repr=False)
class CollinearityResult(CheckResult):
    """Result of :func:`check_collinearity`.

    ``statistic`` is the largest VIF. ``table`` has one row per model term
    with the same columns as R: ``Term``, ``VIF``, ``VIF_CI_low``,
    ``VIF_CI_high``, ``SE_factor``, ``Tolerance``, ``Tolerance_CI_low`` and
    ``Tolerance_CI_high``.
    """

    key = "collinearity"
    title = "Collinearity"

    table: pd.DataFrame = field(default_factory=pd.DataFrame)

    @property
    def vif(self) -> pd.Series:
        """VIF per term."""
        return self.table.set_index("Term")["VIF"]

    def to_dataframe(self) -> pd.DataFrame:
        """Return the VIF table, one row per term."""
        return self.table.copy()


def _det(m: np.ndarray) -> float:
    return float(np.linalg.det(m)) if m.size else 1.0


def check_collinearity(
    model: Model,
    X: ArrayLike | None = None,
    y: ArrayLike | None = None,
    *,
    ci: float = 0.95,
    threshold: float = HIGH_VIF,
) -> CollinearityResult:
    """Compute variance inflation factors for each model term.

    Port of ``performance::check_collinearity()``. VIFs come from the
    correlation matrix of the coefficient estimates, so a term encoded by
    several columns (a categorical variable) gets one *generalized* VIF
    (Fox & Monette, 1992). Confidence intervals follow Marcoulides et al.
    (2019).

    Parameters
    ----------
    model
        Any supported model with at least two terms.
    X, y
        Training data; required for scikit-learn models.
    ci
        Confidence level of the VIF intervals.
    threshold
        The check fails when any VIF reaches this value. VIF < 5 is
        considered low, 5-10 moderate and >= 10 high.

    Returns
    -------
    CollinearityResult

    Raises
    ------
    CheckNotApplicableError
        If the model has fewer than two terms.

    Notes
    -----
    For scikit-learn models the coefficient correlation is computed from
    ``X`` (weighted by ``p(1-p)`` for logistic regression), which equals the
    unpenalized fit. The VIF describes the predictors, not the penalty.
    """
    adapter = as_adapter(model, X, y)
    terms = adapter.terms()
    if len(terms) < 2:
        raise CheckNotApplicableError(
            "Not enough model terms to check for multicollinearity (need at least two)."
        )
    icpt = adapter.intercept_index()
    if icpt is None:
        warnings.warn("Model without intercept. VIFs may not be sensible.", stacklevel=2)
    keep = [j for j in range(adapter.n_params) if j != icpt]
    position = {col: i for i, col in enumerate(keep)}
    v = adapter.vcov_unscaled[np.ix_(keep, keep)]
    sd = np.sqrt(np.diag(v))
    R = v / np.outer(sd, sd)
    det_R = _det(R)
    if not np.isfinite(det_R) or det_R <= 0:
        warnings.warn("Model matrix is rank deficient. VIFs may not be sensible.", stacklevel=2)

    vif = np.empty(len(terms))
    for i, term in enumerate(terms):
        subs = [position[c] for c in term.columns]
        rest = [k for k in range(len(keep)) if k not in subs]
        with np.errstate(divide="ignore", invalid="ignore"):
            vif[i] = np.divide(_det(R[np.ix_(subs, subs)]) * _det(R[np.ix_(rest, rest)]), det_R)

    # Marcoulides, Raykov et al. (2019): interval for R² = 1 - 1/VIF on the logit scale.
    n, p = adapter.n_obs, adapter.n_params
    with np.errstate(divide="ignore", invalid="ignore"):
        r = 1.0 - 1.0 / vif
        z = stats.norm.ppf((1.0 + ci) / 2.0)
        se = np.sqrt((1.0 - r**2) ** 2 * (n - p - 1) ** 2 / ((n**2 - 1) * (n + 3)))
        se_logit = se / (r * (1.0 - r))
        ci_low = special.expit(special.logit(r) - z * se_logit)
        ci_high = special.expit(special.logit(r) + z * se_logit)
        dof = np.array([len(t.columns) if t.is_factor else 1 for t in terms])
        table = pd.DataFrame(
            {
                "Term": [t.name for t in terms],
                "VIF": vif,
                "VIF_CI_low": 1.0 / (1.0 - ci_low),
                "VIF_CI_high": 1.0 / (1.0 - ci_high),
                "SE_factor": vif ** (1.0 / (2.0 * dof)),
                "Tolerance": 1.0 / vif,
                "Tolerance_CI_low": 1.0 - ci_high,
                "Tolerance_CI_high": 1.0 - ci_low,
            }
        )

    worst = float(np.nanmax(vif))
    high = [t.name for t, v_ in zip(terms, vif, strict=True) if v_ >= threshold]
    moderate = [t.name for t, v_ in zip(terms, vif, strict=True) if LOW_VIF <= v_ < threshold]
    passed = not high
    if high:
        message = f"Warning: High collinearity (VIF >= {threshold:g}) for {', '.join(high)}."
    elif moderate:
        message = (
            f"Note: Moderate collinearity (VIF {LOW_VIF:g}-{threshold:g}) for "
            f"{', '.join(moderate)}."
        )
    else:
        message = f"OK: Low collinearity (all VIF < {LOW_VIF:g}; max {worst:.2f})."
    return CollinearityResult(
        passed=passed,
        method="Variance inflation factor",
        statistic=worst,
        p_value=None,
        message=message,
        adapter=adapter,
        table=table,
    )
