"""Normality of residuals and of random effects."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, overload

import numpy as np
import pandas as pd
from scipy import stats

from .._introspection import BaseAdapter, StatsmodelsMixedAdapter, as_adapter
from .._types import ArrayLike, CheckNotApplicableError, FloatArray, Model
from ._base import CheckResult, format_p, warn_about_fit

__all__ = ["NormalityResult", "RandomEffectsNormalityResult", "check_normality"]

_KS_MIN_N = 5000


@dataclass(repr=False)
class NormalityResult(CheckResult):
    """Result of :func:`check_normality`.

    Attributes
    ----------
    residuals
        The standardized residuals that were tested.
    """

    key = "normality"
    title = "Normality of residuals"

    residuals: FloatArray = field(default_factory=lambda: np.empty(0))


@dataclass(repr=False)
class RandomEffectsNormalityResult(CheckResult):
    """Result of ``check_normality(model, effects="random")``.

    ``p_value`` is the smallest p-value across random-effect terms; ``table``
    has one row per term and ``effects`` the conditional modes (BLUPs) tested.
    """

    key = "random_effects"
    title = "Normality of random effects"

    table: pd.DataFrame = field(default_factory=pd.DataFrame)
    effects: pd.DataFrame = field(default_factory=pd.DataFrame)

    def to_dataframe(self) -> pd.DataFrame:
        """Return one row per random-effect term."""
        return self.table.copy()


def _test(values: FloatArray) -> tuple[str, float, float]:
    if values.shape[0] >= _KS_MIN_N:
        ks = stats.kstest(values, "norm", method="asymp")
        return "Kolmogorov-Smirnov", float(ks.statistic), float(ks.pvalue)
    sw = stats.shapiro(values)
    return "Shapiro-Wilk", float(sw.statistic), float(sw.pvalue)


@overload
def check_normality(
    model: Model,
    X: ArrayLike | None = ...,
    y: ArrayLike | None = ...,
    *,
    alpha: float = ...,
    effects: Literal["fixed"] = ...,
) -> NormalityResult: ...


@overload
def check_normality(
    model: Model,
    X: ArrayLike | None = ...,
    y: ArrayLike | None = ...,
    *,
    alpha: float = ...,
    effects: Literal["random"],
) -> RandomEffectsNormalityResult: ...


def check_normality(
    model: Model,
    X: ArrayLike | None = None,
    y: ArrayLike | None = None,
    *,
    alpha: float = 0.05,
    effects: Literal["fixed", "random"] = "fixed",
) -> NormalityResult | RandomEffectsNormalityResult:
    """Check whether the residuals of a linear model are normally distributed.

    Port of ``performance::check_normality()``. The studentized residuals
    (``rstandard()`` in R, ``rstudent()`` for mixed models) are tested with
    Shapiro-Wilk when there are fewer than 5000 of them, and with a
    Kolmogorov-Smirnov test against the standard normal otherwise.

    Parameters
    ----------
    model
        A fitted linear model (statsmodels OLS/WLS/MixedLM, a Gaussian GLM with
        identity link, or a scikit-learn linear regressor).
    X, y
        Training data; required for scikit-learn models.
    alpha
        Significance level. The check fails when ``p < alpha``.
    effects
        ``"fixed"`` (default) tests the residuals. ``"random"`` tests the
        conditional modes of each random-effect term of a mixed model.

    Returns
    -------
    NormalityResult or RandomEffectsNormalityResult

    Raises
    ------
    CheckNotApplicableError
        For non-Gaussian models, where normal residuals are not expected, or
        ``effects="random"`` on a model without random effects.

    Warns
    -----
    UserWarning
        For a penalized fit, whose standardized residuals ignore the penalty.

    Notes
    -----
    Formal normality tests reject trivial departures in large samples. Look
    at the Q-Q plot (``result.plot()``) before acting on a small p-value.

    Examples
    --------
    >>> import statsmodels.formula.api as smf
    >>> from checkmodel import check_normality
    >>> fit = smf.ols("mpg ~ wt + hp", data=mtcars).fit()  # doctest: +SKIP
    >>> check_normality(fit)  # doctest: +SKIP
    Warning: Non-normality of residuals detected (p = 0.021).
    """
    adapter = as_adapter(model, X, y)
    if effects == "random":
        return _check_random_effects(adapter, alpha)
    if effects != "fixed":
        raise ValueError(f"`effects` must be 'fixed' or 'random', not {effects!r}.")
    if not adapter.info.is_linear:
        raise CheckNotApplicableError(
            "Checking normality of residuals is only appropriate for linear models."
        )
    warn_about_fit(adapter, separation=False)
    resid = adapter.std_residuals
    resid = resid[np.isfinite(resid)]
    if resid.shape[0] < 3:
        raise CheckNotApplicableError("At least 3 residuals are needed to test normality.")
    method, statistic, p = _test(resid)
    passed = p >= alpha
    if passed:
        message = f"OK: residuals appear as normally distributed ({format_p(p)})."
    else:
        message = f"Warning: Non-normality of residuals detected ({format_p(p)})."
        if method == "Kolmogorov-Smirnov":
            message += (
                " With large samples this test flags trivial departures; "
                "inspect the Q-Q plot (`result.plot()`)."
            )
    return NormalityResult(
        passed=passed,
        method=method,
        statistic=statistic,
        p_value=p,
        message=message,
        adapter=adapter,
        residuals=resid,
    )


def _check_random_effects(adapter: BaseAdapter, alpha: float) -> RandomEffectsNormalityResult:
    if not isinstance(adapter, StatsmodelsMixedAdapter):
        raise CheckNotApplicableError("`effects='random'` needs a mixed model.")
    blups = adapter.random_effects
    # lme4's isSingular(): a relative standard deviation (theta) below 1e-4.
    theta = np.sqrt(np.diag(adapter.cov_re) / adapter.scale)
    rows, lines = [], []
    for term, term_theta in zip(blups.columns, theta, strict=True):
        values = blups[term].to_numpy(dtype=float)
        label = f"{adapter.group_name}: {term}"
        if values.shape[0] < 3 or term_theta < 1e-4:
            raise CheckNotApplicableError(
                f"Random effects '{label}' have no variance (singular fit); "
                "their normality cannot be tested."
            )
        method, statistic, p = _test(values)
        ok = p >= alpha
        rows.append(
            {
                "Group": adapter.group_name,
                "Term": term,
                "Method": method,
                "Statistic": statistic,
                "p": p,
                "Passed": ok,
            }
        )
        if ok:
            lines.append(
                f"OK: Random effects '{label}' appear as normally distributed ({format_p(p)})."
            )
        else:
            lines.append(
                f"Warning: Non-normality for random effects '{label}' detected ({format_p(p)})."
            )
    table = pd.DataFrame(rows)
    return RandomEffectsNormalityResult(
        passed=bool(table["Passed"].all()),
        method=str(table["Method"].iloc[0]),
        statistic=None,
        p_value=float(table["p"].min()),
        message="\n".join(lines),
        adapter=adapter,
        table=table,
        effects=blups,
    )
