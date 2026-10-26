"""Simulated quantile residuals (DHARMa) and the test of their uniformity."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd
from scipy import stats

from .._introspection import BaseAdapter, as_adapter
from .._types import ArrayLike, FloatArray, Model
from ._base import CheckResult, format_p, warn_about_fit

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

__all__ = ["ResidualsResult", "SimulatedResiduals", "check_residuals", "simulate_residuals"]

Alternative = Literal["two-sided", "less", "greater"]


def _alternative(value: str) -> Alternative:
    value = {"two.sided": "two-sided"}.get(value, value)
    if value not in ("two-sided", "less", "greater"):
        raise ValueError(f"`alternative` must be 'two-sided', 'less' or 'greater', got {value!r}.")
    return value  # type: ignore[return-value]


def simulation_p_value(observed: float, simulated: FloatArray, alternative: str) -> float:
    """p-value of an observed statistic against its simulated distribution.

    Matches ``.simres_statistics()`` in R's ``performance``: the share of
    simulated values at least as extreme, doubled for a two-sided test.
    """
    alternative = _alternative(alternative)
    if alternative == "greater":
        return float(np.mean(simulated >= observed))
    if alternative == "less":
        return float(np.mean(simulated <= observed))
    lower = float(np.mean(simulated <= observed))
    upper = float(np.mean(simulated >= observed))
    return min(2.0 * min(lower, upper), 1.0)


@dataclass(repr=False)
class SimulatedResiduals:
    """Randomized quantile residuals simulated from a fitted model.

    Returned by :func:`simulate_residuals`; can be passed to
    :func:`check_residuals`, :func:`~checkmodel.check_overdispersion` and
    :func:`~checkmodel.check_zeroinflation` so that they all use the same
    simulations, as R's ``performance_simres`` objects.

    Attributes
    ----------
    adapter
        The model the responses were simulated from.
    observed
        Observed response.
    fitted
        Fitted values on the response scale.
    simulated
        Simulated responses, shape ``(iterations, n_obs)``: one replicated
        dataset per row.
    scaled_residuals
        Quantile residuals in ``[0, 1]``: where each observation falls in the
        distribution of its simulated values. Uniform if the model is right.
    """

    adapter: BaseAdapter
    observed: FloatArray
    fitted: FloatArray
    simulated: FloatArray
    scaled_residuals: FloatArray = field(default_factory=lambda: np.empty(0))

    @property
    def iterations(self) -> int:
        """Number of simulated datasets."""
        return int(self.simulated.shape[0])

    def residuals(
        self,
        quantile_function: Callable[[FloatArray], FloatArray] | None = None,
        outlier_values: tuple[float, float] | None = None,
    ) -> FloatArray:
        """Return the scaled residuals, optionally transformed.

        Parameters
        ----------
        quantile_function
            Applied to the residuals, e.g. :func:`scipy.stats.norm.ppf` for
            residuals that are standard normal when the model is right.
        outlier_values
            Values that replace ``-inf`` and ``inf`` after the transformation,
            which arise for observations outside the range of all simulations.

        Returns
        -------
        numpy.ndarray
        """
        res = np.array(self.scaled_residuals, dtype=float)
        if quantile_function is None:
            return res
        res = np.asarray(quantile_function(res), dtype=float)
        if outlier_values is not None:
            if len(outlier_values) != 2:
                raise ValueError("`outlier_values` must have two values, for -inf and inf.")
            res[res == -np.inf] = outlier_values[0]
            res[res == np.inf] = outlier_values[1]
        return res

    def to_dataframe(self) -> pd.DataFrame:
        """One row per observation: observed and fitted values and the scaled residual."""
        return pd.DataFrame(
            {
                "Observed": self.observed,
                "Fitted": self.fitted,
                "Residual": self.scaled_residuals,
            },
            index=self.adapter.index(),
        )

    def plot(self, ax: Axes | None = None) -> Figure:
        """Q-Q plot of the scaled residuals against the uniform distribution."""
        from ..plotting.diagnostic_plots import plot_uniformity

        return plot_uniformity(self, ax=ax)

    def __repr__(self) -> str:
        return (
            f"Simulated residuals from a {self.adapter.info.model_class} model based on "
            f"{self.iterations} simulations. Use `check_residuals()` to check uniformity of "
            "the residuals or `.residuals()` to extract them."
        )

    __str__ = __repr__


def _quantile_residuals(
    simulated: FloatArray, observed: FloatArray, rng: np.random.Generator
) -> FloatArray:
    # DHARMa's PIT residuals: the share of simulations below the observation,
    # randomized between "<" and "<=" when simulations tie with it (discrete data).
    lower = np.mean(simulated < observed, axis=0)
    lower_or_equal = np.mean(simulated <= observed, axis=0)
    tied = lower != lower_or_equal
    out = lower.copy()
    out[tied] = rng.uniform(lower[tied], lower_or_equal[tied])
    return out


def simulate_residuals(
    model: Model,
    X: ArrayLike | None = None,
    y: ArrayLike | None = None,
    *,
    iterations: int = 250,
    random_state: int | np.random.Generator | None = None,
) -> SimulatedResiduals:
    """Simulate randomized quantile residuals from a model.

    Port of ``performance::simulate_residuals()``, which wraps
    ``DHARMa::simulateResiduals()``. New responses are drawn ``iterations``
    times from the fitted model (mixed models draw new group effects); each
    observation's residual is the share of its simulated values that fall
    below it, with ties randomized (Dunn & Smyth, 1996). If the model is right
    the residuals are uniform on ``[0, 1]`` whatever its family, which makes
    them the residuals of choice for non-Gaussian models.

    Parameters
    ----------
    model
        A fitted frequentist model.
    X, y
        Training data; required for scikit-learn models.
    iterations
        Number of simulated datasets, at least 2. R's default is 250.
    random_state
        Seed or generator, for reproducible residuals.

    Returns
    -------
    SimulatedResiduals

    Raises
    ------
    CheckNotApplicableError
        When responses cannot be simulated for the model's family.
    """
    if iterations < 2:
        raise ValueError("`iterations` must be at least 2.")
    adapter = as_adapter(model, X, y)
    rng = np.random.default_rng(random_state)
    simulated = adapter.simulate(iterations, rng)
    observed = adapter.y
    return SimulatedResiduals(
        adapter=adapter,
        observed=observed,
        fitted=adapter.mu,
        simulated=simulated,
        scaled_residuals=_quantile_residuals(simulated, observed, rng),
    )


def ks_test(
    values: FloatArray, cdf: Callable[[FloatArray], FloatArray], alternative: str
) -> tuple[float, float]:
    """One-sample Kolmogorov-Smirnov test, computed as R's ``ks.test()``.

    The p-value is exact below 100 values without ties and asymptotic
    otherwise, as in R.
    """
    alternative = _alternative(alternative)
    x = np.sort(np.asarray(values, dtype=float))
    n = x.shape[0]
    F = np.asarray(cdf(x), dtype=float)
    i = np.arange(1, n + 1)
    d_plus = float(np.max(i / n - F))
    d_minus = float(np.max(F - (i - 1) / n))
    exact = n < 100 and np.unique(x).shape[0] == n
    if alternative == "two-sided":
        d = max(d_plus, d_minus)
        p = stats.kstwo.sf(d, n) if exact else stats.kstwobign.sf(math.sqrt(n) * d)
    else:
        d = d_plus if alternative == "greater" else d_minus
        p = stats.ksone.sf(d, n) if exact else math.exp(-2.0 * n * d**2)
    return d, float(min(max(p, 0.0), 1.0))


def _uniform_cdf(x: FloatArray) -> FloatArray:
    return np.clip(x, 0.0, 1.0)


@dataclass(repr=False)
class ResidualsResult(CheckResult):
    """Result of :func:`check_residuals`.

    ``statistic`` is the Kolmogorov-Smirnov ``D``.

    Attributes
    ----------
    simulation
        The :class:`SimulatedResiduals` that were tested.
    alternative
        Alternative hypothesis of the test.
    """

    key = "residuals"
    title = "Uniformity of residuals"

    simulation: SimulatedResiduals | None = None
    alternative: str = "two-sided"

    @property
    def residuals(self) -> FloatArray:
        """The scaled (quantile) residuals."""
        if self.simulation is None:
            return np.empty(0)
        return self.simulation.scaled_residuals


def check_residuals(
    model: Model | SimulatedResiduals,
    X: ArrayLike | None = None,
    y: ArrayLike | None = None,
    *,
    alternative: str = "two-sided",
    distribution: Callable[[FloatArray], FloatArray] | None = None,
    iterations: int = 250,
    alpha: float = 0.05,
    random_state: int | np.random.Generator | None = None,
) -> ResidualsResult:
    """Test whether simulated quantile residuals are uniformly distributed.

    Port of ``performance::check_residuals()``. The residuals from
    :func:`simulate_residuals` are uniform when the model is right, so a
    Kolmogorov-Smirnov test against the uniform distribution detects a wrong
    family, over- or underdispersion, zero-inflation and similar problems in
    any model, Gaussian or not.

    Parameters
    ----------
    model
        A fitted model, or the output of :func:`simulate_residuals`.
    X, y
        Training data; required for scikit-learn models.
    alternative
        ``"two-sided"`` (default), ``"less"`` or ``"greater"``, as in
        :func:`scipy.stats.kstest`. R's ``"two.sided"`` is accepted too.
    distribution
        Cumulative distribution function to test against; uniform by default.
    iterations
        Number of simulations when ``model`` is a model.
    alpha
        Significance level. The check fails when ``p < alpha``.
    random_state
        Seed for the simulations when ``model`` is a model.

    Returns
    -------
    ResidualsResult
    """
    alternative = _alternative(alternative)
    if isinstance(model, SimulatedResiduals):
        simulation = model
    else:
        simulation = simulate_residuals(
            model, X, y, iterations=iterations, random_state=random_state
        )
    warn_about_fit(simulation.adapter, penalized=False)
    cdf = _uniform_cdf if distribution is None else distribution
    d, p = ks_test(simulation.scaled_residuals, cdf, alternative)
    passed = not p < alpha
    if passed:
        message = f"OK: Simulated residuals appear as uniformly distributed ({format_p(p)})."
    else:
        message = f"Warning: Non-uniformity of simulated residuals detected ({format_p(p)})."
    return ResidualsResult(
        passed=passed,
        method="Kolmogorov-Smirnov test of simulated quantile residuals",
        statistic=d,
        p_value=p,
        message=message,
        adapter=simulation.adapter,
        simulation=simulation,
        alternative=alternative,
    )
