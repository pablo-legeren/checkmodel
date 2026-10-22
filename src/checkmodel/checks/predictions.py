"""Posterior predictive checks: compare data simulated from the model with the data."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .._bayes import BayesianModel, as_bayesian
from .._introspection import as_adapter
from .._types import ArrayLike, CheckNotApplicableError, FloatArray, Model
from ._base import CheckResult, format_p

__all__ = ["PredictionsResult", "check_predictions"]


@dataclass(repr=False)
class PredictionsResult(CheckResult):
    """Result of :func:`check_predictions`.

    Attributes
    ----------
    observed
        The observed response.
    simulated
        Replicated datasets, one per row (posterior predictive draws, or
        simulations from the fitted model).
    table
        For each test statistic: observed value, mean over replications and
        the posterior predictive p-value ``P(T(y_rep) >= T(y))``.
    discrete
        Whether the response takes few integer values (counts, 0/1).
    """

    key = "predictions"
    title = "Posterior predictive check"

    observed: FloatArray = field(default_factory=lambda: np.empty(0))
    simulated: FloatArray = field(default_factory=lambda: np.empty((0, 0)))
    table: pd.DataFrame = field(default_factory=pd.DataFrame)
    discrete: bool = False

    def to_dataframe(self) -> pd.DataFrame:
        """Return one row per test statistic."""
        return self.table.copy()


def _statistics(discrete: bool, binary: bool) -> dict[str, Any]:
    if binary:
        return {"Proportion of 1s": np.mean}
    stats_: dict[str, Any] = {
        "Mean": np.mean,
        "SD": lambda v, axis=None: np.std(v, axis=axis, ddof=1),
    }
    if discrete:
        stats_["Proportion of 0s"] = lambda v, axis=None: np.mean(v == 0, axis=axis)
    return stats_


def check_predictions(
    model: Model,
    X: ArrayLike | None = None,
    y: ArrayLike | None = None,
    *,
    iterations: int = 50,
    alpha: float = 0.05,
    random_state: int | None = None,
) -> PredictionsResult:
    """Compare data replicated from the model with the observed data.

    Port of ``performance::check_predictions()``. For Bayesian models the
    replications are posterior predictive draws; for frequentist models they
    are simulated from the fitted distribution (``simulate()`` in R; mixed
    models draw new group effects). The plot overlays the replicated and
    observed distributions; the table reports posterior predictive p-values
    for the mean, the SD and, for counts, the share of zeros.

    Parameters
    ----------
    model
        A frequentist model, or a Bayesian one from :func:`checkmodel.bayesian`.
    X, y
        Training data; required for scikit-learn models.
    iterations
        Number of replicated datasets to simulate and plot. For Bayesian
        models the p-values use every posterior predictive draw.
    alpha
        The check fails when a p-value falls outside ``[alpha/2, 1 - alpha/2]``.
    random_state
        Seed for the simulations and for choosing which draws to plot.

    Returns
    -------
    PredictionsResult
    """
    rng = np.random.default_rng(random_state)
    bayes: BayesianModel | None = as_bayesian(model)
    if bayes is not None:
        if bayes.y_rep is None:
            raise CheckNotApplicableError(
                "No posterior predictive draws: run `pm.sample_posterior_predictive()` or pass "
                "the model to `bayesian()`."
            )
        observed = bayes.y
        everything = BayesianModel.flat(bayes.y_rep)
        shown = everything[
            rng.choice(everything.shape[0], min(iterations, everything.shape[0]), replace=False)
        ]
        adapter: Any = bayes
        method = "Posterior predictive p-values"
    else:
        adapter = as_adapter(model, X, y)
        observed = adapter.y
        everything = shown = adapter.simulate(iterations, rng)
        method = "Simulation-based predictive p-values"

    binary = bool(np.all(np.isin(observed, (0.0, 1.0))))
    discrete = binary or bool(
        np.all(observed == np.round(observed)) and np.unique(observed).shape[0] <= 50
    )
    rows = []
    for label, fn in _statistics(discrete, binary).items():
        t_obs = float(fn(observed))
        t_rep = np.asarray(fn(everything, axis=1), dtype=float)
        rows.append(
            {
                "Statistic": label,
                "Observed": t_obs,
                "Predicted": float(t_rep.mean()),
                "p": float(np.mean(t_rep >= t_obs)),
            }
        )
    table = pd.DataFrame(rows)
    outside = table[(table["p"] < alpha / 2) | (table["p"] > 1 - alpha / 2)]
    passed = outside.empty
    if passed:
        summary = ", ".join(f"{r.Statistic.lower()} {format_p(r.p)}" for r in table.itertuples())
        message = f"OK: Model-predicted data resemble the observed data ({summary})."
    else:
        worst = ", ".join(f"{r.Statistic.lower()} ({format_p(r.p)})" for r in outside.itertuples())
        message = f"Warning: Observed data fall outside the model-predicted range for: {worst}."
    return PredictionsResult(
        passed=passed,
        method=method,
        statistic=None,
        p_value=float(table["p"].min()),
        message=message,
        adapter=adapter,
        observed=observed,
        simulated=shown,
        table=table,
        discrete=discrete,
    )
