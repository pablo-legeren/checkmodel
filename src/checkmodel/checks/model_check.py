"""``check_model()``: run every check that applies to a model."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Mapping
from functools import partial
from typing import TYPE_CHECKING, Any

import pandas as pd

from .._bayes import BayesianModel, as_bayesian
from .._introspection import BaseAdapter, as_adapter
from .._types import ArrayLike, CheckNotApplicableError, Model, UnsupportedModelError
from ._base import CheckResult, fit_warnings_issued, penalized_note, warn_about_fit
from .autocorrelation import check_autocorrelation
from .binned import binned_residuals
from .collinearity import check_collinearity
from .convergence import check_convergence
from .heteroscedasticity import check_heteroscedasticity
from .normality import check_normality
from .outliers import check_outliers
from .overdispersion import check_overdispersion
from .predictions import check_predictions
from .residuals import check_residuals
from .singularity import check_singularity
from .zeroinflation import check_zeroinflation

if TYPE_CHECKING:
    from matplotlib.figure import Figure

__all__ = ["ModelCheckReport", "check_model"]

_CHECKS: dict[str, Callable[..., CheckResult]] = {
    "predictions": check_predictions,
    "normality": check_normality,
    "heteroscedasticity": check_heteroscedasticity,
    "collinearity": check_collinearity,
    "outliers": check_outliers,
    "autocorrelation": check_autocorrelation,
    "binned_residuals": binned_residuals,
    "random_effects": partial(check_normality, effects="random"),
    "singularity": check_singularity,
    "convergence": check_convergence,
    "overdispersion": check_overdispersion,
    "zeroinflation": check_zeroinflation,
    "residuals": check_residuals,
}
_ALPHA_CHECKS = {
    "predictions",
    "normality",
    "heteroscedasticity",
    "autocorrelation",
    "random_effects",
    "overdispersion",
    "residuals",
}
_RANDOM_CHECKS = {"predictions", "overdispersion", "zeroinflation", "residuals"}


def _relevant_checks(adapter: BaseAdapter | BayesianModel) -> list[str]:
    if isinstance(adapter, BayesianModel):
        return ["predictions", "convergence"]
    return ["predictions", *_frequentist_checks(adapter)]


def _frequentist_checks(adapter: BaseAdapter) -> list[str]:
    info = adapter.info
    if info.is_mixed:
        return [
            "normality",
            "random_effects",
            "heteroscedasticity",
            "collinearity",
            "outliers",
            "singularity",
            "convergence",
        ]
    if info.is_linear:
        checks = ["normality", "heteroscedasticity", "collinearity", "outliers"]
        # Serial correlation only means something when the rows have an order. In
        # cross-sectional data the Durbin-Watson test just reacts to how the rows
        # happen to be sorted, so, like R's check_model(), leave it out there.
        if isinstance(adapter.index(), (pd.DatetimeIndex, pd.PeriodIndex)):
            checks.append("autocorrelation")
        return checks
    # Non-Gaussian models: simulated quantile residuals take the place of the
    # normal Q-Q plot, as `residual_type = "simulated"` does in R.
    if info.is_binomial:
        return ["collinearity", "outliers", "binned_residuals", "residuals"]
    if info.is_count:
        return ["collinearity", "outliers", "overdispersion", "residuals"]
    return ["collinearity", "outliers", "residuals"]


class ModelCheckReport(Mapping[str, CheckResult]):
    """All checks run by :func:`check_model`, accessible by name.

    Behaves like a read-only dictionary, ``report["normality"]``, and adds
    :meth:`summary` and :meth:`plot`.

    Attributes
    ----------
    skipped
        Checks that were requested but do not apply, with the reason.
    """

    def __init__(
        self,
        adapter: BaseAdapter | BayesianModel,
        results: dict[str, CheckResult],
        skipped: dict[str, str],
    ) -> None:
        self.adapter = adapter
        self._results = results
        self.skipped = skipped

    def __getitem__(self, key: str) -> CheckResult:
        try:
            return self._results[key]
        except KeyError:
            if key in self.skipped:
                raise KeyError(f"{key!r} was skipped: {self.skipped[key]}") from None
            raise KeyError(f"{key!r} not in report; available: {list(self._results)}") from None

    def __iter__(self) -> Iterator[str]:
        return iter(self._results)

    def __len__(self) -> int:
        return len(self._results)

    @property
    def passed(self) -> bool:
        """``True`` when every check that ran passed."""
        return all(r.passed for r in self._results.values())

    def summary(self) -> pd.DataFrame:
        """One row per check: method, statistic, p-value, verdict and message."""
        rows = [r._summary_row() for r in self._results.values()]
        return pd.DataFrame(rows, index=pd.Index(list(self._results), name="check"))

    def to_dataframe(self) -> pd.DataFrame:
        """Alias of :meth:`summary`."""
        return self.summary()

    def plot(self, **kwargs: Any) -> Figure:
        """Draw every diagnostic in one grid (up to 2 rows x 3 columns).

        Keyword arguments are passed to
        :func:`checkmodel.plotting.plot_model_check`.
        """
        from ..plotting.diagnostic_plots import plot_model_check

        return plot_model_check(self, **kwargs)

    def __repr__(self) -> str:
        info = self.adapter.info
        header = (
            f"# Model checks: {info.model_class} ({info.family}, {info.link} link), "
            f"n = {self.adapter.n_obs}"
        )
        lines = [header]
        note = penalized_note(self.adapter) if isinstance(self.adapter, BaseAdapter) else None
        if note is not None:
            lines.append(f"Note: {note}")
        lines.append("")
        lines += [f"{r.title}: " + r.message.replace("\n", "\n  ") for r in self._results.values()]
        if self.skipped:
            lines += ["", "Skipped:"]
            lines += [f"  {k}: {why}" for k, why in self.skipped.items()]
        return "\n".join(lines)

    __str__ = __repr__


def check_model(
    model: Model,
    X: ArrayLike | None = None,
    y: ArrayLike | None = None,
    *,
    checks: str | Iterable[str] = "all",
    alpha: float = 0.05,
    random_state: int | None = 0,
) -> ModelCheckReport:
    """Run every assumption check that applies to ``model``.

    The model type decides what runs, as in ``performance::check_model()``.
    Every model gets a posterior predictive check first, then:

    * linear models: normality, heteroscedasticity, collinearity and
      outliers, plus autocorrelation when the observations are indexed by
      time (a :class:`pandas.DatetimeIndex` or :class:`pandas.PeriodIndex`);
    * linear mixed models: normality of residuals and of random effects,
      heteroscedasticity, collinearity, outliers, singularity and convergence;
    * logistic regression: collinearity, outliers, binned residuals and
      uniformity of simulated residuals;
    * count models (Poisson, negative binomial): collinearity, outliers,
      overdispersion and uniformity of simulated residuals;
    * other GLMs: collinearity, outliers and uniformity of simulated residuals;
    * Bayesian models (see :func:`checkmodel.bayesian`): convergence.

    Parameters
    ----------
    model
        A fitted statsmodels or scikit-learn model, or a Bayesian model.
    X, y
        Training data; required for scikit-learn models.
    checks
        ``"all"`` for every relevant check, or an iterable of names among
        ``"predictions"``, ``"normality"``, ``"random_effects"``, ``"heteroscedasticity"``,
        ``"collinearity"``, ``"outliers"``, ``"autocorrelation"``,
        ``"binned_residuals"``, ``"singularity"``, ``"convergence"``,
        ``"overdispersion"``, ``"zeroinflation"`` and ``"residuals"``.
        Named checks run even when they are not in the default set, so
        ``checks=[..., "autocorrelation"]`` tests serial correlation of a
        model fitted on rows without a time index.
    alpha
        Significance level for the formal tests.
    random_state
        Seed for the simulations (predictive check, simulated residuals), so
        that reports are reproducible; ``None`` draws a fresh seed.

    Returns
    -------
    ModelCheckReport

    Warns
    -----
    UserWarning
        Once per call, for a penalized fit (ridge, lasso, ...), whose leverage,
        Cook's distance and standardized residuals ignore the penalty, and for
        a binomial model with complete separation.

    Examples
    --------
    >>> report = check_model(fit)  # doctest: +SKIP
    >>> report.plot()  # doctest: +SKIP
    >>> report["normality"].p_value  # doctest: +SKIP
    >>> check_model(fit, checks=["normality", "autocorrelation"])  # doctest: +SKIP
    """
    bayes = as_bayesian(model)
    adapter: BaseAdapter | BayesianModel = bayes if bayes is not None else as_adapter(model, X, y)
    relevant = _relevant_checks(adapter)
    if checks == "all":
        wanted = relevant
    else:
        wanted = [checks] if isinstance(checks, str) else list(checks)
        unknown = [c for c in wanted if c not in _CHECKS]
        if unknown:
            raise ValueError(f"Unknown checks {unknown}; choose from {list(_CHECKS)}.")
    if isinstance(adapter, BaseAdapter):
        warn_about_fit(adapter)
    results: dict[str, CheckResult] = {}
    skipped: dict[str, str] = {}
    with fit_warnings_issued():
        for name in wanted:
            kwargs: dict[str, Any] = {"alpha": alpha} if name in _ALPHA_CHECKS else {}
            if name in _RANDOM_CHECKS:
                kwargs["random_state"] = random_state
            try:
                results[name] = _CHECKS[name](adapter, **kwargs)
            except (CheckNotApplicableError, UnsupportedModelError) as exc:
                skipped[name] = str(exc)
    return ModelCheckReport(adapter, results, skipped)
