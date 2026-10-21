"""``check_model()``: run every check that applies to a model."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Mapping
from functools import partial
from typing import TYPE_CHECKING, Any

import pandas as pd

from .._introspection import BaseAdapter, as_adapter
from .._types import ArrayLike, CheckNotApplicableError, Model
from ._base import CheckResult
from .autocorrelation import check_autocorrelation
from .binned import binned_residuals
from .collinearity import check_collinearity
from .convergence import check_convergence
from .heteroscedasticity import check_heteroscedasticity
from .normality import check_normality
from .outliers import check_outliers
from .overdispersion import check_overdispersion
from .singularity import check_singularity
from .zeroinflation import check_zeroinflation

if TYPE_CHECKING:
    from matplotlib.figure import Figure

__all__ = ["ModelCheckReport", "check_model"]

_CHECKS: dict[str, Callable[..., CheckResult]] = {
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
}
_ALPHA_CHECKS = {
    "normality",
    "heteroscedasticity",
    "autocorrelation",
    "random_effects",
    "overdispersion",
}


def _relevant_checks(adapter: BaseAdapter) -> list[str]:
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
        return ["normality", "heteroscedasticity", "collinearity", "outliers", "autocorrelation"]
    if info.is_binomial:
        return ["collinearity", "outliers", "binned_residuals"]
    if info.family == "poisson":
        return ["collinearity", "outliers", "overdispersion"]
    return ["collinearity", "outliers"]


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
        adapter: BaseAdapter,
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
        lines = [header, ""]
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
) -> ModelCheckReport:
    """Run every assumption check that applies to ``model``.

    The model type decides what runs, as in ``performance::check_model()``:

    * linear models: normality, heteroscedasticity, collinearity, outliers
      and autocorrelation;
    * linear mixed models: normality of residuals and of random effects,
      heteroscedasticity, collinearity, outliers, singularity and convergence;
    * logistic regression: collinearity, outliers and binned residuals;
    * Poisson regression: collinearity, outliers and overdispersion;
    * other GLMs: collinearity and outliers.

    Parameters
    ----------
    model
        A fitted statsmodels or scikit-learn model.
    X, y
        Training data; required for scikit-learn models.
    checks
        ``"all"`` for every relevant check, or an iterable of names among
        ``"normality"``, ``"random_effects"``, ``"heteroscedasticity"``,
        ``"collinearity"``, ``"outliers"``, ``"autocorrelation"``,
        ``"binned_residuals"``, ``"singularity"``, ``"convergence"``,
        ``"overdispersion"`` and ``"zeroinflation"``.
    alpha
        Significance level for the formal tests.

    Returns
    -------
    ModelCheckReport

    Examples
    --------
    >>> report = check_model(fit)  # doctest: +SKIP
    >>> report.plot()  # doctest: +SKIP
    >>> report["normality"].p_value  # doctest: +SKIP
    """
    adapter = as_adapter(model, X, y)
    relevant = _relevant_checks(adapter)
    if checks == "all":
        wanted = relevant
    else:
        wanted = [checks] if isinstance(checks, str) else list(checks)
        unknown = [c for c in wanted if c not in _CHECKS]
        if unknown:
            raise ValueError(f"Unknown checks {unknown}; choose from {list(_CHECKS)}.")
    results: dict[str, CheckResult] = {}
    skipped: dict[str, str] = {}
    for name in wanted:
        kwargs: dict[str, Any] = {"alpha": alpha} if name in _ALPHA_CHECKS else {}
        try:
            results[name] = _CHECKS[name](adapter, **kwargs)
        except CheckNotApplicableError as exc:
            skipped[name] = str(exc)
    return ModelCheckReport(adapter, results, skipped)
