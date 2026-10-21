"""Coefficients of determination and pseudo-R² measures."""

from __future__ import annotations

from collections.abc import Iterator, Mapping

import numpy as np
import pandas as pd

from .._introspection import BaseAdapter, as_adapter
from .._types import ArrayLike, CheckNotApplicableError, Model

__all__ = ["R2Result", "r2", "r2_coxsnell", "r2_mcfadden", "r2_nagelkerke", "r2_tjur"]

_LABELS = {
    "R2": "R2",
    "R2_adjusted": "adj. R2",
    "R2_Tjur": "Tjur's R2",
    "R2_Nagelkerke": "Nagelkerke's R2",
    "R2_conditional": "Conditional R2",
    "R2_marginal": "Marginal R2",
}


class R2Result(Mapping[str, float]):
    """Read-only mapping of R² measures, e.g. ``{"R2": 0.83, "R2_adjusted": 0.81}``."""

    def __init__(self, values: dict[str, float], model_type: str) -> None:
        self._values = values
        self.model_type = model_type

    def __getitem__(self, key: str) -> float:
        return self._values[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._values)

    def __len__(self) -> int:
        return len(self._values)

    def to_dataframe(self) -> pd.DataFrame:
        """Return the measures as a one-row :class:`pandas.DataFrame`."""
        return pd.DataFrame([self._values])

    def __repr__(self) -> str:
        width = max(len(_LABELS.get(k, k)) for k in self._values)
        lines = [f"# R2 for {self.model_type}"]
        lines += [f"  {_LABELS.get(k, k):>{width}}: {v:.3f}" for k, v in self._values.items()]
        return "\n".join(lines)

    __str__ = __repr__


def _require_bernoulli(adapter: BaseAdapter) -> None:
    if adapter.info.is_binomial and not adapter.info.is_bernoulli:
        raise CheckNotApplicableError(
            "Can't calculate accurate R2 for binomial models that are not Bernoulli models."
        )


def _r2_linear(adapter: BaseAdapter) -> R2Result:
    # summary.lm(): weighted, centered when there is an intercept.
    w, f, e = adapter.weights, adapter.mu, adapter.resid
    rss = float(np.sum(w * e**2))
    if adapter.info.is_penalized:
        # Fitted values and residuals are not orthogonal under a penalty, so
        # use 1 - RSS/TSS (sklearn's `score`) rather than MSS / (MSS + RSS).
        ybar = np.average(adapter.y, weights=w)
        value = 1.0 - rss / float(np.sum(w * (adapter.y - ybar) ** 2))
    else:
        if adapter.has_intercept:
            mss = float(np.sum(w * (f - np.average(f, weights=w)) ** 2))
        else:
            mss = float(np.sum(w * f**2))
        value = mss / (mss + rss)
    n = adapter.n_obs - int(np.sum(w == 0))
    df_int = int(adapter.has_intercept)
    adjusted = 1.0 - (1.0 - value) * ((n - df_int) / adapter.df_resid)
    return R2Result({"R2": value, "R2_adjusted": adjusted}, "Linear Regression")


def r2(model: Model, X: ArrayLike | None = None, y: ArrayLike | None = None) -> R2Result:
    """Compute the R² measure that suits the model, as ``performance::r2()``.

    * Linear models: R² and adjusted R².
    * Gaussian GLMs: R² (``1 - SSE/SST``).
    * Logistic regression with a 0/1 outcome: Tjur's R².
    * Other GLMs (Poisson, ...): Nagelkerke's R².
    * Linear mixed models: Nakagawa's conditional and marginal R².

    Parameters
    ----------
    model
        Any supported model.
    X, y
        Training data; required for scikit-learn models.

    Returns
    -------
    R2Result
        Mapping from measure name to value.
    """
    adapter = as_adapter(model, X, y)
    info = adapter.info
    if info.is_mixed:
        from .icc import r2_nakagawa

        return r2_nakagawa(adapter)
    if info.is_linear and info.model_class != "GLM":
        return _r2_linear(adapter)
    if info.family in {"gaussian", "inverse.gaussian"}:
        tss = float(np.sum((adapter.y - adapter.y.mean()) ** 2))
        return R2Result(
            {"R2": 1.0 - float(np.sum(adapter.resid**2)) / tss}, "Generalized Linear Regression"
        )
    if info.is_binomial:
        _require_bernoulli(adapter)
        return R2Result({"R2_Tjur": r2_tjur(adapter)}, "Logistic Regression")
    return R2Result({"R2_Nagelkerke": r2_nagelkerke(adapter)}, "Generalized Linear Regression")


def r2_tjur(model: Model, X: ArrayLike | None = None, y: ArrayLike | None = None) -> float:
    """Tjur's coefficient of discrimination for binary outcomes.

    The difference between the mean predicted probability of the two
    outcome groups (Tjur, 2009).
    """
    adapter = as_adapter(model, X, y)
    if not adapter.info.is_binomial:
        raise CheckNotApplicableError("Tjur's R2 requires a binomial model.")
    y0, p = adapter.y, adapter.mu
    return float(abs(p[y0 == 1].mean() - p[y0 == 0].mean()))


def r2_coxsnell(model: Model, X: ArrayLike | None = None, y: ArrayLike | None = None) -> float:
    """Cox & Snell's pseudo-R², ``1 - exp((D - D0) / n)``."""
    adapter = as_adapter(model, X, y)
    _require_bernoulli(adapter)
    return float(1.0 - np.exp((adapter.deviance - adapter.null_deviance) / adapter.n_obs))


def r2_nagelkerke(model: Model, X: ArrayLike | None = None, y: ArrayLike | None = None) -> float:
    """Nagelkerke's pseudo-R²: Cox & Snell's rescaled to reach 1."""
    adapter = as_adapter(model, X, y)
    _require_bernoulli(adapter)
    upper = 1.0 - np.exp(-adapter.null_deviance / adapter.n_obs)
    return float(r2_coxsnell(adapter) / upper)


def r2_mcfadden(model: Model, X: ArrayLike | None = None, y: ArrayLike | None = None) -> R2Result:
    """McFadden's pseudo-R² and its adjusted version, ``1 - (LL - k) / LL0``."""
    adapter = as_adapter(model, X, y)
    _require_bernoulli(adapter)
    ll, ll0 = adapter.loglik, adapter.loglik_null
    if ll is None or ll0 is None:
        raise CheckNotApplicableError("McFadden's R2 needs the model log-likelihood.")
    k = adapter.n_params
    return R2Result(
        {"R2": 1.0 - ll / ll0, "R2_adjusted": 1.0 - (ll - k) / ll0},
        "Generalized Linear Regression",
    )
