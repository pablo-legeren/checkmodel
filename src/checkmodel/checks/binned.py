"""Binned residuals for logistic regression (Gelman & Hill, 2007)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd
from scipy import stats

from .._introspection import as_adapter
from .._types import ArrayLike, CheckNotApplicableError, Model
from ._base import CheckResult

__all__ = ["BinnedResidualsResult", "binned_residuals"]

_MIN_SHARE_INSIDE = 0.8
_GOOD_SHARE_INSIDE = 0.95


@dataclass(repr=False)
class BinnedResidualsResult(CheckResult):
    """Result of :func:`binned_residuals`.

    ``statistic`` is the share of bins whose mean residual lies inside the
    error bounds; ``table`` has one row per bin.
    """

    key = "binned_residuals"
    title = "Binned residuals"

    table: pd.DataFrame = field(default_factory=pd.DataFrame)

    def to_dataframe(self) -> pd.DataFrame:
        """Return the per-bin table."""
        return self.table.copy()


def _clopper_pearson(successes: int, n: int, level: float) -> tuple[float, float]:
    a = (1.0 - level) / 2.0
    lo = 0.0 if successes == 0 else stats.beta.ppf(a, successes, n - successes + 1)
    hi = 1.0 if successes == n else stats.beta.ppf(1.0 - a, successes + 1, n - successes)
    return float(lo), float(hi)


def binned_residuals(
    model: Model,
    X: ArrayLike | None = None,
    y: ArrayLike | None = None,
    *,
    term: str | None = None,
    n_bins: int | None = None,
    ci: float = 0.95,
    ci_type: Literal["exact", "gaussian"] = "exact",
    residuals: Literal["response", "pearson", "deviance"] | None = None,
) -> BinnedResidualsResult:
    """Average residuals within bins of the fitted values of a binomial model.

    Port of ``performance::binned_residuals()``. With a well-specified model
    about 95% of the binned mean residuals fall inside the error bounds; the
    check fails when fewer than 80% do.

    Parameters
    ----------
    model
        A fitted binomial model (logistic regression).
    X, y
        Training data; required for scikit-learn models.
    term
        Name of a design-matrix column to bin on instead of the fitted values.
    n_bins
        Number of bins; defaults to ``round(sqrt(n))``.
    ci
        Coverage of the error bounds.
    ci_type
        ``"exact"`` (Clopper-Pearson, the default for 0/1 outcomes) or
        ``"gaussian"``.
    residuals
        Residual type; ``"response"`` for 0/1 outcomes and ``"deviance"``
        otherwise by default.

    Returns
    -------
    BinnedResidualsResult
    """
    adapter = as_adapter(model, X, y)
    if not adapter.info.is_binomial:
        raise CheckNotApplicableError("Binned residuals are only used for binomial models.")
    bernoulli = adapter.info.is_bernoulli
    if not bernoulli:
        ci_type = "gaussian"
    if residuals is None:
        residuals = "response" if bernoulli else "deviance"
    fitted, y0 = adapter.mu, adapter.y
    if term is None:
        pred = fitted
    else:
        names = adapter.column_names()
        if term not in names:
            raise ValueError(f"Unknown term {term!r}; choose one of {names}.")
        pred = adapter.X[:, names.index(term)]
    if residuals == "response":
        resid = y0 - fitted
    elif residuals == "pearson":
        resid = (y0 - fitted) / np.sqrt(fitted * (1.0 - fitted))
    else:
        resid = adapter.deviance_residuals

    n = pred.shape[0]
    if n_bins is None:
        n_bins = round(np.sqrt(n))
    cut_idx = np.floor(n * np.arange(1, n_bins) / n_bins).astype(int) - 1
    breaks = np.unique(np.concatenate([[-np.inf], np.sort(pred)[cut_idx], [np.inf]]))
    bins = np.searchsorted(breaks, pred, side="left")  # R's cut(): right-closed intervals
    z = stats.norm.ppf((1.0 + ci) / 2.0)

    rows = []
    for b in range(1, n_bins + 1):
        members = bins == b
        m = int(members.sum())
        if m < 2:  # R drops bins whose standard deviation is undefined
            continue
        ybar = float(resid[members].mean())
        sdev = float(resid[members].std(ddof=1))
        if ci_type == "gaussian":
            lo, hi = stats.norm.ppf(
                [(1 - ci) / 2, (1 + ci) / 2], loc=ybar, scale=sdev / np.sqrt(m)
            )
        else:
            lo, hi = _clopper_pearson(round(float(y0[members].sum())), m, ci)
            width = hi - lo
            lo, hi = ybar - width / 2.0, ybar + width / 2.0
        se = z * sdev / np.sqrt(m)
        rows.append(
            {
                "xbar": float(pred[members].mean()),
                "ybar": ybar,
                "n": m,
                "x_lo": float(pred[members].min()),
                "x_hi": float(pred[members].max()),
                "se": se,
                "CI_low": float(lo),
                "CI_high": float(hi),
                "inside": abs(ybar) <= abs(se),
            }
        )
    table = pd.DataFrame(rows)
    share = float(table["inside"].mean()) if len(table) else float("nan")
    passed = bool(share >= _MIN_SHARE_INSIDE)
    pct = f"{share:.0%}"
    if share >= _GOOD_SHARE_INSIDE:
        message = f"OK: About {pct} of the residuals are inside the error bounds."
    elif passed:
        message = (
            f"Note: About {pct} of the residuals are inside the error bounds "
            "(~95% or higher would be good)."
        )
    else:
        message = (
            f"Warning: Probably bad model fit. Only about {pct} of the residuals are inside "
            "the error bounds."
        )
    return BinnedResidualsResult(
        passed=passed,
        method="Binned residuals",
        statistic=share,
        p_value=None,
        message=message,
        adapter=adapter,
        table=table,
    )
