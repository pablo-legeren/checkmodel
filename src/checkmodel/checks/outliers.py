"""Influential observations via Cook's distance."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd
from scipy import stats

from .._introspection import BaseAdapter, as_adapter
from .._types import ArrayLike, Model
from ._base import CheckResult

__all__ = ["OutliersResult", "check_outliers"]


@dataclass(repr=False)
class OutliersResult(CheckResult):
    """Result of :func:`check_outliers`.

    Attributes
    ----------
    distances
        Cook's distance per observation, indexed by the observation labels.
    threshold
        Cut-off above which an observation counts as influential.
    outliers
        Labels of the influential observations.
    """

    key = "outliers"
    title = "Influential observations"

    distances: pd.Series = field(default_factory=pd.Series)
    threshold: float = float("nan")
    outliers: list[Any] = field(default_factory=list)

    def to_dataframe(self) -> pd.DataFrame:
        """Return one row per observation with its influence measures."""
        a = self.adapter
        return pd.DataFrame(
            {
                "Cooks_Distance": self.distances.to_numpy(),
                "Leverage": a.hat,
                "Std_Residual": a.std_residuals,
                "Outlier": self.distances.to_numpy() > self.threshold,
            },
            index=self.distances.index,
        )


def _default_threshold(adapter: BaseAdapter) -> float:
    # performance:::.check_outliers_thresholds(): median of F(k, n - k), where k
    # is the number of numeric variables in the model data.
    k = adapter.n_numeric_variables()
    return float(stats.f.ppf(0.5, k, adapter.n_obs - k))


def check_outliers(
    model: Model,
    X: ArrayLike | None = None,
    y: ArrayLike | None = None,
    *,
    threshold: float | Literal["f", "4/n"] = "f",
) -> OutliersResult:
    """Find influential observations with Cook's distance.

    Port of ``performance::check_outliers(method = "cook")``.

    Parameters
    ----------
    model
        Any supported model.
    X, y
        Training data; required for scikit-learn models.
    threshold
        ``"f"`` (default) uses R's rule, the median of an F distribution
        with ``k`` and ``n - k`` degrees of freedom, where ``k`` is the number
        of numeric variables in the model. ``"4/n"`` uses the common
        rule of thumb (which flags many more points). A number sets the
        cut-off directly.

    Returns
    -------
    OutliersResult

    Notes
    -----
    For penalized scikit-learn models (ridge, lasso, elastic net) leverage
    comes from the design matrix, as if the model were unpenalized; the
    residuals are the penalized model's own.
    """
    adapter = as_adapter(model, X, y)
    n = adapter.n_obs
    if threshold == "f":
        cutoff = _default_threshold(adapter)
        rule = "Cook's distance > F median"
    elif threshold == "4/n":
        cutoff = 4.0 / n
        rule = "Cook's distance > 4/n"
    else:
        cutoff = float(threshold)
        rule = "Cook's distance > threshold"
    distances = pd.Series(adapter.cooks_distance, index=adapter.index(), name="Cooks_Distance")
    flagged = np.nan_to_num(distances.to_numpy(), nan=0.0) > cutoff
    outliers = list(distances.index[flagged])
    passed = not outliers
    if passed:
        message = f"OK: No outliers detected (Cook's distance > {cutoff:.3g})."
    else:
        noun = "outlier" if len(outliers) == 1 else "outliers"
        cases = "case" if len(outliers) == 1 else "cases"
        listed = ", ".join(str(o) for o in outliers[:10])
        if len(outliers) > 10:
            listed += f" and {len(outliers) - 10} more"
        message = (
            f"Warning: {len(outliers)} {noun} detected: {cases} {listed} "
            f"(Cook's distance > {cutoff:.3g})."
        )
    return OutliersResult(
        passed=passed,
        method=rule,
        statistic=float(np.nanmax(distances.to_numpy())),
        p_value=None,
        message=message,
        adapter=adapter,
        distances=distances,
        threshold=cutoff,
        outliers=outliers,
    )
