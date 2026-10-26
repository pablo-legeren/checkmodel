"""Shared result type and formatting helpers for all checks."""

from __future__ import annotations

import math
import warnings
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
import pandas as pd

from .._introspection import BaseAdapter

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

# Fitted probabilities this close to 0 or 1 count as perfectly predicted.
_SEPARATION_EPS = 1e-6

# Set while check_model() runs its checks, so that it warns about the fit once
# instead of once per check.
_FIT_WARNINGS_ISSUED: ContextVar[bool] = ContextVar("_FIT_WARNINGS_ISSUED", default=False)


def penalized_note(adapter: BaseAdapter) -> str | None:
    """Explain why influence measures of a penalized fit are approximate, or ``None``."""
    if not adapter.info.is_penalized:
        return None
    return (
        f"`{adapter.info.model_class}` is a penalized fit; leverage, Cook's distance and "
        "standardized residuals are computed as for the unpenalized model and are approximate."
    )


def is_separated(adapter: BaseAdapter) -> bool:
    """Whether every fitted probability of a binomial model is 0 or 1 (complete separation)."""
    if not adapter.info.is_binomial:
        return False
    mu = adapter.mu
    return bool(np.all((mu < _SEPARATION_EPS) | (mu > 1.0 - _SEPARATION_EPS)))


def warn_about_fit(
    adapter: BaseAdapter, *, penalized: bool = True, separation: bool = True
) -> None:
    """Warn when the diagnostics of ``adapter`` cannot be taken at face value.

    Covers penalized fits, whose leverage and standardized residuals ignore the
    penalty, and binomial models with complete separation. Silent inside
    :func:`check_model`, which issues these warnings once for all its checks.
    """
    if _FIT_WARNINGS_ISSUED.get():
        return
    note = penalized_note(adapter) if penalized else None
    if note is not None:
        warnings.warn(note, UserWarning, stacklevel=3)
    if separation and is_separated(adapter):
        warnings.warn(
            "Possible complete separation: fitted probabilities are 0 or 1; "
            "diagnostics are unreliable.",
            UserWarning,
            stacklevel=3,
        )


@contextmanager
def fit_warnings_issued() -> Iterator[None]:
    """Silence :func:`warn_about_fit` in the checks run inside this block."""
    token = _FIT_WARNINGS_ISSUED.set(True)
    try:
        yield
    finally:
        _FIT_WARNINGS_ISSUED.reset(token)


def format_p(p: float) -> str:
    """Format a p-value the way ``insight::format_p()`` does in R."""
    if math.isnan(p):
        return "p = NA"
    if p < 0.001:
        return "p < .001"
    if p > 0.999:
        return "p > .999"
    return f"p = {p:.3f}"


@dataclass(repr=False)
class CheckResult:
    """Outcome of a single assumption check.

    Attributes
    ----------
    passed
        Whether the assumption holds at the chosen threshold.
    method
        Name of the test or rule that was applied.
    statistic
        Test statistic, when the check is a formal test.
    p_value
        p-value of the test, when there is one.
    message
        One-line verdict in the style of R's ``performance`` package.
    """

    key: ClassVar[str] = ""
    title: ClassVar[str] = ""

    passed: bool
    method: str
    statistic: float | None
    p_value: float | None
    message: str
    adapter: BaseAdapter = field(compare=False)

    def summary(self) -> str:
        """Return the one-line verdict."""
        return self.message

    def __str__(self) -> str:
        return self.message

    def __repr__(self) -> str:
        return self.message

    def _summary_row(self) -> dict[str, Any]:
        return {
            "Check": self.title,
            "Method": self.method,
            "Statistic": self.statistic,
            "p": self.p_value,
            "Passed": self.passed,
            "Message": self.message,
        }

    def to_dataframe(self) -> pd.DataFrame:
        """Return the result as a one-row :class:`pandas.DataFrame`."""
        return pd.DataFrame([self._summary_row()])

    def plot(self, ax: Axes | None = None) -> Figure:
        """Draw the diagnostic plot for this check.

        Parameters
        ----------
        ax
            Axes to draw on. When omitted a new figure is created; some checks
            then add a second panel (normality adds a histogram).

        Returns
        -------
        matplotlib.figure.Figure
        """
        from ..plotting.diagnostic_plots import plot_check

        return plot_check(self, ax=ax)
