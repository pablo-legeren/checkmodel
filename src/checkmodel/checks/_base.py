"""Shared result type and formatting helpers for all checks."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar

import pandas as pd

from .._introspection import BaseAdapter

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure


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
