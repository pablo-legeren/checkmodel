"""Diagnostic plots for each check and the combined ``check_model()`` panel.

The look follows the ``see`` package in R: one idea per panel, a short title
saying what is shown, a subtitle saying what a good model looks like,
recessive axes and grid, and color only where it carries meaning.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from functools import partial
from typing import TYPE_CHECKING, Any, cast

import numpy as np
from scipy import stats

from .._introspection import BaseAdapter

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from ..checks._base import CheckResult
    from ..checks.model_check import ModelCheckReport

__all__ = [
    "PALETTE",
    "plot_autocorrelation",
    "plot_binned_residuals",
    "plot_check",
    "plot_collinearity",
    "plot_convergence",
    "plot_homogeneity",
    "plot_linearity",
    "plot_model_check",
    "plot_normality",
    "plot_outliers",
    "plot_overdispersion",
    "plot_predictions",
    "plot_qq",
    "plot_random_effects",
    "plot_random_effects_term",
    "plot_zeroinflation",
]

PALETTE = {
    "surface": "#fcfcfb",
    "ink": "#0b0b0b",
    "ink_secondary": "#52514e",
    "muted": "#898781",
    "grid": "#e1e0d9",
    "axis": "#c3c2b7",
    "points": "#2a78d6",
    "fit": "#eb6834",
    "good": "#0ca30c",
    "warning": "#fab219",
    "critical": "#d03b3b",
}
_P = PALETTE

_RC: dict[str, Any] = {
    "figure.facecolor": _P["surface"],
    "axes.facecolor": _P["surface"],
    "savefig.facecolor": _P["surface"],
    "axes.edgecolor": _P["axis"],
    "axes.linewidth": 0.8,
    "axes.labelcolor": _P["ink_secondary"],
    "axes.labelsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.color": _P["grid"],
    "grid.linewidth": 0.6,
    "xtick.color": _P["axis"],
    "ytick.color": _P["axis"],
    "xtick.labelcolor": _P["muted"],
    "ytick.labelcolor": _P["muted"],
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.frameon": False,
    "legend.fontsize": 8,
    "font.size": 9,
}

_MAX_SMOOTH_POINTS = 5000


@contextmanager
def _style() -> Iterator[None]:
    import matplotlib.pyplot as plt

    with plt.rc_context(cast(Any, _RC)):
        yield


def _new_axes(ncols: int = 1, width: float = 5.2, height: float = 4.0) -> tuple[Figure, Any]:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, ncols, figsize=(width * ncols, height), layout="constrained")
    return fig, axes


def _titles(ax: Axes, title: str, subtitle: str) -> None:
    ax.set_title(
        f"{title}\n",
        loc="left",
        fontsize=11,
        fontweight="bold",
        color=_P["ink"],
        pad=6,
    )
    ax.text(
        0.0,
        1.02,
        subtitle,
        transform=ax.transAxes,
        fontsize=8.5,
        color=_P["ink_secondary"],
        va="bottom",
        ha="left",
    )


def _scatter(ax: Axes, x: np.ndarray, y: np.ndarray, **kwargs: Any) -> None:
    opts: dict[str, Any] = {"s": 16, "color": _P["points"], "alpha": 0.75, "linewidths": 0}
    opts.update(kwargs)
    ax.scatter(x, y, rasterized=x.shape[0] > 5000, **opts)


def _smooth(ax: Axes, x: np.ndarray, y: np.ndarray) -> float:
    """Draw a lowess smoother and return its highest value (nan if none was drawn)."""
    from statsmodels.nonparametric.smoothers_lowess import lowess

    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if x.shape[0] < 5 or np.ptp(x) == 0:
        return float("nan")
    if x.shape[0] > _MAX_SMOOTH_POINTS:
        idx = np.random.default_rng(0).choice(x.shape[0], _MAX_SMOOTH_POINTS, replace=False)
        x, y = x[idx], y[idx]
    curve = lowess(y, x, frac=2 / 3, it=0)
    ax.plot(curve[:, 0], curve[:, 1], color=_P["fit"], linewidth=2, solid_capstyle="round")
    return float(np.max(curve[:, 1]))


def _refline(ax: Axes, y: float = 0.0, **kwargs: Any) -> None:
    opts: dict[str, Any] = {"color": _P["muted"], "linewidth": 1, "linestyle": (0, (4, 3))}
    opts.update(kwargs)
    ax.axhline(y, **opts)


# ------------------------------------------------------------------ panels
def plot_linearity(adapter: BaseAdapter, ax: Axes) -> Axes:
    """Residuals against fitted values with a smoother."""
    with _style():
        x, r = adapter.mu, adapter.resid
        if not adapter.info.is_linear:
            r = adapter.pearson_residuals
        _scatter(ax, x, r)
        _refline(ax)
        _smooth(ax, x, r)
        _titles(ax, "Linearity", "Reference line should be flat and horizontal")
        ax.set_xlabel("Fitted values")
        ax.set_ylabel("Residuals")
    return ax


def plot_homogeneity(adapter: BaseAdapter, ax: Axes, result: CheckResult | None = None) -> Axes:
    """Scale-location plot: sqrt(|standardized residuals|) against fitted values."""
    with _style():
        x = adapter.mu
        yv = np.sqrt(np.abs(adapter.std_residuals))
        _scatter(ax, x, yv)
        _smooth(ax, x, yv)
        subtitle = "Reference line should be flat and horizontal"
        if result is not None and result.p_value is not None:
            subtitle += f"  ·  Breusch-Pagan p = {result.p_value:.3g}"
        _titles(ax, "Homogeneity of Variance", subtitle)
        ax.set_xlabel("Fitted values")
        ax.set_ylabel(r"$\sqrt{|\mathrm{Std.\ residuals}|}$")
    return ax


def plot_outliers(result: CheckResult, ax: Axes) -> Axes:
    """Cook's distance per observation with the threshold line."""
    distances = result.distances  # type: ignore[attr-defined]
    threshold = float(result.threshold)  # type: ignore[attr-defined]
    with _style():
        d = distances.to_numpy()
        pos = np.arange(d.shape[0])
        flagged = d > threshold
        ax.vlines(pos[~flagged], 0, d[~flagged], color=_P["points"], linewidth=1.2)
        ax.vlines(pos[flagged], 0, d[flagged], color=_P["critical"], linewidth=1.6)
        ax.scatter(pos[flagged], d[flagged], s=22, color=_P["critical"], zorder=3, linewidths=0)
        ax.axhline(threshold, color=_P["critical"], linewidth=1, linestyle=(0, (4, 3)))
        ax.annotate(
            f"threshold {threshold:.3g}",
            xy=(0.0, threshold),
            xycoords=("axes fraction", "data"),
            xytext=(0, 3),
            textcoords="offset points",
            ha="left",
            va="bottom",
            fontsize=8,
            color=_P["ink_secondary"],
        )
        top = np.argsort(np.where(flagged, d, -np.inf))[::-1][: min(5, int(flagged.sum()))]
        for i in top:
            ax.annotate(
                str(distances.index[i]),
                xy=(pos[i], d[i]),
                xytext=(4, 2),
                textcoords="offset points",
                fontsize=7.5,
                color=_P["ink"],
            )
        ax.set_ylim(bottom=0, top=max(float(np.nanmax(d)), threshold) * 1.15)
        ax.margins(x=0.02)
        _titles(ax, "Influential Observations", "Points should be below the threshold line")
        ax.set_xlabel("Observation")
        ax.set_ylabel("Cook's distance")
    return ax


def plot_collinearity(result: CheckResult, ax: Axes) -> Axes:
    """Horizontal VIF bars, colored by severity, with 5 and 10 reference lines."""
    from matplotlib.patches import Patch

    from ..checks.collinearity import HIGH_VIF, LOW_VIF

    table = result.table  # type: ignore[attr-defined]
    with _style():
        vif = table["VIF"].to_numpy()
        lo = table["VIF_CI_low"].to_numpy()
        hi = table["VIF_CI_high"].to_numpy()
        levels = np.where(vif >= HIGH_VIF, "critical", np.where(vif >= LOW_VIF, "warning", "good"))
        colors = [_P[level] for level in levels]
        pos = np.arange(len(vif))[::-1]
        log_scale = bool(np.nanmax(np.concatenate([vif, hi])) > 30)
        left = 1.0 if log_scale else 0.0
        ax.barh(pos, vif - left, left=left, color=colors, height=0.45, edgecolor=_P["surface"])
        finite = np.isfinite(lo) & np.isfinite(hi)
        ax.hlines(pos[finite], lo[finite], hi[finite], color=_P["ink_secondary"], linewidth=1)
        label_at = np.where(np.isfinite(hi), np.fmax(vif, hi), vif)
        for p_, v, at in zip(pos, vif, label_at, strict=True):
            ax.annotate(
                f"{v:.2f}",
                xy=(at, p_),
                xytext=(4, 0),
                textcoords="offset points",
                va="center",
                fontsize=7.5,
                color=_P["ink"],
            )
        for ref in (LOW_VIF, HIGH_VIF):
            ax.axvline(ref, color=_P["muted"], linewidth=1, linestyle=(0, (4, 3)))
        if log_scale:
            ax.set_xscale("log")
        else:
            ax.set_xlim(0, max(HIGH_VIF, float(np.nanmax(np.concatenate([vif, hi])))) * 1.12)
        ax.set_yticks(pos, table["Term"].tolist())
        ax.set_ylim(-0.75, len(vif) - 0.25)
        ax.tick_params(axis="y", labelcolor=_P["ink_secondary"])
        ax.grid(axis="y", visible=False)
        handles = [
            Patch(color=_P["good"], label=f"Low (< {LOW_VIF:g})"),
            Patch(color=_P["warning"], label=f"Moderate ({LOW_VIF:g}–{HIGH_VIF:g})"),
            Patch(color=_P["critical"], label=f"High (≥ {HIGH_VIF:g})"),
        ]
        ax.legend(
            handles=handles,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.16),
            ncol=3,
            handlelength=1,
            handleheight=1,
            columnspacing=1.2,
        )
        _titles(ax, "Collinearity", "High collinearity (VIF) may inflate parameter uncertainty")
        ax.set_xlabel("Variance inflation factor (VIF)")
    return ax


def _qq(ax: Axes, values: np.ndarray, *, loc: float = 0.0, scale: float = 1.0) -> None:
    """Q-Q points against N(loc, scale²) with its line and a 95% pointwise band."""
    values = np.sort(values)
    n = values.shape[0]
    i = np.arange(1, n + 1)
    theo = stats.norm.ppf((i - 0.5) / n)
    band_lo = loc + scale * stats.norm.ppf(stats.beta.ppf(0.025, i, n - i + 1))
    band_hi = loc + scale * stats.norm.ppf(stats.beta.ppf(0.975, i, n - i + 1))
    ax.fill_between(theo, band_lo, band_hi, color=_P["grid"], alpha=0.8, linewidth=0)
    ax.plot(theo, loc + scale * theo, color=_P["fit"], linewidth=2)
    _scatter(ax, theo, values)
    ax.set_xlabel("Standard normal quantiles")


def plot_qq(result: CheckResult, ax: Axes) -> Axes:
    """Draw a normal Q-Q plot of the standardized residuals with a 95% pointwise band."""
    with _style():
        _qq(ax, result.residuals)  # type: ignore[attr-defined]
        subtitle = "Dots should fall along the line"
        if result.p_value is not None:
            subtitle += f"  ·  {result.method} p = {result.p_value:.3g}"
        _titles(ax, "Normality of Residuals", subtitle)
        ax.set_ylabel("Standardized residuals")
    return ax


def plot_random_effects_term(result: CheckResult, ax: Axes, term: str | None = None) -> Axes:
    """Q-Q plot of the conditional modes of one random-effect term."""
    table = result.table  # type: ignore[attr-defined]
    effects = result.effects  # type: ignore[attr-defined]
    term = str(table["Term"].iloc[0]) if term is None else term
    row = table.set_index("Term").loc[term]
    values = effects[term].to_numpy(dtype=float)
    with _style():
        _qq(ax, values, loc=float(values.mean()), scale=float(values.std(ddof=1)))
        subtitle = f"{row['Group']}: {term}  ·  {row['Method']} p = {row['p']:.3g}"
        _titles(ax, "Normality of Random Effects", subtitle)
        ax.set_ylabel("Conditional modes (BLUPs)")
    return ax


def plot_random_effects(result: CheckResult, axes: Any = None) -> Figure:
    """One Q-Q plot per random-effect term, side by side."""
    terms = list(result.table["Term"])  # type: ignore[attr-defined]
    with _style():
        if axes is None:
            fig, axes = _new_axes(ncols=len(terms), width=4.8)
        else:
            fig = cast("Figure", np.atleast_1d(axes)[0].figure)
        for term, ax in zip(terms, np.atleast_1d(axes), strict=False):
            plot_random_effects_term(result, ax, term)
    return fig


def plot_overdispersion(result: CheckResult, ax: Axes) -> Axes:
    """Squared residuals against the fitted mean, with the Poisson variance line."""
    from matplotlib.lines import Line2D

    adapter = result.adapter
    mu, sq = adapter.mu, (adapter.y - adapter.mu) ** 2
    with _style():
        _scatter(ax, mu, sq, s=10, alpha=0.35)
        grid = np.linspace(np.min(mu), np.max(mu), 100)
        ax.plot(grid, grid, color=_P["muted"], linewidth=1.5, linestyle=(0, (4, 3)))
        peak = _smooth(ax, mu, sq)
        top = np.nanmax([float(np.quantile(sq, 0.98)), float(np.max(mu)), peak]) * 1.1
        ax.set_ylim(0, top)
        handles = [
            Line2D([], [], color=_P["fit"], linewidth=2, label="Observed variance (smoothed)"),
            Line2D(
                [],
                [],
                color=_P["muted"],
                linewidth=1.5,
                linestyle=(0, (4, 3)),
                label="Poisson variance (= mean)",
            ),
        ]
        ax.legend(handles=handles, loc="upper left")
        ratio = result.dispersion_ratio  # type: ignore[attr-defined]
        subtitle = (
            f"Smoothed line should follow the dashed line  ·  dispersion ratio = {ratio:.2f}"
        )
        _titles(ax, "Overdispersion", subtitle)
        ax.set_xlabel("Fitted mean")
        ax.set_ylabel("Squared residuals")
    return ax


def plot_zeroinflation(result: CheckResult, ax: Axes) -> Axes:
    """Observed and expected frequency of each count, zeros first."""
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    adapter = result.adapter
    y0, mu = adapter.y, adapter.mu
    top = int(min(np.max(y0), max(10, np.quantile(y0, 0.99)), 30))
    counts = np.arange(top + 1)
    observed = np.array([np.sum(y0 == k) for k in counts])
    expected = stats.poisson.pmf(counts[:, None], mu[None, :]).sum(axis=1)
    with _style():
        ax.bar(counts, observed, width=0.7, color=_P["points"], edgecolor=_P["surface"])
        ax.plot(counts, expected, color=_P["fit"], linewidth=2, marker="o", markersize=5)
        ax.set_xticks(counts if top <= 15 else counts[::2])
        ax.grid(axis="x", visible=False)
        handles = [
            Patch(color=_P["points"], label="Observed"),
            Line2D([], [], color=_P["fit"], marker="o", linewidth=2, label="Expected (Poisson)"),
        ]
        ax.legend(handles=handles, loc="upper right")
        obs = result.observed_zeros  # type: ignore[attr-defined]
        pred = result.predicted_zeros  # type: ignore[attr-defined]
        _titles(
            ax,
            "Zero-inflation",
            f"Bars should match the line  ·  zeros: {obs} observed, {pred} predicted",
        )
        ax.set_xlabel("Count")
        ax.set_ylabel("Number of observations")
    return ax


def _plot_residual_histogram(result: CheckResult, ax: Axes) -> Axes:
    resid = result.residuals  # type: ignore[attr-defined]
    with _style():
        ax.hist(
            resid,
            bins="auto",
            density=True,
            color=_P["points"],
            alpha=0.85,
            edgecolor=_P["surface"],
            linewidth=1,
        )
        grid = np.linspace(min(resid.min(), -3.5), max(resid.max(), 3.5), 300)
        ax.plot(
            grid,
            stats.norm.pdf(grid, resid.mean(), resid.std(ddof=1)),
            color=_P["fit"],
            linewidth=2,
        )
        ax.grid(axis="x", visible=False)
        _titles(ax, "Distribution of Residuals", "Should be close to the normal curve")
        ax.set_xlabel("Standardized residuals")
        ax.set_ylabel("Density")
    return ax


def plot_normality(result: CheckResult, axes: Any = None) -> Figure:
    """Q-Q plot and histogram of the standardized residuals, side by side."""
    with _style():
        if axes is None:
            fig, axes = _new_axes(ncols=2, width=4.8)
        else:
            fig = cast("Figure", axes[0].figure)
        plot_qq(result, axes[0])
        _plot_residual_histogram(result, axes[1])
    return fig


def plot_autocorrelation(result: CheckResult, ax: Axes) -> Axes:
    """Autocorrelation function of the residuals with a 95% band."""
    adapter = result.adapter
    e = np.sqrt(adapter.weights) * adapter.resid
    e = e - e.mean()
    n = e.shape[0]
    n_lags = int(min(20, max(1, n // 4)))
    denom = float(np.sum(e**2))
    acf = np.array([np.sum(e[k:] * e[: n - k]) / denom for k in range(1, n_lags + 1)])
    band = stats.norm.ppf(0.975) / math.sqrt(n)
    with _style():
        lags = np.arange(1, n_lags + 1)
        ax.axhspan(-band, band, color=_P["grid"], alpha=0.8, linewidth=0)
        outside = np.abs(acf) > band
        ax.vlines(lags[~outside], 0, acf[~outside], color=_P["points"], linewidth=2.5)
        ax.vlines(lags[outside], 0, acf[outside], color=_P["critical"], linewidth=2.5)
        ax.axhline(0, color=_P["axis"], linewidth=1)
        subtitle = "Bars should stay inside the shaded band"
        if result.statistic is not None:
            subtitle += f"  ·  DW = {result.statistic:.2f}"
        _titles(ax, "Autocorrelation", subtitle)
        ax.set_xlabel("Lag")
        ax.set_ylabel("Autocorrelation of residuals")
        ax.set_xticks(lags if n_lags <= 10 else lags[::2])
        lim = max(float(np.max(np.abs(acf))), band) * 1.2
        ax.set_ylim(-lim, lim)
    return ax


def plot_binned_residuals(result: CheckResult, ax: Axes) -> Axes:
    """Mean residual per bin of fitted values with the error bounds."""
    from matplotlib.lines import Line2D

    table = result.table  # type: ignore[attr-defined]
    with _style():
        x = table["xbar"].to_numpy()
        order = np.argsort(x)
        se = table["se"].to_numpy()
        ax.plot(x[order], se[order], color=_P["muted"], linewidth=1)
        ax.plot(x[order], -se[order], color=_P["muted"], linewidth=1)
        _refline(ax)
        inside = table["inside"].to_numpy(dtype=bool)
        colors = np.where(inside, _P["good"], _P["critical"])
        ax.vlines(x, table["CI_low"], table["CI_high"], colors=colors.tolist(), linewidth=1)
        ax.scatter(x, table["ybar"], c=colors, s=26, zorder=3, linewidths=0)
        handles = [
            Line2D([], [], marker="o", linestyle="", color=_P["good"], label="Within bounds"),
            Line2D([], [], marker="o", linestyle="", color=_P["critical"], label="Outside"),
        ]
        ax.legend(handles=handles, loc="best")
        _titles(ax, "Binned Residuals", "Points should be within error bounds")
        ax.set_xlabel("Estimated probability of event")
        ax.set_ylabel("Average residual")
    return ax


# ----------------------------------------------------------------- dispatch
def plot_predictions(result: CheckResult, ax: Axes) -> Axes:
    """Posterior predictive check: replicated data against the observed data."""
    from matplotlib.lines import Line2D

    observed = result.observed  # type: ignore[attr-defined]
    simulated = result.simulated  # type: ignore[attr-defined]
    with _style():
        if result.discrete:  # type: ignore[attr-defined]
            values = np.unique(np.concatenate([np.unique(observed), np.unique(simulated)]))
            values = values[values <= max(np.max(observed), np.quantile(simulated, 0.99))][:31]
            counts = np.array([[np.sum(rep == v) for v in values] for rep in simulated])
            lo, mid, hi = np.quantile(counts, [0.05, 0.5, 0.95], axis=0)
            ax.vlines(values, lo, hi, color=_P["points"], linewidth=2.5, alpha=0.6)
            ax.scatter(values, mid, color=_P["points"], s=22, zorder=3, linewidths=0)
            observed_counts = [np.sum(observed == v) for v in values]
            ax.scatter(values, observed_counts, color=_P["fit"], s=36, zorder=4, linewidths=0)
            ax.set_xticks(values if values.shape[0] <= 15 else values[::2])
            ax.set_xlabel("Value of the response")
            ax.set_ylabel("Number of observations")
            handles = [
                Line2D([], [], marker="o", linestyle="", color=_P["fit"], label="Observed data"),
                Line2D(
                    [],
                    [],
                    marker="o",
                    color=_P["points"],
                    linewidth=2.5,
                    label="Model-predicted (median, 90% interval)",
                ),
            ]
        else:
            lo = min(np.min(observed), np.quantile(simulated, 0.005))
            hi = max(np.max(observed), np.quantile(simulated, 0.995))
            grid = np.linspace(lo, hi, 256)
            for rep in simulated:
                if np.ptp(rep) > 0:
                    density = stats.gaussian_kde(rep)(grid)
                    ax.plot(grid, density, color=_P["points"], linewidth=0.8, alpha=0.25)
            ax.plot(grid, stats.gaussian_kde(observed)(grid), color=_P["fit"], linewidth=2.2)
            ax.set_xlabel("Response")
            ax.set_ylabel("Density")
            handles = [
                Line2D([], [], color=_P["fit"], linewidth=2.2, label="Observed data"),
                Line2D([], [], color=_P["points"], linewidth=1.2, label="Model-predicted data"),
            ]
        ax.legend(handles=handles, loc="best")
        _titles(
            ax,
            "Posterior Predictive Check",
            "Model-predicted data should resemble observed data",
        )
    return ax


def plot_convergence(result: CheckResult, ax: Axes) -> Axes:
    """R-hat of each parameter of a Bayesian model against the 1.01 threshold."""
    from matplotlib.lines import Line2D

    from ..checks.convergence import ESS_PER_CHAIN, RHAT_MAX

    table = result.table  # type: ignore[attr-defined]
    if table.empty:
        raise TypeError("Convergence of frequentist models has no plot; read `.message` instead.")
    with _style():
        chains = cast(Any, result.adapter).chains  # a BayesianModel
        min_ess = ESS_PER_CHAIN * chains
        ok = (table["Rhat"] < RHAT_MAX) & (table[["ESS_bulk", "ESS_tail"]].min(axis=1) >= min_ess)
        pos = np.arange(len(table))[::-1]
        colors = np.where(ok, _P["good"], _P["critical"])
        ax.hlines(pos, 1.0, table["Rhat"], color=_P["grid"], linewidth=1.5)
        ax.scatter(table["Rhat"], pos, c=colors, s=40, zorder=3, linewidths=0)
        ax.axvline(RHAT_MAX, color=_P["critical"], linewidth=1, linestyle=(0, (4, 3)))
        ax.set_yticks(pos, table["Parameter"].tolist())
        ax.tick_params(axis="y", labelcolor=_P["ink_secondary"])
        ax.grid(axis="y", visible=False)
        right = max(float(table["Rhat"].max()), RHAT_MAX) + 0.005
        ax.set_xlim(min(0.995, float(table["Rhat"].min()) - 0.002), right)
        handles = [
            Line2D([], [], marker="o", linestyle="", color=_P["good"], label="Converged"),
            Line2D(
                [],
                [],
                marker="o",
                linestyle="",
                color=_P["critical"],
                label=f"R-hat >= {RHAT_MAX} or ESS < {min_ess}",
            ),
        ]
        ax.legend(handles=handles, loc="lower right")
        _titles(ax, "Convergence", f"R-hat should stay left of {RHAT_MAX}")
        ax.set_xlabel("R-hat (rank-normalized, split)")
    return ax


_SINGLE: dict[str, Callable[[Any, Any], Any]] = {
    "convergence": plot_convergence,
    "predictions": plot_predictions,
    "heteroscedasticity": lambda r, ax: plot_homogeneity(r.adapter, ax, r),
    "collinearity": plot_collinearity,
    "outliers": plot_outliers,
    "autocorrelation": plot_autocorrelation,
    "binned_residuals": plot_binned_residuals,
    "normality": plot_qq,
    "random_effects": plot_random_effects_term,
    "overdispersion": plot_overdispersion,
    "zeroinflation": plot_zeroinflation,
}


def plot_check(result: CheckResult, ax: Axes | None = None) -> Figure:
    """Draw the plot for one check result; see :meth:`CheckResult.plot`."""
    if result.key not in _SINGLE:
        raise TypeError(f"{result.title} has no diagnostic plot; read `.message` instead.")
    if ax is None and result.key == "normality":
        return plot_normality(result)
    if ax is None and result.key == "random_effects":
        return plot_random_effects(result)
    with _style():
        if ax is None:
            fig, ax = _new_axes()
        else:
            fig = cast("Figure", ax.figure)
        _SINGLE[result.key](result, ax)
    return fig


def plot_model_check(
    report: ModelCheckReport,
    ncols: int | None = None,
    panel_size: tuple[float, float] = (4.6, 3.7),
) -> Figure:
    """Draw every diagnostic of a :class:`~checkmodel.ModelCheckReport` in one grid.

    Linear models get seven panels: the posterior predictive check, linearity,
    homogeneity of variance, influential observations, collinearity, normality
    and autocorrelation. Bayesian models show the predictive check and the
    R-hat of each parameter.
    Mixed models add one Q-Q panel per random-effect term; Poisson models an
    overdispersion panel. Singularity and convergence have no plot.

    Parameters
    ----------
    report
        Output of :func:`checkmodel.check_model`.
    ncols
        Number of grid columns; 3 by default (2 when there are four panels or fewer).
    panel_size
        Width and height of each panel in inches.
    """
    import matplotlib.pyplot as plt

    adapter = report.adapter
    het = report.get("heteroscedasticity")
    panels: list[Callable[[Axes], Any]] = []
    if "predictions" in report:
        panels.append(partial(plot_predictions, report["predictions"]))
    if isinstance(adapter, BaseAdapter):  # Bayesian models have no frequentist residuals
        if adapter.info.is_linear:
            panels.append(lambda ax: plot_linearity(adapter, ax))
        panels.append(lambda ax: plot_homogeneity(adapter, ax, het))
    for key in ("outliers", "collinearity", "normality"):
        if key in report:
            panels.append(partial(_SINGLE[key], report[key]))
    if "random_effects" in report:
        res = report["random_effects"]
        for term in res.table["Term"]:  # type: ignore[attr-defined]
            panels.append(partial(plot_random_effects_term, res, term=term))
    for key in ("autocorrelation", "binned_residuals", "overdispersion"):
        if key in report:
            panels.append(partial(_SINGLE[key], report[key]))
    if "convergence" in report and not isinstance(adapter, BaseAdapter):
        panels.append(partial(plot_convergence, report["convergence"]))
    count = len(panels)
    if count == 0:
        raise ValueError("None of the checks in this report has a plot.")
    cols = ncols or (3 if count > 4 else min(2, count))
    rows = math.ceil(count / cols)
    with _style():
        fig, axes = plt.subplots(
            rows,
            cols,
            figsize=(panel_size[0] * cols, panel_size[1] * rows),
            layout="constrained",
            squeeze=False,
        )
        flat = axes.ravel()
        for draw, ax in zip(panels, flat, strict=False):
            draw(ax)
        for ax in flat[count:]:
            ax.set_visible(False)
    return fig
