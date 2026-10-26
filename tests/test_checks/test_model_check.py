from __future__ import annotations

import time
import warnings

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
from sklearn.linear_model import LinearRegression, LogisticRegression, PoissonRegressor, Ridge

from checkmodel import (
    ModelCheckReport,
    binned_residuals,
    check_autocorrelation,
    check_collinearity,
    check_heteroscedasticity,
    check_model,
    check_normality,
    check_outliers,
    check_residuals,
)

LINEAR_CHECKS = ["predictions", "normality", "heteroscedasticity", "collinearity", "outliers"]


def test_linear_model_runs_all_checks(mtcars_ols):
    report = check_model(*mtcars_ols)
    assert isinstance(report, ModelCheckReport)
    assert list(report) == LINEAR_CHECKS
    assert not report.skipped
    assert len(report) == 5


def test_logistic_and_poisson_checks(mtcars_logit, poisson):
    assert list(check_model(*mtcars_logit)) == [
        "predictions",
        "collinearity",
        "outliers",
        "binned_residuals",
        "residuals",
    ]
    assert list(check_model(poisson)) == [
        "predictions",
        "collinearity",
        "outliers",
        "overdispersion",
        "residuals",
    ]


# ------------------------------------------------------- autocorrelation
def _time_series(index):
    rng = np.random.default_rng(3)
    x1, x2, e = rng.normal(size=(3, len(index)))
    return pd.DataFrame({"x1": x1, "x2": x2, "y": 1 + 2 * x1 - x2 + e}, index=index)


def test_cross_sectional_data_has_no_autocorrelation_check(mtcars_ols):
    report = check_model(*mtcars_ols)
    assert "autocorrelation" not in report
    assert "autocorrelation" not in report.skipped
    assert "Autocorrelat" not in str(report)


@pytest.mark.parametrize(
    "index",
    [
        pd.date_range("2020-01-01", periods=60, freq="D"),
        pd.period_range("2010-01", periods=60, freq="M"),
    ],
    ids=["DatetimeIndex", "PeriodIndex"],
)
def test_time_indexed_data_gets_the_autocorrelation_check(index):
    data = _time_series(index)
    ols = smf.ols("y ~ x1 + x2", data).fit()
    report = check_model(ols)
    assert list(report) == [*LINEAR_CHECKS, "autocorrelation"]
    assert report["autocorrelation"].p_value == check_autocorrelation(ols).p_value

    X, y = data[["x1", "x2"]], data["y"]
    assert "autocorrelation" in check_model(LinearRegression().fit(X, y), X, y)


def test_autocorrelation_can_be_requested(ols, mtcars_ols):
    report = check_model(ols, checks=["normality", "autocorrelation"])
    assert list(report) == ["normality", "autocorrelation"]
    assert report["autocorrelation"].p_value == check_autocorrelation(ols).p_value
    # The README's mtcars example: the false positive of 0.1.0 is still there on request.
    assert not report["autocorrelation"].passed
    assert list(check_model(*mtcars_ols, checks="autocorrelation")) == ["autocorrelation"]


def test_mixed_models_still_skip_autocorrelation(fit):
    report = check_model(fit("mixed_sleep"), checks=["autocorrelation"])
    assert "autocorrelation" in report.skipped


def test_inapplicable_checks_are_skipped(fit, logit):
    report = check_model(fit("lm_hetero"))
    assert "collinearity" not in report
    assert "Not enough model terms" in report.skipped["collinearity"]
    with pytest.raises(KeyError, match="was skipped"):
        report["collinearity"]

    explicit = check_model(logit, checks=["normality", "outliers"])
    assert list(explicit) == ["outliers"]
    assert "normality" in explicit.skipped


def test_select_checks(ols):
    assert list(check_model(ols, checks="normality")) == ["normality"]
    with pytest.raises(ValueError, match="Unknown checks"):
        check_model(ols, checks=["normality", "nope"])
    with pytest.raises(KeyError, match="not in report"):
        check_model(ols, checks="normality")["outliers"]


def test_alpha_is_forwarded(ols):
    assert not check_model(ols)["normality"].passed
    assert check_model(ols, alpha=0.001)["normality"].passed


def test_predictive_check_is_reproducible(ols):
    first = check_model(ols)["predictions"]
    assert first.to_dataframe().equals(check_model(ols)["predictions"].to_dataframe())
    other = check_model(ols, random_state=1)["predictions"]
    assert first.simulated.tolist() != other.simulated.tolist()


def test_summary_and_passed(fit):
    good = check_model(fit("lm_good"))
    assert good.passed
    summary = good.summary()
    assert list(summary.index) == list(good)
    assert summary.index.name == "check"
    assert summary["Passed"].all()
    assert good.to_dataframe().equals(summary)
    assert not check_model(fit("lm_skewed")).passed


def test_repr(fit):
    text = repr(check_model(fit("lm_hetero")))
    assert text.startswith("# Model checks: OLS (gaussian, identity link), n = 200")
    assert "Normality of residuals: Warning" in text
    assert "Skipped:" in text
    assert "Note:" not in text
    assert str(check_model(fit("lm_good"))).count("OK:") == 5


@pytest.mark.parametrize(
    ("name", "panels"),
    [
        ("lm_mtcars", 6),
        ("glm_logit", 6),
        ("glm_pois", 6),
        ("lm_hetero", 5),
        ("mixed_sleep_slope", 7),
        ("mixed_synth", 7),
    ],
)
def test_plot_grid(name, panels, fit):
    fig = check_model(fit(name)).plot()
    visible = [ax for ax in fig.axes if ax.get_visible()]
    assert len(visible) == panels
    titles = [ax.get_title(loc="left").strip() for ax in visible]
    assert titles[0] == "Posterior Predictive Check"


def test_plot_options(ols):
    fig = check_model(ols).plot(ncols=2, panel_size=(3, 2.5))
    assert fig.get_size_inches().tolist() == [6.0, 7.5]


def test_time_indexed_plot_has_an_autocorrelation_panel():
    data = _time_series(pd.date_range("2020-01-01", periods=60, freq="D"))
    fig = check_model(smf.ols("y ~ x1 + x2", data).fit()).plot()
    titles = [ax.get_title(loc="left").strip() for ax in fig.axes if ax.get_visible()]
    assert len(titles) == 7
    assert titles[-1] == "Autocorrelation"


def test_penalized_sklearn(mtcars):
    X, y = mtcars[["wt", "hp"]], mtcars["mpg"]
    with pytest.warns(UserWarning, match="penalized fit"):
        report = check_model(Ridge(alpha=1.0).fit(X, y), X=X, y=y)
    assert len(report) == 5
    report.plot()


# ------------------------------------------------------ penalized fits
PENALIZED = (
    "`Ridge` is a penalized fit; leverage, Cook's distance and standardized residuals "
    "are computed as for the unpenalized model and are approximate."
)


def _user_warnings(record):
    return [str(w.message) for w in record if issubclass(w.category, UserWarning)]


def test_penalized_fit_warns_once(mtcars):
    X, y = mtcars[["wt", "hp"]], mtcars["mpg"]
    ridge = Ridge(alpha=10.0).fit(X, y)
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        report = check_model(ridge, X, y)
    assert _user_warnings(record) == [PENALIZED]
    assert record[0].filename == __file__  # attributed to the caller, not to checkmodel
    assert str(report).splitlines()[1] == "Note: " + PENALIZED


@pytest.mark.parametrize(
    "check", [check_outliers, check_normality, check_heteroscedasticity, check_collinearity]
)
def test_standalone_checks_warn_about_penalized_fits(check, mtcars):
    X, y = mtcars[["wt", "hp"]], mtcars["mpg"]
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        check(Ridge(alpha=10.0).fit(X, y), X, y)
    assert _user_warnings(record) == [PENALIZED]
    assert record[0].filename == __file__


def test_unpenalized_fits_do_not_warn(mtcars, mtcars_ols):
    X, y = mtcars[["wt", "hp"]], mtcars["mpg"]
    counts = mtcars[["wt"]], mtcars["carb"]
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        report = check_model(LinearRegression().fit(X, y), X, y)
        check_model(*mtcars_ols)
        check_outliers(PoissonRegressor(alpha=0.0).fit(*counts), *counts)
    assert not [m for m in _user_warnings(record) if "penalized" in m]
    assert "Note:" not in str(report)


def test_penalized_logistic_and_glm_warn(mtcars):
    X = mtcars[["mpg", "wt"]]
    with pytest.warns(UserWarning, match="`LogisticRegression` is a penalized fit"):
        check_model(LogisticRegression().fit(X, mtcars["vs"]), X, mtcars["vs"])
    counts = mtcars[["wt"]], mtcars["carb"]
    with pytest.warns(UserWarning, match="`PoissonRegressor` is a penalized fit"):
        check_outliers(PoissonRegressor(alpha=1.0).fit(*counts), *counts)


# ----------------------------------------------------------- separation
SEPARATION = (
    "Possible complete separation: fitted probabilities are 0 or 1; diagnostics are unreliable."
)


@pytest.fixture(scope="module")
def separated():
    x = np.linspace(-3, 3, 40)
    data = pd.DataFrame({"x": x, "z": np.cos(x), "y": (x > 0.1).astype(float)})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # statsmodels' PerfectSeparationWarning, overflow
        return smf.glm("y ~ x + z", data, family=sm.families.Binomial()).fit()


def test_complete_separation_warns_once(separated):
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        check_model(separated)
    assert [m for m in _user_warnings(record) if "separation" in m] == [SEPARATION]


@pytest.mark.parametrize("check", [check_outliers, check_collinearity, binned_residuals])
def test_standalone_checks_warn_about_separation(check, separated):
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        check(separated)
    assert SEPARATION in _user_warnings(record)


def test_simulated_residuals_warn_about_separation(separated):
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        check_residuals(separated, iterations=20, random_state=0)
    assert SEPARATION in _user_warnings(record)


def test_no_separation_warning_for_ordinary_logit(logit):
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        check_model(logit)
    assert not [m for m in _user_warnings(record) if "separation" in m]


# ------------------------------------------------------- large samples
def _large_ols(n, seed=1):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 3))
    X[:3] = 8.0  # three high-leverage points far off the line: Cook's outliers
    y = X @ [1.0, 2.0, 3.0] + rng.normal(size=n)
    y[:3] -= 60.0
    return sm.OLS(y, sm.add_constant(X)).fit()


def _offsets(ax, i=0):
    return np.asarray(ax.collections[i].get_offsets())


def test_small_n_plot_draws_every_point(ols):
    report = check_model(ols)
    fig = report.plot()
    pred, lin, hom, cook, _, qq = fig.axes[:6]
    a = report.adapter
    np.testing.assert_array_equal(_offsets(lin), np.column_stack([a.mu, a.resid]))
    np.testing.assert_array_equal(
        _offsets(hom), np.column_stack([a.mu, np.sqrt(np.abs(a.std_residuals))])
    )
    n = a.n_obs
    resid = np.sort(report["normality"].residuals)
    theo = stats.norm.ppf((np.arange(1, n + 1) - 0.5) / n)
    np.testing.assert_array_equal(_offsets(qq, 1), np.column_stack([theo, resid]))
    assert sum(len(c.get_segments()) for c in cook.collections[:2]) == n
    simulated = report["predictions"].simulated
    observed = report["predictions"].observed
    lines = pred.get_lines()
    assert len(lines) == simulated.shape[0] + 1
    grid = lines[-1].get_xdata()
    np.testing.assert_array_equal(lines[-1].get_ydata(), stats.gaussian_kde(observed)(grid))
    np.testing.assert_array_equal(lines[0].get_ydata(), stats.gaussian_kde(simulated[0])(grid))


def test_large_n_plot_subsamples_but_keeps_extremes():
    from checkmodel.plotting.diagnostic_plots import _MAX_PLOT_POINTS

    report = check_model(_large_ols(20_000))
    fig = report.plot()
    _, lin, hom, cook, _, qq = fig.axes[:6]
    a = report.adapter
    outliers = report["outliers"].outliers
    assert set(outliers) >= {0, 1, 2}
    for ax, values in ((lin, a.resid), (hom, np.sqrt(np.abs(a.std_residuals)))):
        drawn = _offsets(ax)
        assert _MAX_PLOT_POINTS <= drawn.shape[0] < 2 * _MAX_PLOT_POINTS
        assert set(values[outliers]) <= set(drawn[:, 1])
    flagged = cook.collections[1].get_segments()
    assert sorted(int(seg[0][0]) for seg in flagged) == sorted(outliers)
    drawn = _offsets(qq, 1)
    resid = np.sort(report["normality"].residuals)
    assert drawn[0, 1] == resid[0] and drawn[-1, 1] == resid[-1]
    assert set(resid[-3:]) <= set(drawn[:, 1])
    # The subsample is the same every time.
    again = check_model(_large_ols(20_000)).plot()
    np.testing.assert_array_equal(_offsets(again.axes[1]), _offsets(lin))


def test_thin_is_deterministic_and_keeps_flagged_points():
    from checkmodel.plotting.diagnostic_plots import _thin

    np.testing.assert_array_equal(_thin(10, limit=20), np.arange(10))
    keep = np.zeros(1000, dtype=bool)
    keep[[3, 500, 999]] = True
    idx = _thin(1000, keep, limit=100)
    assert idx.shape[0] == 103 and set(idx) >= {3, 500, 999}
    assert np.all(np.diff(idx) > 0)
    np.testing.assert_array_equal(idx, _thin(1000, keep, limit=100))


def test_large_n_qq_keeps_points_outside_the_band():
    from checkmodel.plotting.diagnostic_plots import _MAX_PLOT_POINTS, _qq_ranks

    rng = np.random.default_rng(0)
    for power in (1.0, 1.05):  # uniform, and a departure that puts most points outside
        u = np.sort(np.concatenate([rng.uniform(size=20_000) ** power, [0.0, 1.0]]))
        n = u.shape[0]
        ranks, band = _qq_ranks(u)
        assert band.shape[0] == _MAX_PLOT_POINTS + 2 and {1, n} <= set(band)
        i = np.arange(1, n + 1)
        lo, hi = stats.beta.ppf(0.025, i, n - i + 1), stats.beta.ppf(0.975, i, n - i + 1)
        forced = set(i[(u < lo) | (u > hi)]) | {1, n}
        assert forced <= set(ranks)
        assert ranks.shape[0] == len(forced) + min(_MAX_PLOT_POINTS, n - len(forced))


def test_binned_density_matches_the_exact_kde():
    from checkmodel.plotting.diagnostic_plots import _MAX_KDE_POINTS, _density

    rng = np.random.default_rng(0)
    values = np.concatenate([rng.normal(size=30_000), rng.normal(4, 0.5, size=10_000)])
    grid = np.linspace(-4, 6, 256)
    exact = stats.gaussian_kde(values)(grid)
    np.testing.assert_allclose(_density(values, grid), exact, atol=1e-3 * exact.max())
    small = values[:_MAX_KDE_POINTS]
    np.testing.assert_array_equal(_density(small, grid), stats.gaussian_kde(small)(grid))


@pytest.mark.slow
def test_plot_scales_to_200k_observations():
    report = check_model(_large_ols(200_000))
    start = time.perf_counter()
    fig = report.plot()
    fig.canvas.draw()
    assert time.perf_counter() - start < 15
