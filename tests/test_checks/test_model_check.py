from __future__ import annotations

import pytest
from sklearn.linear_model import Ridge

from checkmodel import ModelCheckReport, check_model


def test_linear_model_runs_all_checks(mtcars_ols):
    report = check_model(*mtcars_ols)
    assert isinstance(report, ModelCheckReport)
    assert list(report) == [
        "predictions",
        "normality",
        "heteroscedasticity",
        "collinearity",
        "outliers",
        "autocorrelation",
    ]
    assert not report.skipped
    assert len(report) == 6


def test_logistic_and_poisson_checks(mtcars_logit, poisson):
    assert list(check_model(*mtcars_logit)) == [
        "predictions",
        "collinearity",
        "outliers",
        "binned_residuals",
    ]
    assert list(check_model(poisson)) == [
        "predictions",
        "collinearity",
        "outliers",
        "overdispersion",
    ]


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
    assert str(check_model(fit("lm_good"))).count("OK:") == 6


@pytest.mark.parametrize(
    ("name", "panels"),
    [
        ("lm_mtcars", 7),
        ("glm_logit", 5),
        ("glm_pois", 5),
        ("lm_hetero", 6),
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
    assert fig.get_size_inches().tolist() == [6.0, 10.0]


def test_penalized_sklearn(mtcars):
    X, y = mtcars[["wt", "hp"]], mtcars["mpg"]
    report = check_model(Ridge(alpha=1.0).fit(X, y), X=X, y=y)
    assert len(report) == 6
    report.plot()
