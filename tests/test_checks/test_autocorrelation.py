from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import statsmodels.formula.api as smf
from statsmodels.stats.stattools import durbin_watson

from checkmodel import CheckNotApplicableError, check_autocorrelation
from tests.conftest import assert_r_close

UNWEIGHTED = [
    "lm_mtcars",
    "lm_mtcars_cyl",
    "lm_iris",
    "lm_good",
    "lm_hetero",
    "lm_skewed",
    "lm_autocorr",
    "lm_autocorr_large",
    "lm_large",
    "lm_collinear",
]


@pytest.mark.parametrize("name", UNWEIGHTED)
def test_matches_lmtest(name, fit, reference):
    """Statistic and two-sided p-values vs lmtest::dwtest()."""
    ref = reference["models"][name]["dw"]
    model = fit(name)
    res = check_autocorrelation(model)
    assert_r_close(res.statistic, ref["statistic"])
    assert_r_close(res.p_value, ref["p_auto"])
    assert_r_close(check_autocorrelation(model, method="normal").p_value, ref["p_normal"])
    if ref.get("p_exact") not in (None, {}):
        assert_r_close(check_autocorrelation(model, method="exact").p_value, ref["p_exact"])


def test_auto_picks_exact_below_100(fit):
    assert check_autocorrelation(fit("lm_mtcars")).method == "Durbin-Watson (exact)"
    assert check_autocorrelation(fit("lm_iris")).method == "Durbin-Watson (normal)"


def test_weighted_model_uses_weighted_residuals(fit, reference):
    res = check_autocorrelation(fit("lm_mtcars_wls"))
    assert_r_close(res.statistic, reference["models"]["lm_mtcars_wls"]["dw"]["statistic"])


def test_statistic_equals_statsmodels(ols):
    assert check_autocorrelation(ols).statistic == pytest.approx(durbin_watson(ols.resid))


def test_passing_and_failing_models(fit):
    assert check_autocorrelation(fit("lm_good")).passed
    bad = check_autocorrelation(fit("lm_autocorr"))
    assert not bad.passed
    assert bad.message == "Warning: Autocorrelated residuals detected (p < .001)."


def test_bootstrap_is_reproducible_and_close(fit):
    model = fit("lm_good")
    a = check_autocorrelation(model, method="bootstrap", random_state=7)
    b = check_autocorrelation(model, method="bootstrap", random_state=7)
    assert a.p_value == b.p_value
    assert a.method == "Durbin-Watson (bootstrap)"
    exact = check_autocorrelation(model, method="exact").p_value
    assert abs(a.p_value - exact) < 0.1


def test_same_result_for_statsmodels_and_sklearn(mtcars_ols, reference):
    res = check_autocorrelation(*mtcars_ols)
    assert_r_close(res.p_value, reference["models"]["lm_mtcars"]["dw"]["p_auto"])


def test_exact_refuses_huge_samples(fit):
    with pytest.raises(ValueError, match="method='normal'"):
        check_autocorrelation(fit("lm_large"), method="exact")


def test_unknown_method(ols):
    with pytest.raises(ValueError, match="Unknown method"):
        check_autocorrelation(ols, method="magic")  # type: ignore[arg-type]


def test_tiny_sample():
    df = pd.DataFrame({"x": [1.0, 2.0], "y": [1.0, 3.0]})
    with pytest.warns(UserWarning, match="Not enough observations"):
        res = check_autocorrelation(smf.ols("y ~ 1", df).fit())
    assert res.p_value == 1.0


def test_extreme_autocorrelation_does_not_fall_back(fit):
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        res = check_autocorrelation(fit("lm_autocorr"), method="exact")
    assert 0.0 <= res.p_value < 1e-10


def test_not_applicable_to_glm(logit):
    with pytest.raises(CheckNotApplicableError):
        check_autocorrelation(logit)


def test_plot(fit):
    fig = check_autocorrelation(fit("lm_autocorr")).plot()
    (ax,) = fig.axes
    assert ax.get_title(loc="left").startswith("Autocorrelation")
    assert np.isfinite(ax.get_ylim()).all()
