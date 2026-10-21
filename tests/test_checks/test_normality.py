from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import statsmodels.formula.api as smf

from checkmodel import CheckNotApplicableError, NormalityResult, check_normality
from tests.conftest import assert_r_close
from tests.models import MODELS

LINEAR = [n for n in MODELS if n.startswith("lm_")]


@pytest.mark.parametrize("name", LINEAR)
def test_matches_r(name, fit, reference):
    ref = reference["models"][name]["normality"]
    res = check_normality(fit(name))
    assert_r_close(res.statistic, ref["statistic"])
    assert_r_close(res.p_value, ref["p"])
    assert_r_close(res.p_value, ref["performance_p"])
    assert res.passed == (ref["performance_p"] >= 0.05)


def test_shapiro_below_5000_ks_above(fit):
    assert check_normality(fit("lm_mtcars")).method == "Shapiro-Wilk"
    assert check_normality(fit("lm_large")).method == "Kolmogorov-Smirnov"


def test_passing_and_failing_models(fit):
    good = check_normality(fit("lm_good"))
    bad = check_normality(fit("lm_skewed"))
    assert good.passed and good.message.startswith("OK: residuals appear as normally distributed")
    assert (
        not bad.passed
        and bad.message == "Warning: Non-normality of residuals detected (p < .001)."
    )


def test_same_result_for_statsmodels_and_sklearn(mtcars_ols, reference):
    res = check_normality(*mtcars_ols)
    assert_r_close(res.p_value, reference["models"]["lm_mtcars"]["normality"]["p"])


def test_gaussian_glm_equals_ols(fit):
    assert check_normality(fit("glm_gauss")).p_value == pytest.approx(
        check_normality(fit("lm_mtcars")).p_value
    )


def test_alpha(fit):
    res = check_normality(fit("lm_mtcars"), alpha=0.001)
    assert res.passed


def test_not_applicable_to_glm(logit, poisson):
    for model in (logit, poisson):
        with pytest.raises(CheckNotApplicableError, match="linear models"):
            check_normality(model)


def test_too_few_residuals():
    df = pd.DataFrame({"y": [1.0, 2.5]})
    with pytest.raises(CheckNotApplicableError, match="At least 3"):
        check_normality(smf.ols("y ~ 1", df).fit())


def test_result_api(ols):
    res = check_normality(ols)
    assert isinstance(res, NormalityResult)
    assert repr(res) == str(res) == res.summary() == res.message
    df = res.to_dataframe()
    assert list(df.columns) == ["Check", "Method", "Statistic", "p", "Passed", "Message"]
    assert df.loc[0, "p"] == res.p_value
    assert res.residuals.shape == (32,)
    assert np.isfinite(res.residuals).all()


def test_plot(ols):
    import matplotlib.pyplot as plt

    fig = check_normality(ols).plot()
    assert len(fig.axes) == 2
    assert fig.axes[0].get_title(loc="left").startswith("Normality of Residuals")

    fig2, ax = plt.subplots()
    assert check_normality(ols).plot(ax=ax) is fig2
