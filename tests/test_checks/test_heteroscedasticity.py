from __future__ import annotations

import numpy as np
import pytest
from statsmodels.stats.diagnostic import het_breuschpagan

from checkmodel import CheckNotApplicableError, check_heteroscedasticity
from tests.conftest import assert_r_close
from tests.models import MODELS

LINEAR = [n for n in MODELS if n.startswith("lm_")]


@pytest.mark.parametrize("name", LINEAR)
def test_matches_r(name, fit, reference):
    """Against performance::check_heteroscedasticity() (p) and car::ncvTest() (chi²)."""
    ref = reference["models"][name]["heteroscedasticity"]
    res = check_heteroscedasticity(fit(name))
    assert_r_close(res.statistic, ref["statistic"])
    assert_r_close(res.p_value, ref["p"])
    assert_r_close(res.p_value, ref["p_car"])
    assert res.passed == (ref["p"] >= 0.05)
    assert res.df == 1


def test_equals_non_robust_breusch_pagan(ols):
    """Same as statsmodels' het_breuschpagan with robust=False on [1, fitted]."""
    exog = np.column_stack([np.ones(len(ols.fittedvalues)), ols.fittedvalues])
    lm, lm_p, _, _ = het_breuschpagan(ols.resid, exog, robust=False)
    res = check_heteroscedasticity(ols)
    assert res.statistic == pytest.approx(lm)
    assert res.p_value == pytest.approx(lm_p)


def test_passing_and_failing_models(fit):
    assert check_heteroscedasticity(fit("lm_good")).passed
    bad = check_heteroscedasticity(fit("lm_hetero"))
    assert not bad.passed
    assert bad.message.startswith("Warning: Heteroscedasticity (non-constant error variance)")


def test_same_result_for_statsmodels_and_sklearn(mtcars_ols, reference):
    res = check_heteroscedasticity(*mtcars_ols)
    assert_r_close(res.p_value, reference["models"]["lm_mtcars"]["heteroscedasticity"]["p"])


def test_not_applicable_to_glm(logit, poisson):
    with pytest.raises(CheckNotApplicableError, match="Gaussian"):
        check_heteroscedasticity(logit)
    with pytest.raises(CheckNotApplicableError, match="overdispersion"):
        check_heteroscedasticity(poisson)


def test_plot(ols):
    fig = check_heteroscedasticity(ols).plot()
    (ax,) = fig.axes
    assert ax.get_title(loc="left").startswith("Homogeneity of Variance")
    assert len(ax.lines) == 1  # the smoother
