from __future__ import annotations

import numpy as np
import pytest
import statsmodels.api as sm
from sklearn.linear_model import Lasso, LinearRegression

from checkmodel import (
    CheckNotApplicableError,
    R2Result,
    r2,
    r2_coxsnell,
    r2_mcfadden,
    r2_nagelkerke,
    r2_tjur,
)
from tests.conftest import assert_r_close
from tests.models import MODELS

PSEUDO = {
    "tjur": r2_tjur,
    "coxsnell": r2_coxsnell,
    "nagelkerke": r2_nagelkerke,
    "mcfadden": lambda *args: r2_mcfadden(*args)["R2"],
    "mcfadden_adjusted": lambda *args: r2_mcfadden(*args)["R2_adjusted"],
}


@pytest.mark.parametrize("name", list(MODELS))
def test_r2_matches_r(name, fit, reference):
    ref = reference["models"][name]["r2"]
    result = r2(fit(name))
    assert list(result) == list(ref)
    for key, value in ref.items():
        assert_r_close(result[key], value)


@pytest.mark.parametrize("name", ["glm_logit", "glm_logit_synth", "glm_pois"])
def test_pseudo_r2_match_r(name, fit, reference):
    for key, value in reference["models"][name]["pseudo_r2"].items():
        assert_r_close(PSEUDO[key](fit(name)), value)


def test_default_measure_per_model_type(ols, logit, poisson, fit):
    assert list(r2(ols)) == ["R2", "R2_adjusted"]
    assert list(r2(fit("glm_gauss"))) == ["R2"]
    assert list(r2(logit)) == ["R2_Tjur"]
    assert list(r2(poisson)) == ["R2_Nagelkerke"]


def test_ols_r2_equals_statsmodels(ols, fit):
    assert r2(ols)["R2"] == pytest.approx(ols.rsquared)
    assert r2(ols)["R2_adjusted"] == pytest.approx(ols.rsquared_adj)
    wls = fit("lm_mtcars_wls")
    assert r2(wls)["R2"] == pytest.approx(wls.rsquared)


def test_linear_backends_agree_with_r(mtcars_ols, reference):
    ref = reference["models"]["lm_mtcars"]["r2"]
    result = r2(*mtcars_ols)
    assert_r_close(result["R2"], ref["R2"])
    assert_r_close(result["R2_adjusted"], ref["R2_adjusted"])


@pytest.mark.parametrize("key", list(PSEUDO))
def test_logistic_backends_agree_with_r(key, mtcars_logit, reference):
    expected = reference["models"]["glm_logit"]["pseudo_r2"][key]
    assert_r_close(PSEUDO[key](*mtcars_logit), expected)


def test_penalized_model_uses_one_minus_rss_over_tss(mtcars):
    X, y = mtcars[["wt", "hp"]], mtcars["mpg"]
    lasso = Lasso(alpha=1.0).fit(X, y)
    assert r2(lasso, X, y)["R2"] == pytest.approx(lasso.score(X, y))


def test_model_without_intercept_uses_uncentered_r2(mtcars):
    X, y = mtcars[["wt", "hp"]], mtcars["mpg"]
    ours = r2(LinearRegression(fit_intercept=False).fit(X, y), X, y)
    expected = sm.OLS(y, X).fit()
    assert ours["R2"] == pytest.approx(expected.rsquared)
    assert ours["R2_adjusted"] == pytest.approx(expected.rsquared_adj)


def test_binomial_with_trials_is_refused():
    successes = np.array([3, 5, 8, 9, 2, 6])
    endog = np.column_stack([successes, 10 - successes])
    model = sm.GLM(endog, sm.add_constant(np.arange(6.0)), family=sm.families.Binomial()).fit()
    for fn in (r2, r2_coxsnell, r2_nagelkerke, r2_mcfadden):
        with pytest.raises(CheckNotApplicableError, match="Bernoulli"):
            fn(model)


def test_tjur_needs_binomial(ols):
    with pytest.raises(CheckNotApplicableError, match="binomial"):
        r2_tjur(ols)


def test_result_api(ols, logit):
    result = r2(ols)
    assert isinstance(result, R2Result)
    assert len(result) == 2
    assert result.to_dataframe().columns.tolist() == ["R2", "R2_adjusted"]
    assert repr(result).splitlines() == [
        "# R2 for Linear Regression",
        f"       R2: {result['R2']:.3f}",
        f"  adj. R2: {result['R2_adjusted']:.3f}",
    ]
    assert str(r2(logit)).startswith("# R2 for Logistic Regression")
