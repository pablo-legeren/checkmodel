from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression, Ridge
from statsmodels.stats.outliers_influence import variance_inflation_factor

from checkmodel import CheckNotApplicableError, CollinearityResult, check_collinearity
from tests.conftest import assert_r_close
from tests.models import MODELS, R_TERM_NAMES

WITH_VIF = [
    "lm_mtcars",
    "lm_mtcars_cyl",
    "lm_mtcars_wls",
    "lm_iris",
    "lm_good",
    "lm_collinear",
    "glm_gauss",
    "glm_logit",
    "glm_logit_synth",
    "glm_pois",
    "glm_pois_bio",
    "glm_negbin",
    "glm_probit",
    "glm_cloglog",
    "glm_gamma",
    "glm_invgauss",
]
COLUMNS = [
    "VIF",
    "VIF_CI_low",
    "VIF_CI_high",
    "SE_factor",
    "Tolerance",
    "Tolerance_CI_low",
    "Tolerance_CI_high",
]


def _r_name(term: str) -> str:
    return R_TERM_NAMES.get(term, term)


def test_every_reference_model_is_covered(reference):
    with_vif = {n for n, ref in reference["models"].items() if "vif" in ref}
    assert with_vif == set(WITH_VIF)
    assert set(WITH_VIF) <= set(MODELS)


@pytest.mark.parametrize("name", WITH_VIF)
def test_matches_r(name, fit, reference):
    ref = pd.DataFrame(reference["models"][name]["vif"])
    ref["Term"] = ref["Term"].str.lower().str.replace(".", "_", regex=False)
    table = check_collinearity(fit(name)).table
    table["Term"] = table["Term"].map(_r_name)
    assert list(table["Term"]) == list(ref["Term"])
    for column in COLUMNS:
        assert_r_close(table[column], ref[column])


def test_generalized_vif_for_factor(fit):
    table = check_collinearity(fit("lm_mtcars_cyl")).table.set_index("Term")
    gvif = table.loc["C(cyl)"]
    assert gvif["SE_factor"] == pytest.approx(gvif["VIF"] ** (1 / 4))
    assert table.loc["wt", "SE_factor"] == pytest.approx(np.sqrt(table.loc["wt", "VIF"]))


def test_equals_classical_vif_for_numeric_predictors(ols):
    X = ols.model.exog
    classical = [variance_inflation_factor(X, j) for j in (1, 2)]
    np.testing.assert_allclose(check_collinearity(ols).table["VIF"], classical)


def test_passing_and_failing_models(fit):
    ok = check_collinearity(fit("lm_mtcars"))
    assert ok.passed and ok.message.startswith("OK: Low collinearity")
    bad = check_collinearity(fit("lm_collinear"))
    assert not bad.passed
    assert bad.message == "Warning: High collinearity (VIF >= 10) for x1, x2."
    assert bad.statistic == pytest.approx(bad.table["VIF"].max())


def test_moderate_collinearity_is_a_note(fit):
    res = check_collinearity(fit("lm_collinear"), threshold=1000)
    assert res.passed
    assert res.message.startswith("Note: Moderate collinearity")


def test_sklearn_matches_statsmodels(mtcars_ols, mtcars_logit, reference):
    for model_args, name in ((mtcars_ols, "lm_mtcars"), (mtcars_logit, "glm_logit")):
        table = check_collinearity(*model_args).table
        assert_r_close(table["VIF"], reference["models"][name]["vif"]["VIF"])


def test_penalized_sklearn_uses_design_matrix(mtcars, ols):
    X, y = mtcars[["wt", "hp"]], mtcars["mpg"]
    with pytest.warns(UserWarning, match="penalized fit"):
        ridge = check_collinearity(Ridge(alpha=10.0).fit(X, y), X, y)
    np.testing.assert_allclose(ridge.table["VIF"], check_collinearity(ols).table["VIF"])


def test_needs_two_terms(fit, mtcars):
    X, y = mtcars[["wt"]], mtcars["mpg"]
    with pytest.raises(CheckNotApplicableError, match="Not enough model terms"):
        check_collinearity(LinearRegression().fit(X, y), X, y)
    with pytest.raises(CheckNotApplicableError):
        check_collinearity(fit("lm_hetero"))


def test_warns_without_intercept(mtcars):
    X, y = mtcars[["wt", "hp"]], mtcars["mpg"]
    with pytest.warns(UserWarning, match="without intercept"):
        check_collinearity(LinearRegression(fit_intercept=False).fit(X, y), X, y)


def test_warns_on_rank_deficiency(mtcars):
    X = mtcars[["wt", "hp"]].assign(wt2=mtcars["wt"] * 2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # statsmodels' own SingularMatrixWarning
        fit = sm.OLS(mtcars["mpg"], sm.add_constant(X)).fit()
    with pytest.warns(UserWarning, match="rank deficient"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        check_collinearity(fit)


def test_result_api(ols):
    res = check_collinearity(ols)
    assert isinstance(res, CollinearityResult)
    assert list(res.to_dataframe().columns) == ["Term", *COLUMNS]
    assert res.vif.index.tolist() == ["wt", "hp"]
    assert res.p_value is None


@pytest.mark.parametrize("name", ["lm_mtcars", "lm_collinear", "lm_mtcars_cyl"])
def test_plot(name, fit):
    fig = check_collinearity(fit(name)).plot()
    (ax,) = fig.axes
    assert ax.get_title(loc="left").startswith("Collinearity")
    assert ax.get_legend() is not None


def test_plot_switches_to_log_scale_for_huge_vif():
    rng = np.random.default_rng(1)
    x1 = rng.normal(size=200)
    X = pd.DataFrame({"x1": x1, "x2": x1 + rng.normal(scale=0.01, size=200)})
    y = X.sum(axis=1) + rng.normal(size=200)
    fig = check_collinearity(LinearRegression().fit(X, y), X, y).plot()
    assert fig.axes[0].get_xscale() == "log"


# ------------------------------------------------------ perfect collinearity
def test_rank_deficient_design_fails(mtcars):
    # Every level of cyl as a dummy plus the intercept: cyl_8 = 1 - cyl_4 - cyl_6.
    X = pd.concat(
        [mtcars[["wt"]], pd.get_dummies(mtcars["cyl"], prefix="cyl", dtype=float)], axis=1
    )
    y = mtcars["mpg"]
    with pytest.warns(UserWarning, match="Model matrix is rank deficient"):
        res = check_collinearity(LinearRegression().fit(X, y), X, y)
    assert res.passed is False
    assert res.rank_deficient is True
    assert res.aliased == ["cyl_8"]
    assert res.message == (
        "Warning: Perfect collinearity: the model matrix is rank deficient (aliased terms: cyl_8)."
    )


def test_aliased_columns_match_r(mtcars):
    # lm(mpg ~ wt + hp + wt2) in R gives NA for wt2, the later of the two collinear columns.
    X = mtcars[["wt", "hp"]].assign(wt2=mtcars["wt"] * 2, hp_wt=mtcars["hp"] + mtcars["wt"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = sm.OLS(mtcars["mpg"], sm.add_constant(X)).fit()
        res = check_collinearity(fit)
    assert res.aliased == ["wt2", "hp_wt"]
    assert not res.passed and res.rank_deficient
    assert res.message.endswith("(aliased terms: wt2, hp_wt).")


def test_full_rank_design_is_not_rank_deficient(ols, fit):
    for model in (ols, fit("lm_mtcars_cyl"), fit("lm_collinear")):
        res = check_collinearity(model)
        assert res.rank_deficient is False
        assert res.aliased == []
    assert check_collinearity(ols).passed
