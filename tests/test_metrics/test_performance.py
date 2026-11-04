from __future__ import annotations

import math
import warnings

import numpy as np
import pandas as pd
import pytest
import statsmodels.formula.api as smf
from sklearn.linear_model import Ridge

from checkmodel import model_performance
from tests.conftest import assert_r_close
from tests.models import MODELS, sleepstudy


@pytest.mark.parametrize("name", list(MODELS))
def test_matches_r(name, fit, reference):
    ref = reference["models"][name]["performance"]
    perf = model_performance(fit(name))
    assert list(perf.columns) == list(ref)
    for column, value in ref.items():
        expected = value[0] if isinstance(value, list) else value
        if expected is None:  # R's -Inf (the log score underflows) is written as null
            assert perf.loc[0, column] == -math.inf
        else:
            assert_r_close(perf.loc[0, column], expected)


def test_columns_by_model_type(ols, logit, poisson):
    assert list(model_performance(ols)) == [
        "AIC",
        "AICc",
        "BIC",
        "R2",
        "R2_adjusted",
        "RMSE",
        "Sigma",
    ]
    assert list(model_performance(logit)) == [
        "AIC",
        "AICc",
        "BIC",
        "R2_Tjur",
        "RMSE",
        "Sigma",
        "Log_loss",
        "Score_log",
        "Score_spherical",
        "PCP",
    ]
    assert list(model_performance(poisson)) == [
        "AIC",
        "AICc",
        "BIC",
        "R2_Nagelkerke",
        "RMSE",
        "Sigma",
        "Score_log",
        "Score_spherical",
    ]


def test_aic_counts_sigma_unlike_statsmodels(ols):
    """R's AIC(lm) counts the residual variance; statsmodels' OLS aic does not."""
    assert model_performance(ols).loc[0, "AIC"] == pytest.approx(ols.aic + 2)


def test_backends_agree(mtcars_ols, mtcars_logit, reference):
    for args, name in ((mtcars_ols, "lm_mtcars"), (mtcars_logit, "glm_logit")):
        perf = model_performance(*args)
        for column, value in reference["models"][name]["performance"].items():
            assert_r_close(perf.loc[0, column], value)


def test_metric_selection(ols, logit):
    assert list(model_performance(ols, metrics="common")) == [
        "AIC",
        "BIC",
        "R2",
        "R2_adjusted",
        "RMSE",
    ]
    assert list(model_performance(ols, metrics=["rmse", "r2_adj"])) == ["R2_adjusted", "RMSE"]
    assert list(model_performance(ols, metrics="R2")) == ["R2"]
    assert list(model_performance(logit, metrics=["log_loss"])) == ["Log_loss"]
    loglik = model_performance(ols, metrics=["AIC", "Log_lik"])
    assert loglik.loc[0, "Log_lik"] == pytest.approx(ols.llf)
    with pytest.raises(ValueError, match="Unknown metrics"):
        model_performance(ols, metrics=["AIC", "nope"])


def test_penalized_model_has_no_information_criteria(mtcars):
    X, y = mtcars[["wt", "hp"]], mtcars["mpg"]
    perf = model_performance(Ridge(alpha=1.0).fit(X, y), X, y)
    assert list(perf) == ["R2", "R2_adjusted", "RMSE", "Sigma"]


def test_returns_one_row_dataframe(ols):
    perf = model_performance(ols)
    assert isinstance(perf, pd.DataFrame)
    assert perf.shape[0] == 1


# Mirrors `transformed` in generate_reference.R.
TRANSFORMED = {
    "log": lambda d, s: smf.ols("np.log(mpg) ~ wt + hp", d).fit(),
    "log1p": lambda d, s: smf.ols("np.log1p(mpg) ~ wt + hp", d).fit(),
    "log2": lambda d, s: smf.ols("np.log2(mpg) ~ wt + hp", d).fit(),
    "log10": lambda d, s: smf.ols("np.log10(mpg) ~ wt + hp", d).fit(),
    "sqrt": lambda d, s: smf.ols("np.sqrt(mpg) ~ wt + hp", d).fit(),
    "exp": lambda d, s: smf.ols("np.exp(wt) ~ hp + qsec", d).fit(),
    "power": lambda d, s: smf.ols("I(mpg**2) ~ wt + hp", d).fit(),
    "weighted_log": lambda d, s: smf.wls("np.log(mpg) ~ wt + hp", d, weights=d["cyl"]).fit(),
    "glm_log": lambda d, s: smf.glm("np.log(mpg) ~ wt + hp", d).fit(),
    "mixed_log_reml": lambda d, s: smf.mixedlm(
        "np.log(Reaction) ~ Days", s, groups="Subject"
    ).fit(),
    "mixed_log_ml": lambda d, s: smf.mixedlm("np.log(Reaction) ~ Days", s, groups="Subject").fit(
        reml=False
    ),
}


@pytest.mark.parametrize("name", list(TRANSFORMED))
def test_transformed_response_matches_r(name, mtcars, reference):
    model = TRANSFORMED[name](mtcars, sleepstudy())
    # As in R: REML criteria for the REML fit, ML for the rest.
    estimator = "ML" if name == "mixed_log_ml" else None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        perf = model_performance(model, metrics=["AIC", "AICc", "BIC"], estimator=estimator)
    for column, expected in reference["transformed"][name].items():
        assert_r_close(perf.loc[0, column], expected)


def test_transformed_response_keeps_log_lik_uncorrected(mtcars):
    model = smf.ols("np.log(mpg) ~ wt + hp", mtcars).fit()
    perf = model_performance(model, metrics=["AIC", "Log_lik"])
    assert_r_close(perf.loc[0, "Log_lik"], model.llf)
    assert_r_close(perf.loc[0, "AIC"], -2 * (model.llf - np.log(mtcars["mpg"]).sum()) + 2 * 4)


def test_transformed_response_warns_under_reml():
    model = smf.mixedlm("np.log(Reaction) ~ Days", sleepstudy(), groups="Subject").fit()
    with pytest.warns(UserWarning, match="transformed response"):
        model_performance(model, metrics=["AIC"])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        model_performance(model, metrics=["AIC"], estimator="ML")


def test_unrecognized_transformation_is_left_alone(mtcars):
    # A transformation of an expression, not of a plain column, is not corrected.
    model = smf.ols("np.log(mpg / 10) ~ wt + hp", mtcars).fit()
    assert_r_close(model_performance(model, metrics=["AIC"]).loc[0, "AIC"], -2 * model.llf + 2 * 4)
