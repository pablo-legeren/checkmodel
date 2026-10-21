from __future__ import annotations

import math

import pandas as pd
import pytest
from sklearn.linear_model import Ridge

from checkmodel import model_performance
from tests.conftest import assert_r_close
from tests.models import MODELS


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
