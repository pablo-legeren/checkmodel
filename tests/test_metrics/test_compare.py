from __future__ import annotations

import warnings

import pandas as pd
import pytest
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.linear_model import LinearRegression, Ridge

from checkmodel import compare_performance
from tests.conftest import assert_r_close


@pytest.fixture(scope="module")
def nested(mtcars):
    formulas = ["mpg ~ wt", "mpg ~ wt + hp", "mpg ~ wt + hp + qsec"]
    return [smf.ols(f, mtcars).fit() for f in formulas]


@pytest.fixture(scope="module")
def logits(mtcars):
    family = sm.families.Binomial()
    return [smf.glm(f, mtcars, family=family).fit() for f in ["vs ~ mpg", "vs ~ mpg + wt"]]


def _assert_table_matches(table: pd.DataFrame, ref: dict) -> None:
    expected = pd.DataFrame(ref)
    assert list(table.columns) == list(expected.columns)
    assert table["Name"].tolist() == expected["Name"].tolist()
    for column in expected.columns.drop(["Name", "Model"]):
        assert_r_close(table[column], expected[column])


def test_matches_r(nested, reference):
    table = compare_performance(*nested, names=["small", "medium", "big"])
    _assert_table_matches(table, reference["compare"]["lm"])
    assert table["AIC_wt"].sum() == pytest.approx(1.0)


def test_rank_matches_r(nested, reference):
    table = compare_performance(*nested, names=["small", "medium", "big"], rank=True)
    _assert_table_matches(table, reference["compare"]["lm_rank"])


def test_logistic_matches_r(logits, reference):
    table = compare_performance(logit_1=logits[0], logit_2=logits[1])
    _assert_table_matches(table, reference["compare"]["logit"])


def test_default_names_and_model_column(nested):
    table = compare_performance(*nested)
    assert table["Name"].tolist() == ["Model 1", "Model 2", "Model 3"]
    assert set(table["Model"]) == {"OLS"}


def test_mixing_backends(nested, mtcars):
    X, y = mtcars[["wt", "hp"]], mtcars["mpg"]
    sk = LinearRegression().fit(X, y)
    with pytest.warns(UserWarning, match="ignored for statsmodels"):
        table = compare_performance(nested[1], sk, X=X, y=y, names=["statsmodels", "sklearn"])
    numeric = table.drop(columns=["Name", "Model"])
    pd.testing.assert_series_equal(numeric.iloc[0], numeric.iloc[1], check_names=False)
    assert table["Model"].tolist() == ["OLS", "LinearRegression"]


def test_tuples_for_models_with_their_own_data(mtcars):
    X1, X2, y = mtcars[["wt"]], mtcars[["wt", "hp"]], mtcars["mpg"]
    table = compare_performance(
        (LinearRegression().fit(X1, y), X1, y),
        big=(LinearRegression().fit(X2, y), X2, y),
    )
    assert table["Name"].tolist() == ["Model 1", "big"]


def test_missing_information_criteria_give_missing_weights(nested, mtcars):
    X, y = mtcars[["wt", "hp"]], mtcars["mpg"]
    table = compare_performance(nested[1], ridge=(Ridge().fit(X, y), X, y))
    assert table["AIC_wt"].isna().all()


def test_warns_when_data_differ(nested, mtcars):
    other = smf.ols("qsec ~ wt", mtcars).fit()
    with pytest.warns(UserWarning, match="not all models were fit from same data"):
        compare_performance(nested[0], other)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        compare_performance(nested[0], other, verbose=False)


def test_rank_ignores_incomplete_indices(nested, logits):
    with pytest.warns(UserWarning):
        table = compare_performance(nested[0], logits[1], rank=True)
    assert "Performance_Score" in table


def test_argument_errors(nested):
    with pytest.raises(ValueError, match="names"):
        compare_performance(*nested, names=["a"])
    with pytest.raises(ValueError, match="at least one"):
        compare_performance()
