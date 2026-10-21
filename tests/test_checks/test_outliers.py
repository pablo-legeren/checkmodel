from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Lasso

from checkmodel import OutliersResult, check_outliers
from tests.conftest import assert_r_close
from tests.models import MODELS


@pytest.mark.parametrize("name", list(MODELS))
def test_matches_r(name, fit, reference):
    """Threshold qf(0.5, k, n - k) and the flagged observations, vs check_outliers()."""
    ref = reference["models"][name]["outliers"]
    res = check_outliers(fit(name))
    assert_r_close(res.threshold, ref["threshold"])
    flags = (res.distances > res.threshold).to_numpy()
    if len(flags) <= 400:
        assert flags.tolist() == ref["flags"]
    else:  # R stores the 1-based positions of the flagged rows
        assert (np.flatnonzero(flags) + 1).tolist() == list(ref["flags"])
    assert res.passed == (not any(flags))


def test_flags_and_message(fit):
    res = check_outliers(fit("glm_pois"))
    assert not res.passed
    assert res.outliers == [4]
    assert res.message == "Warning: 1 outlier detected: case 4 (Cook's distance > 0.461)."

    ok = check_outliers(fit("lm_mtcars"))
    assert ok.passed and ok.outliers == []
    assert ok.message == "OK: No outliers detected (Cook's distance > 0.808)."


def test_labels_come_from_the_index(fit):
    res = check_outliers(fit("lm_mtcars"), threshold="4/n")
    assert res.threshold == pytest.approx(4 / 32)
    assert "Chrysler Imperial" in res.outliers
    assert "cases" in res.message


def test_numeric_threshold(fit):
    res = check_outliers(fit("lm_mtcars"), threshold=0.1)
    assert res.threshold == 0.1
    assert res.method == "Cook's distance > threshold"
    assert len(res.outliers) == int((res.distances > 0.1).sum())


def test_long_list_is_truncated(fit):
    res = check_outliers(fit("lm_large"), threshold=0.0)
    assert "more" in res.message


def test_same_result_for_statsmodels_and_sklearn(mtcars_ols, mtcars_logit, reference):
    for args, name in ((mtcars_ols, "lm_mtcars"), (mtcars_logit, "glm_logit")):
        res = check_outliers(*args)
        ref = reference["models"][name]
        assert_r_close(res.distances, ref["cooks"])
        assert_r_close(res.threshold, ref["outliers"]["threshold"])


def test_penalized_sklearn_runs(mtcars):
    X, y = mtcars[["wt", "hp"]], mtcars["mpg"]
    res = check_outliers(Lasso(alpha=0.5).fit(X, y), X, y)
    assert res.distances.index.equals(X.index)


def test_result_api(fit):
    res = check_outliers(fit("glm_pois"))
    assert isinstance(res, OutliersResult)
    df = res.to_dataframe()
    assert list(df.columns) == ["Cooks_Distance", "Leverage", "Std_Residual", "Outlier"]
    assert df["Outlier"].sum() == 1
    assert isinstance(res.distances, pd.Series)
    assert res.statistic == pytest.approx(res.distances.max())


def test_plot(fit):
    fig = check_outliers(fit("glm_pois")).plot()
    (ax,) = fig.axes
    assert ax.get_title(loc="left").startswith("Influential Observations")
    labels = {t.get_text() for t in ax.texts}
    assert "4" in labels
