from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

from checkmodel import BinnedResidualsResult, CheckNotApplicableError, binned_residuals
from tests.conftest import assert_r_close

BINOMIAL = ["glm_logit", "glm_logit_synth"]
COLUMNS = {
    "xbar": "xbar",
    "ybar": "ybar",
    "n": "n",
    "x_lo": "x.lo",
    "x_hi": "x.hi",
    "se": "se",
    "CI_low": "CI_low",
    "CI_high": "CI_high",
}


@pytest.mark.parametrize("name", BINOMIAL)
def test_matches_r(name, fit, reference):
    ref = reference["models"][name]["binned"]
    res = binned_residuals(fit(name))
    for ours, theirs in COLUMNS.items():
        assert_r_close(res.table[ours], ref[theirs])
    assert res.table["inside"].tolist() == [g == "yes" for g in ref["group"]]
    assert_r_close(res.statistic, ref["resid_ok"])


def test_messages(fit):
    bad = binned_residuals(fit("glm_logit"))
    assert not bad.passed
    assert bad.message.startswith("Warning: Probably bad model fit. Only about 50%")
    good = binned_residuals(fit("glm_logit_synth"))
    assert good.passed
    assert good.message.startswith(("OK: About", "Note: About"))


def test_sklearn_matches_statsmodels(mtcars_logit, reference):
    res = binned_residuals(*mtcars_logit)
    assert_r_close(res.table["ybar"], reference["models"]["glm_logit"]["binned"]["ybar"])


def test_options(fit):
    model = fit("glm_logit_synth")
    gaussian = binned_residuals(model, ci_type="gaussian", n_bins=10)
    assert len(gaussian.table) == 10
    width = gaussian.table["CI_high"] - gaussian.table["CI_low"]
    np.testing.assert_allclose(width / 2, gaussian.table["se"])
    pearson = binned_residuals(model, residuals="pearson")
    deviance = binned_residuals(model, residuals="deviance")
    assert not np.allclose(pearson.table["ybar"], deviance.table["ybar"])
    by_term = binned_residuals(model, term="x")
    assert by_term.table["xbar"].min() < 0  # binned on x, not on probabilities
    with pytest.raises(ValueError, match="Unknown term"):
        binned_residuals(model, term="nope")


def test_proportion_response_uses_gaussian_bounds():
    rng = np.random.default_rng(3)
    x = rng.normal(size=60)
    trials = np.full(60, 12)
    successes = rng.binomial(trials, 1 / (1 + np.exp(-x)))
    endog = np.column_stack([successes, trials - successes])
    model = sm.GLM(endog, sm.add_constant(x), family=sm.families.Binomial()).fit()
    res = binned_residuals(model)
    width = res.table["CI_high"] - res.table["CI_low"]
    np.testing.assert_allclose(width / 2, res.table["se"])


def test_note_between_80_and_95_percent(fit, monkeypatch):
    import checkmodel.checks.binned as module

    monkeypatch.setattr(module, "_GOOD_SHARE_INSIDE", 1.01)
    res = binned_residuals(fit("glm_logit_synth"))
    assert res.passed and res.message.startswith("Note: About")


def test_not_applicable_to_linear(ols):
    with pytest.raises(CheckNotApplicableError):
        binned_residuals(ols)


def test_result_api_and_plot(logit):
    res = binned_residuals(logit)
    assert isinstance(res, BinnedResidualsResult)
    assert isinstance(res.to_dataframe(), pd.DataFrame)
    fig = res.plot()
    assert fig.axes[0].get_title(loc="left").startswith("Binned Residuals")
