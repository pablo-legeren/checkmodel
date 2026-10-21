"""check_overdispersion() and check_zeroinflation() for Poisson models."""

from __future__ import annotations

import numpy as np
import pytest
import statsmodels.api as sm
import statsmodels.formula.api as smf

from checkmodel import (
    CheckNotApplicableError,
    OverdispersionResult,
    ZeroInflationResult,
    check_overdispersion,
    check_zeroinflation,
)
from tests.conftest import assert_r_close

POISSON = ["glm_pois", "glm_pois_bio", "glm_pois_synth"]


@pytest.mark.parametrize("name", POISSON)
def test_overdispersion_matches_r(name, fit, reference):
    ref = reference["models"][name]["overdispersion"]
    res = check_overdispersion(fit(name))
    assert_r_close(res.statistic, ref["chisq"])
    assert_r_close(res.dispersion_ratio, ref["ratio"])
    assert_r_close(res.p_value, ref["p"])
    assert res.df == ref["df"]


def test_overdispersion_verdicts(fit):
    over = check_overdispersion(fit("glm_pois_bio"))
    assert not over.passed
    assert over.message.startswith("Warning: Overdispersion detected (dispersion ratio = 1.829")
    ok = check_overdispersion(fit("glm_pois_synth"))
    assert ok.passed
    assert ok.message == "OK: No overdispersion detected (dispersion ratio = 0.905, p = 0.878)."


def test_underdispersion():
    x = np.repeat(np.arange(10.0), 5)
    y = np.round(np.exp(0.5 + 0.2 * x))  # counts sit exactly on their mean
    model = sm.GLM(y, sm.add_constant(x), family=sm.families.Poisson()).fit()
    res = check_overdispersion(model, alpha=1.0)
    assert not res.passed and res.dispersion_ratio < 1
    assert res.message.startswith("Warning: Underdispersion detected")


@pytest.mark.parametrize("name", ["glm_pois_bio", "glm_pois_synth"])
def test_zeroinflation_matches_r(name, fit, reference):
    ref = reference["models"][name]["zeroinflation"]
    res = check_zeroinflation(fit(name))
    assert res.predicted_zeros == ref["predicted"]
    assert res.observed_zeros == ref["observed"]
    assert_r_close(res.statistic, ref["ratio"])


def test_zeroinflation_verdicts(fit):
    under = check_zeroinflation(fit("glm_pois_bio"))
    assert not under.passed
    assert under.message == (
        "Warning: Model is underfitting zeros (probable zero-inflation): "
        "275 observed, 191 predicted, ratio 0.69."
    )
    over = check_zeroinflation(fit("glm_pois_synth"))  # ratio 1.051, just above 1.05
    assert not over.passed and over.message.startswith("Warning: Model is overfitting zeros")
    ok = check_zeroinflation(fit("glm_pois_synth"), tolerance=0.1)
    assert ok.passed and ok.message.startswith("OK: Observed and predicted zeros")


def test_zeroinflation_needs_zeros(fit):
    with pytest.raises(CheckNotApplicableError, match="no observed zeros"):
        check_zeroinflation(fit("glm_pois"))


@pytest.mark.parametrize("check", [check_overdispersion, check_zeroinflation])
def test_only_poisson(check, ols, logit, mtcars):
    with pytest.raises(CheckNotApplicableError, match="Poisson"):
        check(ols)
    with pytest.raises(CheckNotApplicableError, match="Bernoulli"):
        check(logit)
    negbin = smf.glm("carb ~ wt", mtcars, family=sm.families.NegativeBinomial(alpha=1.0)).fit()
    with pytest.raises(CheckNotApplicableError, match="simulated residuals"):
        check(negbin)


def test_results_and_plots(fit):
    od = check_overdispersion(fit("glm_pois_bio"))
    assert isinstance(od, OverdispersionResult)
    assert od.to_dataframe().loc[0, "Statistic"] == od.statistic
    ax = od.plot().axes[0]
    assert ax.get_title(loc="left").startswith("Overdispersion")
    zi = check_zeroinflation(fit("glm_pois_bio"))
    assert isinstance(zi, ZeroInflationResult)
    ax = zi.plot().axes[0]
    assert ax.get_title(loc="left").startswith("Zero-inflation")
    assert len(ax.patches) >= 10  # one bar per count
