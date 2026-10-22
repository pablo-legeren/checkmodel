"""check_overdispersion() and check_zeroinflation() for count and other GLMs."""

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
    simulate_residuals,
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


def test_overdispersion_needs_a_glm(ols, sk_linear):
    for model, X, y in [(ols, None, None), sk_linear]:
        with pytest.raises(CheckNotApplicableError, match="check_heteroscedasticity"):
            check_overdispersion(model, X, y)


def test_zeroinflation_needs_a_count_model(ols, logit):
    for model in (ols, logit):
        with pytest.raises(CheckNotApplicableError, match="count models"):
            check_zeroinflation(model)


@pytest.mark.parametrize(
    ("name", "method"),
    [
        ("glm_pois_bio", "Pearson chi-squared"),
        ("glm_logit", "Simulated residuals"),
        ("glm_gauss", "Simulated residuals"),
        ("mixed_sleep", "Simulated residuals"),
    ],
)
def test_overdispersion_residuals_follow_r(name, method, fit):
    # performance's `.use_simulated_residuals()`: Pearson residuals for Poisson GLMs only.
    res = check_overdispersion(fit(name), random_state=0)
    assert res.method.startswith(method)
    assert (res.simulation is None) == (method == "Pearson chi-squared")


def test_negative_binomial_uses_simulated_residuals(mtcars):
    # carb varies far less than a negative binomial with alpha = 1 allows.
    negbin = smf.glm("carb ~ wt", mtcars, family=sm.families.NegativeBinomial(alpha=1.0)).fit()
    od = check_overdispersion(negbin, random_state=0)
    assert od.simulation is not None and od.statistic is None and od.df is None
    assert od.dispersion_ratio < 1
    assert od.message.startswith("Warning: Underdispersion detected (dispersion ratio = ")


def test_residual_type(fit, logit):
    bio = fit("glm_pois_bio")
    pearson = check_overdispersion(bio)
    assert check_overdispersion(bio, residual_type="pearson").statistic == pearson.statistic
    assert check_overdispersion(bio, residual_type="normal").statistic == pearson.statistic
    simulated = check_overdispersion(bio, residual_type="simulated", random_state=0)
    assert simulated.simulation is not None
    assert not simulated.passed and simulated.dispersion_ratio > 1.5
    with pytest.raises(CheckNotApplicableError, match="Bernoulli"):
        check_overdispersion(logit, residual_type="pearson")
    with pytest.raises(CheckNotApplicableError, match="mixed models"):
        check_overdispersion(fit("mixed_sleep"), residual_type="pearson")
    with pytest.raises(CheckNotApplicableError, match="Poisson families"):
        check_overdispersion(fit("glm_gauss"), residual_type="pearson")
    with pytest.raises(ValueError, match="residual_type"):
        check_overdispersion(bio, residual_type="deviance")


def test_simulated_dispersion_statistic(fit):
    # check_overdispersion.performance_simres(), computed by hand.
    sim = simulate_residuals(fit("glm_pois_bio"), iterations=100, random_state=3)
    res = check_overdispersion(sim)
    variance = np.var(sim.simulated, ddof=1)
    observed = np.var(sim.observed - sim.fitted, ddof=1) / variance
    simulated = np.var(sim.simulated - sim.fitted, axis=1, ddof=1) / variance
    assert res.dispersion_ratio == pytest.approx(observed / simulated.mean())
    p = min(2 * min(np.mean(simulated <= observed), np.mean(simulated >= observed)), 1)
    assert res.p_value == pytest.approx(p)
    assert check_overdispersion(sim, alternative="greater").p_value == np.mean(
        simulated >= observed
    )


def test_simulated_zeroinflation(fit, mtcars):
    sim = simulate_residuals(fit("glm_pois_bio"), random_state=0)
    res = check_zeroinflation(sim)
    zeros = np.sum(sim.simulated == 0, axis=1)
    assert res.tolerance == 0.1
    assert res.observed_zeros == 275
    assert res.predicted_zeros == round(zeros.mean())
    assert res.statistic == pytest.approx(zeros.mean() / 275)
    assert res.p_value == 0.0
    assert res.message.startswith("Warning: Model is underfitting zeros (probable zero-inflation)")
    assert res.message.endswith(", p < .001.")
    assert check_zeroinflation(sim, tolerance=0.5).passed

    negbin = smf.glm(
        "art ~ fem + mar + kid5 + phd + ment",
        fit("glm_pois_bio").model.data.frame,
        family=sm.families.NegativeBinomial(alpha=0.45),
    ).fit()
    nb = check_zeroinflation(negbin, random_state=0)
    assert nb.simulation is not None and nb.tolerance == 0.1
    assert nb.passed, nb.message


def test_results_and_plots(fit):
    od = check_overdispersion(fit("glm_pois_bio"))
    assert isinstance(od, OverdispersionResult)
    assert od.to_dataframe().loc[0, "Statistic"] == od.statistic
    ax = od.plot().axes[0]
    assert ax.get_title(loc="left").startswith("Overdispersion")
    assert ax.get_legend().get_texts()[1].get_text() == "Poisson variance (= mean)"
    zi = check_zeroinflation(fit("glm_pois_bio"))
    assert isinstance(zi, ZeroInflationResult)
    ax = zi.plot().axes[0]
    assert ax.get_title(loc="left").startswith("Zero-inflation")
    assert len(ax.patches) >= 10  # one bar per count


def test_simulated_plots(fit):
    sim = simulate_residuals(fit("glm_pois_bio"), random_state=0)
    ax = check_overdispersion(sim).plot().axes[0]
    assert ax.get_legend().get_texts()[1].get_text() == "Model variance (simulated)"
    ax = check_zeroinflation(sim).plot().axes[0]
    assert ax.get_legend().get_texts()[1].get_text() == "Expected (simulated)"
    ax = check_overdispersion(fit("glm_logit"), random_state=0).plot().axes[0]
    assert ax.get_legend().get_texts()[1].get_text() == "Model variance (simulated)"
