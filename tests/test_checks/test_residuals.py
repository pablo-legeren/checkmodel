"""simulate_residuals(), check_residuals() and the simulation of new responses."""

from __future__ import annotations

import numpy as np
import pytest
import statsmodels.api as sm
from scipy import optimize, special, stats

from checkmodel import (
    CheckNotApplicableError,
    ResidualsResult,
    SimulatedResiduals,
    as_adapter,
    check_model,
    check_overdispersion,
    check_predictions,
    check_residuals,
    simulate_residuals,
)
from checkmodel.checks.residuals import _quantile_residuals, ks_test, simulation_p_value

LOG = sm.families.links.Log()


def _x(seed: int, n: int = 300) -> tuple[np.random.Generator, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    return rng, x, sm.add_constant(x)


def glm(family: str, seed: int = 1, n: int = 300):
    """A GLM fitted to data generated from the same family."""
    rng, x, X = _x(seed, n)
    eta = 0.5 + 0.4 * x
    if family == "poisson":
        return sm.GLM(rng.poisson(np.exp(eta)), X, family=sm.families.Poisson()).fit()
    if family == "negbin":
        y = rng.negative_binomial(2.0, 2.0 / (2.0 + np.exp(eta)))
        return sm.GLM(y, X, family=sm.families.NegativeBinomial(alpha=0.5)).fit()
    if family == "bernoulli":
        y = rng.binomial(1, special.expit(eta))
        return sm.GLM(y, X, family=sm.families.Binomial()).fit()
    if family == "binomial_trials":
        trials = rng.integers(2, 12, size=n)
        successes = rng.binomial(trials, special.expit(eta))
        endog = np.column_stack([successes, trials - successes])
        return sm.GLM(endog, X, family=sm.families.Binomial()).fit()
    if family == "gamma":
        y = rng.gamma(4.0, np.exp(1 + eta) / 4.0)
        return sm.GLM(y, X, family=sm.families.Gamma(LOG)).fit()
    if family == "inverse_gaussian":
        y = rng.wald(np.exp(eta), 8.0)
        return sm.GLM(y, X, family=sm.families.InverseGaussian(LOG)).fit()
    if family == "wls":
        w = rng.uniform(0.5, 3.0, size=n)
        y = 1 + 2 * x + rng.normal(size=n) / np.sqrt(w)
        return sm.WLS(y, X, weights=w).fit()
    raise ValueError(family)


def _theoretical_variance(model, adapter) -> np.ndarray:
    fam, mu = adapter.info.family, adapter.mu
    if fam == "gaussian":
        return adapter.scale / adapter.simulation_weights
    if fam == "binomial":
        return mu * (1 - mu) / adapter.binomial_trials
    if fam == "poisson":
        return mu
    if fam == "negbin":
        return mu + mu**2 / adapter.negbin_theta
    if fam == "gamma":
        return mu**2 / adapter.gamma_shape
    return mu**3 * adapter.scale  # inverse Gaussian: mu^3 / lambda


@pytest.mark.parametrize(
    "family",
    ["wls", "poisson", "negbin", "bernoulli", "binomial_trials", "gamma", "inverse_gaussian"],
)
def test_simulations_follow_the_fitted_distribution(family):
    model = glm(family)
    adapter = as_adapter(model)
    sims = adapter.simulate(4000, np.random.default_rng(0))
    assert sims.shape == (4000, adapter.n_obs)
    variance = _theoretical_variance(model, adapter)
    z = (sims.mean(axis=0) - adapter.mu) / np.sqrt(variance / 4000)
    assert np.max(np.abs(z)) < 5
    ratio = sims.var(axis=0, ddof=1) / variance
    assert abs(np.mean(ratio) - 1) < 0.03


def test_binomial_with_trials_simulates_proportions():
    adapter = as_adapter(glm("binomial_trials"))
    sims = adapter.simulate(5, np.random.default_rng(0))
    successes = sims * adapter.binomial_trials
    np.testing.assert_allclose(successes, np.round(successes), atol=1e-9)
    assert np.all((sims >= 0) & (sims <= 1))


def test_gamma_shape_is_the_maximum_likelihood_estimate():
    adapter = as_adapter(glm("gamma", n=500))
    y, mu = adapter.y, adapter.mu

    def nll(log_shape: float) -> float:
        shape = np.exp(log_shape)
        return -float(np.sum(stats.gamma.logpdf(y, shape, scale=mu / shape)))

    direct = float(np.exp(optimize.minimize_scalar(nll).x))
    assert adapter.gamma_shape == pytest.approx(direct, rel=1e-6)


def test_frequency_weights_replicate_rows_without_narrowing_the_distribution():
    rng, x, X = _x(4, 400)
    fw = rng.integers(1, 6, size=400).astype(float)
    y = rng.binomial(1, special.expit(0.3 + 0.8 * x))
    logit = sm.GLM(y, X, family=sm.families.Binomial(), freq_weights=fw).fit()
    np.testing.assert_array_equal(as_adapter(logit).binomial_trials, np.ones(400))
    assert check_residuals(logit, random_state=0).passed
    ratio = check_overdispersion(logit, random_state=0).dispersion_ratio
    assert ratio == pytest.approx(1, abs=0.15)

    yg = rng.gamma(4.0, np.exp(1 + 0.3 * x) / 4.0)
    gamma = sm.GLM(yg, X, family=sm.families.Gamma(LOG), freq_weights=fw).fit()
    adapter = as_adapter(gamma)

    def nll(log_shape: float) -> float:
        shape = np.exp(log_shape)
        return -float(np.sum(fw * stats.gamma.logpdf(yg, shape, scale=adapter.mu / shape)))

    direct = float(np.exp(optimize.minimize_scalar(nll).x))
    assert adapter.gamma_shape == pytest.approx(direct, rel=1e-6)


def test_families_that_cannot_be_simulated(mtcars):
    tweedie = sm.GLM(
        mtcars["mpg"], sm.add_constant(mtcars["wt"]), family=sm.families.Tweedie(var_power=1.5)
    ).fit()
    with pytest.raises(CheckNotApplicableError, match="tweedie"):
        simulate_residuals(tweedie)
    rng, _, X = _x(0, 50)
    fractional = sm.GLM(
        rng.uniform(size=50),
        X,
        family=sm.families.Binomial(),
        var_weights=rng.uniform(1, 2, size=50),
    ).fit()
    with pytest.raises(CheckNotApplicableError, match="non-integer"):
        simulate_residuals(fractional)


def test_quantile_residuals():
    rng = np.random.default_rng(0)
    simulated = np.array([[0.0, 1.0, 5.0], [1.0, 1.0, 6.0], [2.0, 3.0, 7.0], [3.0, 4.0, 8.0]])
    observed = np.array([1.5, 1.0, 9.0])
    res = _quantile_residuals(simulated, observed, rng)
    assert res[0] == 0.5  # continuous: share of simulations below
    assert 0.0 <= res[1] <= 0.5  # tied with two of four: randomized in [0, 0.5]
    assert res[2] == 1.0  # above every simulation
    ties = np.array([_quantile_residuals(simulated, observed, rng)[1] for _ in range(2000)])
    assert ties.min() >= 0 and ties.max() <= 0.5
    assert ties.mean() == pytest.approx(0.25, abs=0.02)


def test_simulate_residuals_object(ols):
    sim = simulate_residuals(ols, iterations=50, random_state=1)
    assert isinstance(sim, SimulatedResiduals)
    assert sim.iterations == 50
    assert sim.simulated.shape == (50, 32)
    assert np.all((sim.scaled_residuals >= 0) & (sim.scaled_residuals <= 1))
    again = simulate_residuals(ols, iterations=50, random_state=1)
    np.testing.assert_array_equal(sim.scaled_residuals, again.scaled_residuals)
    frame = sim.to_dataframe()
    assert list(frame.columns) == ["Observed", "Fitted", "Residual"]
    assert frame.index[0] == "Mazda RX4"
    assert repr(sim).startswith("Simulated residuals from a OLS model based on 50 simulations.")
    with pytest.raises(ValueError, match="at least 2"):
        simulate_residuals(ols, iterations=1)


def test_residuals_transformations(poisson):
    sim = simulate_residuals(poisson, iterations=5, random_state=0)
    np.testing.assert_array_equal(sim.residuals(), sim.scaled_residuals)
    normal = sim.residuals(quantile_function=stats.norm.ppf)
    assert np.isinf(normal).any()  # with 5 simulations some observations fall outside
    capped = sim.residuals(quantile_function=stats.norm.ppf, outlier_values=(-4, 4))
    assert np.isfinite(capped).all()
    assert set(capped[np.isinf(normal)]) <= {-4.0, 4.0}
    with pytest.raises(ValueError, match="two values"):
        sim.residuals(quantile_function=stats.norm.ppf, outlier_values=(1, 2, 3))


@pytest.mark.parametrize("n", [30, 99, 100, 400])
def test_ks_test_matches_scipy(n):
    values = np.random.default_rng(n).beta(1.2, 1.0, size=n)
    method = "exact" if n < 100 else "asymp"
    for alternative in ("two-sided", "less", "greater"):
        d, p = ks_test(values, stats.uniform.cdf, alternative)
        ref = stats.kstest(values, "uniform", alternative=alternative, method=method)
        assert d == pytest.approx(ref.statistic, abs=1e-12)
        if alternative == "two-sided" or n < 100:
            assert p == pytest.approx(ref.pvalue, rel=1e-6)
        else:  # R's one-sided asymptotic p-value
            assert p == pytest.approx(np.exp(-2 * n * d**2), rel=1e-12)


def test_ks_test_uses_asymptotics_with_ties():
    values = np.repeat(np.linspace(0.05, 0.95, 10), 3)
    d, p = ks_test(values, stats.uniform.cdf, "two-sided")
    assert p == pytest.approx(stats.kstwobign.sf(np.sqrt(30) * d))


def test_simulation_p_value():
    simulated = np.arange(1.0, 101.0)
    assert simulation_p_value(90.5, simulated, "greater") == 0.1
    assert simulation_p_value(90.5, simulated, "less") == 0.9
    assert simulation_p_value(90.5, simulated, "two.sided") == pytest.approx(0.2)
    assert simulation_p_value(50.5, simulated, "two-sided") == 1.0
    with pytest.raises(ValueError, match="alternative"):
        simulation_p_value(1.0, simulated, "both")


def test_check_residuals_detects_a_wrong_family():
    rng, x, X = _x(3, 300)
    y = rng.negative_binomial(1.0, 1.0 / (1.0 + np.exp(0.5 + 0.4 * x)))
    wrong = sm.GLM(y, X, family=sm.families.Poisson()).fit()
    res = check_residuals(wrong, random_state=0)
    assert isinstance(res, ResidualsResult)
    assert not res.passed
    assert res.message == "Warning: Non-uniformity of simulated residuals detected (p < .001)."
    right = sm.GLM(y, X, family=sm.families.NegativeBinomial(alpha=1.0)).fit()
    ok = check_residuals(right, random_state=0)
    assert ok.passed
    assert ok.message.startswith("OK: Simulated residuals appear as uniformly distributed (p = ")


@pytest.mark.parametrize("family", ["poisson", "negbin", "bernoulli", "gamma", "inverse_gaussian"])
def test_check_residuals_passes_for_the_right_family(family):
    assert check_residuals(glm(family), random_state=0).passed


def test_check_residuals_options(poisson):
    sim = simulate_residuals(poisson, random_state=0)
    from_sim = check_residuals(sim)
    from_model = check_residuals(poisson, random_state=0)
    assert from_sim.p_value == from_model.p_value
    assert from_sim.simulation is sim
    np.testing.assert_array_equal(from_sim.residuals, sim.scaled_residuals)
    assert check_residuals(sim, alternative="two.sided").p_value == from_sim.p_value
    greater = check_residuals(sim, alternative="greater")
    assert greater.alternative == "greater"
    assert greater.p_value != from_sim.p_value
    shifted = check_residuals(sim, distribution=lambda v: np.clip(v - 0.2, 0, 1))
    assert not shifted.passed
    assert check_residuals(sim, alpha=1.0).passed is False
    with pytest.raises(ValueError, match="alternative"):
        check_residuals(sim, alternative="sideways")
    row = from_sim.to_dataframe().iloc[0]
    assert row["Check"] == "Uniformity of residuals"
    assert row["Statistic"] == from_sim.statistic


def test_check_model_uses_simulated_residuals_for_non_gaussian_models(logit, poisson):
    assert "residuals" in check_model(logit)
    report = check_model(glm("gamma"))
    assert list(report) == ["predictions", "outliers", "residuals"]
    assert "collinearity" in report.skipped  # a single predictor
    assert check_model(poisson)["residuals"].p_value == check_model(poisson)["residuals"].p_value
    other = check_model(poisson, random_state=5)["residuals"]
    assert other.p_value != check_model(poisson)["residuals"].p_value


def test_predictive_check_for_more_families():
    for family in ("negbin", "binomial_trials", "gamma", "inverse_gaussian"):
        assert check_predictions(glm(family), random_state=0).simulated.shape[0] == 50


def test_plots(poisson):
    res = check_residuals(poisson, random_state=0)
    ax = res.plot().axes[0]
    assert ax.get_title(loc="left").startswith("Uniformity of Residuals")
    assert ax.get_xlabel() == "Expected uniform quantiles"
    fig = simulate_residuals(poisson, random_state=0).plot()
    assert fig.axes[0].get_title(loc="left").startswith("Uniformity of Residuals")


def test_protocol_adapters_can_supply_their_own_simulations(ols):
    class Wrapped:
        def __init__(self, inner):
            self.inner = as_adapter(inner)

        def model_type(self):
            return self.inner.info

        def get_response(self):
            return self.inner.y

        def get_fitted(self):
            return self.inner.mu

        def get_residuals(self):
            return self.inner.resid

        def get_model_matrix(self):
            return self.inner.X

        def simulate(self, n_sim, rng):
            return np.tile(self.inner.mu, (n_sim, 1))

    sim = simulate_residuals(Wrapped(ols), iterations=10, random_state=0)
    np.testing.assert_array_equal(sim.scaled_residuals, np.asarray(ols.resid > 0, dtype=float))
