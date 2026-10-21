"""Bayesian models: R², LOO, WAIC, convergence and predictive checks against R.

The posterior draws are rstanarm's (tests/data/bayes_*.csv.gz); the reference
values were computed from those same draws by R (performance, loo, posterior).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

pytest.importorskip("arviz")

from checkmodel import (
    BayesianModel,
    BayesR2Result,
    CheckNotApplicableError,
    UnsupportedModelError,
    _mcmc,
    as_adapter,
    bayesian,
    check_convergence,
    check_model,
    check_normality,
    check_predictions,
    compare_performance,
    looic,
    model_performance,
    r2,
    r2_bayes,
)
from tests import bayes_models as bm
from tests.conftest import assert_r_close

MODELS = ["bayes_lm", "bayes_logit", "bayes_mixed"]
WITH_LOGLIK = ["bayes_lm", "bayes_logit"]


def _ref(name):
    return bm.reference()["models"][name]


def test_synthetic_data_matches_r():
    from tests import synthetic

    sums = bm.reference()["synthetic_checksums"]["mixed_small"]
    i = np.arange(1, 61, dtype=float)
    g = (i - 1) % 10 + 1
    x1 = synthetic.hash_e(i + 13000)
    y = 1 + 0.5 * x1 + 1.5 * synthetic.hash_e(g + 13500) + synthetic.hash_e(i + 14000)
    assert x1.sum() == pytest.approx(sums["x1"], rel=1e-9)
    assert y.sum() == pytest.approx(sums["y"], rel=1e-9)


# ------------------------------------------------------------------ reading
def test_bayesian_reads_the_datatree():
    fit = bm.fit("bayes_lm")
    assert isinstance(fit, BayesianModel)
    assert fit.mu.shape == (2, 300, 32)
    assert fit.y_rep.shape == fit.log_lik.shape == (2, 300, 32)
    assert fit.sigma.shape == (2, 300)
    assert list(fit.parameters) == ["X.Intercept.", "wt", "hp", "sigma"]
    assert fit.n_draws == 600 and fit.chains == 2 and fit.n_obs == 32
    info = fit.info
    assert (info.family, info.link, info.backend, info.is_mixed) == (
        "gaussian",
        "identity",
        "bayesian",
        False,
    )
    assert bm.fit("bayes_mixed").is_mixed
    assert bm.fit("bayes_logit").info.is_bernoulli


def test_bayesian_requires_expected_values():
    with pytest.raises(UnsupportedModelError, match=r"pm\.Deterministic"):
        bayesian(bm.datatree("bayes_lm"), mu="nope")
    with pytest.raises(UnsupportedModelError, match="posterior"):
        bayesian(object())


def test_datatree_is_accepted_directly():
    """Without bayesian(), PyMC-style defaults apply: `mu`, `sigma`, Gaussian."""
    tree = bm.datatree("bayes_lm")
    assert r2_bayes(tree)["R2"] == pytest.approx(r2_bayes(bm.fit("bayes_lm"))["R2"])
    assert r2(tree)["R2"] == pytest.approx(_ref("bayes_lm")["r2"]["median"], rel=1e-6)


def test_residual_checks_point_to_bayesian_functions():
    with pytest.raises(UnsupportedModelError, match="Bayesian model"):
        as_adapter(bm.datatree("bayes_lm"))
    with pytest.raises(UnsupportedModelError, match="Bayesian model"):
        check_normality(bm.datatree("bayes_lm"))


# -------------------------------------------------------------------- R²
@pytest.mark.parametrize("name", MODELS)
def test_r2_bayes_matches_r(name):
    ref = _ref(name)
    result = r2_bayes(bm.fit(name))
    assert isinstance(result, BayesR2Result)
    assert_r_close(result["R2"], ref["r2"]["median"])
    assert_r_close(result.se["R2"], ref["r2"]["mad"])
    assert_r_close(result.ci["R2"], ref["r2"]["hdi"])
    assert_r_close(result["R2"], ref["r2_performance"]["R2"])
    assert_r_close(result.ci["R2"], ref["r2_performance"]["CI"])
    assert result.draws["R2"].shape == (600,)


def test_r2_bayes_marginal_for_multilevel_models():
    ref = _ref("bayes_mixed")
    result = r2_bayes(bm.fit("bayes_mixed"))
    assert list(result) == ["R2", "R2_marginal"]
    assert_r_close(result["R2_marginal"], ref["r2_marginal"]["median"])
    assert_r_close(result.ci["R2_marginal"], ref["r2_marginal"]["hdi"])
    assert repr(result).splitlines()[2].startswith("  Conditional R2: 0.598 (95% CI [")


def test_r2_bayes_options():
    fit = bm.fit("bayes_lm")
    mean_based = r2_bayes(fit, robust=False, ci=0.89)
    draws = mean_based.draws["R2"]
    assert mean_based["R2"] == pytest.approx(draws.mean())
    assert mean_based.se["R2"] == pytest.approx(draws.std(ddof=1))
    assert mean_based.ci_level == 0.89
    assert repr(mean_based).splitlines()[2].startswith("  R2: ")
    assert len(mean_based) == 1


def test_r2_bayes_family_restrictions():
    fit = bm.fit("bayes_lm")
    poisson = BayesianModel(y=fit.y, mu=np.abs(fit.mu), family="poisson")
    with pytest.raises(CheckNotApplicableError, match="Gaussian and binomial"):
        r2_bayes(poisson)
    no_sigma = BayesianModel(y=fit.y, mu=fit.mu, family="gaussian")
    with pytest.raises(CheckNotApplicableError, match="sigma"):
        r2_bayes(no_sigma)
    with pytest.raises(UnsupportedModelError):
        r2_bayes(object())


# ------------------------------------------------------------- LOO and WAIC
@pytest.mark.parametrize("name", WITH_LOGLIK)
def test_loo_matches_r(name):
    ref = _ref(name)
    loo = looic(bm.fit(name))
    for key, value in ref["loo"].items():
        assert_r_close(getattr(loo, key), value)
    assert_r_close(loo.pareto_k, ref["pareto_k"])
    assert loo.n_bad_k == int(np.sum(np.asarray(ref["pareto_k"]) > 0.7))


@pytest.mark.parametrize("name", WITH_LOGLIK)
def test_loo_with_relative_efficiency_matches_r(name):
    ref = _ref(name)
    fit = bm.fit(name)
    assert_r_close(_mcmc.relative_eff(fit.log_lik), ref["r_eff"])
    loo = looic(fit, use_relative_eff=True)
    for key, value in ref["loo_reff"].items():
        assert_r_close(getattr(loo, key), value)
    assert_r_close(loo.pareto_k, ref["pareto_k_reff"])


@pytest.mark.parametrize("name", WITH_LOGLIK)
def test_waic_matches_r(name):
    ref = _ref(name)
    value, se = _mcmc.waic(bm.fit(name).log_lik)
    assert_r_close(value, ref["waic"]["waic"])
    assert_r_close(se, ref["waic"]["waic_se"])


def test_loo_needs_log_likelihood():
    with pytest.raises(CheckNotApplicableError, match="log-likelihood"):
        looic(bm.fit("bayes_mixed"))


# ----------------------------------------------------------- diagnostics
@pytest.mark.parametrize("name", MODELS)
def test_rhat_and_ess_match_posterior_package(name):
    table = check_convergence(bm.fit(name)).to_dataframe().set_index("Parameter")
    for parameter, ref in _ref(name)["diagnostics"].items():
        assert_r_close(table.loc[parameter, "Rhat"], ref["rhat"])
        assert_r_close(table.loc[parameter, "ESS_bulk"], ref["ess_bulk"])
        assert_r_close(table.loc[parameter, "ESS_tail"], ref["ess_tail"])


def test_convergence_verdicts():
    ok = check_convergence(bm.fit("bayes_lm"))
    assert ok.passed and ok.message.startswith("OK: Chains converged (max R-hat = 1.008")
    low_ess = check_convergence(bm.fit("bayes_logit"))
    assert not low_ess.passed and "ESS < 200 for X.Intercept." in low_ess.message
    bad = check_convergence(bm.fit("bayes_mixed"))
    assert not bad.passed and "R-hat >= 1.01 for X.Intercept., sigma" in bad.message
    assert bad.statistic == pytest.approx(
        _ref("bayes_mixed")["diagnostics"]["X.Intercept."]["rhat"]
    )


def test_convergence_needs_parameters():
    fit = bm.fit("bayes_lm")
    bare = BayesianModel(y=fit.y, mu=fit.mu, family="gaussian")
    with pytest.raises(CheckNotApplicableError, match="no scalar parameters"):
        check_convergence(bare)


def test_mcmc_edge_cases():
    constant = np.ones((2, 50))
    assert math.isnan(_mcmc.rhat(constant))
    assert math.isnan(_mcmc.ess_bulk(constant))
    assert math.isnan(_mcmc.ess_tail(constant))
    assert math.isnan(_mcmc.ess_bulk(np.arange(2.0)))  # too few draws
    assert all(math.isnan(v) for v in _mcmc.hdi(np.arange(3.0)))
    contiguous = np.array([0.0, 1.0, 2.0, 3.0, 10.0, 11.0, 12.0, 13.0])
    assert _mcmc.hdi(contiguous, ci=0.4) == (1.0, 11.0)  # middle of adjacent windows
    split = np.array([0.0, 0.0, 0.0, 5.0, 10.0, 10.0, 10.0])
    assert _mcmc.hdi(split, ci=0.3) == (5.0, 10.0)  # rightmost of separate windows
    assert _mcmc._gpdfit(np.zeros(10))[0] != _mcmc._gpdfit(np.zeros(10))[0]  # nan
    assert math.isnan(_mcmc._qgpd(np.array([0.5]), float("nan"), 0.1)[0])
    assert _mcmc._qgpd(np.array([0.5]), 1.0, 0.0)[0] == pytest.approx(math.log(2))
    lw, k = _mcmc._psis_one(np.zeros(20), tail_len=4)
    assert math.isinf(k) and np.allclose(lw, 0.0)


# ------------------------------------------------------- predictive check
@pytest.mark.parametrize("name", WITH_LOGLIK)
def test_posterior_predictive_check(name):
    fit = bm.fit(name)
    result = check_predictions(fit, random_state=3)
    assert result.passed
    assert result.simulated.shape == (50, 32)
    flat = BayesianModel.flat(fit.y_rep)
    expected = np.mean(flat.mean(axis=1) >= fit.y.mean())
    first = result.table.iloc[0]
    assert first["p"] == pytest.approx(expected)
    assert result.discrete is (name == "bayes_logit")
    assert result.method == "Posterior predictive p-values"


def test_predictive_check_needs_draws():
    with pytest.raises(CheckNotApplicableError, match="posterior predictive"):
        check_predictions(bm.fit("bayes_mixed"))


# ---------------------------------------------------- model_performance
@pytest.mark.parametrize("name", MODELS)
def test_model_performance_matches_r(name):
    ref = _ref(name)
    perf = model_performance(bm.fit(name))
    for column in ("ELPD", "ELPD_SE", "LOOIC", "LOOIC_SE", "WAIC", "R2", "R2_marginal", "Sigma"):
        if column in perf:
            assert_r_close(perf.loc[0, column], ref["performance"][column])
    # RMSE is computed from the posterior median of the expected values; R uses
    # the fitted values at the median coefficients (see docs/comparison-with-r.md).
    assert_r_close(perf.loc[0, "RMSE"], ref["rmse_median_mu"])


def test_model_performance_columns():
    assert list(model_performance(bm.fit("bayes_lm"))) == [
        "ELPD",
        "ELPD_SE",
        "LOOIC",
        "LOOIC_SE",
        "WAIC",
        "R2",
        "RMSE",
        "Sigma",
    ]
    assert list(model_performance(bm.fit("bayes_mixed"))) == [
        "R2",
        "R2_marginal",
        "RMSE",
        "Sigma",
    ]
    assert model_performance(bm.fit("bayes_logit")).loc[0, "Sigma"] == 1.0
    common = model_performance(bm.fit("bayes_lm"), metrics="common")
    assert "Sigma" not in common and "WAIC" in common
    assert list(model_performance(bm.fit("bayes_lm"), metrics=["waic"])) == ["WAIC"]


def test_compare_bayesian_models():
    table = compare_performance(lm=bm.fit("bayes_lm"), logit=bm.fit("bayes_logit"), verbose=False)
    assert table["Model"].tolist() == ["Bayesian", "Bayesian"]
    assert "WAIC_wt" in table and table["WAIC_wt"].sum() == pytest.approx(1.0)
    columns = list(table.columns)
    assert columns.index("WAIC_wt") == columns.index("WAIC") + 1
    ranked = compare_performance(
        lm=bm.fit("bayes_lm"), logit=bm.fit("bayes_logit"), rank=True, verbose=False
    )
    assert not {"LOOIC", "WAIC", "ELPD", "ELPD_SE", "LOOIC_SE"} & set(ranked.columns)


# --------------------------------------------------------------- report
def test_check_model_for_bayesian_models():
    report = check_model(bm.fit("bayes_lm"))
    assert list(report) == ["predictions", "convergence"]
    assert report.passed
    assert repr(report).startswith("# Model checks: Bayesian (gaussian, identity link), n = 32")
    fig = report.plot()
    titles = [ax.get_title(loc="left").strip() for ax in fig.axes if ax.get_visible()]
    assert titles == ["Posterior Predictive Check", "Convergence"]
    explicit = check_model(bm.fit("bayes_lm"), checks=["normality", "convergence"])
    assert "Bayesian model" in explicit.skipped["normality"]


def test_plots_without_predictions_or_with_discrete_data():
    mixed = check_model(bm.fit("bayes_mixed"))
    assert "predictions" in mixed.skipped
    assert [ax.get_title(loc="left").strip() for ax in mixed.plot().axes] == ["Convergence"]
    with pytest.raises(ValueError, match="has a plot"):
        check_model(bm.fit("bayes_mixed"), checks=["predictions"]).plot()
    logit = check_predictions(bm.fit("bayes_logit"), random_state=0)
    assert logit.plot().axes[0].get_xlabel() == "Value of the response"
