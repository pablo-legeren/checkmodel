"""End-to-end runs with real PyMC and bambi fits (slow: they sample)."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

pytestmark = pytest.mark.slow

from checkmodel import (  # noqa: E402
    BayesianModel,
    bayesian,
    check_model,
    model_performance,
    r2_bayes,
)
from tests import models  # noqa: E402

SAMPLE = {"draws": 200, "tune": 200, "chains": 2, "cores": 1, "random_seed": 7}


@pytest.fixture(scope="module")
def quiet():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield


def test_bambi_multilevel_model(quiet):
    bmb = pytest.importorskip("bambi")
    data = models.sleepstudy().iloc[:60]
    model = bmb.Model("Reaction ~ Days + (1|Subject)", data)
    idata = model.fit(progressbar=False, **SAMPLE)
    fit = bayesian(idata, model)
    assert isinstance(fit, BayesianModel)
    assert fit.model_class == "bambi" and fit.response == "Reaction"
    assert fit.family == "gaussian" and fit.is_mixed
    assert fit.y_rep.shape == fit.log_lik.shape == fit.mu.shape == (2, 200, 60)
    assert "Days" in fit.parameters and not any(k.startswith("mu") for k in fit.parameters)
    r2 = r2_bayes(fit)
    assert 0 < r2["R2_marginal"] < r2["R2"] < 1
    perf = model_performance(fit)
    assert {"ELPD", "LOOIC", "WAIC", "R2", "R2_marginal", "RMSE", "Sigma"} <= set(perf.columns)
    report = check_model(fit)
    assert list(report) == ["predictions", "convergence"]
    report.plot()


def test_bambi_logistic_model(quiet):
    bmb = pytest.importorskip("bambi")
    data = models.mtcars()
    model = bmb.Model("vs ~ mpg + wt", data, family="bernoulli")
    idata = model.fit(progressbar=False, **SAMPLE)
    fit = bayesian(idata, model)
    assert fit.family == "binomial" and not fit.is_mixed
    assert set(np.unique(fit.y)) == {0.0, 1.0}
    assert 0 < r2_bayes(fit)["R2"] < 1
    assert check_model(fit)["predictions"].discrete


def test_pymc_model_with_deterministic_mean(quiet):
    pm = pytest.importorskip("pymc")
    data = models.mtcars()
    with pm.Model() as model:
        intercept = pm.Normal("intercept", 20, 10)
        slope = pm.Normal("slope", 0, 10)
        sigma = pm.HalfNormal("sigma", 5)
        mu = pm.Deterministic("mu", intercept + slope * data["wt"].to_numpy())
        pm.Normal("mpg", mu, sigma, observed=data["mpg"].to_numpy())
        idata = pm.sample(progressbar=False, **SAMPLE)
    fit = bayesian(idata, model)
    assert fit.model_class == "PyMC" and fit.family == "gaussian"
    assert fit.y_rep is not None and fit.log_lik is not None
    assert set(fit.parameters) == {"intercept", "slope", "sigma"}
    assert 0.5 < r2_bayes(fit)["R2"] < 0.9
    assert check_model(fit)["convergence"].table.shape[0] == 3
