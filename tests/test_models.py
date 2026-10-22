"""statsmodels discrete models, scikit-learn GLMs and the family-generic computations.

R's `glm()` and `MASS::glm.nb()` have no separate "discrete" classes, so these
models are validated against the equivalent statsmodels GLMs, which the rest
of the suite validates against R.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import special, stats
from sklearn.linear_model import GammaRegressor, PoissonRegressor

from checkmodel import (
    UnsupportedModelError,
    as_adapter,
    check_model,
    check_overdispersion,
    model_performance,
    r2,
)
from checkmodel._introspection import ProtocolAdapter
from tests import models

L = sm.families.links
BIO = "art ~ fem + mar + kid5 + phd + ment"
QUANTITIES = [
    "y",
    "mu",
    "hat",
    "pearson_residuals",
    "deviance_residuals",
    "std_residuals",
    "cooks_distance",
    "deviance",
    "null_deviance",
    "vcov_unscaled",
    "scale",
]


def assert_same(a, b, quantities=QUANTITIES, rtol=1e-6):
    for name in quantities:
        np.testing.assert_allclose(
            np.asarray(getattr(a, name), dtype=float),
            np.asarray(getattr(b, name), dtype=float),
            rtol=rtol,
            atol=1e-8,
            err_msg=name,
        )


def _bio():
    return models.biochemists().assign(e=np.linspace(0.5, 2.0, 915))


def _pair(kind: str):
    """A discrete model and the GLM it is equivalent to, both tightly converged."""
    bio, mt = _bio(), models.mtcars()
    if kind == "logit":
        return (
            smf.logit("vs ~ mpg + wt", mt).fit(disp=0, method="newton", tol=1e-12, maxiter=200),
            smf.glm("vs ~ mpg + wt", mt, family=sm.families.Binomial()).fit(tol=1e-14),
        )
    if kind == "probit":
        return (
            smf.probit("vs ~ mpg + wt", mt).fit(disp=0, method="newton", tol=1e-12, maxiter=200),
            smf.glm("vs ~ mpg + wt", mt, family=sm.families.Binomial(L.Probit())).fit(tol=1e-14),
        )
    kwargs = {"exposure": bio.e} if kind.endswith("exposure") else {}
    if kind.startswith("poisson"):
        return (
            smf.poisson(BIO, bio, **kwargs).fit(disp=0, method="newton", tol=1e-12, maxiter=200),
            smf.glm(BIO, bio, family=sm.families.Poisson(), **kwargs).fit(tol=1e-14),
        )
    nb = smf.negativebinomial(BIO, bio, **kwargs).fit(
        disp=0, method="newton", tol=1e-12, maxiter=200
    )
    family = sm.families.NegativeBinomial(alpha=nb.params["alpha"])
    return nb, smf.glm(BIO, bio, family=family, **kwargs).fit(tol=1e-14)


@pytest.mark.parametrize(
    "kind", ["logit", "probit", "poisson", "poisson_exposure", "negbin", "negbin_exposure"]
)
def test_discrete_models_match_the_equivalent_glm(kind):
    discrete, glm = _pair(kind)
    a, b = as_adapter(discrete), as_adapter(glm)
    assert a.info.family == b.info.family and a.info.link == b.info.link
    assert a.info.model_class == type(discrete.model).__name__
    rtol = 1e-4 if kind.startswith("negbin") else 1e-6
    assert_same(a, b, rtol=rtol)
    assert list(a.index()) == list(b.index())
    assert [t.name for t in a.terms()] == [t.name for t in b.terms()]


@pytest.mark.parametrize("kind", ["logit", "poisson"])
def test_metrics_match_the_equivalent_glm(kind):
    discrete, glm = _pair(kind)
    np.testing.assert_allclose(
        model_performance(discrete).to_numpy(), model_performance(glm).to_numpy(), rtol=1e-6
    )
    assert dict(r2(discrete)) == pytest.approx(dict(r2(glm)), rel=1e-6)


def test_negative_binomial_is_treated_like_glm_nb():
    nb, glm = _pair("negbin")
    a = as_adapter(nb)
    assert a.info.family == "negbin"
    assert a.negbin_theta == pytest.approx(1 / nb.params["alpha"])
    assert a.loglik == pytest.approx(nb.llf)
    assert a.loglik_df == 7  # six coefficients and theta, as logLik(glm.nb)
    perf = model_performance(nb)
    assert perf.loc[0, "AIC"] == pytest.approx(-2 * nb.llf + 14)
    assert dict(r2(nb)) == pytest.approx(dict(r2(glm)), rel=1e-5)
    # performance_score() takes exp(Pearson chi-squared / df) as the size.
    size = np.exp(np.sum(a.pearson_residuals**2) / a.df_resid)
    score = np.mean(stats.nbinom.logpmf(a.y, size, size / (size + a.mu)))
    assert perf.loc[0, "Score_log"] == pytest.approx(score)


def test_geometric_negative_binomial():
    geo = smf.negativebinomial(BIO, _bio(), loglike_method="geometric").fit(disp=0, maxiter=500)
    a = as_adapter(geo)
    assert a.negbin_theta == 1.0
    assert a.loglik_df == 6
    glm = smf.glm(BIO, _bio(), family=sm.families.NegativeBinomial(alpha=1.0)).fit()
    assert_same(a, as_adapter(glm), ["mu", "hat", "pearson_residuals", "deviance"], rtol=1e-4)


def test_unsupported_discrete_models(mtcars):
    nb1 = smf.negativebinomial(BIO, _bio(), loglike_method="nb1").fit(disp=0, maxiter=500)
    with pytest.raises(UnsupportedModelError, match="nb1"):
        as_adapter(nb1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # the fit diverges; only the rejection matters here
        mnlogit = sm.MNLogit(mtcars["cyl"], sm.add_constant(mtcars[["mpg"]])).fit(disp=0)
    with pytest.raises(UnsupportedModelError, match="Logit, Probit, Poisson and NegativeBinomial"):
        as_adapter(mnlogit)
    bio = _bio()
    l1 = sm.Poisson(bio["art"], sm.add_constant(bio[["ment"]])).fit_regularized(alpha=5, disp=0)
    with pytest.raises(UnsupportedModelError, match="Regularized"):
        as_adapter(l1)


def test_separated_logistic_regression_keeps_working():
    # Fitted probabilities of exactly 0 or 1 must not turn the IRLS weights into NaN.
    from sklearn.linear_model import LogisticRegression

    x = np.linspace(-3, 3, 100).reshape(-1, 1)
    y = (x[:, 0] > 0).astype(int)
    model = LogisticRegression(C=1e12, max_iter=10_000).fit(x, y)
    a = as_adapter(model, x, y)
    assert np.any((a.mu == 0) | (a.mu == 1))
    assert np.isfinite(a.working_weights).all() and np.isfinite(a.hat).all()
    assert a.rank == 2
    model_performance(model, x, y)


def test_check_model_on_discrete_models():
    expected = {
        "logit": ["predictions", "collinearity", "outliers", "binned_residuals", "residuals"],
        "poisson": ["predictions", "collinearity", "outliers", "overdispersion", "residuals"],
        "negbin": ["predictions", "collinearity", "outliers", "overdispersion", "residuals"],
    }
    for kind, checks in expected.items():
        report = check_model(_pair(kind)[0])
        assert list(report) == checks
        report.plot()
    assert check_overdispersion(_pair("poisson")[0]).method == "Pearson chi-squared"
    nb = check_overdispersion(_pair("negbin")[0], random_state=0)
    assert nb.simulation is not None and nb.passed


def test_null_model_honours_the_offset():
    discrete, glm = _pair("poisson_exposure")
    assert as_adapter(discrete).null_deviance == pytest.approx(glm.null_deviance, rel=1e-8)
    # Without an intercept, glm.fit() takes the null model's mean from the offset alone.
    bio = _bio()
    no_intercept = as_adapter(smf.poisson("art ~ 0 + phd", bio, exposure=bio.e).fit(disp=0))
    y, mu = bio["art"].to_numpy(float), bio["e"].to_numpy(float)
    expected = np.sum(2 * (special.xlogy(y, y / mu) - (y - mu)))
    assert no_intercept.null_deviance == pytest.approx(expected)


# --------------------------------------------------------------- scikit-learn GLMs
def test_sklearn_poisson_regressor_matches_statsmodels():
    bio = _bio()
    X, y = bio[["kid5", "phd", "ment"]], bio["art"]
    sk = PoissonRegressor(alpha=0, tol=1e-12, max_iter=10_000).fit(X, y)
    glm = sm.GLM(y, sm.add_constant(X), family=sm.families.Poisson()).fit(tol=1e-14)
    a = as_adapter(sk, X, y)
    assert a.info.family == "poisson" and not a.info.is_penalized
    assert_same(a, as_adapter(glm), QUANTITIES[:-1], rtol=1e-4)
    perf = model_performance(sk, X, y)
    assert perf.loc[0, "AIC"] == pytest.approx(model_performance(glm).loc[0, "AIC"], rel=1e-7)
    penalized = PoissonRegressor().fit(X, y)
    assert as_adapter(penalized, X, y).info.is_penalized
    assert "AIC" not in model_performance(penalized, X, y)


def test_sklearn_gamma_regressor_matches_statsmodels():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 2))
    y = rng.gamma(3.0, np.exp(1 + X @ [0.3, -0.2]) / 3.0)
    sk = GammaRegressor(alpha=0, tol=1e-12, max_iter=10_000).fit(X, y)
    glm = sm.GLM(y, sm.add_constant(X), family=sm.families.Gamma(L.Log())).fit(tol=1e-14)
    a, b = as_adapter(sk, X, y), as_adapter(glm)
    assert_same(a, b, rtol=1e-5)
    assert a.loglik == pytest.approx(b.loglik, rel=1e-7)
    assert list(check_model(sk, X, y)) == ["predictions", "collinearity", "outliers", "residuals"]


# --------------------------------------------------- family-generic computations
class _Wrapped:
    """Expose only the five protocol methods, so every quantity is derived generically."""

    def __init__(self, result):
        self.inner = as_adapter(result)

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


def _glm(kind: str):
    rng = np.random.default_rng(0)
    x = rng.normal(size=300)
    X = sm.add_constant(x)
    if kind == "cloglog":
        y = rng.binomial(1, 1 - np.exp(-np.exp(0.2 + 0.5 * x)))
        return sm.GLM(y, X, family=sm.families.Binomial(L.CLogLog())).fit()
    if kind == "gamma_log":
        return sm.GLM(
            rng.gamma(3, np.exp(1 + 0.3 * x) / 3), X, family=sm.families.Gamma(L.Log())
        ).fit()
    if kind == "gamma_inverse":
        y = rng.gamma(3, 1 / (1 + 0.1 * x) / 3)
        with warnings.catch_warnings():
            # statsmodels warns that the canonical (inverse) link can leave the Gamma domain.
            warnings.simplefilter("ignore", sm.tools.sm_exceptions.DomainWarning)
            return sm.GLM(y, X, family=sm.families.Gamma()).fit()
    y = rng.wald(np.exp(0.5 + 0.3 * x), 5)
    return sm.GLM(y, X, family=sm.families.InverseGaussian(L.Log())).fit()


@pytest.mark.parametrize("kind", ["cloglog", "gamma_log", "gamma_inverse", "inverse_gaussian"])
def test_generic_formulas_match_statsmodels(kind):
    glm = _glm(kind)
    # vcov differs at the level of IRLS convergence: statsmodels keeps the weights
    # of the previous iteration.
    assert_same(ProtocolAdapter(_Wrapped(glm)), as_adapter(glm), rtol=1e-5)


@pytest.mark.parametrize("kind", ["cloglog", "gamma_log"])
def test_leverage_uses_fisher_weights_as_r(kind):
    # R's hatvalues() uses the IRLS weights; statsmodels' default the observed information.
    glm = _glm(kind)
    hat = as_adapter(glm).hat
    np.testing.assert_allclose(hat, glm.get_hat_matrix_diag(observed=False))
    assert not np.allclose(hat, glm.get_hat_matrix_diag(observed=True))


def test_gamma_and_inverse_gaussian_likelihoods_follow_r():
    for kind in ("gamma_log", "inverse_gaussian"):
        glm = _glm(kind)
        a = as_adapter(glm)
        y, mu, n = a.y, a.mu, a.n_obs
        disp = glm.deviance / n
        if kind == "gamma_log":
            expected = np.sum(stats.gamma.logpdf(y, 1 / disp, scale=mu * disp))
        else:
            expected = -0.5 * (n * (np.log(disp * 2 * np.pi) + 1) + 3 * np.sum(np.log(y)))
        assert a.loglik == pytest.approx(expected)
        assert a.loglik != pytest.approx(glm.llf)  # statsmodels uses the Pearson dispersion
        assert a.loglik_df == 3


def test_leverage_falls_back_to_qr(monkeypatch, logit):
    expected = ProtocolAdapter(_Wrapped(logit)).hat

    def failing_svd(*args, **kwargs):
        raise np.linalg.LinAlgError("SVD did not converge")

    monkeypatch.setattr(np.linalg, "svd", failing_svd)
    np.testing.assert_allclose(ProtocolAdapter(_Wrapped(logit)).hat, expected)
