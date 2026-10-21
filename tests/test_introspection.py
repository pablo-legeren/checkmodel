"""Adapters: model detection and the diagnostic quantities derived from each backend."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, LogisticRegression, Ridge

from checkmodel import (
    BaseAdapter,
    ModelAdapter,
    ModelType,
    UnsupportedModelError,
    as_adapter,
    check_model,
    check_normality,
    model_performance,
    register_adapter,
)
from checkmodel._introspection import _REGISTRY, ProtocolAdapter
from tests import models
from tests.conftest import assert_r_close, per_obs

REFERENCE_MODELS = list(models.MODELS)


@pytest.mark.parametrize("name", REFERENCE_MODELS)
def test_influence_measures_match_r(name, fit, reference):
    """Fitted values, leverage, standardized residuals and Cook's distance vs R."""
    ref = reference["models"][name]
    a = as_adapter(fit(name))
    assert a.n_obs == ref["n"]
    for values, key in [
        (a.mu, "fitted"),
        (a.hat, "hat"),
        (a.std_residuals, "rstandard"),  # R >= 4.4: rstandard(glm) is Pearson-based
        (a.cooks_distance, "cooks"),
    ]:
        assert_r_close(*per_obs(values, ref[key]))


@pytest.mark.parametrize("name", REFERENCE_MODELS)
def test_loglik_matches_r(name, fit, reference):
    ref = reference["models"][name]
    a = as_adapter(fit(name))
    assert_r_close(a.loglik, ref["loglik"])
    assert a.loglik_df == ref["loglik_df"]


def test_synthetic_data_matches_r(reference):
    from tests import synthetic

    for name, sums in reference["synthetic_checksums"].items():
        df = synthetic.DATASETS[name]()
        for column, total in sums.items():
            assert df[column].sum() == pytest.approx(total, rel=1e-9, abs=1e-7)


def test_model_type_flags(fit, sk_linear, sk_logistic):
    ols = as_adapter(fit("lm_mtcars")).info
    assert (ols.family, ols.link, ols.backend, ols.model_class) == (
        "gaussian",
        "identity",
        "statsmodels",
        "OLS",
    )
    assert ols.is_linear and ols.has_dispersion and not ols.is_binomial

    logit = as_adapter(fit("glm_logit")).info
    assert logit.is_binomial and logit.is_bernoulli and logit.link == "logit"

    pois = as_adapter(fit("glm_pois")).info
    assert pois.is_count and not pois.has_dispersion

    assert as_adapter(fit("lm_mtcars_wls")).info.model_class == "WLS"
    assert as_adapter(*sk_linear).info.backend == "sklearn"
    assert as_adapter(*sk_logistic).info.is_penalized is False


def test_term_grouping_follows_formula(fit):
    terms = as_adapter(fit("lm_mtcars_cyl")).terms()
    assert [t.name for t in terms] == ["wt", "hp", "C(cyl)"]
    cyl = terms[-1]
    assert cyl.is_factor and len(cyl.columns) == 2


def test_observation_labels_come_from_data(fit):
    index = as_adapter(fit("lm_mtcars")).index()
    assert "Maserati Bora" in index


def test_statsmodels_without_formula(mtcars):
    X = sm.add_constant(mtcars[["wt", "hp"]])
    a = as_adapter(sm.OLS(mtcars["mpg"], X).fit())
    assert a.intercept_index() == 0
    assert [t.name for t in a.terms()] == ["wt", "hp"]
    assert a.n_numeric_variables() == 3


def test_binomial_glm_with_trials_is_not_bernoulli():
    successes = np.array([3, 5, 8, 9, 2, 6])
    trials = np.full(6, 10)
    x = np.arange(6.0)
    endog = np.column_stack([successes, trials - successes])
    result = sm.GLM(endog, sm.add_constant(x), family=sm.families.Binomial()).fit()
    info = as_adapter(result).info
    assert info.is_binomial and not info.is_bernoulli


def test_sklearn_requires_data(sk_linear):
    model, _, _ = sk_linear
    with pytest.raises(TypeError, match=r"X=X, y=y"):
        as_adapter(model)


def test_sklearn_rejects_mismatched_data(sk_linear, mtcars):
    model, X, y = sk_linear
    with pytest.raises(ValueError, match="rows"):
        as_adapter(model, X, y.iloc[:10])
    with pytest.raises(ValueError, match="features"):
        as_adapter(model, mtcars[["wt"]], y)


def test_sklearn_unfitted_and_unsupported(mtcars):
    with pytest.raises(UnsupportedModelError, match="not fitted"):
        as_adapter(LinearRegression(), mtcars[["wt"]], mtcars["mpg"])
    forest = RandomForestRegressor(n_estimators=2).fit(mtcars[["wt"]], mtcars["mpg"])
    with pytest.raises(UnsupportedModelError, match="not supported"):
        as_adapter(forest, mtcars[["wt"]], mtcars["mpg"])


def test_sklearn_multiclass_rejected():
    from sklearn.datasets import load_iris

    X, y = load_iris(return_X_y=True)
    model = LogisticRegression(max_iter=1000).fit(X, y)
    with pytest.raises(UnsupportedModelError, match="multiclass"):
        as_adapter(model, X, y)


def test_sklearn_numpy_input_gets_default_names(mtcars):
    X, y = mtcars[["wt", "hp"]].to_numpy(), mtcars["mpg"].to_numpy()
    a = as_adapter(LinearRegression().fit(X, y), X, y)
    assert a.column_names() == ["Intercept", "x0", "x1"]
    assert isinstance(a.index(), pd.RangeIndex)


def test_sklearn_without_intercept(mtcars):
    X, y = mtcars[["wt", "hp"]], mtcars["mpg"]
    a = as_adapter(LinearRegression(fit_intercept=False).fit(X, y), X, y)
    assert a.intercept_index() is None
    assert a.X.shape[1] == 2


def test_penalized_sklearn_models_are_flagged(mtcars):
    X, y = mtcars[["wt", "hp"]], mtcars["mpg"]
    for est in (Ridge(alpha=1.0), Lasso(alpha=0.1)):
        a = as_adapter(est.fit(X, y), X, y)
        assert a.info.is_penalized
    assert as_adapter(LogisticRegression().fit(X, mtcars["vs"]), X, mtcars["vs"]).info.is_penalized


def test_sklearn_logistic_recodes_string_labels(mtcars):
    X = mtcars[["mpg", "wt"]]
    labels = np.where(mtcars["vs"] == 1, "straight", "v-shaped")
    model = LogisticRegression(C=np.inf, max_iter=10_000).fit(X, labels)
    a = as_adapter(model, X, labels)
    assert set(np.unique(a.y)) == {0.0, 1.0}
    assert a.y.sum() == (labels == model.classes_[1]).sum()


def test_statsmodels_ignores_and_warns_about_data(ols, mtcars):
    with pytest.warns(UserWarning, match="ignored"):
        as_adapter(ols, mtcars[["wt"]], mtcars["mpg"])


def test_unfitted_statsmodels_rejected(mtcars):
    with pytest.raises(UnsupportedModelError, match="fit"):
        as_adapter(smf.ols("mpg ~ wt", mtcars))


def test_unsupported_statsmodels_class(mtcars):
    result = smf.quantreg("mpg ~ wt", mtcars).fit(q=0.5)
    with pytest.raises(UnsupportedModelError, match="not supported"):
        as_adapter(result)


def test_unknown_object_rejected():
    with pytest.raises(UnsupportedModelError, match="Don't know"):
        as_adapter(object())


def test_adapter_passthrough(ols):
    a = as_adapter(ols)
    assert as_adapter(a) is a


class _Lm:
    """A minimal third-party model implementing the ModelAdapter protocol."""

    def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
        self.X = np.column_stack([np.ones(len(y)), X])
        self.y = y
        self.beta, *_ = np.linalg.lstsq(self.X, y, rcond=None)

    def model_type(self) -> ModelType:
        return ModelType("gaussian", "identity", "custom", "MyLm")

    def get_response(self) -> np.ndarray:
        return self.y

    def get_fitted(self) -> np.ndarray:
        return self.X @ self.beta

    def get_residuals(self) -> np.ndarray:
        return self.y - self.get_fitted()

    def get_model_matrix(self) -> np.ndarray:
        return self.X

    def column_names(self) -> list[str]:
        return ["(Intercept)", "wt", "hp"]


def test_protocol_objects_are_supported(mtcars, ols):
    custom = _Lm(mtcars[["wt", "hp"]].to_numpy(), mtcars["mpg"].to_numpy())
    assert isinstance(custom, ModelAdapter)
    a = as_adapter(custom)
    assert isinstance(a, ProtocolAdapter)
    assert a.column_names() == ["(Intercept)", "wt", "hp"]
    assert [t.name for t in a.terms()] == ["wt", "hp"]
    assert check_normality(custom).p_value == pytest.approx(check_normality(ols).p_value)
    report = check_model(custom)
    assert set(report) >= {"normality", "collinearity", "outliers"}


def test_register_adapter_takes_precedence(mtcars):
    class Wrapped:
        def __init__(self, inner):
            self.inner = inner

    class WrappedAdapter(BaseAdapter):
        def model_type(self):
            return self.model.inner.model_type()

        def get_response(self):
            return self.model.inner.get_response()

        def get_fitted(self):
            return self.model.inner.get_fitted()

        def get_model_matrix(self):
            return self.model.inner.get_model_matrix()

    inner = _Lm(mtcars[["wt", "hp"]].to_numpy(), mtcars["mpg"].to_numpy())
    register_adapter(Wrapped, lambda m, X=None, y=None: WrappedAdapter(m))
    try:
        a = as_adapter(Wrapped(inner))
        assert isinstance(a, WrappedAdapter)
        assert a.column_names() == ["x0", "x1", "x2"]
        assert a.intercept_index() == 0
        assert not model_performance(a).empty
    finally:
        _REGISTRY.pop(0)

    register_adapter(lambda m: m == "magic", lambda m, X=None, y=None: as_adapter(inner))
    try:
        assert as_adapter("magic").info.model_class == "MyLm"
    finally:
        _REGISTRY.pop(0)


def test_generic_glm_quantities_match_statsmodels(sk_logistic, logit):
    """The generic (numpy) derivations agree with statsmodels' own GLM values."""
    generic = as_adapter(*sk_logistic)
    sm_glm = as_adapter(logit)
    for attr in ("mu", "hat", "pearson_residuals", "deviance_residuals", "cooks_distance"):
        np.testing.assert_allclose(getattr(generic, attr), getattr(sm_glm, attr), atol=1e-6)
    for attr in ("loglik", "loglik_null", "deviance", "null_deviance", "scale"):
        assert getattr(generic, attr) == pytest.approx(getattr(sm_glm, attr), abs=1e-6)


def test_generic_poisson_quantities(poisson):
    """Generic Poisson formulas, exercised through a protocol wrapper."""
    a = as_adapter(poisson)

    class Pois:
        def model_type(self):
            return a.info

        def get_response(self):
            return a.y

        def get_fitted(self):
            return a.mu

        def get_residuals(self):
            return a.y - a.mu

        def get_model_matrix(self):
            return a.X

    g = as_adapter(Pois())
    np.testing.assert_allclose(g.hat, a.hat, atol=1e-8)
    np.testing.assert_allclose(g.deviance_residuals, a.deviance_residuals, atol=1e-8)
    assert g.loglik == pytest.approx(a.loglik)
    assert g.loglik_null == pytest.approx(a.loglik_null)
    assert g.null_deviance == pytest.approx(a.null_deviance)


def test_generic_adapter_rejects_unknown_family():
    class Gamma:
        def model_type(self):
            return ModelType("gamma", "inverse", "custom", "G")

        def get_response(self):
            return np.array([1.0, 2.0, 3.0])

        def get_fitted(self):
            return np.array([1.1, 1.9, 3.0])

        def get_residuals(self):
            return self.get_response() - self.get_fitted()

        def get_model_matrix(self):
            return np.column_stack([np.ones(3), [1.0, 2.0, 3.0]])

    a = as_adapter(Gamma())
    with pytest.raises(UnsupportedModelError):
        _ = a.hat
    with pytest.raises(UnsupportedModelError):
        _ = a.pearson_residuals
    with pytest.raises(UnsupportedModelError):
        _ = a.deviance_residuals
    with pytest.raises(UnsupportedModelError):
        _ = a.deviance
    assert a.loglik is None


def test_model_without_intercept_null_model(mtcars):
    X = mtcars[["wt"]].to_numpy()
    y = mtcars["mpg"].to_numpy()
    a = as_adapter(LinearRegression(fit_intercept=False).fit(X, y), X, y)
    assert not a.has_intercept
    assert a.loglik_null is not None
