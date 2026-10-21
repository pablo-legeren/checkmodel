"""Linear mixed models: influence, assumption checks, singularity and convergence vs lme4."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import statsmodels.formula.api as smf

from checkmodel import (
    CheckNotApplicableError,
    ConvergenceResult,
    RandomEffectsNormalityResult,
    SingularityResult,
    UnsupportedModelError,
    as_adapter,
    check_autocorrelation,
    check_collinearity,
    check_convergence,
    check_heteroscedasticity,
    check_model,
    check_normality,
    check_outliers,
    check_singularity,
)
from tests.conftest import assert_r_close
from tests.models import MIXED

# ML estimates from statsmodels and lme4 agree to ~1e-4 relative; everything
# derived from them is compared at that looser tolerance.
TOL = {"mixed_sleep_ml": 5e-4}


def _tol(name: str) -> float:
    return TOL.get(name, 1e-4)


def _varcorr(ref: dict) -> np.ndarray:
    (columns,) = ref["varcorr"].values()
    return np.column_stack([np.atleast_1d(v) for v in columns.values()])


@pytest.mark.parametrize("name", list(MIXED))
def test_variance_components_match_lme4(name, fit, reference):
    ref = reference["mixed"][name]
    a = as_adapter(fit(name))
    np.testing.assert_allclose(a.cov_re, _varcorr(ref), rtol=_tol(name), atol=1e-4)
    assert_r_close(a.sigma, ref["sigma"], _tol(name))
    assert_r_close(np.asarray(fit(name).fe_params), ref["fixef"], _tol(name))


@pytest.mark.parametrize("name", list(MIXED))
def test_influence_matches_lme4(name, fit, reference):
    """hatvalues(), rstudent() and cooks.distance() for lmer fits."""
    ref = reference["mixed"][name]
    a = as_adapter(fit(name))
    assert_r_close(a.mu, ref["fitted"], _tol(name))
    assert_r_close(a.hat, ref["hat"], _tol(name))
    assert_r_close(a.std_residuals, ref["rstudent"], _tol(name))
    assert_r_close(a.cooks_distance, ref["cooks"], 1e-3)


@pytest.mark.parametrize("name", list(MIXED))
def test_likelihood_matches_lme4(name, fit, reference):
    ref = reference["mixed"][name]
    a = as_adapter(fit(name))
    assert_r_close(a.loglik, ref["loglik"], _tol(name))
    assert a.loglik_df == ref["loglik_df"]
    assert a.df_resid == ref["df_residual"]


@pytest.mark.parametrize("name", list(MIXED))
def test_residual_normality_matches_r(name, fit, reference):
    ref = reference["mixed"][name]["normality"]
    assert_r_close(check_normality(fit(name)).p_value, ref["fixed"], _tol(name))


@pytest.mark.parametrize("name", ["mixed_sleep", "mixed_sleep_slope", "mixed_synth"])
def test_random_effect_normality_matches_r(name, fit, reference):
    expected = np.atleast_1d(reference["mixed"][name]["normality"]["random"])
    res = check_normality(fit(name), effects="random")
    assert isinstance(res, RandomEffectsNormalityResult)
    assert_r_close(res.table["p"], expected, _tol(name))
    assert res.p_value == pytest.approx(min(expected), rel=1e-3)


def test_random_effect_normality_messages(fit):
    res = check_normality(fit("mixed_sleep_slope"), effects="random")
    assert res.passed
    assert res.message.splitlines() == [
        "OK: Random effects 'Subject: (Intercept)' appear as normally distributed (p = 0.428).",
        "OK: Random effects 'Subject: Days' appear as normally distributed (p = 0.898).",
    ]
    assert list(res.effects.columns) == ["(Intercept)", "Days"]
    assert res.to_dataframe()["Group"].unique().tolist() == ["Subject"]


def test_random_effect_normality_needs_variance(fit, ols):
    with pytest.raises(CheckNotApplicableError, match="singular"):
        check_normality(fit("mixed_singular"), effects="random")
    with pytest.raises(CheckNotApplicableError, match="mixed model"):
        check_normality(ols, effects="random")
    with pytest.raises(ValueError, match="effects"):
        check_normality(ols, effects="both")  # type: ignore[call-overload]


@pytest.mark.parametrize("name", list(MIXED))
def test_heteroscedasticity_matches_r_with_sigma(name, fit, reference):
    """Against R's computation with sigma(m); see docs/comparison-with-r.md."""
    ref = reference["mixed"][name]["heteroscedasticity"]
    res = check_heteroscedasticity(fit(name))
    assert_r_close(res.statistic, ref["statistic"], _tol(name))
    assert_r_close(res.p_value, ref["p"], _tol(name))


def test_collinearity_matches_r(fit, reference):
    ref = reference["mixed"]["mixed_synth"]["vif"]
    table = check_collinearity(fit("mixed_synth")).table
    assert table["Term"].tolist() == ref["Term"]
    for column in ("VIF", "VIF_CI_low", "VIF_CI_high", "SE_factor", "Tolerance"):
        assert_r_close(table[column], ref[column])


@pytest.mark.parametrize("name", list(MIXED))
def test_outliers_match_r(name, fit, reference):
    ref = reference["mixed"][name]["outliers"]
    res = check_outliers(fit(name))
    assert_r_close(res.threshold, ref["threshold"])
    assert (res.distances > res.threshold).tolist() == ref["flags"]


@pytest.mark.parametrize("name", list(MIXED))
def test_singularity_matches_r(name, fit, reference):
    res = check_singularity(fit(name))
    assert isinstance(res, SingularityResult)
    assert res.passed is not reference["mixed"][name]["singular"]


def test_singularity_details(fit):
    res = check_singularity(fit("mixed_singular"))
    assert res.message.startswith("Warning: Singular fit")
    assert res.to_dataframe().columns.tolist() == ["Term", "Variance"]
    assert res.to_dataframe()["Term"].tolist() == ["g: (Intercept)"]
    ok = check_singularity(fit("mixed_sleep_slope"))
    assert ok.statistic == pytest.approx(
        np.linalg.det(as_adapter(fit("mixed_sleep_slope")).cov_re)
    )
    with pytest.raises(TypeError, match="no diagnostic plot"):
        ok.plot()


def test_singularity_needs_random_effects(ols):
    with pytest.raises(CheckNotApplicableError, match="mixed models"):
        check_singularity(ols)


@pytest.mark.parametrize("name", list(MIXED))
def test_convergence_agrees_with_r(name, fit, reference):
    res = check_convergence(fit(name))
    assert isinstance(res, ConvergenceResult)
    assert res.passed is reference["mixed"][name]["converged"]
    assert res.statistic is not None and res.statistic < 1e-3


def test_convergence_failure_is_detected():
    import warnings

    from tests import synthetic

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        stopped = smf.mixedlm("y ~ x1", synthetic.mixed_singular(), groups="g").fit()
    res = check_convergence(stopped)  # default BFGS stops short of the boundary
    assert not res.passed
    assert "method='powell'" in res.message


def test_convergence_other_models(ols, logit, mtcars):
    from sklearn.linear_model import LinearRegression, LogisticRegression

    assert check_convergence(ols).method == "closed form"
    assert check_convergence(logit).passed and check_convergence(logit).method == "IRLS"
    X, y = mtcars[["wt", "hp"]], mtcars["vs"]
    stopped = LogisticRegression(max_iter=1)
    with pytest.warns(Warning):
        stopped.fit(X, y)
    assert not check_convergence(stopped, X, y).passed
    assert check_convergence(stopped, X, y).message == "Warning: Model failed to converge."
    lin = LinearRegression().fit(X, mtcars["mpg"])
    assert check_convergence(lin, X, mtcars["mpg"]).passed


def test_convergence_needs_information():
    from tests.test_introspection import _Lm

    rng = np.random.default_rng(0)
    custom = _Lm(rng.normal(size=(20, 2)), rng.normal(size=20))
    with pytest.raises(CheckNotApplicableError, match="convergence"):
        check_convergence(custom)
    custom.converged = False
    assert not check_convergence(custom).passed


def test_autocorrelation_not_applicable(fit):
    with pytest.raises(CheckNotApplicableError, match="random effects"):
        check_autocorrelation(fit("mixed_sleep"))


def test_check_model_for_mixed_models(fit):
    report = check_model(fit("mixed_sleep_slope"))
    assert list(report) == [
        "normality",
        "random_effects",
        "heteroscedasticity",
        "outliers",
        "singularity",
        "convergence",
    ]
    assert "collinearity" in report.skipped
    text = repr(report)
    assert text.startswith("# Model checks: MixedLM (gaussian, identity link), n = 180")
    assert "\n  OK: Random effects 'Subject: Days'" in text
    singular = check_model(fit("mixed_singular"))
    assert "random_effects" in singular.skipped
    assert not singular["singularity"].passed


def test_random_effect_plots(fit):
    res = check_normality(fit("mixed_sleep_slope"), effects="random")
    fig = res.plot()
    assert [ax.get_title(loc="left").strip() for ax in fig.axes] == [
        "Normality of Random Effects"
    ] * 2


def test_variance_components_formula_rejected(mtcars):
    df = mtcars.assign(g=np.arange(32) % 4, h=np.arange(32) % 2)
    model = smf.mixedlm("mpg ~ wt", df, groups="g", vc_formula={"h": "0 + C(h)"}).fit()
    with pytest.raises(UnsupportedModelError, match="vc_formula"):
        as_adapter(model)


def test_groups_given_as_array_are_named_group():
    from tests.models import sleepstudy

    d = sleepstudy()
    a = as_adapter(smf.mixedlm("Reaction ~ Days", d, groups=d["Subject"]).fit())
    assert a.group_name == "Group"
    assert a.random_terms() == ["(Intercept)"]
    assert isinstance(a.random_effects, pd.DataFrame)
