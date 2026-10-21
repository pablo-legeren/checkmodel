"""icc(), r2_nakagawa() and model_performance() for linear mixed models vs R."""

from __future__ import annotations

import math

import pytest

from checkmodel import (
    CheckNotApplicableError,
    ICCResult,
    compare_performance,
    icc,
    model_performance,
    r2,
    r2_nakagawa,
    variance_components,
)
from tests.conftest import assert_r_close
from tests.models import MIXED

TOL = {"mixed_sleep_ml": 5e-4}
NON_SINGULAR = [n for n in MIXED if n != "mixed_singular"]


@pytest.mark.parametrize("name", NON_SINGULAR)
def test_icc_matches_r(name, fit, reference):
    ref = reference["mixed"][name]["icc"]
    result = icc(fit(name))
    assert isinstance(result, ICCResult)
    assert_r_close(result["ICC_adjusted"], ref["adjusted"], TOL.get(name, 1e-4))
    assert_r_close(result["ICC_unadjusted"], ref["unadjusted"], TOL.get(name, 1e-4))
    assert result["ICC_conditional"] == result["ICC_unadjusted"]


@pytest.mark.parametrize("name", NON_SINGULAR)
def test_r2_nakagawa_matches_r(name, fit, reference):
    ref = reference["mixed"][name]["r2"]
    result = r2(fit(name))
    assert list(result) == ["R2_conditional", "R2_marginal"]
    assert_r_close(result["R2_conditional"], ref["conditional"], TOL.get(name, 1e-4))
    assert_r_close(result["R2_marginal"], ref["marginal"], TOL.get(name, 1e-4))


def test_singular_fit_behaves_like_r(fit, reference):
    model = fit("mixed_singular")
    with pytest.warns(UserWarning, match="Can't compute ICC"):
        result = icc(model)
    assert math.isnan(result["ICC_adjusted"])
    with pytest.warns(UserWarning, match="Random effect variances not available"):
        nakagawa = r2_nakagawa(model)
    assert math.isnan(nakagawa["R2_conditional"])
    assert_r_close(nakagawa["R2_marginal"], reference["mixed"]["mixed_singular"]["r2"]["marginal"])
    assert math.isnan(variance_components(model).random)


@pytest.mark.parametrize("name", list(MIXED))
def test_model_performance_matches_r(name, fit, reference):
    ref = reference["mixed"][name]["performance"]
    with pytest.warns(UserWarning) if name == "mixed_singular" else _NoWarning():
        perf = model_performance(fit(name))
    assert list(perf.columns) == list(ref)
    for column, value in ref.items():
        if value is None:  # R keeps the missing conditional R2 of a singular fit
            assert math.isnan(perf.loc[0, column])
        else:
            assert_r_close(perf.loc[0, column], value, TOL.get(name, 1e-4))


def test_estimator_choice(fit):
    reml = fit("mixed_sleep")
    ml = fit("mixed_sleep_ml")
    assert model_performance(reml, estimator="REML").loc[0, "AIC"] == pytest.approx(
        -2 * reml.llf + 2 * 4
    )
    assert model_performance(ml, estimator="ML").loc[0, "AIC"] == pytest.approx(
        -2 * ml.llf + 2 * 4
    )
    # The other criterion evaluated at the fitted parameters, as R does.
    assert model_performance(reml, estimator="ML").loc[0, "AIC"] != pytest.approx(
        model_performance(reml).loc[0, "AIC"]
    )


def test_icc_metric_and_comparison(fit):
    perf = model_performance(fit("mixed_sleep"), metrics=["ICC", "R2"])
    assert list(perf.columns) == ["R2_conditional", "R2_marginal", "ICC"]
    table = compare_performance(intercept=fit("mixed_sleep"), slope=fit("mixed_sleep_slope"))
    assert table["Model"].tolist() == ["MixedLM", "MixedLM"]
    assert table["AIC_wt"].sum() == pytest.approx(1.0)


def test_needs_random_effects(ols):
    for fn in (icc, r2_nakagawa, variance_components):
        with pytest.raises(CheckNotApplicableError, match="no random effects"):
            fn(ols)


def test_result_repr(fit):
    assert repr(icc(fit("mixed_sleep"))).splitlines() == [
        "# Intraclass Correlation Coefficient",
        "    Adjusted ICC: 0.589",
        "  Unadjusted ICC: 0.424",
    ]
    assert repr(r2(fit("mixed_sleep"))).splitlines() == [
        "# R2 for Mixed Models",
        "  Conditional R2: 0.704",
        "     Marginal R2: 0.280",
    ]


class _NoWarning:
    def __enter__(self):
        import warnings

        self._ctx = warnings.catch_warnings()
        self._ctx.__enter__()
        warnings.simplefilter("error", UserWarning)

    def __exit__(self, *exc):
        return self._ctx.__exit__(*exc)
