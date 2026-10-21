"""``model_performance()``: the metric table for one model."""

from __future__ import annotations

import contextlib
import warnings
from collections.abc import Iterable
from typing import Literal

import numpy as np
import pandas as pd
from scipy import stats

from .._bayes import BayesianModel, as_bayesian
from .._introspection import BaseAdapter, as_adapter
from .._mcmc import psis_loo, waic
from .._types import ArrayLike, CheckNotApplicableError, Model
from .bayes import r2_bayes
from .r2 import r2

__all__ = ["model_performance"]

ALL_METRICS = (
    "AIC",
    "AICC",
    "BIC",
    "R2",
    "R2_ADJ",
    "ICC",
    "RMSE",
    "SIGMA",
    "LOGLOSS",
    "PCP",
    "SCORE",
)
COMMON_METRICS = ("AIC", "BIC", "R2", "R2_ADJ", "RMSE")
EXTRA_METRICS = ("LOGLIK",)
BAYES_METRICS = ("LOOIC", "WAIC", "R2", "RMSE", "SIGMA")
BAYES_COMMON_METRICS = ("LOOIC", "WAIC", "R2", "RMSE")


def _resolve_metrics(metrics: str | Iterable[str], bayesian: bool = False) -> set[str]:
    if isinstance(metrics, str):
        if metrics.lower() == "all":
            return set(BAYES_METRICS if bayesian else ALL_METRICS)
        if metrics.lower() == "common":
            return set(BAYES_COMMON_METRICS if bayesian else COMMON_METRICS)
        metrics = [metrics]
    aliases = {"LOG_LOSS": "LOGLOSS", "LOG_LIK": "LOGLIK"}
    wanted = {aliases.get(m.upper(), m.upper()) for m in metrics}
    known = ALL_METRICS + EXTRA_METRICS + BAYES_METRICS
    unknown = wanted - set(known)
    if unknown:
        raise ValueError(f"Unknown metrics {sorted(unknown)}; choose from {sorted(set(known))}.")
    return wanted


def _bayesian_performance(bayes: BayesianModel, wanted: set[str]) -> pd.DataFrame:
    # performance::model_performance.stanreg(), on the posterior draws.
    out: dict[str, float] = {}
    if "LOOIC" in wanted and bayes.log_lik is not None:
        loo = psis_loo(bayes.log_lik, r_eff=np.ones(bayes.n_obs))
        out.update(
            ELPD=loo.elpd_loo, ELPD_SE=loo.elpd_loo_se, LOOIC=loo.looic, LOOIC_SE=loo.looic_se
        )
    if "WAIC" in wanted and bayes.log_lik is not None:
        out["WAIC"] = waic(bayes.log_lik)[0]
    if "R2" in wanted:
        with contextlib.suppress(CheckNotApplicableError):
            out.update(r2_bayes(bayes))
    if "RMSE" in wanted:
        point = np.median(BayesianModel.flat(bayes.mu), axis=0)
        out["RMSE"] = float(np.sqrt(np.mean((bayes.y - point) ** 2)))
    if "SIGMA" in wanted:
        if bayes.sigma is not None:
            out["Sigma"] = float(np.median(bayes.sigma))
        elif bayes.family in {"binomial", "poisson"}:
            out["Sigma"] = 1.0
    return pd.DataFrame([out])


def _scoring_rules(adapter: BaseAdapter) -> tuple[float, float] | None:
    info, y, mu = adapter.info, adapter.y, adapter.mu
    if info.is_binomial:
        # performance::performance_score() passes sum(y) as the binomial size;
        # kept as is so the numbers match R.
        p_y = stats.binom.pmf(np.round(y), round(float(y.sum())), mu)
    elif info.family == "poisson":
        p_y = stats.poisson.pmf(np.round(y), mu)
    else:
        return None
    with np.errstate(divide="ignore"):  # p_y can underflow to 0; R then reports -Inf too
        log_score = float(np.mean(np.log(p_y)))
    spherical = float(np.mean(p_y / np.sqrt(np.sum(p_y**2))))
    return log_score, spherical


def model_performance(
    model: Model,
    X: ArrayLike | None = None,
    y: ArrayLike | None = None,
    *,
    metrics: str | Iterable[str] = "all",
    estimator: Literal["ML", "REML"] | None = None,
) -> pd.DataFrame:
    """Compute performance metrics for one model, as ``performance::model_performance()``.

    Parameters
    ----------
    model
        Any supported model.
    X, y
        Training data; required for scikit-learn models.
    metrics
        ``"all"`` (default), ``"common"`` (AIC, BIC, R², adjusted R², RMSE) or
        a list of names among ``AIC``, ``AICc``, ``BIC``, ``R2``, ``R2_adj``,
        ``RMSE``, ``Sigma``, ``Log_loss``, ``PCP``, ``Score`` and ``Log_lik``.
        ``Log_lik`` is not part of R's output and must be requested explicitly.
        Mixed models also report ``ICC``.
    estimator
        Likelihood behind AIC, AICc and BIC for mixed models: ``"REML"``
        (the default, as in R) or ``"ML"``. As in R, the chosen criterion is
        evaluated at the fitted parameters whichever way the model was fitted.
        Ignored for other models.

    Returns
    -------
    pandas.DataFrame
        One row. Metrics that do not apply to the model are left out, as in R:
        for example ``Log_loss`` and ``PCP`` only appear for binomial models.
        Mixed models report ``R2_conditional``, ``R2_marginal`` and ``ICC``.
        Bayesian models report ``ELPD``, ``ELPD_SE``, ``LOOIC``, ``LOOIC_SE``
        (PSIS-LOO), ``WAIC``, the Bayesian ``R2`` (plus ``R2_marginal`` for
        multilevel models), ``RMSE`` and ``Sigma``, as for ``stanreg`` fits.

    Notes
    -----
    AIC and BIC count the residual variance as a parameter for Gaussian
    models, like R's ``AIC()``. statsmodels' ``OLSResults.aic`` does not, so
    its values are 2 lower. Penalized scikit-learn models have no
    likelihood-based criteria; AIC, AICc and BIC are omitted for them.
    """
    bayes = as_bayesian(model)
    if bayes is not None:
        return _bayesian_performance(bayes, _resolve_metrics(metrics, bayesian=True))
    adapter = as_adapter(model, X, y)
    wanted = _resolve_metrics(metrics)
    info, n = adapter.info, adapter.n_obs
    out: dict[str, float] = {}

    if estimator is None:
        estimator = "REML" if info.is_mixed else "ML"
    ll = None if info.is_penalized else adapter.loglik_with(estimator)
    k = adapter.loglik_df
    if ll is not None:
        if "AIC" in wanted:
            out["AIC"] = -2.0 * ll + 2.0 * k
        if "AICC" in wanted:
            out["AICc"] = -2.0 * ll + 2.0 * k * n / (n - k - 1)
        if "BIC" in wanted:
            out["BIC"] = -2.0 * ll + k * np.log(n)
    if wanted & {"R2", "R2_ADJ"}:
        try:
            measures = dict(r2(adapter))
        except CheckNotApplicableError:
            measures = {}
        if "R2" in measures or "R2_adjusted" in measures:
            if "R2" in wanted and "R2" in measures:
                out["R2"] = measures["R2"]
            if "R2_ADJ" in wanted and "R2_adjusted" in measures:
                out["R2_adjusted"] = measures["R2_adjusted"]
        else:
            out.update(measures)
    if "ICC" in wanted and info.is_mixed:
        from .icc import icc

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            out["ICC"] = icc(adapter)["ICC_adjusted"]
    if "RMSE" in wanted:
        out["RMSE"] = float(np.sqrt(np.mean(adapter.resid**2)))
    if "SIGMA" in wanted:
        out["Sigma"] = adapter.sigma
    if "LOGLOSS" in wanted and info.is_binomial:
        out["Log_loss"] = float(np.mean(-np.log(1.0 - np.abs(adapter.y - adapter.mu))))
    if "SCORE" in wanted and (info.is_binomial or info.is_count):
        rules = _scoring_rules(adapter)
        if rules is not None:
            out["Score_log"], out["Score_spherical"] = rules
    if "PCP" in wanted and info.is_binomial:
        y0, p = adapter.y, adapter.mu
        out["PCP"] = float((np.sum(1.0 - p[y0 == 0]) + np.sum(p[y0 == 1])) / n)
    if "LOGLIK" in wanted and ll is not None:
        out["Log_lik"] = ll
    # Like insight::compact_list(remove_na = TRUE): drop missing values, keep infinities.
    # R keeps a missing conditional R² of a singular mixed model, so do we.
    keep_nan = {"R2_conditional"} if info.is_mixed else set()
    clean = {
        key: float(v)
        for key, v in out.items()
        if v is not None and (key in keep_nan or not np.isnan(v))
    }
    return pd.DataFrame([clean])
