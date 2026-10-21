"""Turn fitted model objects into one uniform view of their data and diagnostics.

Every check and metric in `checkmodel` works on a :class:`BaseAdapter`. An
adapter only has to supply the five primitives of
:class:`~checkmodel.ModelAdapter` (response, fitted values, residuals, model
matrix and model type); leverage, standardized residuals, Cook's distance,
log-likelihood and the rest are derived here from those primitives, using the
same definitions as R's ``stats`` and ``insight`` packages. Backends override a
derived quantity only when they can supply it more faithfully (statsmodels GLMs,
for instance, already know their prior weights, offsets and IRLS weights).
"""

from __future__ import annotations

import copy
import re
import warnings
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from functools import cached_property
from typing import Any, cast

import numpy as np
import pandas as pd
from scipy import special, stats

from ._types import (
    ArrayLike,
    FloatArray,
    Model,
    ModelAdapter,
    ModelType,
    UnsupportedModelError,
)

__all__ = ["BaseAdapter", "Term", "as_adapter", "register_adapter"]


@dataclass(frozen=True)
class Term:
    """A model term and the design-matrix columns that encode it.

    ``is_factor`` marks a term made of a single categorical variable. Its
    generalized VIF is adjusted by the number of columns, as in R.
    """

    name: str
    columns: tuple[int, ...]
    is_factor: bool = False


class BaseAdapter(ABC):
    """Uniform, cached view of a fitted model.

    Subclasses implement the five :class:`~checkmodel.ModelAdapter`
    primitives and, optionally, the metadata hooks (``column_names``,
    ``terms``, ``prior_weights``, ...). Everything else is derived.
    """

    def __init__(self, model: Model) -> None:
        self.model = model

    # ------------------------------------------------------------------ primitives
    @abstractmethod
    def model_type(self) -> ModelType:
        """Describe the model family, link and backend."""

    @abstractmethod
    def get_response(self) -> FloatArray:
        """Observed response (0/1 for binary models)."""

    @abstractmethod
    def get_fitted(self) -> FloatArray:
        """Fitted values on the response scale."""

    @abstractmethod
    def get_model_matrix(self) -> FloatArray:
        """Design matrix including the intercept column, if any."""

    def get_residuals(self) -> FloatArray:
        """Response residuals, ``y - fitted``."""
        return self.y - self.mu

    # ------------------------------------------------------------ metadata hooks
    def column_names(self) -> list[str]:
        """Names of the design-matrix columns."""
        return [f"x{i}" for i in range(self.X.shape[1])]

    def intercept_index(self) -> int | None:
        """Position of the intercept column, or ``None`` for models without one."""
        X = self.X
        for j in range(X.shape[1]):
            if np.all(X[:, j] == 1.0):
                return j
        return None

    def terms(self) -> list[Term]:
        """Model terms; by default every non-intercept column is its own term."""
        names = self.column_names()
        icpt = self.intercept_index()
        return [Term(names[j], (j,)) for j in range(len(names)) if j != icpt]

    def prior_weights(self) -> FloatArray | None:
        """Observation weights the model was fitted with, or ``None``."""
        return None

    def index(self) -> pd.Index:
        """Labels identifying the observations (row names in R)."""
        return pd.RangeIndex(self.n_obs)

    def n_numeric_variables(self) -> int:
        """Count the numeric variables in the model data.

        ``check_outliers()`` in R derives its default Cook's distance threshold
        from this count: response plus predictors, or predictors only for
        binomial models.
        """
        k = len(self.terms())
        return k if self.info.is_binomial else k + 1

    # ------------------------------------------------------------- cached basics
    @cached_property
    def info(self) -> ModelType:
        """The :class:`ModelType` of the wrapped model."""
        return self.model_type()

    @cached_property
    def y(self) -> FloatArray:
        """Observed response."""
        return np.asarray(self.get_response(), dtype=float).ravel()

    @cached_property
    def mu(self) -> FloatArray:
        """Fitted values on the response scale."""
        return np.asarray(self.get_fitted(), dtype=float).ravel()

    @cached_property
    def resid(self) -> FloatArray:
        """Response residuals."""
        return np.asarray(self.get_residuals(), dtype=float).ravel()

    @cached_property
    def X(self) -> FloatArray:
        """Design matrix."""
        X = np.asarray(self.get_model_matrix(), dtype=float)
        return X.reshape(-1, 1) if X.ndim == 1 else X

    @cached_property
    def weights(self) -> FloatArray:
        """Prior weights, ones when the model is unweighted."""
        w = self.prior_weights()
        if w is None:
            return np.ones(self.n_obs)
        return np.broadcast_to(np.asarray(w, dtype=float), (self.n_obs,)).copy()

    @property
    def n_obs(self) -> int:
        """Number of observations."""
        return int(self.y.shape[0])

    @property
    def n_params(self) -> int:
        """Number of regression coefficients, including the intercept."""
        return int(self.X.shape[1])

    @property
    def has_intercept(self) -> bool:
        """Whether the model has an intercept."""
        return self.intercept_index() is not None

    @cached_property
    def rank(self) -> int:
        """Rank of the (weighted) design matrix."""
        return int(np.linalg.matrix_rank(self._wX))

    @property
    def df_resid(self) -> int:
        """Residual degrees of freedom."""
        return self.n_obs - self.rank

    # ------------------------------------------------------ family-level helpers
    @cached_property
    def working_weights(self) -> FloatArray:
        """Final IRLS weights, ``w * (dmu/deta)^2 / V(mu)``."""
        info, mu, w = self.info, self.mu, self.weights
        if info.is_linear:
            return w
        if info.family == "binomial" and info.link == "logit":
            return w * mu * (1.0 - mu)
        if info.family == "poisson" and info.link == "log":
            return w * mu
        raise UnsupportedModelError(
            f"Cannot derive working weights for family={info.family!r}, link={info.link!r}; "
            "the adapter must override `working_weights` or `hat`."
        )

    def _variance(self, mu: FloatArray) -> FloatArray:
        fam = self.info.family
        if fam == "gaussian":
            return np.ones_like(mu)
        if fam == "binomial":
            return mu * (1.0 - mu)
        if fam == "poisson":
            return mu
        raise UnsupportedModelError(f"No variance function for family={fam!r}.")

    @cached_property
    def _wX(self) -> FloatArray:
        return np.sqrt(self.working_weights)[:, None] * self.X

    # ------------------------------------------------------------ residual types
    @cached_property
    def pearson_residuals(self) -> FloatArray:
        """Pearson residuals, ``sqrt(w) * (y - mu) / sqrt(V(mu))``."""
        return np.sqrt(self.weights) * self.resid / np.sqrt(self._variance(self.mu))

    @cached_property
    def deviance_residuals(self) -> FloatArray:
        """Deviance residuals (equal to Pearson residuals for Gaussian models)."""
        fam, y, mu, w = self.info.family, self.y, self.mu, self.weights
        if fam == "gaussian":
            return self.pearson_residuals
        if fam == "binomial":
            unit = -2.0 * (special.xlogy(y, mu) + special.xlogy(1.0 - y, 1.0 - mu))
            unit = unit + 2.0 * (special.xlogy(y, y) + special.xlogy(1.0 - y, 1.0 - y))
        elif fam == "poisson":
            unit = 2.0 * (special.xlogy(y, y / mu) - (y - mu))
        else:
            raise UnsupportedModelError(f"No deviance residuals for family={fam!r}.")
        return np.sign(y - mu) * np.sqrt(np.clip(w * unit, 0.0, None))

    @cached_property
    def scale(self) -> float:
        """Dispersion: 1 for binomial/Poisson, Pearson chi² / df otherwise."""
        if self.info.family in {"binomial", "poisson"}:
            return 1.0
        return float(np.sum(self.pearson_residuals**2) / self.df_resid)

    @property
    def sigma(self) -> float:
        """Residual standard deviation, as returned by ``stats::sigma()`` in R."""
        return float(np.sqrt(self.scale))

    # ------------------------------------------------------------- influence
    @cached_property
    def hat(self) -> FloatArray:
        """Leverage: diagonal of the (weighted) hat matrix."""
        U, s, _ = np.linalg.svd(self._wX, full_matrices=False)
        keep = s > s.max() * max(self._wX.shape) * np.finfo(float).eps
        return np.asarray(np.sum(U[:, keep] ** 2, axis=1))

    @cached_property
    def std_residuals(self) -> FloatArray:
        """Internally studentized Pearson residuals (``rstandard()`` in R)."""
        return self.pearson_residuals / np.sqrt(self.scale * (1.0 - self.hat))

    @cached_property
    def std_deviance_residuals(self) -> FloatArray:
        """Standardized deviance residuals (``rstandard(type = "deviance")``)."""
        return self.deviance_residuals / np.sqrt(self.scale * (1.0 - self.hat))

    @cached_property
    def studentized_residuals(self) -> FloatArray:
        """Externally studentized residuals (``rstudent()`` for linear models)."""
        r = self.std_residuals
        df = self.df_resid
        return r * np.sqrt((df - 1) / np.clip(df - r**2, np.finfo(float).tiny, None))

    @cached_property
    def cooks_distance(self) -> FloatArray:
        """Cook's distance (``cooks.distance()`` in R)."""
        h = self.hat
        return self.pearson_residuals**2 * h / (self.scale * self.rank * (1.0 - h) ** 2)

    @cached_property
    def vcov_unscaled(self) -> FloatArray:
        """``(X' W X)^-1``: the coefficient covariance up to the dispersion."""
        return np.asarray(np.linalg.pinv(self._wX.T @ self._wX))

    # ------------------------------------------------------------- likelihood
    def _loglik_at(self, mu: FloatArray) -> float | None:
        info, y, w = self.info, self.y, self.weights
        if info.family == "gaussian":
            keep = w > 0
            rss = np.sum(w[keep] * (y[keep] - mu[keep]) ** 2)
            s2 = rss / keep.sum()
            return float(
                0.5 * np.sum(np.log(w[keep]) - (np.log(2 * np.pi) + np.log(s2))) - 0.5 * rss / s2
            )
        if info.family == "binomial":
            return float(np.sum(w * (special.xlogy(y, mu) + special.xlogy(1.0 - y, 1.0 - mu))))
        if info.family == "poisson":
            return float(np.sum(w * stats.poisson.logpmf(np.round(y), mu)))
        return None

    @cached_property
    def loglik(self) -> float | None:
        """Log-likelihood at the fitted values (ML estimate of the dispersion)."""
        return self._loglik_at(self.mu)

    @property
    def loglik_df(self) -> int:
        """Parameters counted by AIC/BIC: coefficients plus dispersion, if estimated."""
        return self.rank + int(self.info.has_dispersion)

    def loglik_with(self, estimator: str | None) -> float | None:
        """Log-likelihood under ``"ML"`` or ``"REML"``; only mixed models distinguish them."""
        return self.loglik

    @cached_property
    def _null_mu(self) -> FloatArray:
        if self.has_intercept:
            return np.full(self.n_obs, np.average(self.y, weights=self.weights))
        link = self.info.link
        value = {"logit": 0.5, "log": 1.0}.get(link, 0.0)
        return np.full(self.n_obs, value)

    @cached_property
    def loglik_null(self) -> float | None:
        """Log-likelihood of the intercept-only model."""
        return self._loglik_at(self._null_mu)

    def _deviance_at(self, mu: FloatArray) -> float:
        fam, y, w = self.info.family, self.y, self.weights
        if fam == "gaussian":
            return float(np.sum(w * (y - mu) ** 2))
        if fam == "binomial":
            unit = -2.0 * (special.xlogy(y, mu) + special.xlogy(1.0 - y, 1.0 - mu))
            unit = unit + 2.0 * (special.xlogy(y, y) + special.xlogy(1.0 - y, 1.0 - y))
        elif fam == "poisson":
            unit = 2.0 * (special.xlogy(y, y / mu) - (y - mu))
        else:
            raise UnsupportedModelError(f"No deviance for family={fam!r}.")
        return float(np.sum(w * unit))

    @cached_property
    def deviance(self) -> float:
        """Residual deviance."""
        return self._deviance_at(self.mu)

    @cached_property
    def null_deviance(self) -> float:
        """Deviance of the intercept-only model."""
        return self._deviance_at(self._null_mu)


# --------------------------------------------------------------------------- utils
_IDENT = re.compile(r"[A-Za-z_.][A-Za-z0-9_.]*")
_QUOTED = re.compile(r"""["']([^"']+)["']""")


def _variables_in(expr: str, columns: pd.Index) -> list[str]:
    """Find the data columns referenced by a formula fragment such as ``C(cyl) + np.log(x)``."""
    found = set(_IDENT.findall(expr)) | set(_QUOTED.findall(expr))
    return [c for c in columns if isinstance(c, str) and c in found]


def _is_numeric(series: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series)


def _formula_terms(sm_model: Any, n_cols: int) -> list[Term] | None:
    """Group design-matrix columns into formula terms (patsy or formulaic)."""
    spec = getattr(sm_model.data, "model_spec", None) or getattr(
        sm_model.data, "design_info", None
    )
    if spec is None:
        return None
    slices = getattr(spec, "term_name_slices", None)
    if slices is None:  # formulaic, via statsmodels' engine-agnostic manager
        try:
            from statsmodels.formula._manager import FormulaManager

            slices = FormulaManager().get_term_name_slices(spec)
        except Exception:
            return None
    factor_terms: set[str] = set()
    for term in getattr(spec, "terms", []):
        factors: tuple[Any, ...] = tuple(getattr(term, "factors", ()))
        infos = getattr(spec, "factor_infos", {})
        if len(factors) == 1 and getattr(infos.get(factors[0]), "type", "") == "categorical":
            factor_terms.add(term.name())
    terms = [
        Term(name, tuple(range(n_cols)[sl]), name in factor_terms)
        for name, sl in slices.items()
        if name != "Intercept"
    ]
    return _formula_order(sm_model, terms)


def _formula_order(sm_model: Any, terms: list[Term]) -> list[Term]:
    """Put terms back in the order they were written; patsy sorts them by type."""
    formula = getattr(sm_model, "formula", None)
    if not isinstance(formula, str):
        return terms
    try:
        from patsy import ModelDesc

        written = [t.name() for t in ModelDesc.from_formula(formula).rhs_termlist]
    except Exception:
        return terms
    position = {name: i for i, name in enumerate(written)}
    return sorted(terms, key=lambda t: position.get(t.name, len(position)))


# ----------------------------------------------------------------- statsmodels
class _StatsmodelsMixin:
    """Metadata shared by statsmodels OLS/WLS and GLM results."""

    model: Any
    info: ModelType
    X: FloatArray

    @property
    def _sm(self) -> Any:
        return self.model.model

    def get_model_matrix(self) -> FloatArray:
        return np.asarray(self._sm.exog, dtype=float)

    def column_names(self) -> list[str]:
        return list(self._sm.exog_names)

    def intercept_index(self) -> int | None:
        idx = getattr(self._sm.data, "const_idx", None)
        return None if idx is None else int(idx)

    def terms(self) -> list[Term]:
        terms = _formula_terms(self._sm, self.X.shape[1])
        if terms is not None:
            return terms
        names = self.column_names()
        icpt = self.intercept_index()
        return [Term(names[j], (j,)) for j in range(len(names)) if j != icpt]

    def index(self) -> pd.Index:
        labels = getattr(self._sm.data, "row_labels", None)
        if labels is not None:
            return pd.Index(labels)
        return pd.RangeIndex(self.X.shape[0])

    def n_numeric_variables(self) -> int:
        formula = getattr(self._sm, "formula", None)
        frame = getattr(self._sm.data, "frame", None)
        if isinstance(formula, str) and isinstance(frame, pd.DataFrame) and "~" in formula:
            lhs, rhs = formula.split("~", 1)
            numeric = [c for c in frame.columns if _is_numeric(frame[c])]
            cols = pd.Index(numeric)
            k = len(_variables_in(rhs, cols))
            if not self.info.is_binomial:
                k += len(_variables_in(lhs, cols))
            if k > 0:
                return k
        k = len(self.terms())
        return k if self.info.is_binomial else k + 1


class StatsmodelsLinearAdapter(_StatsmodelsMixin, BaseAdapter):
    """Adapter for statsmodels ``OLS`` and ``WLS`` results."""

    def model_type(self) -> ModelType:
        return ModelType(
            family="gaussian",
            link="identity",
            backend="statsmodels",
            model_class=type(self._sm).__name__,
        )

    def get_response(self) -> FloatArray:
        return np.asarray(self._sm.endog, dtype=float)

    def get_fitted(self) -> FloatArray:
        return np.asarray(self.model.fittedvalues, dtype=float)

    def prior_weights(self) -> FloatArray | None:
        w = getattr(self._sm, "weights", None)
        if w is None or np.ndim(w) == 0:
            return None
        return np.asarray(w, dtype=float)


_SM_FAMILIES = {
    "Binomial": "binomial",
    "Poisson": "poisson",
    "Gaussian": "gaussian",
    "Gamma": "gamma",
    "InverseGaussian": "inverse.gaussian",
    "NegativeBinomial": "negbin",
    "Tweedie": "tweedie",
}
_SM_LINKS = {
    "Logit": "logit",
    "Probit": "probit",
    "CLogLog": "cloglog",
    "LogLog": "loglog",
    "Cauchy": "cauchit",
    "Log": "log",
    "Identity": "identity",
    "InversePower": "inverse",
    "InverseSquared": "1/mu^2",
    "Sqrt": "sqrt",
}


class StatsmodelsGLMAdapter(_StatsmodelsMixin, BaseAdapter):
    """Adapter for statsmodels ``GLM`` results.

    Influence measures, residuals and likelihoods are taken from statsmodels
    itself, so prior weights, offsets and exposure are honoured.
    """

    def model_type(self) -> ModelType:
        family = self._sm.family
        fam = _SM_FAMILIES.get(type(family).__name__, type(family).__name__.lower())
        link = _SM_LINKS.get(type(family.link).__name__, type(family.link).__name__.lower())
        n_trials = np.asarray(getattr(self._sm, "n_trials", 1.0))
        endog = np.asarray(self._sm.endog, dtype=float)
        bernoulli = (
            fam == "binomial"
            and bool(np.all(n_trials == 1))
            and bool(np.all((endog == 0) | (endog == 1)))
        )
        return ModelType(
            family=fam,
            link=link,
            backend="statsmodels",
            model_class="GLM",
            is_bernoulli=bernoulli,
        )

    def get_response(self) -> FloatArray:
        return np.asarray(self._sm.endog, dtype=float)

    def get_fitted(self) -> FloatArray:
        return np.asarray(self.model.mu, dtype=float)

    def prior_weights(self) -> FloatArray | None:
        w = np.asarray(self._sm.var_weights, dtype=float) * np.asarray(
            self._sm.freq_weights, dtype=float
        )
        return None if np.all(w == 1.0) else w

    @cached_property
    def _influence(self) -> Any:
        return self.model.get_influence()

    @cached_property
    def working_weights(self) -> FloatArray:
        mu = self.mu
        family = self._sm.family
        n_trials = getattr(self._sm, "n_trials", 1.0)
        return np.asarray(self._sm.iweights * n_trials * family.weights(mu))

    @cached_property
    def pearson_residuals(self) -> FloatArray:
        return np.asarray(self.model.resid_pearson, dtype=float)

    @cached_property
    def deviance_residuals(self) -> FloatArray:
        return np.asarray(self.model.resid_deviance, dtype=float)

    @cached_property
    def scale(self) -> float:
        return float(self.model.scale)

    @cached_property
    def hat(self) -> FloatArray:
        return np.asarray(self._influence.hat_matrix_diag, dtype=float)

    @cached_property
    def vcov_unscaled(self) -> FloatArray:
        return np.asarray(self.model.normalized_cov_params, dtype=float)

    @cached_property
    def loglik(self) -> float | None:
        return float(self.model.llf)

    @cached_property
    def loglik_null(self) -> float | None:
        return float(self.model.llnull)

    @cached_property
    def deviance(self) -> float:
        return float(self.model.deviance)

    @cached_property
    def null_deviance(self) -> float:
        return float(self.model.null_deviance)


class StatsmodelsMixedAdapter(_StatsmodelsMixin, BaseAdapter):
    """Adapter for statsmodels ``MixedLM`` results (linear mixed models).

    Fitted values and residuals are conditional on the random effects, as in
    ``lme4``. Leverage, studentized residuals and Cook's distance follow
    ``hatvalues()``, ``rstudent()`` and ``cooks.distance()`` for ``lmer`` fits.
    """

    def __init__(self, model: Model) -> None:
        super().__init__(model)
        if getattr(self._sm, "k_vc", 0):
            raise UnsupportedModelError(
                "MixedLM models with variance components (vc_formula) are not supported yet; "
                "use `groups=` with `re_formula=`."
            )

    def model_type(self) -> ModelType:
        return ModelType(
            family="gaussian",
            link="identity",
            backend="statsmodels",
            model_class="MixedLM",
            is_mixed=True,
        )

    def get_response(self) -> FloatArray:
        return np.asarray(self._sm.endog, dtype=float)

    def get_fitted(self) -> FloatArray:
        return np.asarray(self.model.fittedvalues, dtype=float)

    # -------------------------------------------------------- random effects
    @cached_property
    def Z(self) -> FloatArray:
        """Random-effects design matrix."""
        Z = np.asarray(self._sm.exog_re, dtype=float)
        return Z.reshape(-1, 1) if Z.ndim == 1 else Z

    @property
    def group_name(self) -> str:
        """Name of the grouping factor."""
        names = list(getattr(self._sm.data, "exog_re_names", []) or ["Group"])
        return str(names[0]) if np.all(self.Z[:, 0] == 1.0) else "Group"

    def random_terms(self) -> list[str]:
        """Random-effect terms, named as in R: ``(Intercept)``, ``Days``, ..."""
        names = [str(n) for n in getattr(self._sm.data, "exog_re_names", [])]
        if len(names) != self.Z.shape[1]:
            names = [f"z{j}" for j in range(self.Z.shape[1])]
        if np.all(self.Z[:, 0] == 1.0):
            names[0] = "(Intercept)"
        return names

    @cached_property
    def cov_re(self) -> FloatArray:
        """Covariance matrix of the random effects (variance scale, like ``VarCorr``)."""
        return np.atleast_2d(np.asarray(self.model.cov_re, dtype=float))

    @cached_property
    def random_effects(self) -> pd.DataFrame:
        """Conditional modes (BLUPs): one row per group, one column per term."""
        blups = pd.DataFrame.from_dict(
            {k: np.asarray(v, dtype=float) for k, v in self.model.random_effects.items()},
            orient="index",
        )
        blups.columns = self.random_terms()
        return blups

    # ---------------------------------------------------------- influence
    @cached_property
    def _gls(self) -> tuple[FloatArray, FloatArray, FloatArray]:
        """``V^-1 X``, ``diag(V^-1)`` and ``(X' V^-1 X)^-1``, built group by group."""
        s2, G, X, Z = self.scale, self.cov_re, self.X, self.Z
        Vinv_X = np.empty_like(X)
        diag_Vinv = np.empty(self.n_obs)
        for rows in self._sm.row_indices.values():
            Zg = Z[rows]
            Vg_inv = np.linalg.inv(Zg @ G @ Zg.T + s2 * np.eye(len(rows)))
            Vinv_X[rows] = Vg_inv @ X[rows]
            diag_Vinv[rows] = np.diag(Vg_inv)
        return Vinv_X, diag_Vinv, np.linalg.pinv(X.T @ Vinv_X)

    @cached_property
    def hat(self) -> FloatArray:
        # Conditional fitted values are H y with H = I - s2 V^-1 (I - P), where
        # V = Z G Z' + s2 I and P = X (X' V^-1 X)^-1 X' V^-1; V is block diagonal.
        Vinv_X, diag_Vinv, A = self._gls
        return np.asarray(1.0 - self.scale * (diag_Vinv - np.sum((Vinv_X @ A) * Vinv_X, axis=1)))

    @cached_property
    def pearson_residuals(self) -> FloatArray:
        return self.resid

    @cached_property
    def deviance_residuals(self) -> FloatArray:
        return self.resid

    @cached_property
    def scale(self) -> float:
        return float(self.model.scale)

    @property
    def df_resid(self) -> int:
        return self.n_obs - self.loglik_df

    @cached_property
    def vcov_unscaled(self) -> FloatArray:
        # lme4's vcov(): (X' V^-1 X)^-1 given the variance parameters. statsmodels'
        # cov_params() inverts the Hessian of all parameters, which differs slightly.
        return self._gls[2]

    # --------------------------------------------------------- likelihood
    @property
    def is_reml(self) -> bool:
        """Whether the model was fitted by REML."""
        return bool(self._sm.reml)

    @cached_property
    def loglik(self) -> float | None:
        return float(self.model.llf)

    @property
    def loglik_df(self) -> int:
        sm = self._sm
        return int(sm.k_fe + sm.k_re2 + sm.k_vc + 1)

    def loglik_with(self, estimator: str | None) -> float | None:
        if estimator is None or (estimator.upper() == "REML") == self.is_reml:
            return self.loglik
        # The other criterion evaluated at the fitted parameters, as lme4 does
        # for logLik(REML = ...) and performance for `estimator =`.
        other = copy.copy(self._sm)
        other.reml = estimator.upper() == "REML"
        return float(other.loglike(self.model.params_object, profile_fe=False))

    @cached_property
    def loglik_null(self) -> float | None:
        return None


# ------------------------------------------------------------------ scikit-learn
_SKLEARN_PENALIZED = {
    "Ridge",
    "RidgeCV",
    "Lasso",
    "LassoCV",
    "ElasticNet",
    "ElasticNetCV",
    "LassoLars",
    "LassoLarsCV",
    "LassoLarsIC",
}
_SKLEARN_LINEAR = {"LinearRegression"} | _SKLEARN_PENALIZED
_SKLEARN_LOGISTIC = {"LogisticRegression", "LogisticRegressionCV"}


def _sklearn_logistic_is_penalized(model: Any) -> bool:
    penalty = getattr(model, "penalty", "l2")
    if penalty is None or penalty == "none":
        return False
    C = getattr(model, "C_", getattr(model, "C", 1.0))
    return not bool(np.all(np.isinf(np.asarray(C, dtype=float))))


class SklearnAdapter(BaseAdapter):
    """Adapter for scikit-learn linear models.

    scikit-learn estimators do not keep their training data, so ``X`` and
    ``y`` must be passed alongside the model.
    """

    def __init__(self, model: Model, X: ArrayLike | None, y: ArrayLike | None) -> None:
        super().__init__(model)
        name = type(model).__name__
        if X is None or y is None:
            raise TypeError(
                f"scikit-learn models do not store their training data; call e.g. "
                f"`check_model(model, X=X, y=y)` with the data `{name}` was fitted on."
            )
        if not hasattr(model, "coef_"):
            raise UnsupportedModelError(f"`{name}` is not fitted yet; call `.fit(X, y)` first.")
        self._columns = (
            [str(c) for c in X.columns]
            if isinstance(X, pd.DataFrame)
            else [str(c) for c in getattr(model, "feature_names_in_", [])]
        )
        if isinstance(X, (pd.DataFrame, pd.Series)):
            self._index: pd.Index = X.index
        elif isinstance(y, pd.Series):
            self._index = y.index
        else:
            self._index = pd.RangeIndex(len(np.asarray(y)))
        # Predict with X as given, so estimators fitted on a DataFrame see feature names.
        self._Xpredict = X
        self._Xraw = np.asarray(X, dtype=float)
        if self._Xraw.ndim == 1:
            self._Xraw = self._Xraw.reshape(-1, 1)
        self._yraw = np.asarray(y).ravel()
        if self._Xraw.shape[0] != self._yraw.shape[0]:
            raise ValueError(
                f"X has {self._Xraw.shape[0]} rows but y has {self._yraw.shape[0]} values."
            )
        coef = np.atleast_2d(model.coef_)
        if coef.shape[0] != 1:
            raise UnsupportedModelError(
                f"Multi-output and multiclass `{name}` models are not supported."
            )
        if coef.shape[1] != self._Xraw.shape[1]:
            raise ValueError(
                f"`{name}` was fitted on {coef.shape[1]} features but X has "
                f"{self._Xraw.shape[1]} columns."
            )
        if not self._columns:
            self._columns = [f"x{i}" for i in range(self._Xraw.shape[1])]
        self._fit_intercept = bool(getattr(model, "fit_intercept", True))

    def model_type(self) -> ModelType:
        name = type(self.model).__name__
        if name in _SKLEARN_LOGISTIC:
            return ModelType(
                family="binomial",
                link="logit",
                backend="sklearn",
                model_class=name,
                is_bernoulli=True,
                is_penalized=_sklearn_logistic_is_penalized(self.model),
            )
        return ModelType(
            family="gaussian",
            link="identity",
            backend="sklearn",
            model_class=name,
            is_penalized=name in _SKLEARN_PENALIZED,
        )

    def get_response(self) -> FloatArray:
        if self.info.is_binomial:
            return np.asarray(self._yraw == self.model.classes_[1], dtype=float)
        return self._yraw.astype(float)

    def get_fitted(self) -> FloatArray:
        if self.info.is_binomial:
            return np.asarray(self.model.predict_proba(self._Xpredict)[:, 1], dtype=float)
        return np.asarray(self.model.predict(self._Xpredict), dtype=float).ravel()

    def get_model_matrix(self) -> FloatArray:
        if self._fit_intercept:
            return np.column_stack([np.ones(self._Xraw.shape[0]), self._Xraw])
        return self._Xraw

    def column_names(self) -> list[str]:
        return (["Intercept"] if self._fit_intercept else []) + self._columns

    def intercept_index(self) -> int | None:
        return 0 if self._fit_intercept else None

    def index(self) -> pd.Index:
        return self._index


class ProtocolAdapter(BaseAdapter):
    """Wrap a third-party object implementing :class:`~checkmodel.ModelAdapter`.

    Optional hooks (``column_names``, ``prior_weights``, ``index``, ...) are
    used when the wrapped object defines them.
    """

    _HOOKS = ("column_names", "intercept_index", "terms", "prior_weights", "index")

    def model_type(self) -> ModelType:
        return cast(ModelType, self.model.model_type())

    def get_response(self) -> FloatArray:
        return np.asarray(self.model.get_response(), dtype=float)

    def get_fitted(self) -> FloatArray:
        return np.asarray(self.model.get_fitted(), dtype=float)

    def get_residuals(self) -> FloatArray:
        return np.asarray(self.model.get_residuals(), dtype=float)

    def get_model_matrix(self) -> FloatArray:
        return np.asarray(self.model.get_model_matrix(), dtype=float)

    def __getattribute__(self, name: str) -> Any:
        if name in ProtocolAdapter._HOOKS:
            hook = getattr(object.__getattribute__(self, "model"), name, None)
            if callable(hook):
                return hook
        return object.__getattribute__(self, name)


# ---------------------------------------------------------------------- registry
AdapterFactory = Callable[..., BaseAdapter]
_REGISTRY: list[tuple[Callable[[Any], bool], AdapterFactory]] = []


def register_adapter(match: type | Callable[[Any], bool], factory: AdapterFactory) -> None:
    """Teach `checkmodel` to read a new kind of model.

    Parameters
    ----------
    match
        A class (instances and subclasses match) or a predicate
        ``model -> bool``.
    factory
        Called as ``factory(model, X=X, y=y)``; must return a
        :class:`BaseAdapter`. Registered adapters take precedence over the
        built-in ones.

    Examples
    --------
    >>> from checkmodel import BaseAdapter, register_adapter
    >>> class MyAdapter(BaseAdapter): ...  # doctest: +SKIP
    >>> register_adapter(MyModel, lambda m, X=None, y=None: MyAdapter(m))  # doctest: +SKIP
    """
    if isinstance(match, type):
        cls = match
        predicate: Callable[[Any], bool] = lambda model: isinstance(model, cls)  # noqa: E731
    else:
        predicate = match
    _REGISTRY.insert(0, (predicate, factory))


def _statsmodels_adapter(model: Any) -> BaseAdapter:
    inner = getattr(model, "model", None)
    if inner is None or not hasattr(model, "params"):
        raise UnsupportedModelError(
            f"Got an unfitted statsmodels `{type(model).__name__}`; pass the result of `.fit()`."
        )
    from statsmodels.genmod.generalized_linear_model import GLM
    from statsmodels.regression.mixed_linear_model import MixedLM

    if isinstance(inner, GLM):
        return StatsmodelsGLMAdapter(model)
    if isinstance(inner, MixedLM):
        return StatsmodelsMixedAdapter(model)
    if type(inner).__name__ in {"OLS", "WLS"} and hasattr(model, "fittedvalues"):
        if "Regularized" in type(model).__name__:
            raise UnsupportedModelError("Regularized statsmodels fits are not supported yet.")
        return StatsmodelsLinearAdapter(model)
    raise UnsupportedModelError(
        f"statsmodels `{type(inner).__name__}` models are not supported yet. "
        "Supported: OLS, WLS, GLM and MixedLM results."
    )


def as_adapter(
    model: Model, X: ArrayLike | None = None, y: ArrayLike | None = None
) -> BaseAdapter:
    """Wrap ``model`` in the adapter that knows how to read it.

    Parameters
    ----------
    model
        A fitted statsmodels result, a fitted scikit-learn estimator, an
        existing adapter, or any object implementing
        :class:`~checkmodel.ModelAdapter`.
    X, y
        Training data; required for scikit-learn models, ignored otherwise.

    Raises
    ------
    UnsupportedModelError
        If no adapter can read the model.
    """
    if isinstance(model, BaseAdapter):
        return model
    for predicate, factory in _REGISTRY:
        if predicate(model):
            return factory(model, X=X, y=y)
    module = type(model).__module__ or ""
    if module.startswith("statsmodels"):
        if X is not None or y is not None:
            warnings.warn(
                "`X` and `y` are ignored for statsmodels models, which store their own data.",
                UserWarning,
                stacklevel=3,
            )
        return _statsmodels_adapter(model)
    if module.startswith("sklearn"):
        name = type(model).__name__
        if name not in _SKLEARN_LINEAR | _SKLEARN_LOGISTIC:
            raise UnsupportedModelError(
                f"scikit-learn `{name}` is not supported. Supported estimators: "
                + ", ".join(sorted(_SKLEARN_LINEAR | _SKLEARN_LOGISTIC))
                + ". For a Pipeline, pass its final estimator and the transformed X."
            )
        return SklearnAdapter(model, X, y)
    if isinstance(model, ModelAdapter):
        return ProtocolAdapter(model)
    raise UnsupportedModelError(
        f"Don't know how to read a `{type(model).__name__}`. Pass a statsmodels OLS/WLS/GLM "
        "result, a scikit-learn linear model, or an object implementing `ModelAdapter`."
    )
