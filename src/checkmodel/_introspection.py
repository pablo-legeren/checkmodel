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
    CheckNotApplicableError,
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

    @cached_property
    def aliased(self) -> list[int]:
        """Positions of the design-matrix columns that are linear combinations of earlier ones.

        These are the coefficients R reports as ``NA``. Going left to right, a
        column is aliased when adding it does not raise the rank, judged by a
        pivoted QR decomposition with the tolerance :attr:`hat` uses. Empty for
        a full-rank design.
        """
        if self.rank == self.n_params:
            return []
        from scipy import linalg

        wX = self._wX
        tol = max(wX.shape) * np.finfo(float).eps

        def rank(cols: list[int]) -> int:
            R, _ = linalg.qr(wX[:, cols], mode="r", pivoting=True)
            d = np.abs(np.diag(R))
            return int(np.sum(d > d.max() * tol)) if d.max() > 0 else 0

        kept: list[int] = []
        aliased: list[int] = []
        for j in range(wX.shape[1]):
            (kept if rank([*kept, j]) > len(kept) else aliased).append(j)
        return aliased

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
        if info.is_binomial:
            # As binomial()$mu.eta in R: keep fitted probabilities of separated fits
            # off the boundary, where d mu / d eta and the variance are both 0.
            eps = np.finfo(float).eps
            mu = np.clip(mu, eps, 1.0 - eps)
        try:
            num, var = w * self._dmu_deta(mu) ** 2, self._variance(mu)
        except UnsupportedModelError:
            raise UnsupportedModelError(
                f"Cannot derive working weights for family={info.family!r}, "
                f"link={info.link!r}; the adapter must override `working_weights` or `hat`."
            ) from None
        return np.divide(num, var, out=np.zeros_like(num), where=var > 0)

    def _linkinv(self, eta: FloatArray) -> FloatArray:
        link = self.info.link
        if link == "identity":
            return eta
        if link == "logit":
            return np.asarray(special.expit(eta))
        if link == "probit":
            return np.asarray(stats.norm.cdf(eta))
        if link == "cloglog":
            return -np.expm1(-np.exp(eta))
        if link == "log":
            return np.exp(eta)
        if link == "inverse":
            return 1.0 / eta
        if link == "sqrt":
            return eta**2
        raise UnsupportedModelError(f"Unknown link function {link!r}.")

    def _dmu_deta(self, mu: FloatArray) -> FloatArray:
        link = self.info.link
        if link == "identity":
            return np.ones_like(mu)
        if link == "logit":
            return mu * (1.0 - mu)
        if link == "probit":
            return np.asarray(stats.norm.pdf(stats.norm.ppf(mu)))
        if link == "cloglog":
            return -(1.0 - mu) * np.log1p(-mu)
        if link == "log":
            return mu
        if link == "inverse":
            return -(mu**2)
        if link == "sqrt":
            return 2.0 * np.sqrt(mu)
        raise UnsupportedModelError(f"Unknown link function {link!r}.")

    def _variance(self, mu: FloatArray) -> FloatArray:
        fam = self.info.family
        if fam == "gaussian":
            return np.ones_like(mu)
        if fam == "binomial":
            return mu * (1.0 - mu)
        if fam == "poisson":
            return mu
        if fam == "negbin":
            return mu + mu**2 / self.negbin_theta
        if fam == "gamma":
            return mu**2
        if fam == "inverse.gaussian":
            return mu**3
        raise UnsupportedModelError(f"No variance function for family={fam!r}.")

    def _unit_deviance(self, mu: FloatArray) -> FloatArray:
        fam, y = self.info.family, self.y
        if fam == "gaussian":
            return (y - mu) ** 2
        if fam == "binomial":
            unit = -2.0 * (special.xlogy(y, mu) + special.xlogy(1.0 - y, 1.0 - mu))
            return unit + 2.0 * (special.xlogy(y, y) + special.xlogy(1.0 - y, 1.0 - y))
        if fam == "poisson":
            return 2.0 * (special.xlogy(y, y / mu) - (y - mu))
        if fam == "negbin":
            theta = self.negbin_theta
            return 2.0 * (
                special.xlogy(y, y / mu) - (y + theta) * np.log((y + theta) / (mu + theta))
            )
        if fam == "gamma":
            return 2.0 * (-np.log(y / mu) + (y - mu) / mu)
        if fam == "inverse.gaussian":
            return (y - mu) ** 2 / (y * mu**2)
        raise UnsupportedModelError(f"No deviance for family={fam!r}.")

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
        if self.info.family == "gaussian":
            return self.pearson_residuals
        unit = self._unit_deviance(self.mu)
        return np.sign(self.y - self.mu) * np.sqrt(np.clip(self.weights * unit, 0.0, None))

    @cached_property
    def scale(self) -> float:
        """Dispersion: 1 for binomial and count models, Pearson chi² / df otherwise."""
        if self.info.family in {"binomial", "poisson", "negbin"}:
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
        wX = self._wX
        try:
            U, s, _ = np.linalg.svd(wX, full_matrices=False)
        except np.linalg.LinAlgError:  # LAPACK's SVD occasionally fails to converge
            from scipy import linalg

            Q, R, _ = linalg.qr(wX, mode="economic", pivoting=True)
            d = np.abs(np.diag(R))
            rank = int(np.sum(d > d.max() * max(wX.shape) * np.finfo(float).eps))
            return np.asarray(np.sum(Q[:, :rank] ** 2, axis=1))
        keep = s > s.max() * max(wX.shape) * np.finfo(float).eps
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
        if info.family == "negbin":
            theta = self.negbin_theta
            return float(np.sum(w * stats.nbinom.logpmf(np.round(y), theta, theta / (theta + mu))))
        if info.family == "gamma":
            # Gamma()$aic in R: the dispersion is estimated as deviance / sum(w).
            disp = self._deviance_at(mu) / np.sum(w)
            return float(np.sum(w * stats.gamma.logpdf(y, 1.0 / disp, scale=mu * disp)))
        if info.family == "inverse.gaussian":
            disp = self._deviance_at(mu) / np.sum(w)
            return float(
                -0.5 * (np.sum(w) * (np.log(disp * 2 * np.pi) + 1) + 3 * np.sum(np.log(y) * w))
            )
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
        return np.asarray(self._linkinv(np.zeros(self.n_obs)), dtype=float)

    @cached_property
    def loglik_null(self) -> float | None:
        """Log-likelihood of the intercept-only model."""
        return self._loglik_at(self._null_mu)

    def _deviance_at(self, mu: FloatArray) -> float:
        return float(np.sum(self.weights * self._unit_deviance(mu)))

    @cached_property
    def deviance(self) -> float:
        """Residual deviance."""
        return self._deviance_at(self.mu)

    @cached_property
    def null_deviance(self) -> float:
        """Deviance of the intercept-only model."""
        return self._deviance_at(self._null_mu)

    # ------------------------------------------------------------- simulation
    @property
    def simulation_weights(self) -> FloatArray:
        """Weights that scale each observation's variance (R's prior weights)."""
        return self.weights

    @property
    def frequency_weights(self) -> FloatArray:
        """How many identical observations each row stands for."""
        return np.ones(self.n_obs)

    @cached_property
    def binomial_trials(self) -> FloatArray:
        """Number of trials per observation of a binomial model (R's prior weights)."""
        trials = self.simulation_weights
        if np.any(trials % 1 != 0):
            raise CheckNotApplicableError(
                "Cannot simulate from a binomial model with non-integer prior weights."
            )
        return trials

    @property
    def negbin_theta(self) -> float:
        """Shape ``theta`` of a negative binomial model, with variance ``mu + mu^2/theta``."""
        raise CheckNotApplicableError(
            "The adapter does not expose the negative binomial dispersion (`negbin_theta`)."
        )

    @cached_property
    def gamma_shape(self) -> float:
        """Maximum-likelihood shape of a Gamma model, as ``MASS::gamma.shape()``."""
        y, A, u, f = self.y, self.simulation_weights, self.mu, self.frequency_weights
        dbar = self.deviance / self.df_resid
        alpha = (6.0 + 2.0 * dbar) / (dbar * (6.0 + dbar))
        fixed = -y / u - np.log(u) + np.log(A) + 1.0 + np.log(y + (y == 0))
        eps_max = np.finfo(float).eps ** 0.25
        for _ in range(10):  # MASS's it.lim
            score = np.sum(f * A * (fixed + np.log(alpha) - special.digamma(A * alpha)))
            info = np.sum(f * A * (A * special.polygamma(1, A * alpha) - 1.0 / alpha))
            eps = score / info
            alpha += eps
            if abs(eps) <= eps_max:
                break
        return float(alpha)

    def simulate(self, n_sim: int, rng: np.random.Generator) -> FloatArray:
        """Draw new responses from the fitted model, as R's ``simulate()``.

        Parameters are held at their estimates. Binomial responses are drawn
        as proportions of ``binomial_trials``; Gamma models use the
        maximum-likelihood shape and inverse Gaussian models the Pearson
        dispersion, as R does.

        Parameters
        ----------
        n_sim
            Number of replicated datasets.
        rng
            Random number generator.

        Returns
        -------
        numpy.ndarray
            Shape ``(n_sim, n_obs)``: one replicated response per row.
        """
        fam, mu, size = self.info.family, self.mu, (n_sim, self.n_obs)
        w = self.simulation_weights
        if fam == "gaussian":
            with np.errstate(divide="ignore"):
                sd = np.sqrt(self.scale / w)
            return mu + rng.standard_normal(size) * sd
        if fam == "binomial":
            trials = self.binomial_trials
            draws = rng.binomial(trials.astype(np.int64), np.clip(mu, 0.0, 1.0), size=size)
            return np.asarray(draws / trials, dtype=float)
        if fam == "poisson":
            return rng.poisson(mu, size=size).astype(float)
        if fam == "negbin":
            theta = self.negbin_theta
            return rng.negative_binomial(theta, theta / (theta + mu), size=size).astype(float)
        if fam == "gamma":
            shape = self.gamma_shape * w
            return rng.gamma(shape, mu / shape, size=size)
        if fam == "inverse.gaussian":
            return rng.wald(mu, w / self.scale, size=size)
        raise CheckNotApplicableError(
            f"Simulating new responses is not implemented for family={fam!r}."
        )


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
        # Fisher (IRLS) weights, as R's hatvalues(); statsmodels' default uses the
        # observed information, which differs for non-canonical links.
        return np.asarray(self.model.get_hat_matrix_diag(observed=False), dtype=float)

    @cached_property
    def vcov_unscaled(self) -> FloatArray:
        return np.asarray(self.model.normalized_cov_params, dtype=float)

    @cached_property
    def loglik(self) -> float | None:
        if self.info.family in ("gamma", "inverse.gaussian"):
            # R's likelihood (Gamma()$aic) estimates the dispersion as deviance / n;
            # statsmodels uses the Pearson dispersion.
            return self._loglik_at(self.mu)
        return float(self.model.llf)

    @cached_property
    def loglik_null(self) -> float | None:
        if self.info.family in ("gamma", "inverse.gaussian"):
            return self._loglik_at(np.asarray(self.model.null, dtype=float))
        return float(self.model.llnull)

    @cached_property
    def deviance(self) -> float:
        return float(self.model.deviance)

    @cached_property
    def null_deviance(self) -> float:
        return float(self.model.null_deviance)

    @property
    def simulation_weights(self) -> FloatArray:
        # statsmodels' var_weights are R's prior weights; freq_weights replicate rows.
        return np.broadcast_to(np.asarray(self._sm.var_weights, dtype=float), (self.n_obs,)).copy()

    @property
    def frequency_weights(self) -> FloatArray:
        return np.broadcast_to(
            np.asarray(self._sm.freq_weights, dtype=float), (self.n_obs,)
        ).copy()

    @cached_property
    def binomial_trials(self) -> FloatArray:
        n_trials = np.asarray(getattr(self._sm, "n_trials", 1.0), dtype=float)
        trials = np.broadcast_to(n_trials, (self.n_obs,)) * self.simulation_weights
        if np.any(trials % 1 != 0):
            raise CheckNotApplicableError(
                "Cannot simulate from a binomial model with non-integer prior weights."
            )
        return trials

    @property
    def negbin_theta(self) -> float:
        return 1.0 / float(self._sm.family.alpha)


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

    def simulate(self, n_sim: int, rng: np.random.Generator) -> FloatArray:
        # simulate.merMod(re.form = NA), as DHARMa does: new group effects, new residuals.
        fixed = self.X @ np.asarray(self.model.fe_params, dtype=float)
        chol = np.linalg.cholesky(self.cov_re + 1e-12 * np.eye(len(self.cov_re)))
        out = np.empty((n_sim, self.n_obs))
        for s in range(n_sim):
            values = fixed + rng.normal(scale=self.sigma, size=self.n_obs)
            for rows in self._sm.row_indices.values():
                values[rows] += self.Z[rows] @ (chol @ rng.standard_normal(chol.shape[0]))
            out[s] = values
        return out


# ----------------------------------------------------- statsmodels discrete models
_SM_DISCRETE = {
    "Logit": ("binomial", "logit"),
    "Probit": ("binomial", "probit"),
    "Poisson": ("poisson", "log"),
    "NegativeBinomial": ("negbin", "log"),
}


class StatsmodelsDiscreteAdapter(_StatsmodelsMixin, BaseAdapter):
    """Adapter for statsmodels ``Logit``, ``Probit``, ``Poisson`` and ``NegativeBinomial``.

    These are the maximum-likelihood counterparts of the GLMs, so leverage,
    residuals and deviances follow ``glm()`` in R; a negative binomial (NB2)
    fit is treated like ``MASS::glm.nb()``: a GLM with ``theta = 1 / alpha``.
    Offsets and exposure are honoured.
    """

    def __init__(self, model: Model) -> None:
        super().__init__(model)
        method = getattr(self._sm, "loglike_method", "nb2")
        if type(self._sm).__name__ == "NegativeBinomial" and method not in ("nb2", "geometric"):
            raise UnsupportedModelError(
                f"Negative binomial models with loglike_method={method!r} are not supported; "
                "use 'nb2' (the variance mu + alpha * mu^2 of R's glm.nb) or 'geometric'."
            )

    def model_type(self) -> ModelType:
        name = type(self._sm).__name__
        family, link = _SM_DISCRETE[name]
        endog = np.asarray(self._sm.endog, dtype=float)
        return ModelType(
            family=family,
            link=link,
            backend="statsmodels",
            model_class=name,
            is_bernoulli=family == "binomial" and bool(np.all((endog == 0) | (endog == 1))),
        )

    def get_response(self) -> FloatArray:
        return np.asarray(self._sm.endog, dtype=float)

    def get_fitted(self) -> FloatArray:
        # `fittedvalues` is the linear predictor for these models; predict() gives the mean.
        return np.asarray(self.model.predict(), dtype=float)

    @property
    def negbin_theta(self) -> float:
        if getattr(self._sm, "loglike_method", "nb2") == "geometric":
            return 1.0
        return 1.0 / float(np.asarray(self.model.params)[-1])

    @cached_property
    def loglik(self) -> float | None:
        return float(self.model.llf)

    @cached_property
    def loglik_null(self) -> float | None:
        return float(self.model.llnull)

    @property
    def loglik_df(self) -> int:
        # Coefficients plus the estimated alpha of a negative binomial model.
        return self.rank + int(getattr(self._sm, "k_extra", 0))

    def _offset(self) -> FloatArray:
        total = np.zeros(self.n_obs)
        for name in ("offset", "exposure"):  # statsmodels keeps log(exposure)
            value = getattr(self._sm, name, None)
            if value is not None:
                total = total + np.asarray(value, dtype=float)
        return total

    @cached_property
    def _null_mu(self) -> FloatArray:
        offset = self._offset()
        if not np.any(offset):
            return super()._null_mu
        if not self.has_intercept:
            return np.asarray(self._linkinv(offset), dtype=float)
        # The intercept-only GLM with the same offset, as R's null deviance.
        import statsmodels.api as sm

        links = sm.families.links
        family = {
            "logit": lambda: sm.families.Binomial(links.Logit()),
            "probit": lambda: sm.families.Binomial(links.Probit()),
            "poisson": lambda: sm.families.Poisson(),
            "negbin": lambda: sm.families.NegativeBinomial(alpha=1.0 / self.negbin_theta),
        }[self.info.link if self.info.is_binomial else self.info.family]()
        null = sm.GLM(self.y, np.ones((self.n_obs, 1)), family=family, offset=offset).fit()
        return np.asarray(null.mu, dtype=float)


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
_SKLEARN_GLM = {"PoissonRegressor": "poisson", "GammaRegressor": "gamma"}


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
        if name in _SKLEARN_GLM:
            return ModelType(
                family=_SKLEARN_GLM[name],
                link="log",
                backend="sklearn",
                model_class=name,
                is_penalized=float(getattr(self.model, "alpha", 1.0)) > 0,
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

    _HOOKS = ("column_names", "intercept_index", "terms", "prior_weights", "index", "simulate")

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
    if type(inner).__name__ in _SM_DISCRETE and type(inner).__module__.startswith(
        "statsmodels.discrete"
    ):
        if "L1" in type(model).__name__ or "Regularized" in type(model).__name__:
            raise UnsupportedModelError("Regularized statsmodels fits are not supported yet.")
        return StatsmodelsDiscreteAdapter(model)
    if type(inner).__name__ in {"OLS", "WLS"} and hasattr(model, "fittedvalues"):
        if "Regularized" in type(model).__name__:
            raise UnsupportedModelError("Regularized statsmodels fits are not supported yet.")
        return StatsmodelsLinearAdapter(model)
    raise UnsupportedModelError(
        f"statsmodels `{type(inner).__name__}` models are not supported yet. "
        "Supported: OLS, WLS, GLM, MixedLM, Logit, Probit, Poisson and NegativeBinomial results."
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
        supported = _SKLEARN_LINEAR | _SKLEARN_LOGISTIC | set(_SKLEARN_GLM)
        if name not in supported:
            raise UnsupportedModelError(
                f"scikit-learn `{name}` is not supported. Supported estimators: "
                + ", ".join(sorted(supported))
                + ". For a Pipeline, pass its final estimator and the transformed X."
            )
        return SklearnAdapter(model, X, y)
    if isinstance(model, ModelAdapter):
        return ProtocolAdapter(model)
    from ._bayes import looks_bayesian

    if looks_bayesian(model):
        raise UnsupportedModelError(
            "This is a Bayesian model. It works with check_model(), check_predictions(), "
            "check_convergence(), r2(), r2_bayes(), looic(), model_performance() and "
            "compare_performance(); the residual-based checks need a frequentist fit."
        )
    raise UnsupportedModelError(
        f"Don't know how to read a `{type(model).__name__}`. Pass a fitted statsmodels or "
        "scikit-learn model, or an object implementing `ModelAdapter`."
    )
