"""Serial correlation of residuals (Durbin-Watson)."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy import integrate, stats

from .._introspection import as_adapter
from .._types import ArrayLike, CheckNotApplicableError, FloatArray, Model
from ._base import CheckResult, format_p

__all__ = ["AutocorrelationResult", "check_autocorrelation"]

Method = Literal["auto", "exact", "normal", "bootstrap"]

_EXACT_MAX_N = 2000


@dataclass(repr=False)
class AutocorrelationResult(CheckResult):
    """Result of :func:`check_autocorrelation`. ``statistic`` is the DW statistic."""

    key = "autocorrelation"
    title = "Autocorrelation"


def _durbin_watson(e: FloatArray) -> float:
    return float(np.sum(np.diff(e) ** 2) / np.sum(e**2))


def _prob_quadratic_form_negative(lam: FloatArray) -> float:
    """P(sum_j lam_j * z_j^2 < 0) for iid standard normal z (Imhof, 1961)."""

    def integrand(u: float) -> float:
        lu = lam * u
        theta = 0.5 * np.sum(np.arctan(lu))
        log_rho = 0.25 * np.sum(np.log1p(lu**2))
        return float(np.sin(theta) * np.exp(-log_rho) / u)

    value, _ = integrate.quad(integrand, 0.0, np.inf, limit=2000, epsabs=1e-12, epsrel=1e-10)
    prob = 0.5 - value / np.pi
    # Imhof's formula loses precision in the far tails (p ~ 1e-15 and below);
    # clip that rounding noise instead of reporting a failure.
    if -1e-10 < prob < 0.0 or 1.0 < prob < 1.0 + 1e-10:
        prob = min(max(prob, 0.0), 1.0)
    return float(prob)


def _p_exact(dw: float, X: FloatArray, n: int) -> float:
    # lmtest::dwtest(exact = TRUE): DW = e'Ae / e'e with e = M eps, so
    # P(DW < d) = P(sum (nu_j - d) z_j^2 < 0), nu_j the non-null eigenvalues of MA.
    Q, _ = np.linalg.qr(X)
    M = np.eye(n) - Q @ Q.T
    A = 2.0 * np.eye(n) - np.eye(n, k=1) - np.eye(n, k=-1)
    A[0, 0] = A[-1, -1] = 1.0
    ev = np.sort(np.linalg.eigvalsh(M @ A @ M))[::-1][: n - X.shape[1]]
    ev = ev[ev > 1e-10]
    p_low = _prob_quadratic_form_negative(ev - dw)
    return 2.0 * min(p_low, 1.0 - p_low)


def _p_normal(dw: float, X: FloatArray, n: int) -> float:
    # lmtest::dwtest(exact = FALSE): normal approximation with exact moments.
    k = X.shape[1]
    AX = np.empty_like(X)
    AX[1:-1] = -X[:-2] + 2.0 * X[1:-1] - X[2:]
    AX[0] = X[0] - X[1]
    AX[-1] = X[-1] - X[-2]
    Q1 = np.linalg.inv(X.T @ X)
    XAXQ = X.T @ AX @ Q1
    P = 2.0 * (n - 1) - np.trace(XAXQ)
    Qm = 2.0 * (3 * n - 4) - 2.0 * np.trace(AX.T @ AX @ Q1) + np.trace(XAXQ @ XAXQ)
    dmean = P / (n - k)
    dvar = 2.0 / ((n - k) * (n - k + 2)) * (Qm - P * dmean)
    return float(2.0 * stats.norm.sf(abs(dw - dmean), scale=np.sqrt(dvar)))


def _p_bootstrap(
    dw: float, X: FloatArray, e: FloatArray, mu: FloatArray, nsim: int, seed: int | None
) -> float:
    # performance::check_autocorrelation(): resample residuals, refit, compare DW.
    rng = np.random.default_rng(seed)
    n = e.shape[0]
    Y = rng.choice(e, size=(n, nsim), replace=True) + mu[:, None]
    coef, *_ = np.linalg.lstsq(X, Y, rcond=None)
    E = Y - X @ coef
    dws = np.sum(np.diff(E, axis=0) ** 2, axis=0) / np.sum(E**2, axis=0)
    p = float(np.mean(dw < dws))
    return 2.0 * min(p, 1.0 - p)


def check_autocorrelation(
    model: Model,
    X: ArrayLike | None = None,
    y: ArrayLike | None = None,
    *,
    alpha: float = 0.05,
    method: Method = "auto",
    nsim: int = 1000,
    random_state: int | None = None,
) -> AutocorrelationResult:
    """Test residuals of a linear model for first-order autocorrelation.

    Computes the Durbin-Watson statistic with a two-sided p-value. Rows are
    taken in the order the model was fitted on, so sort time series by time
    before fitting.

    Parameters
    ----------
    model
        A fitted linear model.
    X, y
        Training data; required for scikit-learn models.
    alpha
        Significance level. The check fails when ``p < alpha``.
    method
        How to compute the p-value:

        * ``"auto"`` (default): ``"exact"`` below 100 observations,
          ``"normal"`` otherwise, which is ``lmtest::dwtest()``'s rule.
        * ``"exact"``: exact distribution under normal errors (Imhof's
          method), as ``lmtest::dwtest(exact = TRUE)``.
        * ``"normal"``: normal approximation with exact moments.
        * ``"bootstrap"``: residual bootstrap with ``nsim`` draws, which is
          what ``performance::check_autocorrelation()`` uses. Random, so pass
          ``random_state`` for reproducible results.
    nsim
        Bootstrap replicates (``method="bootstrap"`` only).
    random_state
        Seed for the bootstrap.

    Returns
    -------
    AutocorrelationResult

    Raises
    ------
    CheckNotApplicableError
        For non-Gaussian models.
    """
    adapter = as_adapter(model, X, y)
    if not adapter.info.is_linear or adapter.info.is_mixed:
        raise CheckNotApplicableError(
            "The Durbin-Watson test is only implemented for linear models without random effects."
        )
    sw = np.sqrt(adapter.weights)
    e = sw * adapter.resid
    Xw = sw[:, None] * adapter.X
    n = e.shape[0]
    dw = _durbin_watson(e)
    chosen = method
    if chosen == "auto":
        chosen = "exact" if n < 100 else "normal"
    if n < 3:
        warnings.warn("Not enough observations to compute a p-value; set to 1.", stacklevel=2)
        p = 1.0
    elif chosen == "exact":
        if n > _EXACT_MAX_N:
            raise ValueError(
                f"The exact p-value needs an n x n eigendecomposition; it is limited to "
                f"{_EXACT_MAX_N} observations (got {n}). Use method='normal'."
            )
        p = _p_exact(dw, Xw, n)
        if not 0.0 <= p <= 1.0 or np.isnan(p):
            warnings.warn(
                "Exact p-value could not be computed; using the normal approximation.",
                stacklevel=2,
            )
            chosen, p = "normal", _p_normal(dw, Xw, n)
    elif chosen == "normal":
        p = _p_normal(dw, Xw, n)
    elif chosen == "bootstrap":
        p = _p_bootstrap(dw, Xw, e, sw * adapter.mu, nsim, random_state)
    else:
        raise ValueError(f"Unknown method {method!r}.")
    passed = p >= alpha
    if passed:
        message = f"OK: Residuals appear to be independent and not autocorrelated ({format_p(p)})."
    else:
        message = f"Warning: Autocorrelated residuals detected ({format_p(p)})."
    return AutocorrelationResult(
        passed=passed,
        method=f"Durbin-Watson ({chosen})",
        statistic=dw,
        p_value=float(p),
        message=message,
        adapter=adapter,
    )
