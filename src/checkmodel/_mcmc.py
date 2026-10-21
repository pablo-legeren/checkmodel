"""MCMC diagnostics and leave-one-out estimates, ported from R's ``posterior`` and ``loo``.

Everything here works on plain numpy arrays so results match R draw for draw:

* :func:`rhat`, :func:`ess_bulk`, :func:`ess_tail` follow ``posterior`` 1.7
  (Vehtari, Gelman, Simpson, Carpenter & Bürkner, 2021);
* :func:`psis_loo` and :func:`waic` follow ``loo`` 2.10 (Vehtari, Gelman &
  Gabry, 2017), including per-observation relative efficiencies;
* :func:`hdi` follows ``bayestestR``.

Draw arrays for one quantity are shaped ``(chains, draws)``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import fft, special, stats

from ._types import FloatArray

__all__ = [
    "LooResult",
    "ess_bulk",
    "ess_mean",
    "ess_tail",
    "hdi",
    "mad",
    "psis_loo",
    "relative_eff",
    "rhat",
    "waic",
]


# ----------------------------------------------------------------- utilities
def _as_chains(x: FloatArray) -> FloatArray:
    x = np.asarray(x, dtype=float)
    return x.reshape(1, -1) if x.ndim == 1 else x


def _split_chains(x: FloatArray) -> FloatArray:
    """Split each chain in two halves (dropping the middle draw of odd lengths)."""
    x = _as_chains(x)
    n = x.shape[1]
    if n == 1:
        return x
    half = n / 2
    first = x[:, : math.floor(half)]
    second = x[:, math.ceil(half + 1) - 1 :]
    return np.concatenate([first, second], axis=0)


def _is_constant(x: FloatArray) -> bool:
    return bool(np.ptp(x) <= np.finfo(float).eps) if x.size else True


def _should_return_nan(x: FloatArray) -> bool:
    return not np.all(np.isfinite(x)) or _is_constant(x)


def _z_scale(x: FloatArray) -> FloatArray:
    """Rank-normalize draws (average ties, offset 3/8)."""
    r = stats.rankdata(x, method="average").reshape(x.shape)
    return np.asarray(stats.norm.ppf((r - 3 / 8) / (x.size - 2 * (3 / 8) + 1)))


def _autocovariance(x: FloatArray) -> FloatArray:
    n = x.shape[0]
    var = np.var(x, ddof=1)
    if var == 0:
        return np.zeros(n)
    size = fft.next_fast_len(2 * n)
    centred = x - x.mean()
    spectrum = fft.rfft(centred, size)
    ac = fft.irfft(spectrum * np.conj(spectrum), size)[:n].real
    return np.asarray(ac / ac[0] * var * (n - 1) / n)


def _rhat_basic(x: FloatArray) -> float:
    if _should_return_nan(x):
        return float("nan")
    chains, n = x.shape
    chain_mean = x.mean(axis=1)
    chain_var = x.var(axis=1, ddof=1)
    var_between = n * np.var(chain_mean, ddof=1) if chains > 1 else 0.0
    var_within = chain_var.mean()
    return float(np.sqrt((var_between / var_within + n - 1) / n))


def _ess_basic(x: FloatArray) -> float:
    """``posterior:::.ess()``: Geyer's initial monotone sequence over all chains."""
    chains, n = x.shape
    if n < 3 or _should_return_nan(x):
        return float("nan")
    acov = np.stack([_autocovariance(x[c]) for c in range(chains)], axis=1)
    acov_means = acov.mean(axis=1)
    mean_var = acov_means[0] * n / (n - 1)
    var_plus = mean_var * (n - 1) / n
    if chains > 1:
        var_plus += np.var(x.mean(axis=1), ddof=1)
    rho = np.zeros(n)
    t = 0
    rho_even = 1.0
    rho[0] = rho_even
    rho_odd = 1 - (mean_var - acov_means[1]) / var_plus
    rho[1] = rho_odd
    while t < n - 5 and not np.isnan(rho_even + rho_odd) and rho_even + rho_odd > 0:
        t += 2
        rho_even = 1 - (mean_var - acov_means[t]) / var_plus
        rho_odd = 1 - (mean_var - acov_means[t + 1]) / var_plus
        if rho_even + rho_odd >= 0:
            rho[t] = rho_even
            rho[t + 1] = rho_odd
    max_t = t
    if rho_even > 0:
        rho[max_t] = rho_even
    t = 0
    while t <= max_t - 4:
        t += 2
        if rho[t] + rho[t + 1] > rho[t - 2] + rho[t - 1]:
            rho[t] = (rho[t - 2] + rho[t - 1]) / 2
            rho[t + 1] = rho[t]
    total = chains * n
    tau = -1 + 2 * rho[:max_t].sum() + rho[max_t]
    tau = max(tau, 1 / math.log10(total))
    return float(total / tau)


# ------------------------------------------------------------- diagnostics
def rhat(x: FloatArray) -> float:
    """Rank-normalized split R-hat: the larger of the bulk and tail versions."""
    x = _as_chains(x)
    split = _split_chains(x)
    bulk = _rhat_basic(_z_scale(split))
    tail = _rhat_basic(_z_scale(_split_chains(np.abs(x - np.median(x)))))
    return max(bulk, tail)


def ess_bulk(x: FloatArray) -> float:
    """Bulk effective sample size (rank-normalized)."""
    return _ess_basic(_z_scale(_split_chains(_as_chains(x))))


def ess_mean(x: FloatArray) -> float:
    """Effective sample size of the mean."""
    return _ess_basic(_split_chains(_as_chains(x)))


def _ess_quantile(x: FloatArray, prob: float) -> float:
    if _should_return_nan(x):
        return float("nan")
    indicator = (x <= np.quantile(x, prob)).astype(float)
    return _ess_basic(_split_chains(indicator))


def ess_tail(x: FloatArray) -> float:
    """Tail effective sample size: the smaller of the 5% and 95% quantile ESS."""
    x = _as_chains(x)
    return min(_ess_quantile(x, 0.05), _ess_quantile(x, 0.95))


def mad(x: FloatArray) -> float:
    """Median absolute deviation scaled to the normal SD, as R's ``mad()``."""
    x = np.asarray(x, dtype=float)
    return float(1.4826 * np.median(np.abs(x - np.median(x))))


def hdi(x: FloatArray, ci: float = 0.95) -> tuple[float, float]:
    """Highest density interval of draws, as ``bayestestR::hdi()``."""
    xs = np.sort(np.asarray(x, dtype=float).ravel())
    n = xs.shape[0]
    window = math.ceil(ci * n)
    if window < 2 or n - window < 1:
        return float("nan"), float("nan")
    widths = xs[window:] - xs[: n - window]
    candidates = np.flatnonzero(widths == float(np.min(widths)))
    if candidates.size > 1 and np.any(np.diff(candidates) != 1):
        i = int(np.max(candidates))  # separate segments: R takes the rightmost
    else:
        i = math.floor(float(np.mean(candidates)))
    return float(xs[i]), float(xs[i + window])


# ---------------------------------------------------------------- PSIS-LOO
def relative_eff(log_lik: FloatArray) -> FloatArray:
    """Relative efficiency of ``exp(log_lik)`` per observation, as ``loo::relative_eff()``.

    ``log_lik`` is shaped ``(chains, draws, observations)``.
    """
    chains, draws, n_obs = log_lik.shape
    lik = np.exp(log_lik)
    return np.array([ess_mean(lik[:, :, i]) for i in range(n_obs)]) / (chains * draws)


def _gpdfit(x: FloatArray) -> tuple[float, float]:
    """Generalized Pareto fit of Zhang & Stephens (2009) with a weak prior on k."""
    x = np.asarray(x, dtype=float)
    n = x.shape[0]
    prior = 3.0
    m = 30 + math.floor(math.sqrt(n))
    jj = np.arange(1, m + 1)
    xstar = x[math.floor(n / 4 + 0.5) - 1]
    if not xstar > x[0]:
        return float("nan"), float("nan")
    theta = 1 / x[-1] + (1 - np.sqrt(m / (jj - 0.5))) / prior / xstar
    k = np.log1p(-np.outer(theta, x)).mean(axis=1)
    l_theta = n * (np.log(-theta / k) - k - 1)
    w_theta = np.exp(l_theta - special.logsumexp(l_theta))
    theta_hat = float(np.sum(theta * w_theta))
    k_hat = float(np.mean(np.log1p(-theta_hat * x)))
    sigma_hat = -k_hat / theta_hat
    k_hat = (k_hat * n + 0.5 * 10) / (n + 10)
    if math.isnan(k_hat):
        return float("inf"), float("nan")
    return k_hat, sigma_hat


def _qgpd(p: FloatArray, sigma: float, k: float) -> FloatArray:
    if math.isnan(sigma) or sigma <= 0:
        return np.full_like(p, np.nan)
    if k == 0:
        return np.asarray(-sigma * np.log1p(-p))
    return np.asarray(sigma * np.expm1(-k * np.log1p(-p)) / k)


def _psis_one(log_ratios: FloatArray, tail_len: int) -> tuple[FloatArray, float]:
    s = log_ratios.shape[0]
    lw = log_ratios - float(np.max(log_ratios))
    khat = float("inf")
    if tail_len >= 5:
        order = np.argsort(lw, kind="stable")
        sorted_lw = lw[order]
        tail_ids = np.arange(s - tail_len, s)
        lw_tail = sorted_lw[tail_ids]
        if abs(float(np.max(lw_tail)) - float(np.min(lw_tail))) >= np.finfo(float).eps / 100:
            cutoff = float(sorted_lw[s - tail_len - 1])
            exp_cutoff = math.exp(cutoff)
            k, sigma = _gpdfit(np.exp(lw_tail) - exp_cutoff)
            if math.isnan(k):
                k = float("inf")
            if math.isfinite(k):
                p = (np.arange(1, tail_len + 1) - 0.5) / tail_len
                lw[order[tail_ids]] = np.log(_qgpd(p, sigma, k) + exp_cutoff)
            khat = k
    lw[lw > 0] = 0
    return lw + float(np.max(log_ratios)), khat


@dataclass(frozen=True)
class LooResult:
    """Pointwise and summed PSIS-LOO estimates, as ``loo::loo()``."""

    elpd_loo: float
    elpd_loo_se: float
    p_loo: float
    p_loo_se: float
    looic: float
    looic_se: float
    pareto_k: FloatArray
    pointwise_elpd: FloatArray

    @property
    def n_bad_k(self) -> int:
        """Observations whose Pareto k exceeds 0.7 (unreliable importance sampling)."""
        return int(np.sum(self.pareto_k > 0.7))


def psis_loo(log_lik: FloatArray, r_eff: FloatArray | None = None) -> LooResult:
    """Pareto-smoothed importance sampling leave-one-out cross-validation.

    Parameters
    ----------
    log_lik
        Pointwise log-likelihood, shaped ``(chains, draws, observations)``.
    r_eff
        Relative efficiency per observation; computed from the chains when
        omitted, as ``rstanarm`` does.
    """
    chains, draws, n_obs = log_lik.shape
    if r_eff is None:
        r_eff = relative_eff(log_lik)
    s = chains * draws
    ll = log_lik.reshape(s, n_obs)
    tail = np.ceil(np.minimum(0.2 * s, 3 * np.sqrt(s / r_eff))).astype(int)
    lw = np.empty_like(ll)
    pareto_k = np.empty(n_obs)
    for i in range(n_obs):
        lw[:, i], pareto_k[i] = _psis_one(-ll[:, i], int(tail[i]))
    lw = lw - special.logsumexp(lw, axis=0)
    elpd_i = special.logsumexp(ll + lw, axis=0)
    lpd_i = special.logsumexp(ll, axis=0) - math.log(s)
    p_i = lpd_i - elpd_i

    def total(v: FloatArray) -> tuple[float, float]:
        return float(v.sum()), float(math.sqrt(n_obs * np.var(v, ddof=1)))

    elpd, elpd_se = total(elpd_i)
    p, p_se = total(p_i)
    looic, looic_se = total(-2 * elpd_i)
    return LooResult(elpd, elpd_se, p, p_se, looic, looic_se, pareto_k, elpd_i)


def waic(log_lik: FloatArray) -> tuple[float, float]:
    """WAIC and its standard error, as ``loo::waic()``."""
    s = log_lik.shape[0] * log_lik.shape[1]
    ll = log_lik.reshape(s, -1)
    lpd = special.logsumexp(ll, axis=0) - math.log(s)
    p_waic = ll.var(axis=0, ddof=1)
    pointwise = -2 * (lpd - p_waic)
    return float(pointwise.sum()), float(math.sqrt(pointwise.shape[0] * np.var(pointwise, ddof=1)))
