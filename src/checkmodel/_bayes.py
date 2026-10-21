"""Read Bayesian models (PyMC, bambi) into posterior draw arrays.

A fitted Bayesian model is an ArviZ ``DataTree`` (ArviZ >= 1.0) or
``InferenceData`` (older ArviZ) with ``posterior`` and ``observed_data``
groups, plus ``posterior_predictive`` and ``log_likelihood`` when available.
:func:`bayesian` gathers what checkmodel needs from it into a
:class:`BayesianModel`; with a bambi model it computes the missing pieces
itself (expected values, predictions, pointwise log-likelihood).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ._types import FloatArray, ModelType, UnsupportedModelError

__all__ = ["BayesianModel", "as_bayesian", "bayesian", "looks_bayesian"]

_BAMBI_FAMILIES = {"bernoulli": "binomial", "negativebinomial": "negbin"}


@dataclass
class BayesianModel:
    """Posterior draws of a Bayesian regression, shaped ``(chains, draws, ...)``.

    Build it with :func:`bayesian`. Attributes ending in ``_draws`` and
    ``y_rep``/``log_lik`` hold one row per observation in the last axis.

    Attributes
    ----------
    y
        Observed response (0/1 for binary models).
    mu
        Draws of the expected value of each observation (``posterior_epred``).
    family
        ``"gaussian"``, ``"binomial"``, ``"poisson"``, ...
    sigma
        Draws of the residual standard deviation (Gaussian models).
    mu_marginal
        Expected values without the group-level effects (mixed models).
    y_rep
        Posterior predictive draws.
    log_lik
        Pointwise log-likelihood draws.
    parameters
        Draws of each scalar parameter, used for convergence diagnostics.
    """

    y: FloatArray
    mu: FloatArray
    family: str
    sigma: FloatArray | None = None
    mu_marginal: FloatArray | None = None
    y_rep: FloatArray | None = None
    log_lik: FloatArray | None = None
    parameters: dict[str, FloatArray] = field(default_factory=dict)
    model_class: str = "Bayesian"
    response: str = "y"

    @property
    def n_obs(self) -> int:
        """Number of observations."""
        return int(self.y.shape[0])

    @property
    def chains(self) -> int:
        """Number of chains."""
        return int(self.mu.shape[0])

    @property
    def n_draws(self) -> int:
        """Total number of posterior draws (all chains)."""
        return int(self.mu.shape[0] * self.mu.shape[1])

    @property
    def is_mixed(self) -> bool:
        """Whether marginal (population-level) expected values are available."""
        return self.mu_marginal is not None

    @property
    def info(self) -> ModelType:
        """A :class:`~checkmodel.ModelType` describing the model."""
        link = {"gaussian": "identity", "binomial": "logit", "poisson": "log"}.get(
            self.family, "unknown"
        )
        return ModelType(
            family=self.family,
            link=link,
            backend="bayesian",
            model_class=self.model_class,
            is_bernoulli=self.family == "binomial",
            is_mixed=self.is_mixed,
        )

    @staticmethod
    def flat(draws: FloatArray) -> FloatArray:
        """Stack chains: ``(chains, draws, n) -> (chains * draws, n)``."""
        return draws.reshape(draws.shape[0] * draws.shape[1], -1)


# ----------------------------------------------------------------- DataTree
def _group(idata: Any, name: str) -> Any:
    """Return a group as an xarray Dataset, or ``None`` (DataTree or InferenceData)."""
    try:
        node = idata[name]
    except (KeyError, TypeError, AttributeError):
        node = getattr(idata, name, None)
    if node is None:
        return None
    return node.to_dataset() if hasattr(node, "to_dataset") else node


def looks_bayesian(obj: Any) -> bool:
    """Whether ``obj`` is a BayesianModel or an ArviZ object with a posterior."""
    if isinstance(obj, BayesianModel):
        return True
    try:
        return _group(obj, "posterior") is not None and _group(obj, "observed_data") is not None
    except Exception:
        return False


def _draws(data_array: Any) -> FloatArray:
    da = data_array.transpose("chain", "draw", ...)
    values = np.asarray(da.values, dtype=float)
    return values.reshape(values.shape[0], values.shape[1], -1)


def _bambi_info(model: Any) -> tuple[str, str, str]:
    # bambi >= 0.21 exposes `response_term`; older versions `response_component.term`.
    term = getattr(model, "response_term", None)
    if term is None:
        term = model.response_component.term
    response = str(term.name)
    family = str(model.family.name)
    return response, _BAMBI_FAMILIES.get(family, family), str(model.family.likelihood.parent)


def _pymc_family(model: Any, response: str) -> str | None:
    try:
        op = type(model[response].owner.op).__name__.lower()
    except Exception:
        return None
    for key, family in (("normal", "gaussian"), ("bernoulli", "binomial"), ("poisson", "poisson")):
        if key in op:
            return family
    return None


def bayesian(
    idata: Any,
    model: Any = None,
    *,
    response: str | None = None,
    mu: str = "mu",
    sigma: str = "sigma",
    family: str | None = None,
    mu_marginal: str | None = None,
) -> BayesianModel:
    """Prepare a fitted Bayesian model for checkmodel.

    Parameters
    ----------
    idata
        Result of ``pm.sample()`` or ``bambi.Model.fit()``: an ArviZ
        ``DataTree`` or ``InferenceData``.
    model
        The ``bambi.Model`` or ``pymc.Model``. With bambi, everything else is
        worked out automatically. With PyMC it is used to infer the family and
        to compute posterior predictive draws and the pointwise
        log-likelihood when ``idata`` lacks them.
    response
        Name of the observed variable; needed only when ``observed_data``
        holds several.
    mu
        PyMC only: posterior variable holding the expected value of each
        observation, e.g. ``pm.Deterministic("mu", a + b * x)``.
    sigma
        PyMC only: posterior variable holding the residual SD.
    family
        PyMC only: ``"gaussian"``, ``"binomial"`` or ``"poisson"``; inferred
        from ``model`` when given, else ``"gaussian"``.
    mu_marginal
        PyMC only: posterior variable with expected values that leave out the
        group-level effects, for marginal R² of multilevel models.

    Returns
    -------
    BayesianModel

    Examples
    --------
    >>> import bambi as bmb  # doctest: +SKIP
    >>> m = bmb.Model("Reaction ~ Days + (1|Subject)", sleepstudy)  # doctest: +SKIP
    >>> fit = bayesian(m.fit(), m)  # doctest: +SKIP
    >>> check_model(fit)  # doctest: +SKIP
    """
    if _group(idata, "posterior") is None or _group(idata, "observed_data") is None:
        raise UnsupportedModelError(
            "Expected an ArviZ DataTree/InferenceData with `posterior` and `observed_data` groups."
        )
    module = type(model).__module__ if model is not None else ""
    is_bambi = module.startswith("bambi")
    model_class = "bambi" if is_bambi else ("PyMC" if module.startswith("pymc") else "Bayesian")

    if is_bambi:
        response, family, parent = _bambi_info(model)
        # bambi names group-level parameters like "1|Subject" or "Days|Subject".
        mixed = any("|" in str(name) for name in _group(idata, "posterior").data_vars)
        with_mu = model.predict(idata, kind="response_params", inplace=False)
        mu_draws = _draws(_group(with_mu, "posterior")[parent])
        marginal = None
        if mixed:
            without = model.predict(
                idata, kind="response_params", include_group_specific=False, inplace=False
            )
            marginal = _draws(_group(without, "posterior")[parent])
        posterior = _group(idata, "posterior")
        sigma_draws = posterior["sigma"] if family == "gaussian" and "sigma" in posterior else None
        predictive = _group(idata, "posterior_predictive")
        if predictive is None or response not in predictive:
            predictive = _group(
                model.predict(idata, kind="response", inplace=False), "posterior_predictive"
            )
        loglik = _group(idata, "log_likelihood")
        if loglik is None or response not in loglik:
            loglik = _group(model.compute_log_likelihood(idata, inplace=False), "log_likelihood")
    else:
        observed = _group(idata, "observed_data")
        if response is None:
            names = list(observed.data_vars)
            if len(names) != 1:
                raise ValueError(f"`observed_data` holds {names}; pass `response=`.")
            response = str(names[0])
        posterior = _group(idata, "posterior")
        if mu not in posterior:
            raise UnsupportedModelError(
                f"No posterior variable {mu!r} with the expected value of each observation. "
                f"Add one to the PyMC model, e.g. `pm.Deterministic('{mu}', ...)`, or pass `mu=`."
            )
        mu_draws = _draws(posterior[mu])
        marginal = _draws(posterior[mu_marginal]) if mu_marginal else None
        if family is None:
            family = (_pymc_family(model, response) if model is not None else None) or "gaussian"
        sigma_draws = posterior[sigma] if family == "gaussian" and sigma in posterior else None
        predictive = _group(idata, "posterior_predictive")
        loglik = _group(idata, "log_likelihood")
        if model is not None and module.startswith("pymc"):
            import pymc as pm

            if predictive is None or response not in predictive:
                predictive = _group(
                    pm.sample_posterior_predictive(idata, model=model, progressbar=False),
                    "posterior_predictive",
                )
            if loglik is None or response not in loglik:
                computed = pm.compute_log_likelihood(
                    idata, model=model, extend_inferencedata=False, progressbar=False
                )
                # PyMC 6 returns the log-likelihood Dataset itself; older versions a tree.
                loglik = (
                    computed
                    if response in getattr(computed, "data_vars", {})
                    else _group(computed, "log_likelihood")
                )

    observed = _group(idata, "observed_data")
    y = np.asarray(observed[response].values, dtype=float).ravel()
    obs_dims = set(observed[response].dims)
    exclude = {mu, mu_marginal, "mu", "p"} if not is_bambi else {"mu", "p"}
    parameters: dict[str, FloatArray] = {}
    for name, da in posterior.data_vars.items():
        if name in exclude or obs_dims & set(da.dims):
            continue
        values = _draws(da)
        for j in range(values.shape[2]):
            label = str(name) if values.shape[2] == 1 else f"{name}[{j}]"
            parameters[label] = values[:, :, j]
    return BayesianModel(
        y=y,
        mu=mu_draws,
        family=family,
        sigma=None if sigma_draws is None else _draws(sigma_draws)[:, :, 0],
        mu_marginal=marginal,
        y_rep=_draws(predictive[response])
        if predictive is not None and response in predictive
        else None,
        log_lik=_draws(loglik[response]) if loglik is not None and response in loglik else None,
        parameters=parameters,
        model_class=model_class,
        response=str(response),
    )


def as_bayesian(obj: Any) -> BayesianModel | None:
    """Return ``obj`` as a :class:`BayesianModel`, or ``None`` if it is not Bayesian."""
    if isinstance(obj, BayesianModel):
        return obj
    if looks_bayesian(obj):
        return bayesian(obj)
    return None
