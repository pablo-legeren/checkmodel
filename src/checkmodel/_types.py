"""Type aliases, model descriptors and the adapter protocol."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, TypeAlias, runtime_checkable

import numpy as np
import numpy.typing as npt

FloatArray: TypeAlias = npt.NDArray[np.float64]
ArrayLike: TypeAlias = npt.ArrayLike
Model: TypeAlias = Any
"""Any fitted model object `checkmodel` knows how to adapt."""


class CheckNotApplicableError(ValueError):
    """Raised when a check does not make sense for the given model.

    For example, testing residuals for normality is only meaningful for
    linear (Gaussian, identity-link) models. :func:`checkmodel.check_model`
    catches this error and simply skips the check.
    """


class UnsupportedModelError(TypeError):
    """Raised when `checkmodel` does not know how to read a model object."""


@dataclass(frozen=True)
class ModelType:
    """What kind of model an adapter wraps.

    The flags mirror ``insight::model_info()`` in R, which is what decides
    which checks and metrics apply to a model.

    Parameters
    ----------
    family
        Error distribution: ``"gaussian"``, ``"binomial"``, ``"poisson"``,
        ``"gamma"``, ``"inverse.gaussian"``, ``"negbin"`` or ``"tweedie"``.
    link
        Link function name, e.g. ``"identity"``, ``"logit"``, ``"log"``.
    backend
        Library the model comes from (``"statsmodels"``, ``"sklearn"``, ...).
    model_class
        Class name of the wrapped model, used in comparison tables.
    is_bernoulli
        Binomial model whose response is a single 0/1 outcome.
    is_penalized
        Fitted with a penalty (ridge, lasso, ...). Likelihood-based criteria
        such as AIC are not defined for these and are reported as missing.
    is_mixed
        Has random effects (a mixed model).
    """

    family: str
    link: str
    backend: str
    model_class: str
    is_bernoulli: bool = False
    is_penalized: bool = False
    is_mixed: bool = False

    @property
    def is_linear(self) -> bool:
        """Gaussian family with identity link (an ordinary linear model)."""
        return self.family == "gaussian" and self.link == "identity"

    @property
    def is_binomial(self) -> bool:
        """Binomial family (logistic, probit, ... regression)."""
        return self.family == "binomial"

    @property
    def is_count(self) -> bool:
        """Count-data family (Poisson or negative binomial)."""
        return self.family in {"poisson", "negbin"}

    @property
    def has_dispersion(self) -> bool:
        """Whether the family estimates a dispersion parameter (counts in AIC)."""
        return self.family in {"gaussian", "gamma", "inverse.gaussian"}


@runtime_checkable
class ModelAdapter(Protocol):
    """Minimal interface `checkmodel` needs to diagnose a model.

    Implement these five methods on any object (or pass a wrapper around your
    model) and every check and metric in `checkmodel` will accept it. See
    ``CONTRIBUTING.md`` for a worked example.
    """

    def model_type(self) -> ModelType:
        """Describe the model family, link and backend."""
        ...

    def get_response(self) -> FloatArray:
        """Return the observed response, recoded to 0/1 for binary models."""
        ...

    def get_fitted(self) -> FloatArray:
        """Return fitted values on the response scale."""
        ...

    def get_residuals(self) -> FloatArray:
        """Return response residuals (observed minus fitted)."""
        ...

    def get_model_matrix(self) -> FloatArray:
        """Return the design matrix, including the intercept column if any."""
        ...
