"""Shared fixtures: R reference values, datasets and fitted models."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from sklearn.linear_model import LinearRegression, LogisticRegression

from . import models

REFERENCE = Path(__file__).parent / "r_reference" / "reference.json"
TOL = 1e-4


@pytest.fixture(scope="session")
def reference() -> dict[str, Any]:
    """Everything tests/r_reference/generate_reference.R computed in R."""
    return json.loads(REFERENCE.read_text())


def assert_r_close(actual: Any, expected: Any, tol: float = TOL) -> None:
    """Assert agreement with R to `tol`, relative for large values, absolute near 0."""
    a = np.asarray(actual, dtype=float)
    e = np.asarray(expected, dtype=float)
    assert a.shape == e.shape, f"shape {a.shape} != {e.shape}"
    np.testing.assert_allclose(a, e, rtol=tol, atol=tol)


def per_obs(values: Any, ref_values: Any) -> tuple[np.ndarray, Any]:
    """Align a per-observation vector with R's, which keeps only 50 values past n = 400."""
    values = np.asarray(values, dtype=float)
    if isinstance(ref_values, dict):
        return values[:50], ref_values["head"]
    return values, ref_values


@pytest.fixture(scope="session")
def mtcars():
    return models.mtcars()


@pytest.fixture(scope="session")
def fit():
    """Fit (and cache) a reference model by name: ``fit("lm_mtcars")``."""
    return models.fit


@pytest.fixture(scope="session")
def ols(fit):
    return fit("lm_mtcars")


@pytest.fixture(scope="session")
def logit(fit):
    return fit("glm_logit")


@pytest.fixture(scope="session")
def poisson(fit):
    return fit("glm_pois")


@pytest.fixture(scope="session")
def sk_linear(mtcars):
    X, y = mtcars[["wt", "hp"]], mtcars["mpg"]
    return LinearRegression().fit(X, y), X, y


@pytest.fixture(scope="session")
def sk_logistic(mtcars):
    X, y = mtcars[["mpg", "wt"]], mtcars["vs"]
    model = LogisticRegression(C=np.inf, tol=1e-12, max_iter=10_000).fit(X, y)
    return model, X, y


@pytest.fixture(params=["statsmodels", "sklearn"])
def mtcars_ols(request, ols, sk_linear):
    """``mpg ~ wt + hp`` from both backends, as ``(model, X, y)``."""
    if request.param == "statsmodels":
        return ols, None, None
    return sk_linear


@pytest.fixture(params=["statsmodels", "sklearn"])
def mtcars_logit(request, logit, sk_logistic):
    """``vs ~ mpg + wt`` logistic regression from both backends, as ``(model, X, y)``."""
    if request.param == "statsmodels":
        return logit, None, None
    return sk_logistic


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    import matplotlib.pyplot as plt

    plt.close("all")
