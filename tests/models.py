"""The models of tests/r_reference/generate_reference.R, fitted with statsmodels."""

from __future__ import annotations

import warnings
from collections.abc import Callable
from functools import cache
from pathlib import Path
from typing import Any

import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

from . import synthetic

DATA = Path(__file__).parent / "data"


@cache
def mtcars() -> pd.DataFrame:
    return pd.read_csv(DATA / "mtcars.csv", index_col=0)


@cache
def iris() -> pd.DataFrame:
    df = pd.read_csv(DATA / "iris.csv")
    df.columns = [c.lower().replace(".", "_") for c in df.columns]
    return df


@cache
def warpbreaks() -> pd.DataFrame:
    return pd.read_csv(DATA / "warpbreaks.csv")


@cache
def sleepstudy() -> pd.DataFrame:
    return pd.read_csv(DATA / "sleepstudy.csv")


@cache
def biochemists() -> pd.DataFrame:
    return pd.read_csv(DATA / "biochemists.csv")


@cache
def dataset(name: str) -> pd.DataFrame:
    return synthetic.DATASETS[name]()


def _ols(formula: str, data: Callable[[], pd.DataFrame]) -> Callable[[], Any]:
    return lambda: smf.ols(formula, data()).fit()


def _synthetic_ols(name: str, formula: str = "y ~ x") -> Callable[[], Any]:
    return lambda: smf.ols(formula, dataset(name)).fit()


MODELS: dict[str, Callable[[], Any]] = {
    "lm_mtcars": _ols("mpg ~ wt + hp", mtcars),
    "lm_mtcars_cyl": _ols("mpg ~ wt + hp + C(cyl)", mtcars),
    "lm_mtcars_wls": lambda: smf.wls("mpg ~ wt + hp", mtcars(), weights=mtcars().disp / 100).fit(),
    "lm_iris": _ols("sepal_length ~ sepal_width + petal_length + petal_width", iris),
    "lm_good": _synthetic_ols("good", "y ~ x1 + x2"),
    "lm_hetero": _synthetic_ols("hetero"),
    "lm_skewed": _synthetic_ols("skewed"),
    "lm_autocorr": _synthetic_ols("autocorr"),
    "lm_autocorr_large": _synthetic_ols("autocorr_large"),
    "lm_large": _synthetic_ols("large"),
    "lm_collinear": _synthetic_ols("collinear", "y ~ x1 + x2 + x3"),
    "glm_gauss": lambda: smf.glm("mpg ~ wt + hp", mtcars()).fit(),
    "glm_logit": lambda: smf.glm("vs ~ mpg + wt", mtcars(), family=sm.families.Binomial()).fit(),
    "glm_logit_synth": lambda: smf.glm(
        "y ~ x + z", dataset("logit"), family=sm.families.Binomial()
    ).fit(),
    "glm_pois": lambda: smf.glm(
        "breaks ~ wool + tension", warpbreaks(), family=sm.families.Poisson()
    ).fit(),
    "glm_pois_bio": lambda: smf.glm(
        "art ~ fem + mar + kid5 + phd + ment", biochemists(), family=sm.families.Poisson()
    ).fit(),
    "glm_pois_synth": lambda: smf.glm(
        "y ~ x", dataset("pois"), family=sm.families.Poisson()
    ).fit(),
    # MASS::glm.nb() estimates theta by maximum likelihood, like statsmodels' NB2.
    "glm_negbin": lambda: smf.negativebinomial(
        "art ~ fem + mar + kid5 + phd + ment", biochemists()
    ).fit(disp=0, method="newton", tol=1e-12, maxiter=200),
    "glm_probit": lambda: smf.glm(
        "vs ~ mpg + wt", mtcars(), family=sm.families.Binomial(sm.families.links.Probit())
    ).fit(tol=1e-14),
    "glm_cloglog": lambda: smf.glm(
        "vs ~ mpg + wt", mtcars(), family=sm.families.Binomial(sm.families.links.CLogLog())
    ).fit(tol=1e-14),
    "glm_gamma": lambda: smf.glm(
        "mpg ~ wt + hp", mtcars(), family=sm.families.Gamma(sm.families.links.Log())
    ).fit(tol=1e-14),
    "glm_invgauss": lambda: smf.glm(
        "mpg ~ wt + hp", mtcars(), family=sm.families.InverseGaussian(sm.families.links.Log())
    ).fit(tol=1e-14),
}

# Linear mixed models, mirroring `mixed_models` in generate_reference.R.
MIXED: dict[str, Callable[[], Any]] = {
    "mixed_sleep": lambda: smf.mixedlm("Reaction ~ Days", sleepstudy(), groups="Subject").fit(),
    "mixed_sleep_slope": lambda: smf.mixedlm(
        "Reaction ~ Days", sleepstudy(), groups="Subject", re_formula="~Days"
    ).fit(),
    "mixed_sleep_ml": lambda: smf.mixedlm("Reaction ~ Days", sleepstudy(), groups="Subject").fit(
        reml=False
    ),
    "mixed_synth": lambda: smf.mixedlm("y ~ x1 + x2", dataset("mixed"), groups="g").fit(),
    # statsmodels' default optimizer stops just short of the zero-variance boundary
    # here (check_convergence flags it); Powell reaches it, as lme4 does.
    "mixed_singular": lambda: smf.mixedlm("y ~ x1", dataset("mixed_singular"), groups="g").fit(
        method="powell"
    ),
}

# Term names as patsy writes them -> as R writes them.
R_TERM_NAMES = {"C(cyl)": "factor(cyl)"}


@cache
def fit(name: str) -> Any:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # statsmodels' MixedLM convergence chatter
        return {**MODELS, **MIXED}[name]()
