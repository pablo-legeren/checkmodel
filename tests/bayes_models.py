"""Rebuild the rstanarm fits of generate_bayes_reference.R as ArviZ DataTrees.

The draws come from tests/data/bayes_<name>.csv.gz, so checkmodel computes
on exactly the numbers R summarised in bayes_reference.json.
"""

from __future__ import annotations

import json
from functools import cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

DATA = Path(__file__).parent / "data"
REFERENCE = Path(__file__).parent / "r_reference" / "bayes_reference.json"
FAMILIES = {"bayes_lm": "gaussian", "bayes_logit": "binomial", "bayes_mixed": "gaussian"}


@cache
def reference() -> dict[str, Any]:
    return json.loads(REFERENCE.read_text())


@cache
def draws(name: str) -> pd.DataFrame:
    return pd.read_csv(DATA / f"{name}.csv.gz")


def _block(table: pd.DataFrame, prefix: str, chains: int) -> np.ndarray:
    cols = [c for c in table.columns if c.startswith(f"{prefix}_")]
    return table[cols].to_numpy().reshape(chains, -1, len(cols))


def parameter_names(name: str) -> list[str]:
    table = draws(name)
    return [
        c
        for c in table.columns
        if c not in {"chain", "draw"} and c.split("_")[0] not in {"mu", "mum", "yrep", "ll"}
    ]


@cache
def datatree(name: str) -> Any:
    """The draws as a DataTree with posterior, predictive, log-lik and observed groups."""
    import arviz as az

    table = draws(name)
    chains = table["chain"].nunique()
    posterior: dict[str, np.ndarray] = {"mu": _block(table, "mu", chains)}
    for p in parameter_names(name):
        posterior[p] = table[p].to_numpy().reshape(chains, -1)
    groups: dict[str, dict[str, np.ndarray]] = {
        "posterior": posterior,
        "observed_data": {"y": np.asarray(reference()["models"][name]["y"], dtype=float)},
    }
    dims = {"mu": ["obs"], "y": ["obs"]}
    if any(c.startswith("mum_") for c in table.columns):
        posterior["mu_marginal"] = _block(table, "mum", chains)
        dims["mu_marginal"] = ["obs"]
    if any(c.startswith("yrep_") for c in table.columns):
        groups["posterior_predictive"] = {"y": _block(table, "yrep", chains)}
        groups["log_likelihood"] = {"y": _block(table, "ll", chains)}
    return az.from_dict(groups, dims=dims)


@cache
def fit(name: str) -> Any:
    """The draws wrapped by checkmodel.bayesian()."""
    from checkmodel import bayesian

    mixed = any(c.startswith("mum_") for c in draws(name).columns)
    return bayesian(
        datatree(name),
        family=FAMILIES[name],
        mu_marginal="mu_marginal" if mixed else None,
    )
