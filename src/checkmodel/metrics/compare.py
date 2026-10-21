"""``compare_performance()``: metric tables for several models side by side."""

from __future__ import annotations

import warnings
from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np
import pandas as pd

from .._introspection import BaseAdapter, as_adapter
from .._types import ArrayLike
from .performance import model_performance

__all__ = ["compare_performance"]

_IC = ("AIC", "AICc", "BIC")
_LOWER_IS_BETTER = ("RMSE", "Sigma", "Log_loss")


def _ic_weight(ic: pd.Series) -> pd.Series:
    diffs = ic - ic.min(skipna=False)
    f = np.exp(-0.5 * diffs)
    return f / f.sum(skipna=False)


def _same_data(adapters: Sequence[BaseAdapter]) -> bool:
    first = adapters[0].y
    return all(a.y.shape == first.shape and np.allclose(a.y, first) for a in adapters[1:])


def _rank(table: pd.DataFrame, verbose: bool) -> pd.DataFrame:
    # performance:::.rank_performance_indices(): rescale every index to [0, 1],
    # flip the ones where lower is better, and average them.
    numeric = table.select_dtypes("number")
    span = numeric.max() - numeric.min()
    normalized = (numeric - numeric.min()) / span
    for col in _LOWER_IS_BETTER:
        if col in normalized:
            normalized[col] = 1.0 - normalized[col]
    usable = normalized.columns[normalized.notna().all()]
    dropped = [c for c in normalized.columns if c not in usable]
    if dropped and verbose:
        warnings.warn(
            "Following indices with missing values are not used for ranking: "
            + ", ".join(dropped),
            UserWarning,
            stacklevel=3,
        )
    out = table.copy()
    out["Performance_Score"] = normalized[usable].mean(axis=1)
    return out.sort_values("Performance_Score", ascending=False, kind="stable").reset_index(
        drop=True
    )


def compare_performance(
    *models: Any,
    names: Sequence[str] | None = None,
    metrics: str | Iterable[str] = "all",
    rank: bool = False,
    X: ArrayLike | None = None,
    y: ArrayLike | None = None,
    verbose: bool = True,
    **named_models: Any,
) -> pd.DataFrame:
    """Compare the performance of several models, as ``performance::compare_performance()``.

    Parameters
    ----------
    *models
        Fitted models. A scikit-learn model fitted on its own data can be
        given as a ``(model, X, y)`` tuple.
    names
        Names for the positional models; defaults to ``"Model 1"``, ...
    metrics
        Passed to :func:`model_performance`.
    rank
        Add a ``Performance_Score`` (mean of the rescaled indices, higher is
        better) and sort by it. As in R, the raw AIC/AICc/BIC columns are
        dropped and only their weights enter the score.
    X, y
        Shared training data for scikit-learn models.
    verbose
        Warn when the models were apparently not fit on the same data.
    **named_models
        Models passed by keyword use the keyword as their name:
        ``compare_performance(small=fit1, large=fit2)``.

    Returns
    -------
    pandas.DataFrame
        One row per model with ``Name``, ``Model`` and the metrics. Each
        information criterion is followed by its Akaike weight
        (``AIC_wt``, ``AICc_wt``, ``BIC_wt``).
    """
    if names is not None and len(names) != len(models):
        raise ValueError(f"Got {len(names)} names for {len(models)} models.")
    entries: list[tuple[str, BaseAdapter]] = []
    for i, m in enumerate(models):
        name = names[i] if names is not None else f"Model {i + 1}"
        if isinstance(m, tuple):
            entries.append((name, as_adapter(*m)))
        else:
            entries.append((name, as_adapter(m, X, y)))
    for name, m in named_models.items():
        entries.append((name, as_adapter(*m) if isinstance(m, tuple) else as_adapter(m, X, y)))
    if not entries:
        raise ValueError("Pass at least one model.")

    frames = []
    for name, adapter in entries:
        perf = model_performance(adapter, metrics=metrics)
        perf.insert(0, "Model", adapter.info.model_class)
        perf.insert(0, "Name", name)
        frames.append(perf)
    table = pd.concat(frames, ignore_index=True, sort=False)

    for ic in _IC:
        if ic in table:
            table[f"{ic}_wt"] = _ic_weight(table[ic])

    if verbose and len(entries) > 1 and not _same_data([a for _, a in entries]):
        warnings.warn(
            "When comparing models, please note that probably not all models were fit "
            "from same data.",
            UserWarning,
            stacklevel=2,
        )
    if rank:
        # R drops the raw criteria and keeps their weights at the end.
        return _rank(table.drop(columns=[c for c in _IC if c in table]), verbose)
    columns = [c for c in table.columns if not c.endswith("_wt")]
    for ic in _IC:
        if ic in columns:
            columns.insert(columns.index(ic) + 1, f"{ic}_wt")
    return table[columns]
