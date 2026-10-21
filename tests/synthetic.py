"""Deterministic datasets, mirrored from tests/r_reference/generate_reference.R.

Both sides build them from the same closed-form formulas, so no random number
generator has to agree across languages. ``reference.json`` stores column sums
that the test suite checks before trusting any comparison.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import special, stats


def hash_u(i: np.ndarray) -> np.ndarray:
    v = np.sin(i * 12.9898) * 43758.5453
    return v - np.floor(v)


def hash_e(i: np.ndarray) -> np.ndarray:
    return stats.norm.ppf(np.clip(hash_u(i), 1e-6, 1 - 1e-6))


def _i(n: int) -> np.ndarray:
    return np.arange(1, n + 1, dtype=float)


def good() -> pd.DataFrame:
    i = _i(60)
    x1, x2 = i / 6, hash_u(i + 1000) * 10
    return pd.DataFrame({"x1": x1, "x2": x2, "y": 1 + 0.5 * x1 - 0.3 * x2 + hash_e(i)})


def hetero() -> pd.DataFrame:
    i = _i(200)
    x = i / 20
    return pd.DataFrame({"x": x, "y": 1 + 2 * x + (0.2 + x) * hash_e(i)})


def skewed() -> pd.DataFrame:
    i = _i(200)
    x = i / 20
    return pd.DataFrame({"x": x, "y": 1 + 2 * x + np.exp(1.5 * hash_e(i))})


def autocorr() -> pd.DataFrame:
    i = _i(80)
    x = i / 8
    return pd.DataFrame({"x": x, "y": 1 + 0.5 * x + 2 * np.sin(i / 5) + 0.3 * hash_e(i)})


def autocorr_large() -> pd.DataFrame:
    i = _i(300)
    x = i / 30
    return pd.DataFrame({"x": x, "y": 1 + 0.5 * x + 2 * np.sin(i / 8) + 0.5 * hash_e(i)})


def large() -> pd.DataFrame:
    i = _i(6000)
    x = hash_u(i + 5000) * 4
    return pd.DataFrame({"x": x, "y": 2 + x + hash_e(i)})


def collinear() -> pd.DataFrame:
    i = _i(100)
    x1 = hash_e(i)
    x2 = x1 + 0.1 * hash_e(i + 2000)
    x3 = hash_e(i + 3000)
    return pd.DataFrame({"x1": x1, "x2": x2, "x3": x3, "y": x1 + x2 + x3 + hash_e(i + 4000)})


def logit() -> pd.DataFrame:
    i = _i(400)
    x, z = hash_e(i + 6000), hash_e(i + 8000)
    p = special.expit(-0.5 + 1.2 * x - 0.6 * z)
    return pd.DataFrame({"x": x, "z": z, "y": (hash_u(i + 7000) < p).astype(float)})


def pois() -> pd.DataFrame:
    i = _i(300)
    x = hash_e(i + 12000)
    return pd.DataFrame({"x": x, "y": stats.poisson.ppf(hash_u(i + 12500), np.exp(0.3 + 0.5 * x))})


def mixed() -> pd.DataFrame:
    i = _i(200)
    g = (i - 1) % 20 + 1
    x1 = hash_e(i + 9000)
    x2 = 0.5 * x1 + hash_e(i + 9500)
    y = 1 + 0.5 * x1 - 0.3 * x2 + 2 * hash_e(g + 9900) + hash_e(i + 10000)
    return pd.DataFrame({"g": g, "x1": x1, "x2": x2, "y": y})


def mixed_singular() -> pd.DataFrame:
    """No between-group variance at all: the noise is centred within each group."""
    i = _i(200)
    g = (i - 1) % 20 + 1
    x1 = hash_e(i + 9000)
    e = pd.Series(hash_e(i + 11000))
    centred = (e - e.groupby(g).transform("mean")).to_numpy()
    return pd.DataFrame({"g": g, "x1": x1, "y": 1 + 0.5 * x1 + centred})


DATASETS = {
    "good": good,
    "hetero": hetero,
    "skewed": skewed,
    "autocorr": autocorr,
    "autocorr_large": autocorr_large,
    "large": large,
    "collinear": collinear,
    "logit": logit,
    "pois": pois,
    "mixed": mixed,
    "mixed_singular": mixed_singular,
}
