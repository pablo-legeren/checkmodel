# Contributing

Thanks for helping. Bug reports, model types and checks from R's `performance`
that are not ported yet are all welcome.

## Development setup

```bash
git clone https://github.com/pablo-legeren/checkmodel
cd checkmodel
pip install -e ".[dev]"
pre-commit install
pytest
```

`ruff check`, `ruff format --check` and `mypy` run in CI next to the tests.

## Numbers must match R

Every statistic is tested against R. The reference values live in
`tests/r_reference/reference.json` and come from
`tests/r_reference/generate_reference.R`. When you add a check or a model:

1. Add the model or statistic to `generate_reference.R` (and the model to
   `tests/models.py`; synthetic datasets go in both the R script and
   `tests/synthetic.py`).
2. Regenerate the reference values from the repository root:

   ```bash
   Rscript tests/r_reference/generate_reference.R
   ```

   It needs R with `performance`, `lmtest`, `car` and `jsonlite`.
3. Compare with `assert_r_close()` from `tests/conftest.py` (tolerance 1e-4).

If a result differs from R on purpose, document why in
`docs/comparison-with-r.md`.

## Supporting a new model type

A model only has to provide five things. The quickest route is an object
implementing the `ModelAdapter` protocol:

```python
import numpy as np
from checkmodel import ModelType, check_model


class MyModelAdapter:
    def __init__(self, model):
        self.model = model

    def model_type(self) -> ModelType:
        return ModelType(family="gaussian", link="identity",
                         backend="mylib", model_class="MyModel")

    def get_response(self) -> np.ndarray: ...
    def get_fitted(self) -> np.ndarray: ...
    def get_residuals(self) -> np.ndarray: ...
    def get_model_matrix(self) -> np.ndarray: ...  # include the intercept column


check_model(MyModelAdapter(fitted_model))
```

Leverage, standardized residuals, Cook's distance, VIFs and likelihoods are
derived from these for Gaussian, binomial, Poisson, negative binomial (expose
`negbin_theta`), Gamma and inverse Gaussian models with identity, logit,
probit, cloglog, log, inverse or sqrt links.
Optional methods `column_names()`, `prior_weights()` and `index()` improve the
output when defined. `simulate(n_sim, rng)`, returning an `(n_sim, n_obs)`
array of new responses, lets simulation-based checks (`check_predictions()`,
`simulate_residuals()`) work for families `checkmodel` cannot simulate itself.

To make `check_model(fitted_model)` work directly, subclass `BaseAdapter`
and register it:

```python
from checkmodel import BaseAdapter, register_adapter

class MyAdapter(BaseAdapter):
    ...  # same five methods, reading from self.model

register_adapter(MyModel, lambda model, X=None, y=None: MyAdapter(model))
```

Override any derived property (`hat`, `pearson_residuals`, `loglik`, ...) when
your library computes it more faithfully, for example to honour offsets.

## Releasing

1. Update `__version__` in `src/checkmodel/__init__.py` and `CHANGELOG.md`.
2. Commit, then tag: `git tag v0.1.0 && git push origin v0.1.0`.
3. The `publish` workflow builds, checks and uploads to PyPI through
   trusted publishing.
