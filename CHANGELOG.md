# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[semantic versioning](https://semver.org/).

## [Unreleased]

## [0.2.0] - 2026-10-26

### Changed

- **Breaking:** `check_model()` no longer tests linear models for
  autocorrelation by default, so the report has no `"autocorrelation"` key
  and its result no longer counts towards `report.passed`. On cross-sectional
  data the Durbin-Watson test only reacts to how the rows happen to be sorted
  (the `mtcars` example in the README reported p = 0.041), and R's
  `check_model()` does not run it either. It is added automatically when the
  observations are indexed by time (a pandas `DatetimeIndex` or
  `PeriodIndex`). To keep the 0.1.0 behaviour, request it:
  `check_model(m, checks=[..., "autocorrelation"])`. `check_autocorrelation()`
  is unchanged. The combined plot of a linear model has six panels instead of
  seven.
- `check_collinearity()` fails (`passed=False`) when the model matrix is rank
  deficient, with the message "Perfect collinearity: the model matrix is rank
  deficient (aliased terms: ...)". It still emits the "Model matrix is rank
  deficient" warning.
- Normality checks that fail on 5000 or more residuals (Kolmogorov-Smirnov)
  add a note that large samples flag trivial departures and point to the Q-Q
  plot. The verdict and the p-value are unchanged.
- Plots of large models: past 5000 observations each panel draws a
  deterministic subsample that keeps the points outside the bands and the
  observations flagged by Cook's distance, the lowess smoother interpolates
  between nearby fits, and density curves past 10 000 values use a binned
  kernel density estimate. `check_model(fit).plot()` on 200 000 observations
  takes a few seconds instead of about two minutes. Plots of up to 5000
  observations are drawn exactly as before.

### Added

- `CollinearityResult.rank_deficient` and `CollinearityResult.aliased`: the
  design-matrix columns that are linear combinations of earlier ones, the
  coefficients R reports as `NA`. Also available as `BaseAdapter.aliased`.
- A warning for penalized fits (`Ridge`, `Lasso`, `ElasticNet`, penalized
  `LogisticRegression`, `PoissonRegressor` and `GammaRegressor` with
  `alpha > 0`): their leverage, Cook's distance and standardized residuals
  are computed as for the unpenalized model and are approximate.
  `check_model()` warns once and notes it in the report header;
  `check_outliers()`, `check_normality()`, `check_heteroscedasticity()` and
  `check_collinearity()` warn when called on their own.
- A warning for binomial models with complete separation (every fitted
  probability within 1e-6 of 0 or 1) from `check_model()`,
  `check_outliers()`, `check_collinearity()`, `binned_residuals()` and
  `check_residuals()`.
- README: status line, "Open in Colab" badges for the example notebooks, a
  short table of R equivalents and how to request the autocorrelation check.
  The notebooks install `checkmodel` and read their data from GitHub when run
  on Colab.
- Tests of the Bayesian model reader that need neither ArviZ, PyMC nor bambi,
  so `_bayes.py` is fully covered on every supported Python version.

### Fixed

- The `mtcars` example in the README no longer reports a spurious
  "Autocorrelated residuals detected" warning.
- A rank-deficient model (for example a dummy for every category plus the
  intercept) was reported as "OK: Low collinearity" with `passed=True`.
- Diagnostics of penalized models were reported as "OK" with no sign that the
  influence measures ignore the penalty.
- `check_model(fit).plot()` took minutes on large data (about 108 s for
  200 000 rows), almost all of it in kernel density estimates of every
  replicated dataset.

## [0.1.0] - 2026-10-22

First public release.

### Added

- `check_model()`: runs every check that applies to a model and draws a
  combined diagnostic panel. Every model starts with a posterior predictive
  check, as in R; `random_state=` keeps reports reproducible.
- Assumption checks with typed results (`.passed`, `.statistic`, `.p_value`,
  `.plot()`, `.to_dataframe()`): `check_normality()` (residuals and random
  effects), `check_heteroscedasticity()`, `check_collinearity()`,
  `check_outliers()`, `check_autocorrelation()`, `binned_residuals()`,
  `check_overdispersion()`, `check_zeroinflation()`, `check_singularity()`,
  `check_convergence()` and `check_predictions()`.
- Simulated residuals as in DHARMa: `simulate_residuals()` and
  `check_residuals()`, used for non-Gaussian models and wherever R's
  `performance` uses them.
- Metrics: `r2()` and its variants (Tjur, Cox & Snell, Nagelkerke, McFadden,
  Nakagawa, Bayesian), `icc()`, `variance_components()`, `looic()`,
  `model_performance()` and `compare_performance()`.
- Models:
  - statsmodels OLS, WLS, GLM (Gaussian, binomial, Poisson, negative
    binomial, Gamma, inverse Gaussian; logit, probit, cloglog, log, identity,
    inverse and sqrt links), `Logit`, `Probit`, `Poisson`,
    `NegativeBinomial` and `MixedLM`;
  - scikit-learn `LinearRegression`, `Ridge`, `Lasso`, `ElasticNet`,
    binary `LogisticRegression`, `PoissonRegressor` and `GammaRegressor`;
  - PyMC and bambi models through `bayesian()` (`pip install
    "checkmodel[bayes]"`, Python 3.12+);
  - anything implementing the `ModelAdapter` protocol, or registered with
    `register_adapter()`.
- Matplotlib plots for every check, in the style of R's `see` package.
- Validation against R: every statistic is compared with `performance`
  0.18.2, `lme4`, `lmtest`, `car`, `MASS`, `rstanarm`, `loo` and `posterior`
  (to 1e-4) on 27 reference models; Bayesian statistics on the same posterior
  draws. Deliberate differences are listed in `docs/comparison-with-r.md`.

[Unreleased]: https://github.com/pablo-legeren/checkmodel/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/pablo-legeren/checkmodel/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/pablo-legeren/checkmodel/releases/tag/v0.1.0
