# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[semantic versioning](https://semver.org/).

## [Unreleased]

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

[Unreleased]: https://github.com/pablo-legeren/checkmodel/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/pablo-legeren/checkmodel/releases/tag/v0.1.0
