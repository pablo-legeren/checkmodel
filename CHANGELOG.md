# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[semantic versioning](https://semver.org/).

## [Unreleased]

### Added

- Linear mixed models: statsmodels `MixedLM` results (random intercepts and
  slopes). Leverage, studentized residuals and Cook's distance follow lme4.
- `icc()`, `r2_nakagawa()` and `variance_components()`; `r2()` and
  `model_performance()` report Nakagawa's R² and the ICC for mixed models, and
  `model_performance()` gains `estimator=` (REML by default, as in R).
- `check_singularity()` and `check_convergence()`.
- `check_normality(model, effects="random")` for random effects.
- `check_overdispersion()` and `check_zeroinflation()` for Poisson models;
  `check_model()` runs the overdispersion check on Poisson regressions.
- Plots for random effects, overdispersion and zero-inflation.

### Changed

- `check_heteroscedasticity()` scales by `df_resid * sigma² / n`. Identical for
  OLS, WLS and GLMs; for mixed models it uses the REML sigma (R's
  `performance` uses a likelihood deviance there; see
  `docs/comparison-with-r.md`).
- `R2Result` titles name the full model type ("Linear Regression",
  "Mixed Models").

## [0.1.0] - 2026-10-20

First release.

### Added

- `check_model()`: runs every check that applies to a model and draws a
  combined diagnostic panel.
- Checks with typed results (`.passed`, `.statistic`, `.p_value`, `.plot()`,
  `.to_dataframe()`): `check_normality()`, `check_heteroscedasticity()`,
  `check_collinearity()`, `check_outliers()`, `check_autocorrelation()` and
  `binned_residuals()`.
- Metrics: `r2()`, `r2_tjur()`, `r2_coxsnell()`, `r2_nagelkerke()`,
  `r2_mcfadden()`, `model_performance()` and `compare_performance()`.
- Support for statsmodels OLS, WLS and GLM (Gaussian, binomial, Poisson)
  results and scikit-learn `LinearRegression`, `Ridge`, `Lasso`,
  `ElasticNet` and binary `LogisticRegression`.
- `ModelAdapter` protocol and `register_adapter()` for adding model types.
- Test suite validated against R (`performance` 0.18.2, `lmtest`, `car`) to
  a tolerance of 1e-4.

[Unreleased]: https://github.com/pablo-legeren/checkmodel/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/pablo-legeren/checkmodel/releases/tag/v0.1.0
