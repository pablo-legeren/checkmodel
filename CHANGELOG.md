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
- Bayesian models from PyMC and bambi (`pip install "checkmodel[bayes]"`):
  `bayesian()` reads a fitted model; `r2_bayes()`, `looic()` (PSIS-LOO),
  WAIC, R-hat and bulk/tail ESS are ported from R's `performance`, `loo` and
  `posterior` and validated on the same posterior draws; `check_model()`,
  `check_convergence()`, `model_performance()` and `compare_performance()`
  accept Bayesian models.
- `check_predictions()`: posterior predictive checks for Bayesian models and
  simulation-based ones for frequentist models, with p-values and a plot.
  `check_model()` now starts with it, as in R (`random_state=` keeps reports
  reproducible).
- `simulate_residuals()` and `check_residuals()`: randomized quantile
  residuals as in DHARMa, and a Kolmogorov-Smirnov test of their uniformity
  computed as R's `ks.test()`. `check_model()` runs the test on non-Gaussian
  models, with a uniform Q-Q plot.
- `check_overdispersion()` works for every GLM and for mixed models, with
  simulated residuals where R uses them (Bernoulli, binomial, negative
  binomial, Gaussian, Gamma and inverse Gaussian GLMs, and mixed models), and
  gains `residual_type=`, `alternative=`, `iterations=` and `random_state=`.
- `check_zeroinflation()` works for negative binomial models through simulated
  residuals (tolerance 0.1 and a p-value, as in R). Both checks also accept
  the output of `simulate_residuals()`.
- Response simulation (`check_predictions()`, simulated residuals) for
  binomial models with trials, negative binomial, Gamma and inverse Gaussian
  GLMs. Adapters gain a `simulate()` method, which `ModelAdapter`
  implementations may provide.
- statsmodels `Logit`, `Probit`, `Poisson` and `NegativeBinomial` (NB2 and
  geometric) results, treated as the equivalent GLMs (NB2 like R's
  `glm.nb()`); offsets and exposure are honoured.
- scikit-learn `PoissonRegressor` and `GammaRegressor`.
- Generic adapters derive leverage, residuals, deviances and likelihoods for
  negative binomial, Gamma and inverse Gaussian families and for probit,
  cloglog, inverse and sqrt links; `model_performance()` reports scoring rules
  for negative binomial models.

### Changed

- `check_overdispersion()` on Bernoulli and other non-Poisson GLMs and on
  mixed models now runs the simulated-residual test instead of raising, as
  `performance` 0.18.2 does. On linear models fitted by least squares it
  raises with a pointer to `check_heteroscedasticity()`.
- `check_model()` on logistic, count and other non-Gaussian GLMs also reports
  `"residuals"` (uniformity of simulated residuals), and `random_state` seeds
  every simulation-based check.
- `OverdispersionResult.df` is `None` for the simulated test, and the
  overdispersion and zero-inflation plots draw the variance and zero counts
  the model expects instead of assuming Poisson.

- `check_heteroscedasticity()` scales by `df_resid * sigma² / n`. Identical for
  OLS, WLS and GLMs; for mixed models it uses the REML sigma (R's
  `performance` uses a likelihood deviance there; see
  `docs/comparison-with-r.md`).
- `R2Result` titles name the full model type ("Linear Regression",
  "Mixed Models").

### Fixed

- `r2()` (and so `model_performance()`) reports Nagelkerke's R² for binary
  models with probit, cloglog and other non-logit links, as R's `r2.glm()`
  does; it reported Tjur's R², which R keeps for logit links.
- Leverage of statsmodels GLMs uses the IRLS (Fisher) weights, as R's
  `hatvalues()`. statsmodels' default observed-information leverage differed
  for non-canonical links (probit, cloglog, log-link Gamma, negative
  binomial), and with it Cook's distance and standardized residuals.
- Log-likelihood, AIC and BIC of Gamma and inverse Gaussian GLMs estimate the
  dispersion as deviance / n, as R does.
- Leverage no longer fails when LAPACK's SVD does not converge.

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
