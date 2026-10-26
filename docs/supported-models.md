# Supported models

Which function works on which model. The table is produced by running every
function on one model of each kind; a dash means the function raises
`CheckNotApplicableError` (or, for Bayesian models, points you to the
functions that do apply).

**Columns:** *OLS / WLS*: statsmodels `OLS`/`WLS` results. *sklearn linear*:
`LinearRegression`. *sklearn penalized*: `Ridge`, `Lasso`, `ElasticNet` and
their `CV` variants. *Gaussian GLM*, *Logistic GLM*, *Poisson GLM*: statsmodels
`GLM` results. *sklearn logistic*: binary `LogisticRegression`. *MixedLM*:
statsmodels linear mixed models (random intercepts and slopes). *Bayesian*:
PyMC and bambi models wrapped with `bayesian()` (Python 3.12+).

| Function | OLS / WLS | sklearn linear | sklearn penalized | Gaussian GLM | Logistic GLM | sklearn logistic | Poisson GLM | MixedLM | Bayesian |
|---|---|---|---|---|---|---|---|---|---|
| `check_predictions` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `check_normality` | ✓ | ✓ | ✓ | ✓ | – | – | – | ✓ | – |
| `check_heteroscedasticity` | ✓ | ✓ | ✓ | ✓ | – | – | – | ✓ | – |
| `check_collinearity` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | – |
| `check_outliers` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | – |
| `check_autocorrelation` | ✓ | ✓ | ✓ | ✓ | – | – | – | – | – |
| `binned_residuals` | – | – | – | – | ✓ | ✓ | – | – | – |
| `check_overdispersion` | – | – | – | ✓ | ✓ | ✓ | ✓ | ✓ | – |
| `check_zeroinflation` | – | – | – | – | – | – | ✓ | – | – |
| `check_normality(effects='random')` | – | – | – | – | – | – | – | ✓ | – |
| `check_singularity` | – | – | – | – | – | – | – | ✓ | – |
| `check_convergence` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `simulate_residuals`, `check_residuals` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | – |
| `r2` | ✓ R2, R2_adjusted | ✓ R2, R2_adjusted | ✓ R2, R2_adjusted | ✓ R2 | ✓ R2_Tjur | ✓ R2_Tjur | ✓ R2_Nagelkerke | ✓ R2_conditional, R2_marginal | ✓ R2 |
| `icc` | – | – | – | – | – | – | – | ✓ | – |
| `model_performance` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

## Notes per model type

- **scikit-learn** estimators need `X=` and `y=`, because they do not keep
  their training data. Penalized models have no likelihood, so
  `model_performance()` leaves out AIC, AICc and BIC; leverage and VIFs come
  from the design matrix, and the checks that use leverage or standardized
  residuals warn that they are approximate. `PoissonRegressor` and `GammaRegressor` behave like
  the Poisson and other GLM columns (penalized unless `alpha=0`).
- **statsmodels discrete models** (`Logit`, `Probit`, `Poisson`,
  `NegativeBinomial`) behave like the GLM of the same family and link. A
  negative binomial fit (NB2 or geometric) is treated like R's `glm.nb()`;
  NB1, multinomial, generalized Poisson and zero-inflated models are not
  supported yet.
- **Logistic regression** is tested for collinearity, influence, binned
  residuals and the uniformity of simulated residuals; normality and constant
  variance are not expected for a binary outcome. Binomial models with several
  trials per row are supported, but pseudo-R² measures that need 0/1 outcomes
  raise an error, as in R.
- **Poisson regression** adds overdispersion (Pearson chi-squared) and
  zero-inflation. **Negative binomial** GLMs get both checks from simulated
  residuals, as R does. Gamma and inverse Gaussian GLMs are checked through
  simulated residuals too.
- **Overdispersion** uses Pearson residuals for Poisson GLMs and simulated
  residuals for every other GLM and for mixed models, following R's rules;
  `residual_type=` overrides the choice. Linear models fitted by least squares
  have no overdispersion check.
- **Mixed models** use lme4's definitions of leverage, studentized residuals
  and Cook's distance. Variance components (`vc_formula`) are not supported
  yet. If a variance is estimated at (nearly) zero, refit with
  `.fit(method="powell")`; see the note on optimizers in
  [Comparison with R](comparison-with-r.md).
- **Bayesian models** are checked through their posterior draws:
  predictions, convergence, Bayesian R², PSIS-LOO and WAIC. Residual-based
  frequentist checks do not apply to them.
- **Anything else** can be supported by implementing the `ModelAdapter`
  protocol (five methods) or registering an adapter; see `CONTRIBUTING.md`.
