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
| `check_overdispersion` | – | – | – | – | – | – | ✓ | – | – |
| `check_zeroinflation` | – | – | – | – | – | – | ✓ | – | – |
| `check_normality(effects='random')` | – | – | – | – | – | – | – | ✓ | – |
| `check_singularity` | – | – | – | – | – | – | – | ✓ | – |
| `check_convergence` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `r2` | ✓ R2, R2_adjusted | ✓ R2, R2_adjusted | ✓ R2, R2_adjusted | ✓ R2 | ✓ R2_Tjur | ✓ R2_Tjur | ✓ R2_Nagelkerke | ✓ R2_conditional, R2_marginal | ✓ R2 |
| `icc` | – | – | – | – | – | – | – | ✓ | – |
| `model_performance` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

## Notes per model type

- **scikit-learn** estimators need `X=` and `y=`, because they do not keep
  their training data. Penalized models have no likelihood, so
  `model_performance()` leaves out AIC, AICc and BIC; leverage and VIFs come
  from the design matrix.
- **Logistic regression** is tested for collinearity, influence and binned
  residuals; normality and constant variance are not expected for a binary
  outcome. Binomial models with several trials per row are supported, but
  pseudo-R² measures that need 0/1 outcomes raise an error, as in R.
- **Poisson regression** adds overdispersion and zero-inflation. Negative
  binomial and other count families get the general checks only: R handles
  them with simulated residuals, which are not ported yet.
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
