# checkmodel

**Status: alpha – API may change before 1.0**

[![PyPI version](https://img.shields.io/pypi/v/checkmodel.svg)](https://pypi.org/project/checkmodel/)
[![Downloads](https://static.pepy.tech/badge/checkmodel/month)](https://pepy.tech/project/checkmodel)
[![CI](https://github.com/pablo-legeren/checkmodel/actions/workflows/ci.yml/badge.svg)](https://github.com/pablo-legeren/checkmodel/actions/workflows/ci.yml)
[![Python versions](https://img.shields.io/pypi/pyversions/checkmodel.svg)](https://pypi.org/project/checkmodel/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

**One-call diagnostics for statistical models in Python.** A port of R's
[`performance`](https://easystats.github.io/performance/) package from the
easystats ecosystem: `check_model(model)` works out what kind of model you
fitted, runs every assumption check that applies to it and draws a single
diagnostic panel.

Every statistic is tested against the R implementation (`performance`,
`lmtest`, `car`) to a tolerance of 1e-4.

## Installation

```bash
pip install checkmodel
```

Python 3.10 or newer. Depends on numpy, scipy, pandas, statsmodels, scikit-learn
and matplotlib. For Bayesian models (PyMC, bambi; Python 3.12+):

```bash
pip install "checkmodel[bayes]"
```

## Quick example

```python
import statsmodels.formula.api as smf
from checkmodel import check_model

fit = smf.ols("mpg ~ wt + hp", data=mtcars).fit()
report = check_model(fit)
report.plot()
```

```text
# Model checks: OLS (gaussian, identity link), n = 32

Posterior predictive check: OK: Model-predicted data resemble the observed data (mean p = 0.400, sd p = 0.460).
Normality of residuals: Warning: Non-normality of residuals detected (p = 0.021).
Homogeneity of variance: OK: Error variance appears to be homoscedastic (p = 0.402).
Collinearity: OK: Low collinearity (all VIF < 5; max 1.77).
Influential observations: OK: No outliers detected (Cook's distance > 0.808).
```

Autocorrelation is tested only when the observations are indexed by time (a
pandas `DatetimeIndex` or `PeriodIndex`), because in cross-sectional data the
Durbin-Watson test just reacts to how the rows happen to be sorted; R's
`check_model()` leaves it out too. Ask for it by name when the row order means
something:

```python
check_model(fit, checks=["normality", "autocorrelation"])
check_autocorrelation(fit)   # or on its own
```

![check_model() diagnostic panel](https://raw.githubusercontent.com/pablo-legeren/checkmodel/main/docs/images/check_model.png)

scikit-learn estimators don't keep their training data, so pass it along:

```python
from sklearn.linear_model import LinearRegression

model = LinearRegression().fit(X, y)
check_model(model, X=X, y=y)
```

## Examples

Worked notebooks, runnable in the browser:

| Notebook | |
|---|---|
| Quickstart: a linear and a logistic model, then a comparison | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pablo-legeren/checkmodel/blob/main/examples/quickstart.ipynb) |
| Diagnosing a linear regression | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pablo-legeren/checkmodel/blob/main/examples/01_linear_regression.ipynb) |
| Logistic regression: binned and simulated residuals | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pablo-legeren/checkmodel/blob/main/examples/02_logistic_regression.ipynb) |
| Comparing models across statsmodels and scikit-learn | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pablo-legeren/checkmodel/blob/main/examples/03_model_comparison.ipynb) |
| Linear mixed models: random effects, ICC, Nakagawa's R² | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pablo-legeren/checkmodel/blob/main/examples/04_mixed_models.ipynb) |
| Coming from R: the same analysis side by side | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pablo-legeren/checkmodel/blob/main/examples/05_coming_from_r.ipynb) |
| Bayesian models with bambi (Python 3.12+) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pablo-legeren/checkmodel/blob/main/examples/06_bayesian_models.ipynb) |

## What's inside

| Function | What it does | Method |
|---|---|---|
| `check_model(model)` | Runs every check that applies and plots them together | Orchestrator |
| `check_predictions(model)` | Do data simulated from the model look like the data? | Posterior predictive check |
| `check_normality(model)` | Normality of residuals | Shapiro-Wilk (n < 5000), Kolmogorov-Smirnov (n ≥ 5000) |
| `check_heteroscedasticity(model)` | Constant error variance | Breusch-Pagan / Cook-Weisberg score test |
| `check_collinearity(model)` | Multicollinearity | (Generalized) variance inflation factors with CIs |
| `check_outliers(model)` | Influential observations | Cook's distance |
| `check_autocorrelation(model)` | Serially correlated residuals | Durbin-Watson, exact or normal-approximation p-value |
| `binned_residuals(model)` | Fit of logistic regression | Binned residuals (Gelman & Hill) |
| `simulate_residuals(model)` | Residuals that are uniform for any family if the model is right | Randomized quantile residuals (DHARMa) |
| `check_residuals(model)` | Do the simulated residuals look uniform? | Kolmogorov-Smirnov test |
| `check_overdispersion(model)` | Variance larger than the model allows (GLMs, mixed models) | Pearson chi-squared (Poisson) or simulated residuals |
| `check_zeroinflation(model)` | Excess zeros in count models | Predicted (or simulated) vs observed zeros |
| `check_singularity(model)` | Random-effect variances at zero (mixed models) | Determinant of the random-effects covariance |
| `check_convergence(model)` | Whether the fit converged | R-hat and ESS (Bayesian), relative gradient (mixed models) |
| `icc(model)` | Intraclass correlation (mixed models) | Adjusted and unadjusted ICC |
| `r2(model)` | The R² that suits the model | R², adjusted R², Tjur, Nagelkerke, Nakagawa, Bayesian |
| `r2_tjur`, `r2_nagelkerke`, `r2_coxsnell`, `r2_mcfadden`, `r2_nakagawa`, `r2_bayes` | Pseudo-R², mixed-model and Bayesian R² | |
| `looic(model)` | Leave-one-out cross-validation (Bayesian) | PSIS-LOO, as R's `loo` |
| `model_performance(model)` | Metric table for one model | AIC, AICc, BIC, R², RMSE, Sigma, log loss, scoring rules, PCP |
| `compare_performance(*models)` | Metric tables side by side, optional ranking | Adds Akaike weights |

Every check returns a typed result:

```python
result = check_normality(fit)
print(result)           # Warning: Non-normality of residuals detected (p = 0.021).
result.passed           # False
result.statistic        # 0.9205 (Shapiro-Wilk W)
result.p_value          # 0.0214
result.to_dataframe()   # one-row DataFrame
result.plot()           # Q-Q plot + histogram
```

`check_model()` returns a report you can index like a dictionary:

```python
report["normality"].p_value
report.summary()        # one row per check
report.passed           # True only if every check passed
report.skipped          # checks that don't apply to this model, and why
```

## Supported models

| Backend | Models | Notes |
|---|---|---|
| statsmodels | `OLS`, `WLS` results | Formula and array interfaces |
| statsmodels | `GLM` results: Gaussian, binomial (logistic), Poisson, negative binomial, Gamma, inverse Gaussian | Weights, offsets and exposure honoured |
| statsmodels | `Logit`, `Probit`, `Poisson` and `NegativeBinomial` (NB2, geometric) results | Treated as the equivalent GLM; NB2 like R's `glm.nb()` |
| statsmodels | `MixedLM` results (linear mixed models) | Random intercepts and slopes via `groups=` and `re_formula=` |
| scikit-learn | `LinearRegression`, `Ridge`, `Lasso`, `ElasticNet` (and their `CV` variants) | Pass `X=` and `y=`; penalized fits get a warning that leverage and Cook's distance are approximate |
| scikit-learn | `LogisticRegression` (binary) | Pass `X=` and `y=` |
| scikit-learn | `PoissonRegressor`, `GammaRegressor` | Pass `X=` and `y=`; penalized unless `alpha=0` |
| PyMC, bambi | Fitted models (ArviZ `DataTree`) | Wrap with `bayesian(idata, model)`; Python 3.12+ |
| Anything else | Objects implementing the `ModelAdapter` protocol | Five methods; see below |

Which checks run depends on the model, as in R. Every model also gets a
posterior predictive check:

| Model | Checks |
|---|---|
| Linear (OLS, WLS, Gaussian GLM, sklearn regressors) | normality, heteroscedasticity, collinearity, outliers; autocorrelation when the rows are indexed by time |
| Linear mixed models (`MixedLM`) | normality of residuals and random effects, heteroscedasticity, collinearity, outliers, singularity, convergence |
| Logistic regression | collinearity, outliers, binned residuals, uniformity of simulated residuals |
| Count models (Poisson, negative binomial) | collinearity, outliers, overdispersion, uniformity of simulated residuals |
| Other GLMs | collinearity, outliers, uniformity of simulated residuals |
| Bayesian (PyMC, bambi) | convergence (R-hat, effective sample size) |

Your own model types can plug in by implementing five methods (`model_type`,
`get_response`, `get_fitted`, `get_residuals` and `get_model_matrix`) or by
registering an adapter with `register_adapter()`.

## Coming from R?

Function names and arguments follow R. The common cases:

| R | Python |
|---|---|
| `performance::check_model(m)` | `check_model(m)`, then `.plot()` for the panel |
| `check_normality(m)`, `check_collinearity(m)`, ... | same names: `check_normality(m)`, `check_collinearity(m)`, ... |
| `check_outliers(m, method = "cook")` | `check_outliers(m)` |
| `r2(m)`, `model_performance(m)` | `r2(m)`, `model_performance(m)` |
| `compare_performance(m1, m2, rank = TRUE)` | `compare_performance(m1, m2, rank=True)` |
| `r2_bayes(m)` (rstanarm, brms) | `r2_bayes(bayesian(idata, model))` (PyMC, bambi) |
| `lm(mpg ~ wt + factor(cyl), mtcars)` | `smf.ols("mpg ~ wt + C(cyl)", mtcars).fit()` |
| `lmer(Reaction ~ Days + (Days \| Subject), sleepstudy)` | `smf.mixedlm("Reaction ~ Days", sleepstudy, groups="Subject", re_formula="~Days").fit()` |

The full function-by-function mapping, more model formulas and the few places
where the defaults differ on purpose are in
[Comparison with R](https://github.com/pablo-legeren/checkmodel/blob/main/docs/comparison-with-r.md).

## Why not just use X?

- **`statsmodels`' `summary()`** prints coefficients plus a few omnibus tests,
  but it doesn't check assumptions model-aware, doesn't plot, and doesn't
  compare models. The diagnostic functions it has are spread across
  `statsmodels.stats.*` with different signatures and return types.
- **`sklearn.metrics`** scores predictions. It has no notion of residual
  assumptions, influence or information criteria.
- **`scipy.stats`** gives you the tests (Shapiro-Wilk, ...) but not the
  plumbing: which residuals, standardized how, for which kind of model.
- **Assembling it yourself** means knowing that R's AIC counts the residual
  variance and statsmodels' doesn't, that a logistic regression shouldn't be
  tested for normal residuals, or that VIFs for a categorical predictor need
  the generalized form. `checkmodel` encodes those choices once, the way R
  makes them, and tests them against R.

`checkmodel` is an integration and UX layer: the statistics come from scipy,
statsmodels and numpy.

## Validation against R

`tests/r_reference/generate_reference.R` fits the same models in R 4.6 with
`performance` 0.18.2, `lme4`, `lmtest` and `car`, and writes every statistic the
test suite checks to `tests/r_reference/reference.json`. For Bayesian models,
`generate_bayes_reference.R` fits `rstanarm` models and exports their posterior
draws; the tests run checkmodel on those same draws, so Bayesian R², PSIS-LOO,
WAIC, R-hat and effective sample sizes are compared with `performance`, `loo`
and `posterior` exactly, not just approximately. CI compares against that
file, so it doesn't need R. Covered: leverage, standardized residuals, Cook's
distance, log-likelihoods, every test statistic and p-value, VIFs and their
intervals, outlier thresholds and flags, every R² variant, every
`model_performance()` column, binned residuals, `compare_performance()`
tables, overdispersion, zero-inflation, ICC, Nakagawa's R² and lme4's
influence measures, for 27 models (including negative binomial, probit,
cloglog, Gamma and inverse Gaussian GLMs) over mtcars, iris, warpbreaks,
bioChemists, sleepstudy and synthetic data.

## Contributing

Issues and pull requests are welcome. To set up a development environment:

```bash
pip install -e ".[dev]"
pre-commit install
pytest
```

## License

MIT. `checkmodel` is an independent port and is not affiliated with the
easystats project; the R package is the work of Daniel Lüdecke, Dominique
Makowski, Mattan S. Ben-Shachar, Indrajeet Patil and contributors.
