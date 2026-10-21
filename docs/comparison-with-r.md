# Comparison with R

`checkmodel` follows `performance` 0.18.2 (with `insight` 1.6.0) and is tested
against it. This page maps the functions one to one and lists every place where
the Python version deliberately behaves differently.

## Function by function

| R | Python | Same numbers? |
|---|---|---|
| `check_model(m)` | `check_model(m)`, then `.plot()` | Same checks per model type; see the panel notes below |
| `check_normality(m)` | `check_normality(m)` | Yes: Shapiro-Wilk on `rstandard()`, KS against N(0, 1) from 5000 residuals |
| `check_heteroscedasticity(m)` | `check_heteroscedasticity(m)` | Yes, and equals `car::ncvTest(m)` |
| `check_collinearity(m)` | `check_collinearity(m)` | Yes, including generalized VIFs, `SE_factor` and the VIF intervals |
| `check_outliers(m, method = "cook")` | `check_outliers(m)` | Yes, including the default threshold |
| `check_autocorrelation(m)` | `check_autocorrelation(m, method="bootstrap")` | Same statistic; the p-value is random in both |
| `lmtest::dwtest(m, alternative = "two.sided")` | `check_autocorrelation(m)` | Yes (the default) |
| `binned_residuals(m)` | `binned_residuals(m)` | Yes |
| `r2(m)` | `r2(m)` | Yes |
| `r2_tjur(m)` | `r2_tjur(m)` | Yes |
| `r2_coxsnell(m)` | `r2_coxsnell(m)` | Yes |
| `r2_nagelkerke(m)` | `r2_nagelkerke(m)` | Yes |
| `r2_mcfadden(m)` | `r2_mcfadden(m)` | Yes (`R2` and `R2_adjusted`) |
| `model_performance(m)` | `model_performance(m)` | Yes, same columns |
| `compare_performance(m1, m2)` | `compare_performance(m1, m2)` | Yes, including Akaike weights |
| `compare_performance(..., rank = TRUE)` | `compare_performance(..., rank=True)` | Yes |
| `check_overdispersion(m)` (Poisson GLM) | `check_overdispersion(m)` | Yes |
| `check_zeroinflation(m)` (Poisson GLM) | `check_zeroinflation(m)` | Yes |
| `check_singularity(m)` | `check_singularity(m)` | Same rule; see the note on optimizers below |
| `check_convergence(m)` | `check_convergence(m)` | Same verdict; the gradient is on statsmodels' scale |
| `check_normality(m)` (`lmer`) | `check_normality(m)` | Yes, on lme4-style studentized residuals |
| `check_normality(m, effects = "random")` | `check_normality(m, effects="random")` | Yes |
| `check_heteroscedasticity(m)` (`lmer`) | `check_heteroscedasticity(m)` | No, on purpose; see below |
| `icc(m)` | `icc(m)` | Yes |
| `r2_nakagawa(m)`, `r2(m)` (`lmer`) | `r2_nakagawa(m)`, `r2(m)` | Yes |
| `model_performance(m)` (`lmer`) | `model_performance(m)` | Yes, including `estimator` |
| `hatvalues()`, `rstudent()`, `cooks.distance()` (`lmer`) | adapter `.hat`, `.std_residuals`, `.cooks_distance` | Yes |
| `check_predictions(m)` | `check_predictions(m)` | Same idea; adds p-values, see below |
| `r2_bayes(m)` | `r2_bayes(m)` | Yes, on the same draws (median, MAD, HDI) |
| `looic(m)`, `loo::loo(m)` | `looic(m)` | Yes, including Pareto k |
| `loo::waic(m)` | in `model_performance()` | Yes |
| `posterior::rhat()`, `ess_bulk()`, `ess_tail()` | `check_convergence(m).to_dataframe()` | Yes |
| `model_performance(m)` (`stanreg`) | `model_performance(m)` | Yes except `RMSE`; see below |

## Model formulas

| R | statsmodels |
|---|---|
| `lm(y ~ x, data)` | `smf.ols("y ~ x", data).fit()` |
| `lm(y ~ x, data, weights = w)` | `smf.wls("y ~ x", data, weights=data.w).fit()` |
| `glm(y ~ x, binomial, data)` | `smf.glm("y ~ x", data, family=sm.families.Binomial()).fit()` |
| `glm(y ~ x, poisson, data)` | `smf.glm("y ~ x", data, family=sm.families.Poisson()).fit()` |
| `factor(cyl)` | `C(cyl)` |
| `lmer(y ~ x + (1 \| g), data)` | `smf.mixedlm("y ~ x", data, groups="g").fit()` |
| `lmer(y ~ x + (x \| g), data)` | `smf.mixedlm("y ~ x", data, groups="g", re_formula="~x").fit()` |
| `lmer(..., REML = FALSE)` | `.fit(reml=False)` |
| `stan_glm(y ~ x, data)` | `bmb.Model("y ~ x", data)`, then `bayesian(model.fit(), model)` |
| `stan_lmer(y ~ x + (1 \| g), data)` | `bmb.Model("y ~ x + (1\|g)", data)`, then `bayesian(...)` |

## Where the defaults differ, and why

**Autocorrelation p-value.** R's `check_autocorrelation()` gets its p-value by
resampling residuals 1000 times, so it changes between runs. `checkmodel`
defaults to the deterministic p-value of `lmtest::dwtest()`: exact (Imhof's
method) below 100 observations and the normal approximation above.
`method="bootstrap"` reproduces R's procedure, with `random_state` for
repeatable results.

**Normality of Gaussian GLMs.** R refuses to test the residuals of any `glm`
object for normality, even a Gaussian one with identity link. `checkmodel`
treats such a model like the equivalent linear model, which it is.

**Pass/fail for collinearity.** R prints VIFs grouped as low (< 5), moderate
(5–10) and high (≥ 10) without a verdict. `checkmodel` sets `passed = False`
only when a term reaches `threshold` (10 by default) and reports moderate
collinearity as a note.

**Term order.** patsy sorts categorical terms before numeric ones in the design
matrix. `checkmodel` puts terms back in the order they appear in the formula,
like R, but names follow patsy (`C(cyl)` rather than `factor(cyl)`).

**Observation labels.** R numbers outliers from 1. `checkmodel` reports the
labels of the data's index, for example car names for mtcars, or 0-based
positions for unlabelled data.

**Heteroscedasticity of mixed models.** For `lmer` fits,
`performance::check_heteroscedasticity()` takes the residual standard
deviation from `insight::get_deviance()`, which for these models is a
likelihood deviance (about 1794 for `sleepstudy`) rather than the residual sum
of squares (about 155 697). The residual SD it uses is therefore far too small
and the test returns p ≈ 0 regardless of the data. `checkmodel` uses the
model's sigma instead, which is what the function does for `lm` fits. The test
suite checks the result against that computation done in R.

**Optimizers and singular fits.** lme4 reaches a zero variance exactly when
the data call for it. statsmodels' default optimizer (BFGS) can stop just short
of that boundary and still report `converged=True`; `check_convergence()`
catches it through the relative gradient and suggests refitting, for example
with `.fit(method="powell")`. Once the boundary is reached,
`check_singularity()`, `icc()` and `r2_nakagawa()` behave as in R.

**Count models other than Poisson.** For negative binomial and mixed count
models R checks overdispersion and zero-inflation with simulated (DHARMa)
residuals. These are not ported yet, so the checks raise
`CheckNotApplicableError` for those models.

**Bayesian models.**

- *Predictive checks report p-values.* R's `check_predictions()` only plots.
  `checkmodel` also computes posterior predictive p-values for the mean, the SD
  and (for counts) the share of zeros, and fails the check when one falls
  outside [0.025, 0.975]. For frequentist models the replications are
  simulated from the fitted distribution, as R's `simulate()` does.
- *RMSE.* R computes it from `fitted()`, i.e. the linear predictor at the
  posterior median of the coefficients, which only `rstanarm` provides.
  `checkmodel` uses the posterior median of each expected value, which is
  defined for any PyMC or bambi model. The difference is small (2.466 vs
  2.469 for the mtcars test model).
- *Not ported:* `R2_adjusted` (LOO-R², which relies on R's random number
  stream), the Bayesian `ICC`, and `Log_loss` / `Score_*` for Bayesian models.
- *Convergence verdict.* R reports R-hat and ESS without a verdict;
  `checkmodel` flags R-hat >= 1.01 or bulk/tail ESS below 100 per chain
  (Vehtari et al., 2021).
- *Relative efficiency in LOO.* `loo()` on an `rstanarm` fit, and therefore
  `model_performance()` in R, uses `r_eff = 1`; `checkmodel` does the same by
  default. `looic(m, use_relative_eff=True)` matches
  `loo(m, r_eff = relative_eff(...))` instead.

## Faithfully reproduced quirks

These match R exactly, even though you might not expect them:

- **AIC and BIC count the residual variance** as a parameter for Gaussian
  models, like `AIC(lm)`. statsmodels' `OLSResults.aic` does not, so its
  value is 2 lower.
- **The default Cook's distance threshold** is the median of an
  F(k, n − k) distribution, where k is the number of numeric variables in
  the model data (response included, except for binomial models). It is not
  the 4/n rule of thumb; pass `threshold="4/n"` for that.
- **`Score_log` for logistic regression** is computed with the binomial size
  set to the number of successes, as `performance_score()` does. It can
  underflow to `-inf`, as in R.
- **Mixed models are compared with REML AIC by default.** As in
  `model_performance(estimator = "REML")`, AIC, AICc and BIC use the REML
  criterion, evaluated at the ML estimates if the model was fitted by ML. Pass
  `estimator="ML"` for likelihood-ratio style comparisons of fixed effects.
- **A singular mixed model keeps a missing `R2_conditional`** column in
  `model_performance()`, while its `ICC` column is dropped, as in R.
- **Standardized residuals of a GLM** are Pearson-based, matching
  `rstandard()` in the R version the reference values come from (4.6.1),
  whose default type is `"pearson"`.

## Not yet ported

Generalized linear mixed models, simulated (DHARMa-style) residuals and the
checks that depend on them (overdispersion and zero-inflation of negative
binomial and mixed count models), ICC and R² confidence intervals for
frequentist models, `MixedLM` variance components (`vc_formula`), and the
Bayesian pieces listed above.
