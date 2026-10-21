# Getting started

## Install

```bash
pip install checkmodel
```

## Check a model

Fit a model as usual and hand it to `check_model()`:

```python
import pandas as pd
import statsmodels.formula.api as smf
from checkmodel import check_model

mtcars = pd.read_csv(
    "https://raw.githubusercontent.com/pablo-legeren/checkmodel/main/tests/data/mtcars.csv",
    index_col=0,
)
fit = smf.ols("mpg ~ wt + hp", data=mtcars).fit()

report = check_model(fit)
print(report)
```

```text
# Model checks: OLS (gaussian, identity link), n = 32

Normality of residuals: Warning: Non-normality of residuals detected (p = 0.021).
Homogeneity of variance: OK: Error variance appears to be homoscedastic (p = 0.402).
Collinearity: OK: Low collinearity (all VIF < 5; max 1.77).
Influential observations: OK: No outliers detected (Cook's distance > 0.808).
Autocorrelation: Warning: Autocorrelated residuals detected (p = 0.041).
```

`report.plot()` draws all the diagnostics in one figure. Each panel says what
a good model looks like in its subtitle.

!!! note "Formal tests and plots"
    A small p-value on 32 observations and a clean-looking Q-Q plot can
    disagree. Read the plots first: the tests are a prompt to look, not a
    verdict.

## Read individual results

The report behaves like a dictionary of check results:

```python
normality = report["normality"]
normality.passed      # False
normality.p_value     # 0.0214
normality.method      # 'Shapiro-Wilk'
normality.plot()      # Q-Q plot and histogram

report.summary()      # one row per check, as a DataFrame
report.skipped        # checks that do not apply, with the reason
```

Every check is also available on its own:

```python
from checkmodel import check_collinearity, check_outliers

check_collinearity(fit).to_dataframe()   # VIF table, one row per term
check_outliers(fit, threshold="4/n")      # use the 4/n rule instead of R's default
```

## scikit-learn models

scikit-learn estimators do not store their training data, so pass it:

```python
from sklearn.linear_model import LinearRegression

X, y = mtcars[["wt", "hp"]], mtcars["mpg"]
model = LinearRegression().fit(X, y)
check_model(model, X=X, y=y)
```

The numbers are the same as for the statsmodels fit above. Penalized models
(`Ridge`, `Lasso`, `ElasticNet`, penalized `LogisticRegression`) are
supported too; they have no likelihood-based criteria, so `AIC` and `BIC` are
left out of their metric tables.

## Logistic regression

```python
import statsmodels.api as sm

logit = smf.glm("vs ~ mpg + wt", data=mtcars, family=sm.families.Binomial()).fit()
check_model(logit)
```

Normality and homoscedasticity are not expected for a binary outcome, so they
are not run. Binned residuals take their place.

## Count data

Poisson regressions are checked for overdispersion; zero-inflation has its own
check:

```python
from checkmodel import check_zeroinflation

bio = pd.read_csv(
    "https://raw.githubusercontent.com/pablo-legeren/checkmodel/main/tests/data/biochemists.csv"
)
pois = smf.glm(
    "art ~ fem + mar + kid5 + phd + ment", data=bio, family=sm.families.Poisson()
).fit()
check_model(pois)
check_zeroinflation(pois)
```

```text
# Model checks: GLM (poisson, log link), n = 915

Collinearity: OK: Low collinearity (all VIF < 5; max 1.29).
Influential observations: OK: No outliers detected (Cook's distance > 0.84).
Overdispersion: Warning: Overdispersion detected (dispersion ratio = 1.829, p < .001).

Warning: Model is underfitting zeros (probable zero-inflation): 275 observed, 191 predicted, ratio 0.69.
```

## Mixed models

statsmodels `MixedLM` results get the residual checks plus checks on the random
effects, singularity and convergence:

```python
from checkmodel import icc

sleep = pd.read_csv(
    "https://raw.githubusercontent.com/pablo-legeren/checkmodel/main/tests/data/sleepstudy.csv"
)
mixed = smf.mixedlm("Reaction ~ Days", sleep, groups="Subject", re_formula="~Days").fit()
check_model(mixed)
```

```text
# Model checks: MixedLM (gaussian, identity link), n = 180

Normality of residuals: Warning: Non-normality of residuals detected (p < .001).
Normality of random effects: OK: Random effects 'Subject: (Intercept)' appear as normally distributed (p = 0.428).
  OK: Random effects 'Subject: Days' appear as normally distributed (p = 0.898).
Homogeneity of variance: Warning: Heteroscedasticity (non-constant error variance) detected (p < .001).
Influential observations: Warning: 4 outliers detected: cases 7, 56, 59, 130 (Cook's distance > 0.696).
Singularity: OK: The random-effects structure is not singular.
Convergence: OK: Model converged.

Skipped:
  collinearity: Not enough model terms to check for multicollinearity (need at least two).
```

`icc()` and `r2()` decompose the variance as R's `performance` does:

```python
icc(mixed)   # Adjusted ICC: 0.722, Unadjusted ICC: 0.521
r2(mixed)    # Conditional R2: 0.799, Marginal R2: 0.279
```

!!! tip "Singular fits"
    If a random-effect variance belongs at zero, statsmodels' default
    optimizer can stop just short of it. `check_convergence()` reports this;
    refit with `.fit(method="powell")` and `check_singularity()` will then
    flag the singular fit, as lme4 does.

## Performance metrics

```python
from checkmodel import model_performance, compare_performance, r2

r2(fit)
model_performance(fit)
```

```text
       AIC     AICc      BIC     R2  R2_adjusted   RMSE  Sigma
0  156.652  158.134  162.515  0.827        0.815  2.469  2.593
```

Compare several models; keyword names label the rows:

```python
small = smf.ols("mpg ~ wt", data=mtcars).fit()
large = smf.ols("mpg ~ wt + hp + qsec", data=mtcars).fit()

compare_performance(small=small, medium=fit, large=large, rank=True)
```

With `rank=True` the table gains a `Performance_Score` (the mean of every
index rescaled to 0–1, higher is better) and is sorted by it.
