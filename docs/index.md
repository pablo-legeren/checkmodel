# checkmodel

**One-call diagnostics for statistical models in Python**, ported from R's
[`performance`](https://easystats.github.io/performance/) package.

```python
from checkmodel import check_model

report = check_model(fit)   # fit: a statsmodels or scikit-learn model
report.plot()
```

`check_model()` works out what kind of model it was given and runs the
checks that make sense for it: normality, homoscedasticity, multicollinearity,
influential observations and autocorrelation for a linear model; collinearity,
influence and binned residuals for a logistic regression.

![check_model() panel](images/check_model.png)

- **[Getting started](getting-started.md)**: install, check a model, read the
  results, compare models.
- **[Supported models](supported-models.md)**: which check works on which model.
- **[API reference](api-reference.md)**: every function and result type.
- **[Comparison with R](comparison-with-r.md)**: function-by-function mapping
  and the few deliberate differences.

All statistics are tested against R (`performance` 0.18.2, `lme4`, `lmtest`,
`car`, `rstanarm`, `loo` and `posterior`) to a tolerance of 1e-4.
