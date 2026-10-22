# API reference

## Model checks

::: checkmodel.check_model

::: checkmodel.ModelCheckReport

::: checkmodel.check_normality

::: checkmodel.check_heteroscedasticity

::: checkmodel.check_collinearity

::: checkmodel.check_outliers

::: checkmodel.check_autocorrelation

::: checkmodel.binned_residuals

::: checkmodel.check_overdispersion

::: checkmodel.check_zeroinflation

::: checkmodel.check_singularity

::: checkmodel.check_convergence

::: checkmodel.check_predictions

::: checkmodel.check_residuals

## Simulated residuals

::: checkmodel.simulate_residuals

::: checkmodel.SimulatedResiduals

## Result types

::: checkmodel.CheckResult

::: checkmodel.NormalityResult

::: checkmodel.HeteroscedasticityResult

::: checkmodel.CollinearityResult

::: checkmodel.OutliersResult

::: checkmodel.AutocorrelationResult

::: checkmodel.BinnedResidualsResult

::: checkmodel.RandomEffectsNormalityResult

::: checkmodel.OverdispersionResult

::: checkmodel.ZeroInflationResult

::: checkmodel.SingularityResult

::: checkmodel.ConvergenceResult

::: checkmodel.PredictionsResult

::: checkmodel.ResidualsResult

## Performance metrics

::: checkmodel.r2

::: checkmodel.R2Result

::: checkmodel.r2_tjur

::: checkmodel.r2_coxsnell

::: checkmodel.r2_nagelkerke

::: checkmodel.r2_mcfadden

::: checkmodel.r2_nakagawa

::: checkmodel.model_performance

::: checkmodel.compare_performance

## Mixed models

::: checkmodel.icc

::: checkmodel.ICCResult

::: checkmodel.variance_components

::: checkmodel.VarianceComponents

## Bayesian models

::: checkmodel.bayesian

::: checkmodel.BayesianModel
    options:
      members: false

::: checkmodel.r2_bayes

::: checkmodel.BayesR2Result

::: checkmodel.looic

::: checkmodel.LooResult

## Plotting

::: checkmodel.plotting.plot_model_check

::: checkmodel.plotting.plot_uniformity

## Extending checkmodel

::: checkmodel.ModelAdapter

::: checkmodel.ModelType

::: checkmodel.BaseAdapter
    options:
      members: false

::: checkmodel.register_adapter

::: checkmodel.as_adapter

## Errors

::: checkmodel.CheckNotApplicableError

::: checkmodel.UnsupportedModelError
