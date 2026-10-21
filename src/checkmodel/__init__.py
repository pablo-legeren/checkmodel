"""checkmodel: one-call diagnostics for statistical models.

A Python port of the R package ``performance`` (easystats).

>>> from checkmodel import check_model
>>> report = check_model(fit)  # doctest: +SKIP
>>> report.plot()  # doctest: +SKIP
"""

from ._introspection import BaseAdapter, Term, as_adapter, register_adapter
from ._types import CheckNotApplicableError, ModelAdapter, ModelType, UnsupportedModelError
from .checks import (
    AutocorrelationResult,
    BinnedResidualsResult,
    CheckResult,
    CollinearityResult,
    ConvergenceResult,
    HeteroscedasticityResult,
    ModelCheckReport,
    NormalityResult,
    OutliersResult,
    OverdispersionResult,
    RandomEffectsNormalityResult,
    SingularityResult,
    ZeroInflationResult,
    binned_residuals,
    check_autocorrelation,
    check_collinearity,
    check_convergence,
    check_heteroscedasticity,
    check_model,
    check_normality,
    check_outliers,
    check_overdispersion,
    check_singularity,
    check_zeroinflation,
)
from .metrics import (
    ICCResult,
    R2Result,
    VarianceComponents,
    compare_performance,
    icc,
    model_performance,
    r2,
    r2_coxsnell,
    r2_mcfadden,
    r2_nagelkerke,
    r2_nakagawa,
    r2_tjur,
    variance_components,
)

__version__ = "0.1.0"

__all__ = [
    "AutocorrelationResult",
    "BaseAdapter",
    "BinnedResidualsResult",
    "CheckNotApplicableError",
    "CheckResult",
    "CollinearityResult",
    "ConvergenceResult",
    "HeteroscedasticityResult",
    "ICCResult",
    "ModelAdapter",
    "ModelCheckReport",
    "ModelType",
    "NormalityResult",
    "OutliersResult",
    "OverdispersionResult",
    "R2Result",
    "RandomEffectsNormalityResult",
    "SingularityResult",
    "Term",
    "UnsupportedModelError",
    "VarianceComponents",
    "ZeroInflationResult",
    "__version__",
    "as_adapter",
    "binned_residuals",
    "check_autocorrelation",
    "check_collinearity",
    "check_convergence",
    "check_heteroscedasticity",
    "check_model",
    "check_normality",
    "check_outliers",
    "check_overdispersion",
    "check_singularity",
    "check_zeroinflation",
    "compare_performance",
    "icc",
    "model_performance",
    "r2",
    "r2_coxsnell",
    "r2_mcfadden",
    "r2_nagelkerke",
    "r2_nakagawa",
    "r2_tjur",
    "register_adapter",
    "variance_components",
]
