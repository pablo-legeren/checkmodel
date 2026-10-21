"""Assumption checks: each returns a typed result with ``.plot()`` and ``.to_dataframe()``."""

from ._base import CheckResult
from .autocorrelation import AutocorrelationResult, check_autocorrelation
from .binned import BinnedResidualsResult, binned_residuals
from .collinearity import CollinearityResult, check_collinearity
from .convergence import ConvergenceResult, check_convergence
from .heteroscedasticity import HeteroscedasticityResult, check_heteroscedasticity
from .model_check import ModelCheckReport, check_model
from .normality import NormalityResult, RandomEffectsNormalityResult, check_normality
from .outliers import OutliersResult, check_outliers
from .overdispersion import OverdispersionResult, check_overdispersion
from .singularity import SingularityResult, check_singularity
from .zeroinflation import ZeroInflationResult, check_zeroinflation

__all__ = [
    "AutocorrelationResult",
    "BinnedResidualsResult",
    "CheckResult",
    "CollinearityResult",
    "ConvergenceResult",
    "HeteroscedasticityResult",
    "ModelCheckReport",
    "NormalityResult",
    "OutliersResult",
    "OverdispersionResult",
    "RandomEffectsNormalityResult",
    "SingularityResult",
    "ZeroInflationResult",
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
]
