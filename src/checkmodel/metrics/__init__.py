"""Performance metrics: R² variants, information criteria and model comparison."""

from .bayes import BayesR2Result, looic, r2_bayes
from .compare import compare_performance
from .icc import ICCResult, VarianceComponents, icc, r2_nakagawa, variance_components
from .performance import model_performance
from .r2 import R2Result, r2, r2_coxsnell, r2_mcfadden, r2_nagelkerke, r2_tjur

__all__ = [
    "BayesR2Result",
    "ICCResult",
    "R2Result",
    "VarianceComponents",
    "compare_performance",
    "icc",
    "looic",
    "model_performance",
    "r2",
    "r2_bayes",
    "r2_coxsnell",
    "r2_mcfadden",
    "r2_nagelkerke",
    "r2_nakagawa",
    "r2_tjur",
    "variance_components",
]
