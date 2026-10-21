"""Matplotlib plots for every check and the combined ``check_model()`` panel."""

from .diagnostic_plots import (
    PALETTE,
    plot_autocorrelation,
    plot_binned_residuals,
    plot_check,
    plot_collinearity,
    plot_homogeneity,
    plot_linearity,
    plot_model_check,
    plot_normality,
    plot_outliers,
    plot_overdispersion,
    plot_qq,
    plot_random_effects,
    plot_random_effects_term,
    plot_zeroinflation,
)

__all__ = [
    "PALETTE",
    "plot_autocorrelation",
    "plot_binned_residuals",
    "plot_check",
    "plot_collinearity",
    "plot_homogeneity",
    "plot_linearity",
    "plot_model_check",
    "plot_normality",
    "plot_outliers",
    "plot_overdispersion",
    "plot_qq",
    "plot_random_effects",
    "plot_random_effects_term",
    "plot_zeroinflation",
]
