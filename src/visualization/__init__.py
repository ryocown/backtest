"""
Visualization package for the Institutional Backtester.

This package provides modular, type-safe plotting functions organized by category:
- cursor: Interactive cursor and legend interactivity
- rolling: Rolling metric plots (volatility, sharpe, alpha/beta, etc.)
- static: Static charts (distributions, correlations, captures)
- heatmaps: Heatmap visualizations (monthly returns, correlations)
- surfaces: 3D surface plots (Sharpe robustness, drawdown recovery)
- tda: Topological Data Analysis visualizations

All public functions are re-exported from this module for backward compatibility.
"""
from __future__ import annotations

# Re-export all public functions for backward compatibility
from src.visualization.cursor import (
    FinancialCursor,
    add_financial_cursor,
    make_legend_interactive,
)
from src.visualization.rolling import (
    plot_drawdowns,
    plot_rolling_volatility,
    plot_rolling_sharpe,
    plot_rolling_sortino,
    plot_rolling_beta,
    plot_rolling_alpha,
    plot_rolling_alpha_beta,
    plot_rolling_info_ratio,
)
from src.visualization.static import (
    plot_return_distribution,
    plot_upside_downside_capture,
    plot_correlation_matrix,
    plot_risk_contribution,
    plot_return_attribution,
    plot_valuation_comparison,
)
from src.visualization.heatmaps import (
    plot_monthly_returns_heatmap,
    plot_grouped_correlation_matrix,
)
from src.visualization.surfaces import (
    plot_sharpe_robustness_surface,
    plot_drawdown_recovery_surface,
    plot_historical_pe_trend,
)
from src.visualization.tda import (
    plot_persistence_diagram,
    plot_persistence_barcode,
    plot_tda_metrics_trend,
    plot_3d_simplicial_complex,
)

__all__ = [
    # Cursor
    "FinancialCursor",
    "add_financial_cursor",
    "make_legend_interactive",
    # Rolling
    "plot_drawdowns",
    "plot_rolling_volatility",
    "plot_rolling_sharpe",
    "plot_rolling_sortino",
    "plot_rolling_beta",
    "plot_rolling_alpha",
    "plot_rolling_alpha_beta",
    "plot_rolling_info_ratio",
    # Static
    "plot_return_distribution",
    "plot_upside_downside_capture",
    "plot_correlation_matrix",
    "plot_risk_contribution",
    "plot_return_attribution",
    "plot_valuation_comparison",
    # Heatmaps
    "plot_monthly_returns_heatmap",
    "plot_grouped_correlation_matrix",
    # Surfaces
    "plot_sharpe_robustness_surface",
    "plot_drawdown_recovery_surface",
    "plot_historical_pe_trend",
    # TDA
    "plot_persistence_diagram",
    "plot_persistence_barcode",
    "plot_tda_metrics_trend",
    "plot_3d_simplicial_complex",
]
