"""
Plot Orchestrator for the Institutional Backtester.

Handles selection-based rendering of visualizations, decoupled from BacktestEngine.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

import matplotlib.pyplot as plt
from src.visualization.static import plot_sectoral_allocation, plot_risk_contribution_breakdown


if TYPE_CHECKING:
    import pandas as pd
    from bt import Result

logger = logging.getLogger(__name__)


class PlotOrchestrator:
    """
    Orchestrates graph rendering based on user selection from GraphSelector.
    
    This class decouples the visualization logic from BacktestEngine,
    making it easier to test and modify independently.
    """
    
    GLOBAL_PLOTS = [
        "Equity Curve", "Drawdowns", "Rolling Volatility", "Rolling Sharpe",
        "Rolling Sortino", "Return Distribution", "Correlation Matrix",
        "Upside/Downside Capture", "Rolling Info Ratio", "Rolling Alpha/Beta",
        "Valuation Comparison", "Historical P/E Trend", "Global Sector Correlation"
    ]
    
    PER_ENTITY_PLOTS = [
        "Heatmap", "Risk (User Class)", "Risk (Market Sector)",
        "Correlation (User Class)", "Performance Attribution",
        "Sharpe Robustness", "Drawdown-Recovery", "Sectoral Allocation",
        "Risk Contribution Breakdown"  # New: MCTR analysis for portfolios
    ]

    def __init__(
        self,
        results: "Result",
        data: "pd.DataFrame",
        metadata: dict,
        configs: list[dict],
        overrides: dict,
    ) -> None:
        """
        Initialize the orchestrator.
        
        Args:
            results: bt.Result from backtest run.
            data: Price data DataFrame.
            metadata: Dict mapping ticker -> metadata.
            configs: List of portfolio config dicts.
            overrides: CLI overrides dict.
        """
        self.results = results
        self.data = data
        self.metadata = metadata
        self.configs = configs
        self.overrides = overrides
        
        self.strategies = [cfg['_name'] for cfg in configs]
        self.benchmarks = self._get_benchmarks()
    
    def _get_benchmarks(self) -> list[str]:
        """Gets benchmark list from overrides or configs."""
        benchmarks = set()
        if self.overrides.get('benchmarks'):
            benchmarks.update(self.overrides['benchmarks'])
        else:
            for cfg in self.configs:
                benchmarks.update(cfg['settings'].get('benchmarks', []))
        return sorted(list(benchmarks))
    
    def _is_selected(
        self,
        selection: Optional[dict],
        category: str,
        entity: Optional[str],
        plot: str
    ) -> bool:
        """Checks if a plot is selected for rendering."""
        if not selection:
            return True  # Render all if no selection
        if category == 'global':
            return selection['global'].get(plot, True)
        return selection['per_entity'].get(entity, {}).get(plot, True)
    
    def render_all(self, selection: Optional[dict] = None) -> None:
        """
        Renders all selected visualizations.
        
        Args:
            selection: Dict from GraphSelector, or None to render all.
        """
        # Prevent figure memory warning
        plt.close('all')
        
        # Import visualization module
        try:
            import src.visualization as viz
        except ImportError:
            logger.error("Could not import visualization package.")
            return
        
        primary_benchmark = self._get_primary_benchmark()
        market_sector_map = {
            t: self.metadata.get(t, {}).get('sector', 'Other')
            for t in self.data.columns
        }
        
        # --- Global Plots ---
        self._render_global_plots(selection, viz, primary_benchmark, market_sector_map)
        
        # --- Per-Entity Plots ---
        self._render_per_entity_plots(selection, viz, market_sector_map)
        
        plt.show()
    
    def _get_primary_benchmark(self) -> str:
        """Determines the primary benchmark for comparison."""
        primary = self.configs[0]['settings'].get('primary_benchmark')
        if not primary:
            if 'SPY' in self.benchmarks:
                primary = 'SPY'
            elif self.benchmarks:
                primary = self.benchmarks[0]
            else:
                primary = 'SPY'
        return primary
    
    def _render_global_plots(
        self,
        selection: Optional[dict],
        viz,
        primary_benchmark: str,
        market_sector_map: dict
    ) -> None:
        """Renders global plots (applied to all strategies)."""
        is_sel = lambda p: self._is_selected(selection, 'global', None, p)
        
        if is_sel("Equity Curve"):
            self.results.plot(title="Backtest Results")
            viz.make_legend_interactive(plt.gcf())
        
        if is_sel("Drawdowns"):
            viz.plot_drawdowns(self.results)
        if is_sel("Rolling Volatility"):
            viz.plot_rolling_volatility(self.results)
        if is_sel("Rolling Sharpe"):
            viz.plot_rolling_sharpe(self.results)
        if is_sel("Rolling Sortino"):
            viz.plot_rolling_sortino(self.results)
        if is_sel("Return Distribution"):
            viz.plot_return_distribution(self.results)
        if is_sel("Correlation Matrix"):
            viz.plot_correlation_matrix(self.results)
        
        if is_sel("Upside/Downside Capture"):
            viz.plot_upside_downside_capture(self.results, benchmark=primary_benchmark)
        if is_sel("Rolling Info Ratio"):
            viz.plot_rolling_info_ratio(self.results, benchmark=primary_benchmark)
        
        if is_sel("Rolling Alpha/Beta"):
            viz.plot_rolling_alpha_beta(
                self.results.prices[[cfg['_name'] for cfg in self.configs]],
                self.results.prices[[primary_benchmark]],
                title_prefix=f"Strategies vs {primary_benchmark}"
            )
        
        if is_sel("Global Sector Correlation"):
            viz.plot_grouped_correlation_matrix(
                self.data, market_sector_map, title="Correlation: Global Market Sectors"
            )
        
        if is_sel("Valuation Comparison"):
            viz.plot_valuation_comparison(self.results, self.metadata)
        
        # P/E Trend requires FundamentalEngine
        if is_sel("Historical P/E Trend"):
            try:
                from src.fundamentals import FundamentalEngine
                fe = FundamentalEngine()
                viz.plot_historical_pe_trend(self.results, fe, prices=self.data)
            except Exception as e:
                logger.warning(f"Could not render P/E trend: {e}")
    
    def _render_per_entity_plots(
        self,
        selection: Optional[dict],
        viz,
        market_sector_map: dict
    ) -> None:
        """Renders per-entity plots (strategies and benchmarks)."""
        all_entities = self.strategies + self.benchmarks
        
        for name in all_entities:
            if name not in self.results.prices.columns:
                continue
            
            is_sel = lambda p: self._is_selected(selection, 'per_entity', name, p)
            
            if is_sel("Heatmap"):
                viz.plot_monthly_returns_heatmap(self.results, name)
        
        # Strategy-specific plots
        for name in self.strategies:
            if name not in self.results.prices.columns:
                continue
            
            cfg = next((c for c in self.configs if c['_name'] == name), None)
            if not cfg:
                continue
            
            is_sel = lambda p: self._is_selected(selection, 'per_entity', name, p)
            
            custom_map = cfg.get('_custom_sector_map', {})
            fixed_weights = (cfg['portfolio'].get('fixed_weights') or {}).keys()
            user_class_map = custom_map.copy()
            for fw in fixed_weights:
                if fw not in user_class_map:
                    user_class_map[fw] = fw
            
            if is_sel("Risk (User Class)"):
                viz.plot_risk_contribution(
                    self.results, name, prices=self.data,
                    group_map=user_class_map,
                    title=f"Risk: User Asset Class ({name})"
                )
            
            if is_sel("Correlation (User Class)"):
                viz.plot_grouped_correlation_matrix(
                    self.data, user_class_map,
                    title=f"Correlation: User Asset Class ({name})"
                )
            
            if is_sel("Performance Attribution") and '_target_weights' in cfg:
                viz.plot_return_attribution(
                    self.data[list(cfg['_target_weights'].keys())],
                    cfg['_target_weights'],
                    group_map=user_class_map,
                    title=f"Performance Attribution ({name})"
                )
            
            if is_sel("Risk (Market Sector)"):
                viz.plot_risk_contribution(
                    self.results, name, prices=self.data,
                    group_map=market_sector_map,
                    title=f"Risk: Market Sectors ({name})"
                )
            
            if is_sel("Sharpe Robustness"):
                viz.plot_sharpe_robustness_surface(self.results, name)
            
            if is_sel("Drawdown-Recovery"):
                viz.plot_drawdown_recovery_surface(self.results, name)
            
            if is_sel("Sectoral Allocation"):
                plot_sectoral_allocation(self.results, name, self.metadata, custom_map)
            
            if is_sel("Risk Contribution Breakdown"):
                plot_risk_contribution_breakdown(name, cfg, self.data)

    


