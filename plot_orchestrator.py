"""
Plot Orchestrator for the Institutional Backtester.

Handles selection-based rendering of visualizations, decoupled from BacktestEngine.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

import matplotlib.pyplot as plt

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
            import visualization as viz
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
                from fundamentals import FundamentalEngine
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
                self._plot_sectoral_allocation(name, custom_map)
            
            if is_sel("Risk Contribution Breakdown"):
                self._plot_risk_contribution_breakdown(name, cfg)
    
    def _plot_sectoral_allocation(
        self,
        strategy_name: str,
        custom_sector_map: Optional[dict]
    ) -> None:
        """Plots sector allocation over time as stacked area chart."""
        import visualization as viz
        
        weights = self.results.get_security_weights(strategy_name)
        sector_map = {
            t: (custom_sector_map.get(t) if custom_sector_map else None)
            or self.metadata.get(t, {}).get('sector', 'Other')
            for t in weights.columns
        }
        sector_weights = weights.T.groupby(sector_map).sum().T
        
        ax = sector_weights.plot(
            kind='area', stacked=True, figsize=(10, 6),
            title=f"Sector: {strategy_name}"
        )
        ax.set_ylabel("Weight")
        plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
        plt.tight_layout()
        import visualization as viz
        viz.make_legend_interactive(plt.gcf())

    def _plot_risk_contribution_breakdown(
        self,
        strategy_name: str,
        config: dict
    ) -> None:
        """
        Plots risk contribution breakdown for a portfolio.
        
        Shows which holdings contribute most to portfolio volatility (MCTR analysis).
        Only runs for portfolios (not benchmarks) since benchmarks are single tickers.
        """
        import numpy as np
        import pandas as pd
        
        # Get portfolio weights
        portfolio = config.get('portfolio', {})
        fixed_weights = portfolio.get('fixed_weights', {})
        
        if not fixed_weights:
            logger.warning(f"No fixed_weights found for {strategy_name}, skipping risk breakdown")
            return
        
        weights = pd.Series(fixed_weights)
        
        # Get the available tickers from data
        valid_tickers = [t for t in weights.index if t in self.data.columns]
        if len(valid_tickers) < 2:
            logger.warning(f"Not enough tickers with price data for {strategy_name}")
            return
        
        weights = weights[valid_tickers]
        weights = weights / weights.sum()  # Renormalize
        prices = self.data[valid_tickers]
        
        # Calculate returns
        returns = prices.pct_change(fill_method=None).dropna()
        
        # Individual volatilities (annualized)
        individual_vol = returns.std() * np.sqrt(252)
        
        # Portfolio volatility
        cov_matrix = returns.cov() * 252  # Annualized covariance
        port_var = float(weights @ cov_matrix @ weights)
        port_vol = np.sqrt(port_var)
        
        # Marginal Contribution to Risk (MCTR)
        mctr = (cov_matrix @ weights) / port_vol
        
        # Percentage Contribution to Risk
        pct_ctr = (weights * mctr) / port_vol
        
        # Build result DataFrame
        result = pd.DataFrame({
            'weight': weights,
            'vol': individual_vol,
            'mctr': mctr,
            'pct_ctr': pct_ctr
        }).sort_values('pct_ctr', ascending=False)
        
        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. Top left: Percentage Contribution to Risk (bar chart)
        ax1 = axes[0, 0]
        colors = ['#e74c3c' if x > result['pct_ctr'].mean() else '#3498db' 
                  for x in result['pct_ctr']]
        result['pct_ctr'].plot(kind='barh', ax=ax1, color=colors)
        ax1.set_xlabel('% Contribution to Total Risk')
        ax1.set_title('Risk Contribution by Holding\n(Red = Above Average)')
        ax1.axvline(result['pct_ctr'].mean(), color='black', linestyle='--', alpha=0.5)
        
        # 2. Top right: Weight vs Risk Contribution scatter
        ax2 = axes[0, 1]
        ax2.scatter(result['weight'] * 100, result['pct_ctr'] * 100, 
                    s=result['vol'] * 500, alpha=0.6, c='#2ecc71')
        for ticker in result.index:
            ax2.annotate(ticker, 
                        (result.loc[ticker, 'weight'] * 100, 
                         result.loc[ticker, 'pct_ctr'] * 100),
                        fontsize=7)
        ax2.plot([0, result['weight'].max() * 100], 
                 [0, result['weight'].max() * 100], 
                 'k--', alpha=0.3, label='1:1 line')
        ax2.set_xlabel('Portfolio Weight (%)')
        ax2.set_ylabel('Risk Contribution (%)')
        ax2.set_title('Weight vs Risk Contribution\n(Size = Volatility)')
        ax2.legend()
        
        # 3. Bottom left: Individual volatility bar
        ax3 = axes[1, 0]
        result_sorted = result.sort_values('vol', ascending=True)
        result_sorted['vol'].plot(kind='barh', ax=ax3, color='#9b59b6')
        ax3.set_xlabel('Annualized Volatility')
        ax3.set_title('Individual Asset Volatility')
        ax3.axvline(port_vol, color='red', linestyle='-', linewidth=2, 
                    label=f'Portfolio Vol: {port_vol:.1%}')
        ax3.legend()
        
        # 4. Bottom right: Top 10 risk contributors pie chart
        # Use absolute values since some contributions can be negative (hedges)
        ax4 = axes[1, 1]
        top10 = result.head(10)['pct_ctr'].abs()  # Use absolute values for pie
        if len(result) > 10:
            other = result.iloc[10:]['pct_ctr'].abs().sum()
            top10 = pd.concat([top10, pd.Series({'Other': other})])
        # Filter out any zero or very small values
        top10 = top10[top10 > 0.001]
        if len(top10) > 0:
            top10.plot(kind='pie', ax=ax4, autopct='%.1f%%', startangle=90)
        ax4.set_ylabel('')
        ax4.set_title('Top 10 Risk Contributors (absolute)')
        
        plt.suptitle(f'{strategy_name} Risk Contribution Breakdown\nPortfolio Volatility: {port_vol:.1%}', 
                     fontsize=14, fontweight='bold')
        plt.tight_layout()

