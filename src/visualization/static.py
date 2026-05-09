"""
Static chart visualizations for portfolio analysis.

These plots don't involve rolling windows - they show aggregate or distribution data.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.analytics import AnalyticsEngine
from src.visualization.cursor import make_legend_interactive

# Try to import seaborn for better style  
try:
    import seaborn as sns
    HAS_SEABORN = True
    sns.set_theme(style="whitegrid")
except ImportError:
    HAS_SEABORN = False
    sns = None


def plot_return_distribution(res, title: str = "Daily Return Distribution") -> None:
    """
    Plots the distribution (histogram/KDE) of daily returns with statistics.
    
    Args:
        res: bt.Result object.
        title: Chart title.
    """
    rets = res.prices.pct_change().dropna()
    
    plt.figure(figsize=(12, 6))
    if HAS_SEABORN and sns is not None:
        for col in rets.columns:
            sns.kdeplot(rets[col], fill=True, label=col, alpha=0.3)
    else:
        for col in rets.columns:
            plt.hist(rets[col], bins=50, alpha=0.5, label=col)
            
    plt.axvline(0, color='black', linestyle='--')
    plt.title(title)
    plt.xlabel("Daily Return")
    plt.ylabel("Density")
    plt.legend(loc='upper right')  # Fixed location to prevent jumping
    plt.tight_layout()
    make_legend_interactive()


def plot_upside_downside_capture(
    res, 
    benchmark: str = 'SPY', 
    title: str = "Upside/Downside Capture Ratios"
) -> None:
    """
    Plots bar chart of capture ratios relative to a benchmark.
    
    Institutions use this to see 'Beta-up' vs 'Beta-down'.
    
    Args:
        res: bt.Result object.
        benchmark: Benchmark ticker symbol.
        title: Chart title.
    """
    if benchmark not in res.prices.columns:
        return
        
    strategies = [col for col in res.prices.columns if col != benchmark]
    
    captures = []
    for strat in strategies:
        cap = AnalyticsEngine.get_capture_ratios(res.prices[strat], res.prices[benchmark])
        captures.append({
            'Strategy': strat,
            'Upside Capture': cap['upside_capture'],
            'Downside Capture': cap['downside_capture']
        })
    
    df = pd.DataFrame(captures).set_index('Strategy')
    ax = df.plot(kind='bar', figsize=(10, 6), title=f"{title} (Benchmark: {benchmark})")
    ax.set_ylabel("Capture Ratio")
    plt.axhline(1.0, color='gray', linestyle='--')
    plt.xticks(rotation=45)
    plt.tight_layout()


def plot_correlation_matrix(res, title: str = "Strategy Correlation Matrix") -> None:
    """
    Plots a heatmap of return correlations.
    
    Professionals use this to identify hidden concentration/overlap.
    
    Args:
        res: bt.Result object.
        title: Chart title.
    """
    rets = res.prices.pct_change().dropna()
    corr = rets.corr()
    
    plt.figure(figsize=(10, 8))
    if HAS_SEABORN and sns is not None:
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0)
    else:
        plt.imshow(corr, cmap='coolwarm', aspect='auto')
        plt.colorbar()
        
    plt.title(title)
    plt.tight_layout()


def plot_risk_contribution(
    res, 
    strategy_name: str,
    prices: Optional[pd.DataFrame] = None,
    group_map: Optional[dict[str, str]] = None,
    title: str = "Risk Contribution (MCTR)"
) -> None:
    """
    Plots the Percentage Contribution to Risk for each asset (or group) in a strategy.
    
    Institutions use this to see what is 'driving' the volatility.
    
    Args:
        res: bt.Result object.
        strategy_name: Name of the strategy to analyze.
        prices: Asset price DataFrame for covariance calculation.
        group_map: Optional mapping of ticker -> group name for aggregation.
        title: Chart title.
    """
    if strategy_name not in res.prices.columns:
        return
        
    weights = res.get_security_weights(strategy_name).iloc[-1]
    constituents = weights[weights > 0]
    
    if prices is None:
        return  # Need price data to calc covariance
        
    # Align prices with constituents
    valid_tickers = [t for t in constituents.index if t in prices.columns]
    if not valid_tickers:
        return
    
    subset_prices = prices[valid_tickers]
    pct_ctr = AnalyticsEngine.get_risk_contribution(subset_prices, constituents[valid_tickers])
    
    if group_map:
        # Aggregate by group
        group_map_valid = {t: group_map.get(t, 'Other') for t in pct_ctr.index}
        pct_ctr = pct_ctr.groupby(group_map_valid).sum()
        
    plt.figure(figsize=(10, 6))
    sorted_ctr = pct_ctr.sort_values()
    sorted_ctr.plot(kind='barh', color='#e74c3c', alpha=0.8)
    plt.xlabel("Percentage of Total Risk")
    plt.title(title)
    plt.axvline(0, color='black', linewidth=0.8)
    plt.grid(axis='x', linestyle='--', alpha=0.6)
    plt.tight_layout()


def plot_return_attribution(
    prices: pd.DataFrame,
    weights_series: pd.Series,
    group_map: Optional[dict[str, str]] = None,
    title: str = ""
) -> None:
    """
    Performance Attribution Dashboard.
    
    1. Grouped Cumulative Attribution (if group_map provided)
    2. Total Contribution bar chart (Individual Assets)
    
    Args:
        prices: Asset price DataFrame.
        weights_series: Portfolio weights as Series.
        group_map: Optional mapping of ticker -> group name.
        title: Chart title suffix.
    """
    cum_ctr = AnalyticsEngine.get_return_attribution(prices, weights_series)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
    
    # 1. Plot Individual or Grouped Trends
    if group_map:
        group_map_valid = {t: group_map.get(t, 'Other') for t in cum_ctr.columns}
        grouped_cum_ctr = cum_ctr.T.groupby(group_map_valid).sum().T
        grouped_cum_ctr.plot(ax=ax1, title=f"Cumulative Return Attribution (by Group) - {title}")
    else:
        cum_ctr.plot(ax=ax1, title=f"Cumulative Return Attribution (by Asset) - {title}")
        
    ax1.set_ylabel("Cumulative Return %")
    ax1.grid(True, alpha=0.3)
    
    # 2. Final Total Attribution (Bar Chart)
    total_ctr = cum_ctr.iloc[-1]
    total_ctr.sort_values().plot(
        kind='barh', ax=ax2, color='#2ecc71', alpha=0.8, 
        title="Total Contribution by Asset"
    )
    ax2.set_xlabel("Return Contribution (%)")
    ax2.grid(axis='x', linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    make_legend_interactive(fig)


def plot_valuation_comparison(res, metadata: dict, title: str = "Portfolio Valuation Metrics") -> None:
    """
    Compares Trailing PE, Forward PE, and PEG ratios across strategies and benchmarks.
    
    Calculates weighted harmonic mean of constituent metrics.
    
    Args:
        res: bt.Result object.
        metadata: Dict mapping ticker -> metadata dict with valuation fields.
        title: Chart title.
    """
    try:
        metrics = ['trailingPE', 'forwardPE', 'trailingPegRatio']
        plot_labels = ['Trailing P/E', 'Forward P/E', 'PEG Ratio']
        data = []
        
        for col in res.prices.columns:
            try:
                weights = res.get_security_weights(col).iloc[-1]
                constituents = weights[weights > 0.0001]
            except Exception:
                constituents = pd.Series({col: 1.0})

            row = {'Strategy': col}
            for metric in metrics:
                sum_inv_metric = 0.0
                valid_weight = 0.0
                for ticker, weight in constituents.items():
                    m_val = metadata.get(ticker, {}).get(metric)
                    if m_val is not None and not np.isnan(m_val) and m_val > 0.01:
                        sum_inv_metric += weight / m_val
                        valid_weight += weight
                
                row[metric] = valid_weight / sum_inv_metric if sum_inv_metric > 0 else np.nan
            data.append(row)
        
        df = pd.DataFrame(data).set_index('Strategy')
        
        fig, axes = plt.subplots(len(metrics), 1, figsize=(12, 4 * len(metrics)))
        
        for i, metric in enumerate(metrics):
            ax = axes[i] if len(metrics) > 1 else axes
            df[metric].plot(kind='bar', ax=ax, color='#3498db', alpha=0.8)
            ax.set_title(f"{plot_labels[i]} comparison")
            ax.set_ylabel("Ratio")
            ax.grid(axis='y', linestyle='--', alpha=0.6)
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
            
            for p in ax.patches:
                val = p.get_height()
                if not np.isnan(val):
                    ax.annotate(
                        f"{val:.2f}", 
                        (p.get_x() + p.get_width() / 2., val),
                        ha='center', va='center', 
                        xytext=(0, 10), textcoords='offset points'
                    )

        plt.suptitle(title, fontsize=16)
        plt.tight_layout(rect=(0, 0.03, 1, 0.95))
        make_legend_interactive()
    except Exception as e:
        print(f"Valuation comparison plot failed: {e}")


def plot_sectoral_allocation(results, strategy_name, metadata, custom_sector_map):
    import matplotlib.pyplot as plt
    import src.visualization as viz
    
    weights = results.get_security_weights(strategy_name)
    sector_map = {
        t: (custom_sector_map.get(t) if custom_sector_map else None)
        or metadata.get(t, {}).get('sector', 'Other')
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
    viz.make_legend_interactive(plt.gcf())


def plot_risk_contribution_breakdown(strategy_name, config, data):
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import logging
    
    logger = logging.getLogger(__name__)
    
    # Get portfolio weights
    portfolio = config.get('portfolio', {})
    fixed_weights = portfolio.get('fixed_weights', {})
    
    if not fixed_weights:
        logger.warning(f"No fixed_weights found for {strategy_name}, skipping risk breakdown")
        return
    
    weights = pd.Series(fixed_weights)
    
    # Get the available tickers from data
    valid_tickers = [t for t in weights.index if t in data.columns]
    if len(valid_tickers) < 2:
        logger.warning(f"Not enough tickers with price data for {strategy_name}")
        return
    
    weights = weights[valid_tickers]
    weights = weights / weights.sum()  # Renormalize
    prices = data[valid_tickers]
    
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
    ax4 = axes[1, 1]
    top10 = result.head(10)['pct_ctr'].abs()  # Use absolute values for pie
    if len(result) > 10:
        other = result.iloc[10:]['pct_ctr'].abs().sum()
        top10 = pd.concat([top10, pd.Series({'Other': other})])
    top10 = top10[top10 > 0.001]
    if len(top10) > 0:
        top10.plot(kind='pie', ax=ax4, autopct='%.1f%%', startangle=90)
    ax4.set_ylabel('')
    ax4.set_title('Top 10 Risk Contributors (absolute)')
    
    plt.suptitle(f'{strategy_name} Risk Contribution Breakdown\nPortfolio Volatility: {port_vol:.1%}', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

